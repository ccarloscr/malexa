"""
03_train_model.py

Train and evaluate one (task, model, fold) combination.
Called once per combination of task - model - fold by Snakemake rule 'train_model'.
Everything that must NOT leak from test into training happens here, inside the fold boundary:

  1. load_fold_data
        - Restrict expression matrix to train/test sample IDs from the splits JSON.
        - Extract clean, binarized labels.
  2. compute_cpm
        - CPM normalization (per sample) on training samples only.
  3. select_features
        - CPM median filter on training samples only.
        - Variance filter on training samples only.
  4. log1p normalize
        - log1p transform on raw counts.
  5. build_estimator
        - Instantiate the estimator from config (estimator_class + fixed_params).
  6. run_hyperparameter_search
        - Hyperparameter search (GridSearchCV or RandomizedSearchCV) with an
        inner 3-fold stratified CV on the training fold only.
  7. evaluate
        - Score best pipeline on held-out test fold.
  8. extract_importances
        - Get feature importances and map back to gene names.

Outputs (paths from Snakemake rule `train_model`):
  metrics.json            — scalar evaluation metrics for this fold
  predictions.csv         — sample-level predicted probabilities + true labels
  feature_importances.csv — gene-level importance scores for this fold
  model.pkl               — fitted best estimator (full pipeline)

Runnable as a Snakemake script or standalone from the CLI.
"""

import importlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import (average_precision_score, 
                             balanced_accuracy_score,
                             f1_score,matthews_corrcoef,
                             roc_auc_score
                             )
from sklearn.model_selection import (GridSearchCV,
                                     RandomizedSearchCV,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold,
                                     StratifiedShuffleSplit
                                     )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(log_path=None):
    logger = logging.getLogger("train_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_fold_data(expression_path, splits_path, task_name,
                   task_config, fold_idx, logger):
    """Return X_train, X_test, y_train, y_test as aligned dataframes
  
    Labels come from the clean clinical CSV (sample IDs as index). All JSON splits
    contain samples with a valid label (required in 02_generate_cv_splits.py), so
    not necessary to filter labels in this script.
    """

    # --- splits ---
    with open(splits_path) as f:
        splits = json.load(f)

    folds = splits["folds"]

    # Look up the fold by its explicit split_idx field rather than list position
    matches = [f for f in folds if f.get("split_idx") == fold_idx]
    if matches:
        fold_data = matches[0]
    elif 0 <= fold_idx < len(folds) and folds[fold_idx].get("split_idx") is None:
        fold_data = folds[fold_idx]
    else:
        raise ValueError(
            f"Could not find fold_idx={fold_idx} in splits JSON "
            f"'{splits_path}' ({len(folds)} folds available). The splits "
            f"file may be stale, reordered, or generated for a different "
            f"cv configuration."
        )

    train_ids = fold_data["train"]
    test_ids  = fold_data["test"]
    logger.info(
        f"Fold {fold_idx}: {len(train_ids)} train / {len(test_ids)} test samples."
    )

    # --- expression: load only needed columns ---
    all_cols       = pd.read_csv(expression_path, nrows=0).columns.tolist()
    index_col_name = all_cols[0]   # first column is the gene index
    needed         = set(train_ids + test_ids)
    cols_to_load   = [index_col_name] + [c for c in all_cols[1:] if c in needed]
    counts         = pd.read_csv(expression_path, usecols=cols_to_load, index_col=0)    

    # keep only the fold samples, in split order
    train_missing  = [s for s in train_ids if s not in counts.columns]
    test_missing   = [s for s in test_ids  if s not in counts.columns]
    if train_missing or test_missing:
        raise ValueError(
            f"Fold {fold_idx}: {len(train_missing)} train / {len(test_missing)} "
            f"test sample IDs from splits JSON not found in expression matrix. "
            f"First few missing: {(train_missing + test_missing)[:5]}"
        )

    # genes × samples -> transpose to samples × genes
    X_train        = counts[train_ids].T
    X_test         = counts[test_ids].T

    # --- clinical labels (pre-binarized) ---
    labels_map = splits.get("labels")
    if labels_map is None:
        raise ValueError(
            "Splits JSON is missing the 'labels' field. Please re-run 02_generate_cv_splits.py "
        )

    y_train = pd.Series({sid: labels_map[sid] for sid in train_ids}, name=task_config["label_col"]).astype(int)
    y_test  = pd.Series({sid: labels_map[sid] for sid in test_ids},  name=task_config["label_col"]).astype(int)

    logger.info(
        f"Train class counts: {y_train.value_counts().to_dict()}  |  "
        f"Test class counts: {y_test.value_counts().to_dict()}"
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# manual gene exclusion (e.g. driver genes that would  dominate feature
# importance for a mutation-status)
# --------------------------------------------------------------------------- #
def exclude_configured_genes(X_train, X_test, exclude_genes, logger):
    """Drop genes listed in task_config['exclude_genes'] before feature selection

    This runs BEFORE the CPM/variance filters and BEFORE log1p, so excluded
    genes never enter feature selection and can never be reported in
    feature_importances.csv. Applied identically to train and test.
    """
    if not exclude_genes:
        return X_train, X_test

    present = [g for g in exclude_genes if g in X_train.columns]
    missing = [g for g in exclude_genes if g not in X_train.columns]

    if present:
        logger.info(
            f"Excluding {len(present)} configured gene(s) before feature "
            f"selection: {present}"
        )
        X_train = X_train.drop(columns=present)
        X_test = X_test.drop(columns=present)

    if missing:
        # Warning if exluded_genes is not empty but it is not found in the
        # expression matrix
        logger.warning(
            f"exclude_genes configured but not found in expression matrix "
            f"(check ID format, e.g. version suffix): {missing}"
        )

    return X_train, X_test


# --------------------------------------------------------------------------- #
# fold-local feature selection (leakage-safe)
# --------------------------------------------------------------------------- #

def compute_cpm(counts_df):
    """ Calculates CPM per sample from the raw counts expression matrix

    Computes counts per million (CPMs), column-wise (counts_df: samples × genes).
    Returns the normalized CPM matrix.
    """
    lib_sizes = counts_df.sum(axis=1)             # total counts per sample
    lib_sizes = lib_sizes.replace(0, np.nan)      # prevents error on empty samples
    cpm = counts_df.div(lib_sizes, axis=0) * 1e6  # compute CPMs
    return cpm


def select_features(X_train, X_test, fs_config, logger):
    """Applies the expression and variance filters exclusively on the train set
    """

    # --- Define parameters ---
    min_cpm       = fs_config.get("min_cpm", 1.0)
    min_var_pct   = fs_config.get("min_variance_pct", 10)
    n_genes_start = X_train.shape[1]

    # --- CPM filter: median CPM >= min_cpm across training samples ---
    cpm_train = compute_cpm(X_train)
    median_cpm = cpm_train.median(axis=0)
    cpm_mask = median_cpm >= min_cpm
    X_train = X_train.loc[:, cpm_mask]
    X_test  = X_test.loc[:,  cpm_mask]
    logger.info(
        f"CPM filter (median >= {min_cpm}): "
        f"{cpm_mask.sum()} / {n_genes_start} genes retained."
    )

    # --- variance filter: drop bottom min_var_pct % by variance on training ---
    if X_train.shape[1] == 0:
        logger.warning("No genes survived CPM filter. Skipping variance filter.")
        return X_train, X_test, []
    
    train_var = X_train.var(axis=0)
    var_threshold = np.percentile(train_var.values, min_var_pct)
    var_mask = train_var >= var_threshold # Mask defined by the variance on the training set
    
    X_train = X_train.loc[:, var_mask]    # Filter from training set applied on train set
    X_test  = X_test.loc[:,  var_mask]    # Filter from training set applied on test set
    logger.info(
        f"Variance filter (bottom {min_var_pct}% removed): "
        f"{var_mask.sum()} / {cpm_mask.sum()} genes retained."
    )

    selected_genes = X_train.columns.tolist()
    logger.info(f"Feature selection: {len(selected_genes)} genes selected for modelling.")
    
    return X_train, X_test, selected_genes


# --------------------------------------------------------------------------- #
# normalization (log1p: logarithm of 1 plus)
# --------------------------------------------------------------------------- #
def log1p_normalise(X_train, X_test):
    """log1p transformation on raw counts

    Applied after feature selection (filtered by CPM and Variance) to avoid leakage.
    """
    return np.log1p(X_train.values), np.log1p(X_test.values)


# --------------------------------------------------------------------------- #
# estimator construction
# --------------------------------------------------------------------------- #
def build_estimator(model_name, model_config, random_seed, logger, inner_n_splits=None):
    """Instantiate the estimator from config, wrapping LinearSVC if needed.

    inner_n_splits, if given, drives the number of CV folds used internally
    by CalibratedClassifierCV (kept in sync with the same inner CV used for
    hyperparameter search).

    Returns the base estimator (before Pipeline wrapping).
    """
    class_path = model_config["estimator_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    EstimatorClass = getattr(importlib.import_module(module_path), class_name)

    fixed_params = model_config.get("fixed_params", {}) or {}
    # Propagate random_seed from config if the estimator accepts random_state
    import inspect
    sig = inspect.signature(EstimatorClass.__init__)
    if "random_state" in sig.parameters and "random_state" not in fixed_params:
        fixed_params = {**fixed_params, "random_state": random_seed}

    # If estimator requests n_jobs>1 --> force n_jobs=1 to avoid CPU oversubscription
    # since the outer search already parallelizes across CV folds via n_jobs=n_threads
    if "n_jobs" in sig.parameters and fixed_params.get("n_jobs", 1) != 1:
        logger.info(
            f"Overriding fixed_params n_jobs={fixed_params['n_jobs']} -> 1 "
            f"for {class_name}; parallelism is controlled by the search's "
            f"n_jobs (n_threads) instead, to avoid CPU oversubscription."
        )
        fixed_params = {**fixed_params, "n_jobs": 1}

    estimator = EstimatorClass(**fixed_params)
    logger.info(f"Instantiated {class_name} with fixed_params={fixed_params}")

    # LinearSVC doesn't implement predict_proba; wrap for probability calibration
    if isinstance(estimator, LinearSVC):
        calibration_cv = inner_n_splits or 3
        estimator = CalibratedClassifierCV(estimator, cv=calibration_cv, method="sigmoid")
        logger.info(
            f"Wrapped LinearSVC in CalibratedClassifierCV (sigmoid, cv={calibration_cv})."
        )

    return estimator


# --------------------------------------------------------------------------- #
# hyperparameter search
# --------------------------------------------------------------------------- #
def run_hyperparameter_search(estimator, param_grid, model_config,
                               X_train_arr, y_train, random_seed, n_threads, logger,
                               cv_config=None):
    
    """Wrap estimator in a Pipeline + scaler, run CV search on training fold.

    The scaler is inside the pipeline so it is re-fit on each inner fold,
    preventing any leakage from inner-fold validation samples.

    The inner CV method is independently configurable from the outer CV
    (config['cv']['inner_method']), defaulting to StratifiedKFold if not set.

    Returns the fitted SearchCV object.
    """
  
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    estimator),
    ])

    strategy   = model_config.get("search_strategy", "grid")
    scoring    = model_config.get("scoring", "roc_auc")

    cv_config        = cv_config or {}
    inner_method     = cv_config.get("inner_method", "StratifiedKFold")
    inner_n_splits   = cv_config.get("inner_n_splits")
    inner_n_repeats  = cv_config.get("inner_n_repeats")
    inner_test_size  = cv_config.get("inner_test_size")

    SUPPORTED_INNER_CV_METHODS = {"StratifiedKFold", "RepeatedStratifiedKFold", "StratifiedShuffleSplit"}
    if inner_method not in SUPPORTED_INNER_CV_METHODS:
        raise ValueError(
            f"Unsupported cv.inner_method '{inner_method}'. "
            f"Choose one of: {sorted(SUPPORTED_INNER_CV_METHODS)}"
        )
    if not inner_n_splits:
        raise ValueError("cv.inner_n_splits is required for the inner CV.")

    if inner_method == "StratifiedKFold":
        inner_cv = StratifiedKFold(
            n_splits=inner_n_splits, shuffle=True, random_state=random_seed
        )

    elif inner_method == "RepeatedStratifiedKFold":
        if not inner_n_repeats:
            raise ValueError(
                "cv.inner_n_repeats is required when "
                "cv.inner_method == 'RepeatedStratifiedKFold'."
            )
        inner_cv = RepeatedStratifiedKFold(
            n_splits=inner_n_splits, n_repeats=inner_n_repeats, random_state=random_seed
        )

    elif inner_method == "StratifiedShuffleSplit":
        if inner_test_size is None:
            raise ValueError(
                "cv.inner_test_size is required when "
                "cv.inner_method == 'StratifiedShuffleSplit'."
            )
        inner_cv = StratifiedShuffleSplit(
            n_splits=inner_n_splits, test_size=inner_test_size, random_state=random_seed
        )

    logger.info(
        f"Inner CV: {inner_method}(n_splits={inner_n_splits}"
        f"{f', n_repeats={inner_n_repeats}' if inner_method == 'RepeatedStratifiedKFold' else ''}"
        f"{f', test_size={inner_test_size}' if inner_method == 'StratifiedShuffleSplit' else ''})"
    )

    common_kwargs = dict(
        estimator=pipeline,
        scoring=scoring,
        cv=inner_cv,
        refit=True,
        n_jobs=n_threads,
        verbose=1,
    )

    if strategy == "random":
        n_iter = model_config.get("n_iter", 20)
        logger.info(
            f"RandomizedSearchCV: n_iter={n_iter}, scoring={scoring}, "
            f"inner_cv={inner_n_splits}-fold"
        )
        search = RandomizedSearchCV(
            **common_kwargs,
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=random_seed,
        )
    else:
        logger.info(
            f"GridSearchCV: scoring={scoring}, inner_cv={inner_n_splits}-fold"
        )
        search = GridSearchCV(
            **common_kwargs,
            param_grid=param_grid
        )

    search.fit(X_train_arr, y_train.values)
    logger.info(f"Best params : {search.best_params_}")
    logger.info(f"Best CV {scoring}: {search.best_score_:.4f}")
    return search


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
# Registry of supported metrics:
_METRIC_REGISTRY = {
    "roc_auc":            lambda yt, ys, thr: roc_auc_score(yt, ys),
    "average_precision":  lambda yt, ys, thr: average_precision_score(yt, ys),
    "balanced_accuracy":  lambda yt, ys, thr: balanced_accuracy_score(yt, (ys >= thr).astype(int)),
    "f1_weighted":        lambda yt, ys, thr: f1_score(yt, (ys >= thr).astype(int),
                                                        average="weighted", zero_division=0),
    "matthews_corrcoef":  lambda yt, ys, thr: matthews_corrcoef(yt, (ys >= thr).astype(int)),
}


def evaluate(best_pipeline, X_test_arr, y_test, pos_label, metric_names, logger, threshold=0.5):
 
    """Score the fitted pipeline on the test fold.

    Returns
    -------
    metrics : dict[str, float]
    y_prob  : np.ndarray  (probability of pos_label for each test sample)
    """

    classes = best_pipeline.classes_ if hasattr(best_pipeline, "classes_") else None

    proba = best_pipeline.predict_proba(X_test_arr)
    if classes is not None and pos_label in classes:
        pos_idx = list(classes).index(pos_label)
    else:
        # Fallback: sklearn Pipeline exposes classes_ from the final step for
        # sklearn>=1.0, so this should be rare. Warn loudly since assuming
        # index 1 can silently give wrong probabilities if it's ever hit.
        logger.warning(
            f"Could not determine class index for pos_label={pos_label} "
            f"(classes_={classes}). Falling back to column index 1 — "
            f"verify this is correct for this model/sklearn version."
        )
        pos_idx = 1   # fallback
    y_prob = proba[:, pos_idx]

    metrics = {}
    for metric in metric_names:
        fn = _METRIC_REGISTRY.get(metric)
        if fn is None:
            logger.warning(f"Unknown metric '{metric}' — skipping.")
            continue
        try:
            metrics[metric] = float(fn(y_test.values, y_prob, threshold))
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
        except Exception as exc:
            logger.warning(f"  {metric} failed: {exc}")
            metrics[metric] = float("nan")

    return metrics, y_prob


# --------------------------------------------------------------------------- #
# feature importance extraction
# --------------------------------------------------------------------------- #
def extract_importances(best_pipeline, selected_genes, model_name, logger):
    """Extract gene-level importance scores

    Supported patterns:
      - feature_importances_  : RandomForest, XGBoost
      - coef_                 : LogisticRegression (elasticnet)
      - CalibratedClassifierCV -> base_estimator.coef_ : LinearSVC

    Returns a pd.DataFrame with columns [gene, importance, abs_importance]
    sorted descending by abs_importance.
    """
    clf_step = best_pipeline.named_steps["clf"]

    importances = None

    # --- tree-based: feature_importances_ ---
    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_

    # --- linear: coef_ (direct or via CalibratedClassifierCV) ---
    elif hasattr(clf_step, "coef_"):
        coef = clf_step.coef_
        importances = coef.ravel()   # binary: shape (1, n_features) or (n_features,)

    elif hasattr(clf_step, "calibrated_classifiers_"):
        # CalibratedClassifierCV wrapping LinearSVC
        coefs = []
        for cc in clf_step.calibrated_classifiers_:
            sub_est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if sub_est is not None and hasattr(sub_est, "coef_"):
                coefs.append(sub_est.coef_.ravel())
        if coefs:
            # average coef across calibrated folds
            importances = np.mean(coefs, axis=0)

    if importances is None:
        logger.warning(
            f"Could not extract feature importances for model '{model_name}'. "
            f"Writing empty importances file."
        )
        return pd.DataFrame(columns=["gene", "importance", "abs_importance"])

    if len(importances) != len(selected_genes):
        logger.error(
            f"Importance vector length ({len(importances)}) != number of selected "
            f"genes ({len(selected_genes)}). Writing empty importances file."
        )
        return pd.DataFrame(columns=["gene", "importance", "abs_importance"])

    df = pd.DataFrame({
        "gene":            selected_genes,
        "importance":      importances,
        "abs_importance":  np.abs(importances),
    }).sort_values("abs_importance", ascending=False).reset_index(drop=True)

    logger.info(
        f"Top-5 features: "
        + ", ".join(f"{r.gene}({r.importance:.3f})" for _, r in df.head(5).iterrows())
    )
    return df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(expression_path, splits_path,
         task_name, model_name, fold_idx,
         config,
         out_metrics, out_predictions, out_importances, out_model_pkl,
         log_path=None, n_threads=1):

    logger = get_logger(log_path)
    logger.info(
        f"=== train_model | task={task_name} | model={model_name} | "
        f"fold={fold_idx} ==="
    )

    task_config  = config["tasks"][task_name]
    model_config = config["models"][model_name]
    fs_config    = config.get("feature_selection", {})
    eval_config  = config.get("evaluation", {})
    random_seed  = config["cv"]["random_seed"]
    pos_label    = task_config.get("pos_label", 1)
    metric_names = eval_config.get("metrics", ["roc_auc"])
    threshold    = eval_config.get("threshold", 0.5)

    # ------------------------------------------------------------------ #
    # 1. Load fold data
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = load_fold_data(
        expression_path, splits_path,
        task_name, task_config, fold_idx, logger
    )

    # Drop manually-excluded genes
    exclude_genes = task_config.get("exclude_genes", []) or []
    X_train, X_test = exclude_configured_genes(X_train, X_test, exclude_genes, logger)

    # ------------------------------------------------------------------ #
    # 2. Fold-local feature selection (on training data only)
    # ------------------------------------------------------------------ #
    X_train_sel, X_test_sel, selected_genes = select_features(
        X_train, X_test, fs_config, logger
    )

    if len(selected_genes) == 0:
        raise RuntimeError(
            f"No genes survived feature selection for task={task_name}, "
            f"model={model_name}, fold={fold_idx}. "
            f"Check your feature_selection thresholds in config.yaml."
        )

    # ------------------------------------------------------------------ #
    # 3. log1p transform
    # ------------------------------------------------------------------ #
    X_train_arr, X_test_arr = log1p_normalise(X_train_sel, X_test_sel)
    logger.info(
        f"After feature selection + log1p: X_train={X_train_arr.shape}, "
        f"X_test={X_test_arr.shape}"
    )

    # ------------------------------------------------------------------ #
    # 4. Build estimator
    # ------------------------------------------------------------------ #
    estimator = build_estimator(
        model_name, model_config, random_seed, logger,
        inner_n_splits=config["cv"].get("inner_n_splits"),
    )
    # ------------------------------------------------------------------ #
    # 5. Hyperparameter search (inner CV on training fold only)
    # ------------------------------------------------------------------ #
    param_grid = model_config.get("param_grid", {}) or {}

    search = run_hyperparameter_search(
        estimator, param_grid, model_config,
        X_train_arr, y_train, random_seed, n_threads, logger,
        cv_config=config["cv"]
    )

    best_pipeline = search.best_estimator_

    # ------------------------------------------------------------------ #
    # 6. Evaluate on test fold
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on held-out test fold ...")
    metrics, y_prob = evaluate(
        best_pipeline, X_test_arr, y_test, pos_label, metric_names, logger,
        threshold=threshold,
    )

    # augment metrics with provenance fields
    metrics.update({
        "task":          task_name,
        "model":         model_name,
        "fold":          int(fold_idx),
        "n_train":       int(len(y_train)),
        "n_test":        int(len(y_test)),
        "n_genes_selected": len(selected_genes),
        "excluded_genes": exclude_genes,
        "classification_threshold": threshold,
        "best_params":   search.best_params_,
        "best_inner_cv_score": float(search.best_score_),
    })

    # ------------------------------------------------------------------ #
    # 7. Feature importances
    # ------------------------------------------------------------------ #
    importances_df = extract_importances(
        best_pipeline, selected_genes, model_name, logger
    )
    importances_df["task"]  = task_name
    importances_df["model"] = model_name
    importances_df["fold"]  = int(fold_idx)

    # ------------------------------------------------------------------ #
    # 8. Predictions table
    # ------------------------------------------------------------------ #
    predictions_df = pd.DataFrame({
        "sample_id":    y_test.index.tolist(),
        "y_true":       y_test.values,
        "y_prob":       y_prob,
        "y_pred":       (y_prob >= threshold).astype(int),
        "task":         task_name,
        "model":        model_name,
        "fold":         int(fold_idx),
    })

    # ------------------------------------------------------------------ #
    # 9. Write outputs
    # ------------------------------------------------------------------ #
    for path in [out_metrics, out_predictions, out_importances, out_model_pkl]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Wrote metrics     -> {out_metrics}")

    predictions_df.to_csv(out_predictions, index=False)
    logger.info(f"Wrote predictions -> {out_predictions}")

    importances_df.to_csv(out_importances, index=False)
    logger.info(f"Wrote importances -> {out_importances}")

    with open(out_model_pkl, "wb") as f:
        pickle.dump(best_pipeline, f)
    logger.info(f"Wrote model PKL   -> {out_model_pkl}")

    logger.info("=== Done ===")


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if "snakemake" in globals():
        main(
            expression_path=snakemake.input.expression,
            splits_path=snakemake.input.splits,
            task_name=snakemake.params.task,
            model_name=snakemake.params.model,
            fold_idx=int(snakemake.params.fold),
            config=snakemake.config,
            out_metrics=snakemake.output.metrics,
            out_predictions=snakemake.output.predictions,
            out_importances=snakemake.output.importances,
            out_model_pkl=snakemake.output.model_pkl,
            log_path=snakemake.log[0] if len(snakemake.log) else None,
            n_threads=snakemake.threads,
        )
    else:
        import argparse
        import yaml

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--expression",   required=True)
        parser.add_argument("--splits",       required=True)
        parser.add_argument("--task",         required=True)
        parser.add_argument("--model",        required=True)
        parser.add_argument("--fold",         required=True, type=int)
        parser.add_argument("--config",       required=True)
        parser.add_argument("--out-metrics",     required=True)
        parser.add_argument("--out-predictions", required=True)
        parser.add_argument("--out-importances", required=True)
        parser.add_argument("--out-model-pkl",   required=True)
        parser.add_argument("--log",          default=None)
        parser.add_argument("--threads",      default=1, type=int)
        args = parser.parse_args()

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        main(
            expression_path=args.expression,
            splits_path=args.splits,
            task_name=args.task,
            model_name=args.model,
            fold_idx=args.fold,
            config=cfg,
            out_metrics=args.out_metrics,
            out_predictions=args.out_predictions,
            out_importances=args.out_importances,
            out_model_pkl=args.out_model_pkl,
            log_path=args.log,
            n_threads=args.threads,
        )
