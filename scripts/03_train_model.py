"""
03_train_model.py

Train and evaluate one (task, model, fold) combination.

Called once per cell of the task × model × fold grid by Snakemake rule
`train_model`.  Everything that must NOT leak from test into training happens
here, inside the fold boundary:

  1. Restrict expression matrix to train/test sample IDs from the splits JSON.
  2. Extract labels from the clean clinical CSV (already {0,1} or binarized
     by 02_generate_cv_splits.py — we just look them up by sample ID).
  3. Fold-local feature selection on training samples only:
       a. CPM normalisation + median CPM filter  (removes unexpressed genes)
       b. Variance percentile filter             (removes low-information genes)
  4. log1p transform the filtered count matrix (VST-lite, avoids full DESeq2
     dependency; appropriate for downstream linear models and trees alike).
  5. StandardScaler fit on training samples only (required for SVM / ElasticNet;
     harmless for tree ensembles but kept for pipeline uniformity).
  6. Instantiate the estimator from config `estimator_class` + `fixed_params`.
     LinearSVC is wrapped in CalibratedClassifierCV so predict_proba is
     available for ROC-AUC / PR-AUC.
  7. Hyperparameter search (GridSearchCV or RandomizedSearchCV) with an inner
     3-fold stratified CV on the training fold only.
  8. Evaluate the best estimator on the held-out test fold.
  9. Extract feature importances (coefficients for linear models, feature
     importances for trees) and map back to gene names.
 10. Persist: metrics JSON, predictions CSV, feature importances CSV, model PKL.

Outputs (paths from Snakemake rule `train_model`):
  metrics.json            — scalar evaluation metrics for this fold
  predictions.csv         — sample-level predicted probabilities + true labels
  feature_importances.csv — gene-level importance scores for this fold
  model.pkl               — fitted best estimator (full pipeline)

Runnable as a Snakemake `script:` or standalone from the CLI.
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
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
def load_fold_data(expression_path, clinical_path, splits_path,
                   task_name, task_config, fold_idx, logger):
    """Return X_train, X_test, y_train, y_test as aligned DataFrames/Series.

    Labels come from the clean clinical CSV (sample IDs as index).  The splits
    JSON already contains only samples that had a valid label (unknowns were
    dropped by 02_generate_cv_splits.py), so no label filtering is needed here.
    """
    # --- splits ---
    with open(splits_path) as f:
        splits = json.load(f)

    fold_data = splits["folds"][fold_idx]
    train_ids = fold_data["train"]
    test_ids  = fold_data["test"]
    logger.info(
        f"Fold {fold_idx}: {len(train_ids)} train / {len(test_ids)} test samples."
    )

    # --- expression: load only needed columns (memory efficient) ---
    # Read header first to know which columns to pull
    header = pd.read_csv(expression_path, index_col=0, nrows=0).columns.tolist()
    needed = sorted(set(train_ids + test_ids) & set(header))

    counts = pd.read_csv(expression_path, index_col=0, usecols=[""] + needed
                         if "" in pd.read_csv(expression_path, nrows=0).columns
                         else None)
    # Fallback: read everything if the trick above fails (index col name varies)
    try:
        counts = pd.read_csv(expression_path, index_col=0, usecols=lambda c: c == c or c in needed)
    except Exception:
        counts = pd.read_csv(expression_path, index_col=0)

    # Keep only the fold samples, in split order
    train_missing = [s for s in train_ids if s not in counts.columns]
    test_missing  = [s for s in test_ids  if s not in counts.columns]
    if train_missing or test_missing:
        raise ValueError(
            f"Fold {fold_idx}: {len(train_missing)} train / {len(test_missing)} "
            f"test sample IDs from splits JSON not found in expression matrix. "
            f"First few missing: {(train_missing + test_missing)[:5]}"
        )

    # genes × samples -> transpose to samples × genes for sklearn
    X_train = counts[train_ids].T
    X_test  = counts[test_ids].T

    # --- labels ---
    clinical = pd.read_csv(clinical_path, index_col=0)
    label_col = task_config["label_col"]
    y_train = clinical.loc[train_ids, label_col].astype(int)
    y_test  = clinical.loc[test_ids,  label_col].astype(int)

    logger.info(
        f"Train class counts: {y_train.value_counts().to_dict()}  |  "
        f"Test class counts: {y_test.value_counts().to_dict()}"
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# fold-local feature selection (leakage-safe)
# --------------------------------------------------------------------------- #
def compute_cpm(counts_df):
    """Counts per million, column-wise (counts_df: samples × genes)."""
    lib_sizes = counts_df.sum(axis=1)          # total counts per sample
    # avoid division by zero for pathological samples
    lib_sizes = lib_sizes.replace(0, np.nan)
    cpm = counts_df.div(lib_sizes, axis=0) * 1e6
    return cpm


def select_features(X_train, X_test, fs_config, logger):
    """Apply CPM + variance filters fit exclusively on X_train.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame, samples × genes
    fs_config : dict  (config['feature_selection'])

    Returns
    -------
    X_train_sel, X_test_sel : pd.DataFrame
        Filtered, with same gene columns.
    selected_genes : list[str]
    """
    min_cpm       = fs_config.get("min_cpm",          1.0)
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
    var_mask = train_var >= var_threshold
    X_train = X_train.loc[:, var_mask]
    X_test  = X_test.loc[:,  var_mask]
    logger.info(
        f"Variance filter (bottom {min_var_pct}% removed): "
        f"{var_mask.sum()} / {cpm_mask.sum()} genes retained."
    )

    selected_genes = X_train.columns.tolist()
    logger.info(f"Feature selection: {len(selected_genes)} genes selected for modelling.")
    return X_train, X_test, selected_genes


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def log1p_normalise(X_train, X_test):
    """log1p on raw counts (applied after feature selection).

    Fit nothing on test data — log1p is a fixed transformation, no leakage.
    """
    return np.log1p(X_train.values), np.log1p(X_test.values)


# --------------------------------------------------------------------------- #
# estimator construction
# --------------------------------------------------------------------------- #
def build_estimator(model_name, model_config, random_seed, logger):
    """Instantiate the estimator from config, wrapping LinearSVC if needed.

    Returns the base estimator (before Pipeline wrapping).
    """
    class_path = model_config["estimator_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    EstimatorClass = getattr(importlib.import_module(module_path), class_name)

    fixed_params = model_config.get("fixed_params", {}) or {}
    # Propagate random_seed from config if the estimator accepts random_state
    # and it wasn't already overridden in fixed_params
    import inspect
    sig = inspect.signature(EstimatorClass.__init__)
    if "random_state" in sig.parameters and "random_state" not in fixed_params:
        fixed_params = {**fixed_params, "random_state": random_seed}

    estimator = EstimatorClass(**fixed_params)
    logger.info(f"Instantiated {class_name} with fixed_params={fixed_params}")

    # LinearSVC doesn't implement predict_proba; wrap for probability calibration
    if class_name == "LinearSVC":
        estimator = CalibratedClassifierCV(estimator, cv=3, method="sigmoid")
        logger.info("Wrapped LinearSVC in CalibratedClassifierCV (sigmoid).")

    return estimator


# --------------------------------------------------------------------------- #
# hyperparameter search
# --------------------------------------------------------------------------- #
def run_hyperparameter_search(estimator, param_grid, model_config,
                               X_train_arr, y_train, random_seed, n_threads, logger):
    """Wrap estimator in a Pipeline + scaler, run CV search on training fold.

    The scaler is inside the pipeline so it is re-fit on each inner fold,
    preventing any leakage from inner-fold validation samples.

    Returns the fitted SearchCV object.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    estimator),
    ])

    strategy   = model_config.get("search_strategy", "grid")
    scoring    = model_config.get("scoring", "roc_auc")
    inner_cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

    common_kwargs = dict(
        estimator=pipeline,
        param_grid=param_grid,
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
            f"inner_cv={inner_cv.n_splits}-fold"
        )
        search = RandomizedSearchCV(
            **common_kwargs,
            n_iter=n_iter,
            random_state=random_seed,
        )
    else:
        logger.info(
            f"GridSearchCV: scoring={scoring}, inner_cv={inner_cv.n_splits}-fold"
        )
        search = GridSearchCV(**common_kwargs)

    search.fit(X_train_arr, y_train.values)
    logger.info(f"Best params : {search.best_params_}")
    logger.info(f"Best CV {scoring}: {search.best_score_:.4f}")
    return search


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
# Registry of supported metrics — maps config name -> callable(y_true, y_score)
# All callables accept (y_true, y_score) where y_score is continuous probability
# of the positive class, except where noted.
_METRIC_REGISTRY = {
    "roc_auc":            lambda yt, ys: roc_auc_score(yt, ys),
    "average_precision":  lambda yt, ys: average_precision_score(yt, ys),
    "balanced_accuracy":  lambda yt, ys: balanced_accuracy_score(yt, (ys >= 0.5).astype(int)),
    "f1_weighted":        lambda yt, ys: f1_score(yt, (ys >= 0.5).astype(int),
                                                   average="weighted", zero_division=0),
    "matthews_corrcoef":  lambda yt, ys: matthews_corrcoef(yt, (ys >= 0.5).astype(int)),
}


def evaluate(best_pipeline, X_test_arr, y_test, pos_label, metric_names, logger):
    """Score the fitted pipeline on the test fold.

    Returns
    -------
    metrics : dict[str, float]
    y_prob  : np.ndarray  (probability of pos_label for each test sample)
    """
    # predict_proba returns [prob_class0, prob_class1]
    # For binary tasks pos_label is always 1 in our config, so index 1 is safe.
    # Guard against edge cases where classes might be [0] only.
    classes = best_pipeline.classes_ if hasattr(best_pipeline, "classes_") else None

    proba = best_pipeline.predict_proba(X_test_arr)
    if classes is not None and 1 in classes:
        pos_idx = list(classes).index(pos_label)
    else:
        pos_idx = 1   # fallback
    y_prob = proba[:, pos_idx]

    metrics = {}
    for metric in metric_names:
        fn = _METRIC_REGISTRY.get(metric)
        if fn is None:
            logger.warning(f"Unknown metric '{metric}' — skipping.")
            continue
        try:
            metrics[metric] = float(fn(y_test.values, y_prob))
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
        except Exception as exc:
            logger.warning(f"  {metric} failed: {exc}")
            metrics[metric] = float("nan")

    return metrics, y_prob


# --------------------------------------------------------------------------- #
# feature importance extraction
# --------------------------------------------------------------------------- #
def extract_importances(best_pipeline, selected_genes, model_name, logger):
    """Extract gene-level importance scores from the fitted pipeline step 'clf'.

    Supported patterns:
      - feature_importances_  : RandomForest, XGBoost
      - coef_                 : LogisticRegression (elasticnet)
      - CalibratedClassifierCV -> base_estimator.coef_ : LinearSVC

    Returns
    -------
    pd.DataFrame with columns [gene, importance, abs_importance]
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

    elif hasattr(clf_step, "estimator"):
        # CalibratedClassifierCV wrapping LinearSVC
        base = clf_step.estimator
        if hasattr(base, "coef_"):
            importances = base.coef_.ravel()
        elif hasattr(clf_step, "calibrated_classifiers_"):
            # average coef across calibrated folds
            coefs = [cc.estimator.coef_.ravel()
                     for cc in clf_step.calibrated_classifiers_
                     if hasattr(cc.estimator, "coef_")]
            if coefs:
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
def main(expression_path, clinical_path, splits_path,
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

    # ------------------------------------------------------------------ #
    # 1. Load fold data
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = load_fold_data(
        expression_path, clinical_path, splits_path,
        task_name, task_config, fold_idx, logger
    )

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
    estimator = build_estimator(model_name, model_config, random_seed, logger)

    # ------------------------------------------------------------------ #
    # 5. Hyperparameter search (inner CV on training fold only)
    # ------------------------------------------------------------------ #
    # param_grid keys already use pipeline prefix "clf__" per config convention
    param_grid = model_config.get("param_grid", {}) or {}

    search = run_hyperparameter_search(
        estimator, param_grid, model_config,
        X_train_arr, y_train, random_seed, n_threads, logger
    )

    best_pipeline = search.best_estimator_

    # ------------------------------------------------------------------ #
    # 6. Evaluate on test fold
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on held-out test fold ...")
    metrics, y_prob = evaluate(
        best_pipeline, X_test_arr, y_test, pos_label, metric_names, logger
    )

    # augment metrics with provenance fields
    metrics.update({
        "task":          task_name,
        "model":         model_name,
        "fold":          int(fold_idx),
        "n_train":       int(len(y_train)),
        "n_test":        int(len(y_test)),
        "n_genes_selected": len(selected_genes),
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
        "y_pred":       (y_prob >= 0.5).astype(int),
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
            clinical_path=snakemake.input.clinical,
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
        parser.add_argument("--clinical",     required=True)
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
            clinical_path=args.clinical,
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
