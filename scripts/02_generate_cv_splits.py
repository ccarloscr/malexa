"""
02_generate_cv_splits.py

Generate selected method for cross-validation split indices for one
task and write them to a JSON file.  This script runs ONCE per task before
any model training, so all models for a given task are evaluated on identical
folds, so model comparison is fair and reproducible.

What this script does:
  - Reads the clean expression matrix and clinical metadata written by
    01_load_clean_data.py.
  - Extracts and validates the label column for the requested task.
  - For mutation-status tasks: drops samples where the status is NaN (unknown
    after 01_load_clean_data.py standardisation).
  - Writes a JSON file containing, for each fold, the list of sample IDs
    assigned to the training set and the test set.  Indices are sample IDs
    (strings), not integer positions, so they remain valid even if the matrix
    is later reordered.

Output JSON schema
------------------
{
  "task":       "<task name>",
  "label_col":  "<column name>",
  "n_splits":   5,
  "random_seed": 42,
  "label_counts": {"0": 212, "1": 289},
  "samples_dropped_unknown_label": ["TCGA-XX-YYYY", ...],
  "folds": [
    {
      "fold": 0,
      "train": ["TCGA-...", ...],
      "test":  ["TCGA-...", ...]
    },
    ...
  ]
}

Run as a Snakemake script or standalone from the CLI for testing.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold,
                                     RepeatedStratifiedKFold,
                                     StratifiedShuffleSplit)


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(log_path=None):
    logger = logging.getLogger("generate_cv_splits")
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
# label extraction
# --------------------------------------------------------------------------- #
def extract_labels(clinical, task_name, task_config, config, logger):
    """Extract and validate labels for *task_name* from the clean clinical table.

    Returns
    -------
    labels : pd.Series (dtype int, index = sample_id)
        Only samples with a known, valid label are included.
    dropped_samples : list[str]
        Sample IDs excluded due to unknown / unmapped label values.
    """
    label_col = task_config["label_col"]

    if label_col not in clinical.columns:
        raise ValueError(
            f"Task '{task_name}': label column '{label_col}' not found in "
            f"clinical table. Available columns: {list(clinical.columns)}"
        )

    labels = clinical[label_col].copy()
    # guard against any accidental non-binary values surviving script 01
    valid_values = {0, 1, 0.0, 1.0}
    bad_mask = labels.notna() & ~labels.isin(valid_values)
    if bad_mask.any():
        bad_vals = sorted(labels[bad_mask].unique().tolist())
        logger.warning(
            f"Task '{task_name}': {bad_mask.sum()} samples have unexpected "
            f"non-binary label values {bad_vals} treating as unknown and "
            f"dropping."
        )
        labels[bad_mask] = np.nan


    # ------------------------------------------------------------------ #
    # drop unknowns, cast to int
    # ------------------------------------------------------------------ #
    unknown_mask = labels.isna()
    dropped_samples = labels.index[unknown_mask].tolist()
    if dropped_samples:
        logger.info(
            f"Task '{task_name}': dropping {len(dropped_samples)} samples "
            f"with unknown/unmappable labels."
        )

    labels = labels.dropna().astype(int)

    # sanity: require at least two classes
    unique_classes = labels.unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Task '{task_name}': only one class present after filtering "
            f"({unique_classes}). Cannot run stratified CV."
        )

    logger.info(
        f"Task '{task_name}': {len(labels)} samples with known labels. "
        f"Class counts: {labels.value_counts().to_dict()}"
    )
    return labels, dropped_samples


# --------------------------------------------------------------------------- #
# CV split generation
# --------------------------------------------------------------------------- #
SUPPORTED_CV_METHODS = {"StratifiedKFold", "RepeatedStratifiedKFold", "StratifiedShuffleSplit"}


def generate_splits(labels, cv_config, logger):
    """Run the configured CV strategy and return a list of fold dicts.

    Each dict contains:
      {"repeat": int, "fold": int, "train": [sample_id, ...], "test": [sample_id, ...]}

    Supported methods (cv_config["method"]):
      - StratifiedKFold
      - RepeatedStratifiedKFold
      - StratifiedShuffleSplit
    """

    method = cv_config.get("method")
    if method not in SUPPORTED_CV_METHODS:
        raise ValueError(
            f"Unsupported CV method '{method}'. "
            f"Choose one of: {sorted(SUPPORTED_CV_METHODS)}"
        )

    seed = cv_config.get("random_seed", 123)

    if method     == "StratifiedKFold":
        cv        = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=True, random_state=seed)
        n_splits  = cv_config["n_splits"]
        n_repeats = 1

    elif method   == "RepeatedStratifiedKFold":
        cv        = RepeatedStratifiedKFold(n_splits=cv_config["n_splits"], n_repeats=cv_config["n_repeats"], random_state=seed)
        n_splits  = cv_config["n_splits"]
        n_repeats = cv_config["n_repeats"]

    elif method   == "StratifiedShuffleSplit":
        cv        = StratifiedShuffleSplit(n_splits=cv_config["n_splits"], test_size=cv_config["test_size"], random_state=seed)
        n_splits  = cv_config["n_splits"]
        n_repeats = 1

    sample_ids    = labels.index.astype(str).tolist()
    y             = labels.values

    folds = []
    
    for idx, (train_pos, test_pos) in enumerate(cv.split(sample_ids, y)):
        repeat = idx // n_splits
        fold   = idx  % n_splits

        train_ids = [sample_ids[i] for i in train_pos]
        test_ids  = [sample_ids[i] for i in test_pos]

        train_counts = pd.Series(y[train_pos]).value_counts().to_dict()
        test_counts  = pd.Series(y[test_pos]).value_counts().to_dict()
        logger.info(
            f"  [{method}] Repeat {repeat} Fold {fold}: "
            f"train={len(train_ids)} {train_counts} "
            f"test={len(test_ids)} {test_counts}"
        )

        folds.append({
            "split_idx": idx,
            "repeat": repeat,
            "fold": fold,
            "train": train_ids,
            "test": test_ids
        })

    return folds, n_splits, n_repeats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(expression_path, clinical_path, task_name, config,
         out_splits, log_path=None):
    
    logger = get_logger(log_path)

    logger.info(f"Task: {task_name}")
    logger.info(f"Loading expression index from: {expression_path}")

    # Read only the sample_id column from the expression matrix
    expression_samples = pd.read_csv(expression_path, index_col=0, nrows=0).columns.tolist()
    logger.info(f"Expression matrix has {len(expression_samples)} samples.")

    logger.info(f"Loading clinical metadata from: {clinical_path}")
    clinical = pd.read_csv(clinical_path, index_col=0)
    logger.info(f"Clinical table: {clinical.shape[0]} samples x {clinical.shape[1]} columns")

    # Restrict clinical data to samples present in the expression matrix
    common_samples = [s for s in expression_samples if s in clinical.index]
    n_missing = len(expression_samples) - len(common_samples)
    if n_missing:
        logger.warning(
            f"{n_missing} expression samples not found in clinical table — "
            f"they will be excluded."
        )
    clinical = clinical.loc[common_samples]

    task_config = config["tasks"][task_name]
    cv_config   = config["cv"]
    method      = cv_config.get("method")

    labels, dropped_samples = extract_labels(clinical, task_name, task_config, config, logger)

    logger.info(f"Generating CV splits (method={method}) ...")
    folds, n_splits, n_repeats = generate_splits(labels, cv_config, logger)


    label_counts = {str(k): int(v) for k, v in labels.value_counts().items()}

    result = {
        "task":          task_name,
        "label_col":     task_config["label_col"],
        "cv_method":     method,
        "n_splits":      n_splits,
        "n_repeats":     n_repeats,
        "total_splits":  n_splits * n_repeats,
        "random_seed":   cv_config.get("random_seed", 123),
        "label_counts":  label_counts,
        "samples_dropped_unknown_label": dropped_samples,
        "labels":        {str(k): int(v) for k, v in labels.items()},
        "folds":         folds
    }

    Path(out_splits).parent.mkdir(parents=True, exist_ok=True)
    with open(out_splits, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Wrote splits to: {out_splits}")


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # =========================================================================
    # EXECUTION MODE 1: Snakemake Integration
    # =========================================================================
    if "snakemake" in globals():
        main(
            expression_path  = snakemake.input.expression,
            clinical_path    = snakemake.input.clinical,
            task_name        = snakemake.params.task,
            config           = snakemake.config,
            out_splits       = snakemake.output.splits,
            log_path         = snakemake.log[0] if len(snakemake.log) else None,
        )

    # =========================================================================
    # EXECUTION MODE 2: Standalone CLI
    # Parameter definition: CLI Arguments (Priority 1) > config.yaml (Priority 2)
    # =========================================================================  
    else:
        import argparse
        import yaml

        # --- Parse CLI Arguments ---
        parser = argparse.ArgumentParser(description=__doc__)

        # Mandatory arguments
        parser.add_argument("--task",       required=True,  help="Task name as defined in config.yaml (e.g. cancer_stage)")
        parser.add_argument("--config",     required=True,  help="Path to config.yaml")

        # Optional arguments (if not defined --> resolved by the config.yaml)
        parser.add_argument("--expression", required=False, help="Path to clean expression CSV")
        parser.add_argument("--clinical",   required=False, help="Path to clean clinical CSV")
        parser.add_argument("--out-splits", required=False, help="Output path for the splits JSON file")  
        parser.add_argument("--log",        default=None,   help="Path to log file")
        parser.add_argument("--cv-method",  default=None,   help="Override cv.method from config.yaml",
                            choices=["StratifiedKFold", "RepeatedStratifiedKFold", "StratifiedShuffleSplit"]
                            )
        args = parser.parse_args()

        # --- Load Base Configuration File ---
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # Patch cv.method if provided via CLI
        if args.cv_method:
            cfg["cv"]["method"] = args.cv_method

        # --- Resolve Output Directory and Steps Config ---
        out_dir = cfg["data"].get("output_dir", "results")
        step01_cfg = cfg["outputs"]["step_01"]
        step02_cfg = cfg["outputs"]["step_02"]

        # --- Resolve Input Paths (Outputs from Step 01) ---
        expression_path = args.expression or step01_cfg["expression"].format(output_dir=out_dir)
        clinical_path   = args.clinical or step01_cfg["clinical"].format(output_dir=out_dir)

        # --- Resolve Output Path ---
        out_splits = args.out_splits or step02_cfg["splits"].format(output_dir=out_dir, task=args.task)

        # Run
        main(
            expression_path  = expression_path,
            clinical_path    = clinical_path,
            task_name        = args.task,
            config           = cfg,
            out_splits       = out_splits,
            log_path         = args.log,
        )
