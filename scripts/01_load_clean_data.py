"""
01_load_clean_data.py

Load raw RNA-seq counts and clinical metadata, align samples by
sample_id, and clean missing/invalid values.

This step is intentionally generic and task-agnostic:
  - It does NOT binarize cancer stage (that's config-driven, per-task logic
    that belongs in 02_generate_cv_splits.py).
  - It does NOT filter low-expression/low-variance genes (that must happen
    inside each CV fold, on training data only, to avoid leakage — see
    03_train_model.py).

What it DOES do:
  - Align samples common to both the counts matrix and clinical table.
  - Drop genes/samples with excessive missing raw counts (config-controlled
    thresholds), impute rare residual NaNs with 0.
  - Standardize mutation-status encodings (WT/Mutant, yes/no, 0/1, ...) to
    a clean {0, 1} column, leaving true unknowns as NaN.
  - Tidy free-text stage strings (whitespace/case), leaving vocabulary as-is.
  - Write a QC report documenting what was dropped and why, for auditability.

Runnable both as a Snakemake `script:` and standalone from the CLI for
testing outside the pipeline.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(log_path=None):
    logger = logging.getLogger("load_clean_data")
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
# loading
# --------------------------------------------------------------------------- #
def load_counts(path):
    """Load raw counts matrix: genes as rows, samples as columns."""
    path = Path(path)
    if path.suffix == ".parquet":
        counts = pd.read_parquet(path)
    else:
        counts = pd.read_csv(path, index_col=0)

    # Index must be gene indentifiers (strings), if the index is a default RangeIndex
    # it means index_col=0 got a data column instead of the intended gene ID column
    if isinstance(counts.index, pd.RangeIndex):
        raise ValueError(
            f"Counts matrix loaded from '{path}' has a numeric RangeIndex. "
            f"Expected gene identifiers as the row index. "
            f"Check that the first column of the file contains gene IDs."
        )

    if counts.index.duplicated().any():
        n_dupes = int(counts.index.duplicated().sum())
        counts = counts.groupby(counts.index).sum()
        # Non-numeric columns are dropped by groupby().sum()
        # Dropped columns are logged for convenience
        import logging
        logging.getLogger("load_clean_data").info(
            f"load_counts: {n_dupes} duplicate gene IDs detected and summed."
        )

    return counts


def load_clinical(path, sample_id_col):
    clinical = pd.read_csv(path)
    if sample_id_col not in clinical.columns:
        raise ValueError(
            f"Column '{sample_id_col}' not found in clinical file. "
            f"Available columns: {list(clinical.columns)}"
        )
    clinical = clinical.set_index(sample_id_col)
    clinical = clinical[~clinical.index.duplicated(keep="first")]
    return clinical


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def align_samples(counts, clinical, logger):
    common = counts.columns.intersection(clinical.index)
    dropped_counts_only = sorted(set(counts.columns) - set(common))
    dropped_clinical_only = sorted(set(clinical.index) - set(common))

    if dropped_counts_only:
        logger.info(
            f"{len(dropped_counts_only)} samples present in counts but not "
            f"in clinical metadata -> dropped."
        )
    if dropped_clinical_only:
        logger.info(
            f"{len(dropped_clinical_only)} samples present in clinical "
            f"metadata but not in counts -> dropped."
        )

    common = sorted(common)
    return counts[common], clinical.loc[common], dropped_counts_only, dropped_clinical_only


# --------------------------------------------------------------------------- #
# missing-value handling (expression)
# --------------------------------------------------------------------------- #
def clean_expression_missing(counts, max_gene_missing_frac, max_sample_missing_frac, logger):
    """Drop genes/samples with excessive missing raw counts, impute the rest.

    Raw GDC counts are rarely NaN, but this guards against merge artifacts,
    corrupted rows, or genes not profiled uniformly across samples.
    """
    gene_missing_frac = counts.isna().mean(axis=1)
    keep_genes = gene_missing_frac <= max_gene_missing_frac
    n_dropped_genes = int((~keep_genes).sum())
    if n_dropped_genes:
        logger.info(
            f"Dropping {n_dropped_genes} genes with >{max_gene_missing_frac:.0%} "
            f"missing values across samples."
        )
    counts = counts.loc[keep_genes]

    sample_missing_frac = counts.isna().mean(axis=0)
    keep_samples = sample_missing_frac <= max_sample_missing_frac
    dropped_sample_ids = sorted(counts.columns[~keep_samples].tolist())
    n_dropped_samples = int((~keep_samples).sum())
    if n_dropped_samples:
        logger.info(
            f"Dropping {n_dropped_samples} samples with >{max_sample_missing_frac:.0%} "
            f"missing values across genes."
        )
    counts = counts.loc[:, keep_samples]

    n_residual = int(counts.isna().sum().sum())
    if n_residual:
        logger.info(f"Imputing {n_residual} residual missing counts with 0.")
        counts = counts.fillna(0)

    # raw counts should be non-negative integers
    counts = counts.clip(lower=0).round().astype(int)

    return counts, dropped_sample_ids


# --------------------------------------------------------------------------- #
# clinical standardization
# --------------------------------------------------------------------------- #
def standardize_mutation_column(series, colname, logger):
    """Map common mutation-status encodings to {0, 1}; unmapped -> NaN (unknown)."""
    mapping = {
        "false": 0, "wt": 0, "wild type": 0, "wildtype": 0, "wild-type": 0,
        "0": 0, "0.0": 0, "no": 0, "negative": 0, "none": 0,
        "true": 1, "mut": 1, "mutant": 1, "mutated": 1, "1": 1, "1.0": 1,
        "yes": 1, "positive": 1,
    }
    original_na = series.isna()
    cleaned = series.astype(str).str.strip().str.lower().map(mapping)

    n_unmapped = int(cleaned.isna().sum() - original_na.sum())
    if n_unmapped > 0:
        raw_lower = series.dropna().astype(str).str.strip().str.lower()
        unknown_values = sorted(
            set(series.dropna().astype(str)[~raw_lower.isin(mapping.keys())].unique())
        )
        logger.warning(
            f"{colname}: {n_unmapped} value(s) did not match a known "
            f"WT/Mutant encoding and were set to NaN. Examples of unmapped "
            f"raw values: {unknown_values[:10]}"
        )
    return cleaned


def standardize_stage_column(series, logger):
    """Trim whitespace / normalize placeholders in free-text stage strings.

    Deliberately does NOT binarize into Early/Late — that mapping is
    config-driven per task and happens in 02_generate_cv_splits.py.
    """
    cleaned = series.astype(str).str.strip()
    placeholder_na = {"nan", "NaN", "", "not reported", "unknown", "NA", "N/A"}
    cleaned = cleaned.replace({p: np.nan for p in placeholder_na})
    return cleaned


def clean_clinical(clinical, config, logger):
    clinical = clinical.copy()

    stage_col = config.get("clinical_columns", {}).get("stage", "cancer_stage")
    if stage_col in clinical.columns:
        clinical[stage_col] = standardize_stage_column(clinical[stage_col], logger)
    else:
        logger.warning(f"Configured stage column '{stage_col}' not found in clinical metadata.")

    mutation_cols = config.get("clinical_columns", {}).get(
        "mutation_status", ["EGFR_mutation_status", "KRAS_mutation_status"]
    )
    for col in mutation_cols:
        if col in clinical.columns:
            clinical[col] = standardize_mutation_column(clinical[col], col, logger)
        else:
            logger.warning(f"Configured mutation column '{col}' not found in clinical metadata.")

    return clinical


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(counts_path, clinical_path, sample_id_col, config,
         out_expression, out_clinical, out_qc, log_path=None):
    logger = get_logger(log_path)

    logger.info(f"Loading counts from: {counts_path}")
    counts = load_counts(counts_path)
    logger.info(f"Raw counts matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")

    logger.info(f"Loading clinical metadata from: {clinical_path}")
    clinical = load_clinical(clinical_path, sample_id_col)
    logger.info(f"Raw clinical table: {clinical.shape[0]} samples x {clinical.shape[1]} columns")

    counts, clinical, dropped_counts_only, dropped_clinical_only = align_samples(
        counts, clinical, logger
    )
    logger.info(f"{counts.shape[1]} samples common to both inputs after alignment.")

    max_gene_missing_frac = config["preprocessing"].get("max_gene_missing_frac", 0.2)
    max_sample_missing_frac = config["preprocessing"].get("max_sample_missing_frac", 0.2)
    
    counts_clean, dropped_samples_missing = clean_expression_missing(
        counts, max_gene_missing_frac, max_sample_missing_frac, logger
    )

    # re-align clinical metadata to samples surviving the missing-value filter
    clinical_clean = clinical.loc[counts_clean.columns]
    clinical_clean = clean_clinical(clinical_clean, config, logger)

    Path(out_expression).parent.mkdir(parents=True, exist_ok=True)
    Path(out_clinical).parent.mkdir(parents=True, exist_ok=True)
    counts_clean.to_csv(out_expression)
    clinical_clean.to_csv(out_clinical)

    qc_report = {
        "n_genes_input": int(counts.shape[0]),
        "n_samples_input": int(counts.shape[1]),
        "n_genes_output": int(counts_clean.shape[0]),
        "n_samples_output": int(counts_clean.shape[1]),
        "samples_dropped_no_clinical_match": dropped_counts_only,
        "samples_dropped_no_expression_match": dropped_clinical_only,
        "samples_dropped_excessive_missing": dropped_samples_missing,
        "max_gene_missing_frac_threshold": max_gene_missing_frac,
        "max_sample_missing_frac_threshold": max_sample_missing_frac,
    }
    Path(out_qc).parent.mkdir(parents=True, exist_ok=True)
    with open(out_qc, "w") as f:
        json.dump(qc_report, f, indent=2)

    logger.info(
        f"Final clean matrix: {counts_clean.shape[0]} genes x "
        f"{counts_clean.shape[1]} samples"
    )
    logger.info(f"Wrote: {out_expression}, {out_clinical}, {out_qc}")


if __name__ == "__main__":

    # =========================================================================
    # EXECUTION MODE 1: Snakemake Integration
    # =========================================================================
    if "snakemake" in globals():
        main(
            counts_path=snakemake.input.counts,
            clinical_path=snakemake.input.clinical,
            sample_id_col=snakemake.config["data"]["sample_id_col"],
            config=snakemake.config,
            out_expression=snakemake.output.expression,
            out_clinical=snakemake.output.clinical,
            out_qc=snakemake.output.qc_report,
            log_path=snakemake.log[0] if len(snakemake.log) else None,
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
        parser.add_argument("--counts", required=False, help="path to raw counts CSV")
        parser.add_argument("--clinical", required=False, help="path to raw clinical CSV")
        parser.add_argument("--config", required=True, help="path to config.yaml")
        parser.add_argument("--out-expression", required=False)
        parser.add_argument("--out-clinical", required=False)
        parser.add_argument("--out-qc", required=False)
        parser.add_argument("--log", default=None)
        args = parser.parse_args()

        # --- Load Base Configuration File ---
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # --- Resolve Input Paths (CLI Override > YAML Config) ---
        counts_path = args.counts or cfg["data"]["counts_file"]
        clinical_path = args.clinical or cfg["data"]["clinical_file"]

        # --- Resolve Output Paths (CLI Override > YAML Config) ---
        out_dir = cfg["data"].get("output_dir", "results")
        step01_cfg = cfg["outputs"]["step_01"]

        out_expression = args.out_expression or step01_cfg["expression"].format(output_dir=out_dir)
        out_clinical = args.out_clinical or step01_cfg["clinical"].format(output_dir=out_dir)
        out_qc = args.out_qc or step01_cfg["qc_report"].format(output_dir=out_dir)


        main(
            counts_path=counts_path,
            clinical_path=clinical_path,
            sample_id_col=cfg["data"]["sample_id_col"],
            config=cfg,
            out_expression=out_expression,
            out_clinical=out_clinical,
            out_qc=out_qc,
            log_path=args.log,
        )
