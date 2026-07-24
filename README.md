# MALEXA
### MAchine Learning EXpression-based Algorithms

[![Python](https://img.shields.io/badge/Python-%E2%89%A5_3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![Snakemake](https://img.shields.io/badge/Snakemake-%E2%89%A5_7-4B0082?style=flat)](https://snakemake.github.io/) [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%E2%89%A5_1.1-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-%E2%89%A5_1.7-006400?style=flat)](https://xgboost.readthedocs.io/) [![ML Models](https://img.shields.io/badge/ML_Models-ElasticNet_%7C_XGBoost-blueviolet?style=flat)](#)


MALEXA predicts **somatic mutation status** from RNA-seq expression data using nested cross-validation and selected machine learning algorithms. It consists of a multi-step Python pipeline orchestrated by Snakemake.

**Key features:**

- **Leakage-free.** CPM filtering and variance selection are fit on training folds only; CV splits are generated once before training and shared across all models.
- **Fully config-driven.** Adding a new task or model requires only a `config.yaml` edit.
- **HPC-native.** Each `(task, model, fold)` job is an independent SLURM submission.
- **Complete auditability.** Every dropped gene, dropped sample, and data transformation decision is written to `qc_report.json` and per-job log files.
- **Interpretable outputs.** Cross-fold, cross-model gene importance consensus rankings surface the most predictive biomarkers.

---

## Repository Structure

```
.
├── Snakefile                          # DAG definition; reads all config from config.yaml
├── config.yaml                        # Single source for all pipeline settings
├── run_pipeline.sh                    # Convenience launcher for SLURM environments
├── envs/
│   └── pipeline.yaml                  # Conda environment specification
├── profiles/
│   └── slurm/
│       ├── config.yaml                # Snakemake SLURM profile
│       └── slurm-status.py            # sacct-based job status poller
├── scripts/
│   ├── 01_load_clean_data.py          # Load, align, and clean raw inputs
│   ├── 02_generate_cv_splits.py       # Stratified CV split generation (once per task)
│   ├── 03_train_model.py              # Feature selection + train + evaluate (per task × model × fold)
│   ├── 04_aggregate_metrics.py        # Aggregate metrics and produce comparison figures
│   └── 05_interpret_report.py         # Gene importance ranking and cross-model consensus
├── data/                              # Not provided, see Sample Data below
│   └── raw/
│       ├── counts.parquet             # RNA-seq raw counts (genes × samples)
│       └── clinical.csv               # Clinical metadata (samples × features)
└── results/                           # Created at runtime; all outputs land here
```

> **Note:** Input data files are not provided. See [Sample Data](#sample-data) for input setup instructions.

---

## Pipeline Architecture

```
counts.{csv,parquet} ──┐
                        ├─► 01_load_clean_data ──► expression_clean.csv
clinical.csv ──────────┘                        └► clinical_clean.csv
                                                          │
                                       ┌──────────────────┤
                                       ▼                  ▼
                           02_generate_cv_splits   02_generate_cv_splits
                               (EGFR_mutation)      (KRAS_mutation)
                                       │                  │
                           ┌───────────┤      ┌───────────┤
                           ▼           ▼      ▼           ▼
                    03_train_model  (× n_splits × n_repeats × models — parallel)
                           │
                           ├─► metrics.json
                           ├─► predictions.csv
                           ├─► feature_importances.csv
                           └─► model.pkl
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
           04_aggregate_metrics              05_interpret_report
           (aggregated_metrics.csv           (gene_importance_report.csv
            model_comparison.png)             gene_importance_plot.png)
```

Each `03_train_model` invocation is fully independent. With `RepeatedStratifiedKFold` (`n_splits=5`, `n_repeats=3`), 15 parallel jobs are submitted per model per task.

---

## Quick Start

**Requirements:** Python ≥ 3.10, Conda or Mamba, Snakemake ≥ 7.

```bash
# 1. Clone the repository
git clone https://github.com/ccarloscr/malexa.git
cd malexa

# 2. Create and activate the environment
conda env create -f envs/pipeline.yaml
conda activate malexa_env

# 3. Place input data under data/raw/
# See the Sample Data section for setup instructions

# 4. Check the DAG without executing anything
snakemake --dry-run --cores 1
```

---

## Configuration

All pipeline behaviour is controlled from `config.yaml`. No hardcoded logic exists in any script.

### Supported CV methods

| Method | Config key | Parameters used |
|---|---|---|
| `StratifiedKFold` | `method: StratifiedKFold` | `n_splits`, `random_seed` |
| `RepeatedStratifiedKFold` | `method: RepeatedStratifiedKFold` | `n_splits`, `n_repeats`, `random_seed` |
| `StratifiedShuffleSplit` | `method: StratifiedShuffleSplit` | `n_splits`, `test_size`, `random_seed` |

---

## Running the Pipeline

### Dry run

```bash
snakemake --dry-run --cores 1
```

### Local execution

```bash
snakemake --cores 8
```

### HPC — SLURM

The recommended approach for large cohorts. Each `(task, model, fold)` combination becomes an independent SLURM job. The `run_pipeline.sh` launcher is designed to be kept alive in a `screen` or `tmux` session on the login node:

```bash
screen -S malexa
conda activate malexa
./run_pipeline.sh
# Leave screen: Ctrl+A d
# Return screen: screen -r malexa
```

Or equivalently:

```bash
snakemake \
    --profile profiles/slurm \
    --latency-wait 60 \
    --rerun-incomplete \
    --keep-going
```

The SLURM profile (`profiles/slurm/config.yaml`) controls the number of concurrent jobs, default memory/time fallbacks, and the partition. The `slurm-status.py` script polls `sacct` to detect failed jobs immediately rather than waiting for output files to appear.

> **Before running on your HPC:** update the `conda activate` path in `run_pipeline.sh` and the `partition` name in `profiles/slurm/config.yaml` to match your cluster configuration.

### Standalone script execution

Every script can be called directly for debugging without Snakemake:

> The following instructions default to predicting EGFR mutations; update `--task` options and directory paths as needed.

```bash
# Step 1 — load and clean
python scripts/01_load_clean_data.py \
    --counts    data/raw/counts.parquet \
    --clinical  data/raw/clinical.csv \
    --config    config.yaml \
    --out-expression results/data/expression_clean.csv \
    --out-clinical   results/data/clinical_clean.csv \
    --out-qc         results/data/qc_report.json

# Step 2 — generate splits for one task
python scripts/02_generate_cv_splits.py \
    --expression results/data/expression_clean.csv \
    --clinical   results/data/clinical_clean.csv \
    --task       EGFR_mutation \
    --config     config.yaml \
    --out-splits results/splits/EGFR_mutation_splits.json

# Step 4 — aggregate metrics after training
python scripts/04_aggregate_metrics.py \
    --metrics-dir results/EGFR_mutation \
    --task        EGFR_mutation \
    --config      config.yaml \
    --out-table   results/EGFR_mutation/aggregated_metrics.csv \
    --out-plot    results/EGFR_mutation/model_comparison.png

# Step 5 — gene importance report
python scripts/05_interpret_report.py \
    --importances-dir results/EGFR_mutation \
    --task            EGFR_mutation \
    --config          config.yaml \
    --out-report      results/EGFR_mutation/gene_importance_report.csv \
    --out-plot        results/EGFR_mutation/gene_importance_plot.png
```

---

## Scripts

### `01_load_clean_data.py` — Load and clean

Runs **once** for the entire project.

- Loads the raw counts matrix (genes × samples; CSV or Parquet) and clinical metadata CSV.
- Aligns samples by `sample_id`; logs and drops any that appear in only one source.
- Drops genes and samples exceeding configurable missing-value thresholds (`max_gene_missing_frac`, `max_sample_missing_frac`); imputes rare residual NaNs with 0.
- Standardises free-text mutation-status encodings (`WT`, `Mutant`, `yes/no`, `0/1`, etc.) to a clean `{0, 1}` integer; true unknowns become `NaN`.
- Emits a `qc_report.json` documenting every gene/sample dropped and the reason.

> No gene filtering occurs at this step to prevent leakage; these operations happen downstream within fold boundaries.

### `02_generate_cv_splits.py` — Generate CV splits

Runs **once per task**, before any model training.

- Extracts and validates the target label column for the requested task.
- Drops samples with unknown labels (`NaN` after step 01) and logs their IDs.
- Applies the configured CV strategy (`StratifiedKFold`, `RepeatedStratifiedKFold`, or `StratifiedShuffleSplit`).
- Writes a JSON file with train/test **sample ID lists** for every split, ensuring validity even if the matrix is reordered later.

Because splits are generated once and shared, all models for a given task are evaluated on exactly the same data partitions, making performance comparison fair.

### `03_train_model.py` — Train and evaluate

Runs **once per `(task, model, fold)`** combination. All such jobs are fully independent and parallel.

Inside each fold, on training data only:

1. **CPM filter**: removes genes with median CPM below `feature_selection.min_cpm`.
2. **Variance filter**: removes the bottom `feature_selection.min_variance_pct` percent of genes by variance.
3. **log1p normalisation**: applied after filtering.
4. **StandardScaler**: fit on training samples, applied to test.
5. **Hyperparameter search**: `GridSearchCV` or `RandomizedSearchCV` with an inner stratified CV on the training fold.
6. **Evaluation**: best pipeline is scored on the held-out test fold.

The target gene (e.g. EGFR, KRAS) is excluded per-task via `exclude_genes` in `config.yaml` to prevent expression-of-the-target leakage. Feature importances are extracted from `feature_importances_` (XGBoost) or `coef_` (ElasticNet LogisticRegression).

Outputs: `metrics.json`, `predictions.csv`, `feature_importances.csv`, `model.pkl`.

### `04_aggregate_metrics.py` — Aggregate and visualise

- Collects all `metrics.json` files for a task and assembles a long-format DataFrame (one row per model × fold).
- Computes per-model mean ± std for every configured evaluation metric.
- Writes `aggregated_metrics.csv` with per-fold rows plus summary rows (tagged `fold = "mean"` / `"std"`).
- Generates `model_comparison.png`: one panel per metric, scatter + mean ± 1-SD error bars, sorted by `primary_metric`, with reference lines.

### `05_interpret_report.py` — Gene importance and consensus ranking

- Collects all `feature_importances.csv` files for a task (all models × folds).
- Aggregates using the configured strategy:
  - `mean_rank`: ranks genes by |importance| within each fold, then averages ranks across folds. Robust to scale differences between model families.
  - `mean_importance`: averages raw importance scores (appropriate when scores are comparable).
- Computes a **cross-model consensus ranking**: genes that appear as important across multiple models receive a combined score.
- Writes `gene_importance_report.csv` and `gene_importance_plot.png` (one panel per model + one consensus panel).

---

## Outputs

After a full run, `results/` has the following structure:

```
results/
├── data/
│   ├── expression_clean.csv              # QC-filtered, aligned counts matrix
│   ├── clinical_clean.csv                # Standardised clinical metadata
│   └── qc_report.json                    # Audit log: every dropped gene/sample
├── splits/
│   ├── EGFR_mutation_splits.json         # CV fold sample-ID assignments
│   └── KRAS_mutation_splits.json
├── EGFR_mutation/
│   ├── elasticnet_logreg/
│   │   └── fold{0..14}/                  # 5 splits × 3 repeats
│   │       ├── metrics.json
│   │       ├── predictions.csv
│   │       ├── feature_importances.csv
│   │       └── model.pkl
│   ├── xgboost/
│   │   └── fold{0..14}/  ...
│   ├── aggregated_metrics.csv
│   ├── model_comparison.png
│   ├── gene_importance_report.csv
│   └── gene_importance_plot.png
├── KRAS_mutation/  ...                   # Same structure
└── logs/                                 # One log file per rule invocation
    ├── load_clean_data.log
    ├── generate_cv_splits_EGFR_mutation.log
    └── train_EGFR_mutation_xgboost_fold0.log  ...
```

### Key output files

| File | Description |
|---|---|
| `data/qc_report.json` | Full audit trail of genes/samples dropped and thresholds applied |
| `splits/<task>_splits.json` | Stratified fold sample-ID assignments shared by all models |
| `<task>/aggregated_metrics.csv` | ROC-AUC, PR-AUC, balanced accuracy, F1, MCC per model × fold + mean ± std summary |
| `<task>/model_comparison.png` | Multi-metric visual comparison of all models across all folds |
| `<task>/gene_importance_report.csv` | Genes ranked by mean_rank or mean_importance with cross-model consensus scores |
| `<task>/gene_importance_plot.png` | Horizontal bar chart of top-N genes per model and consensus panel |

---

## Sample Data

> ⚠️ Raw data files are **not included** in this repository.

The pipeline is validated on a cohort of **517 samples** from the GDC TCGA Lung Adenocarcinoma (LUAD) project, processed with [PyGDC-RNA-ETL](https://github.com/ccarloscr/pygdc-rna-etl). Sample outputs generated from this cohort are provided in the `sample_results/` directory.

### Benchmarking results — TCGA-LUAD (n = 517)

Results below are cross-validated mean metrics across 15 folds (`RepeatedStratifiedKFold`, 5 splits × 3 repeats). Feature selection retains ~7,660 genes per fold after CPM (median ≥ 10) and variance (bottom 25% removed) filtering. XGBoost hyperparameters were tuned via `RandomizedSearchCV` (15 iterations, 2-fold inner CV, scored by ROC-AUC).

#### EGFR mutation prediction

| Model | ROC-AUC | PR-AUC | Balanced Accuracy | F1 (weighted) | MCC |
|---|---|---|---|---|---|
| ElasticNet | **0.845** | **0.627** | **0.757** | **0.870** | **0.504** |
| XGBoost | 0.835 | 0.542 | 0.590 | 0.833 | 0.304 |

#### KRAS mutation prediction

| Model | ROC-AUC | PR-AUC | Balanced Accuracy | F1 (weighted) | MCC |
|---|---|---|---|---|---|
| ElasticNet | **0.853** | **0.697** | **0.766** | **0.809** | **0.523** |
| XGBoost | 0.824 | 0.646 | 0.644 | 0.751 | 0.378 |

> **Runtime**: ~5 hours clock time on HPC (SLURM) with a concurrency cap of 6 simultaneous jobs (jobs: 6 in profiles/slurm/config.yaml). Each train_model job requests 4 CPUs and 8 GB RAM, with a 4-hour time limit.

Both tasks show strong discriminative performance (ROC-AUC > 0.82 for all model–task combinations). ElasticNet consistently outperforms XGBoost on this cohort, particularly on balanced accuracy and PR-AUC, which are more informative under the class imbalance present in LUAD mutation data. Full per-fold metrics, gene importance rankings, and model comparison plots are available in `sample_results/`.

**Downloading data manually from GDC:**

1. **RNA-seq raw counts**: visit [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov), select Project, Data Category `Transcriptome Profiling`, Data Type `Gene Expression Quantification`, Workflow Type `STAR - Counts`. Export as a merged matrix with Ensembl gene IDs as row index and `sample_id` values as column headers. Save as `data/raw/counts.parquet` (or `.csv`).

2. **Clinical metadata**: download the clinical supplement for your Project. The file must contain at minimum: `sample_id`, `X_mutation_status`, where X refers to the gene of interest. Column names are configurable in `config.yaml`. Save as `data/raw/clinical.csv`.

---

## Design Principles

**Leakage-free.** CV splits are written before any model sees the data. All feature selection (CPM filter, variance filter), normalization (log1p), and scaling (StandardScaler) is fit on training folds only, applied to test folds. Exclusion of the target gene is optional via `exclude_genes` but highly recommended.

**Config is the API.** Every parameter (thresholds, CV strategy, model hyperparameter grids, evaluation metrics, interpretation strategy) is defined in `config.yaml`. No task or model logic is hardcoded in any script.

**Separation of concerns.** Each script has a single responsibility and a CLI entry point. The Snakemake DAG wires them together, but each script is debuggable in isolation.

---

## Extending the Pipeline

### Add a new classification task

```yaml
# config.yaml
tasks:
  TP53_mutation:
    label_col: "TP53_mutation_status"        # must exist in clinical CSV
    models: [elasticnet_logreg, xgboost]
    pos_label: 1
    exclude_genes: ["ENSG00000141510"]       # TP53 Ensembl ID
```

### Add a new model

```yaml
# config.yaml
models:
  lightgbm:
    estimator_class: "lightgbm.LGBMClassifier"
    search_strategy: random
    n_iter: 30
    scoring: "roc_auc"
    fixed_params: {n_jobs: 4, random_state: 123, verbose: -1}
    param_grid:
      clf__n_estimators:  [100, 300, 500]
      clf__max_depth:     [3, 5, 7]
      clf__learning_rate: [0.01, 0.05, 0.1]
      clf__num_leaves:    [15, 31, 63]
```

Then reference `lightgbm` in any task's `models` list. No Snakefile or script changes are required.

---

## Reproducibility

- The random seed (`cv.random_seed` in `config.yaml`) controls both the CV split strategy and all model random states.
- CV splits are generated once and stored as JSON before any model is trained; all models for a given task read the same file.
- Fold-local feature selection is fit exclusively on training samples.
- The `qc_report.json` and per-fold log files in `results/logs/` provide a complete audit trail of every data transformation.

To exactly reproduce a run: fix the seed in `config.yaml`, use the same Conda environment (`envs/pipeline.yaml`), and verify input file integrity against the GDC manifest checksums.

---

## Dependencies

Core dependencies defined in `envs/pipeline.yaml`:

| Package | Role |
|---|---|
| `snakemake-minimal >= 7.0` | Workflow orchestration and job submission |
| `pandas >= 1.5` | Data handling |
| `numpy >= 1.23` | Numerical operations |
| `pyarrow` | Parquet handling |
| `scikit-learn >= 1.1` | ML models, CV, feature selection, evaluation |
| `xgboost >= 1.7` | Gradient boosting classifier |
| `matplotlib >= 3.6` | Figures |
| `pyyaml >= 6.0` | Config parsing |

```bash
conda env create -f envs/pipeline.yaml
conda activate malexa_env
```

---

## License

MIT License. See `LICENSE` for details.
