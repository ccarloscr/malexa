# MALEXA
## MAchine Learning EXpression-based Algorithms

MALEXA uses RNA-seq raw expression data to predict cancer stage and somatic mutational status. It is built on Python and is designed to run via Snakemake locally, on an HPC cluster (SLURM), or on AWS.

---

## Overview

This pipeline addresses two independent classification problems using bulk RNA-seq expression profiles and clinical metadata (required labels to test and train the model): cancer stage and mutation status of set genes.

> For convenience, running PyGDC-RNA-ETL results in the recommended input data to run MALEXA. Alternatively, any pair of RNA-seq raw count matrix and clinical metadata csv file with the specifications listed in the requirements section is suitable.

The sample dataset is a cohort (n = 517) from the GDC TCGA Lung Adenocarcinoma (LUAD) project.

| Question | Target | Models |
|---|---|---|
| Cancer stage prediction | Early (I–II) vs. Late (III–IV) | Random Forest, XGBoost |
| Mutation status prediction | EGFR mutant vs. wild-type | Linear SVM, ElasticNet Logistic Regression |
| Mutation status prediction | KRAS mutant vs. wild-type | Linear SVM, ElasticNet Logistic Regression |

Key design principles:

- **Config-driven.** Tasks, models, hyperparameter grids, evaluation metrics, and clinical column names are all declared in `config.yaml`. Adding a new task or model requires no changes to any script.
- **Leakage-free CV.** Cross-validation splits are generated once per task (stratified, fixed seed) and shared by all models, enabling fair comparison. Fold-local feature selection (CPM filter + variance filter) is applied inside `03_train_model.py` on training data only.
- **Separation of concerns.** Each pipeline stage is a standalone script that can be run via Snakemake or directly from the CLI for debugging.

---

## Repository Structure

```
.
├── Snakefile                    # DAG definition; reads everything from config.yaml
├── config.yaml                  # Single source of truth for all pipeline settings
├── data/
│   └── raw/
│       ├── tcga_luad_counts.csv     # RNA-seq raw counts (genes × samples)
│       └── tcga_luad_clinical.csv   # Clinical metadata (samples × features)
├── scripts/
│   ├── 01_load_clean_data.py        # Load, align, and clean raw inputs
│   ├── 02_generate_cv_splits.py     # Generate stratified CV splits per task
│   ├── 03_train_model.py            # Train and evaluate one (task, model, fold)
│   ├── 04_aggregate_metrics.py      # Aggregate metrics and produce comparison plots
│   └── 05_interpret_report.py       # Gene importance ranking and reporting
├── profiles/
│   ├── slurm/                       # Snakemake SLURM profile (HPC)
│   └── aws/                         # Snakemake AWS/k8s profile
├── envs/
│   └── pipeline.yaml                # Conda environment specification
└── results/                         # Created at runtime; all outputs land here
```

> **Note:** The raw data files are not versioned in this repository. See [Data](#data) below for download instructions.

---

## Pipeline DAG

```
counts.csv ──┐
             ├─► 01_load_clean_data ──► expression_clean.csv
clinical.csv─┘                      └► clinical_clean.csv
                                              │
                          ┌───────────────────┤
                          ▼                   ▼
              02_generate_cv_splits     02_generate_cv_splits
                (cancer_stage)          (EGFR/KRAS_mutation)
                          │                   │
              ┌───────────┤       ┌───────────┤
              ▼           ▼       ▼           ▼
       03_train_model  (×N folds × models, run in parallel)
              │
              ├─► metrics.json
              ├─► predictions.csv
              ├─► feature_importances.csv
              └─► model.pkl
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   04_aggregate_metrics        05_interpret_report
   (aggregated_metrics.csv     (gene_importance_report.csv
    model_comparison.png)       gene_importance_plot.png)
```

Each `03_train_model` job (one per task × model × fold combination) is fully independent and can be parallelised across cluster nodes or AWS workers.

---

## Scripts

### `01_load_clean_data.py` — Load and clean

- Loads the raw counts matrix (genes × samples) and clinical metadata CSV.
- Aligns samples by `sample_id`; logs and drops any that appear in only one source.
- Drops genes or samples exceeding configurable missing-value thresholds (`preprocessing.max_gene_missing_frac`, `preprocessing.max_sample_missing_frac`); imputes rare residual NaNs with 0.
- Standardises mutation-status columns from free-text encodings (e.g. `WT`, `Mutant`, `yes/no`, `0/1`) to a clean `{0, 1}` integer, leaving true unknowns as `NaN`.
- Normalises free-text stage strings (whitespace, case) without binarising — that is a per-task concern handled downstream.
- Emits a `qc_report.json` documenting every sample or gene dropped and why.

### `02_generate_cv_splits.py` — Generate CV splits

Runs **once per task**, before any model training.

- For `cancer_stage`: binarises free-text stage strings to `{0 = Early, 1 = Late}` using the mapping in `config.yaml[stage_binarization]`. Samples with unmapped stage values are excluded and logged.
- For mutation tasks: drops samples where the status is `NaN` (unknown after script 01).
- Runs `StratifiedKFold` with the seed from `config.yaml[cv.random_seed]`.
- Writes a JSON file containing, for each fold, the list of **sample IDs** (not integer positions) assigned to training and test sets. Using sample IDs ensures splits remain valid if the matrix is ever reordered.

Because splits are generated once and shared across models, all models for a given task are evaluated on exactly the same data partitions — making performance comparison fair by construction.

### `03_train_model.py` — Train model

Runs **once per (task, model, fold)** combination; these jobs are fully independent and embarrassingly parallel.

- Loads the fold's train/test sample IDs from the splits JSON.
- Applies fold-local feature selection **on training data only** to avoid leakage:
  1. CPM filter: removes genes with median CPM below `feature_selection.min_cpm`.
  2. Variance filter: removes the bottom `feature_selection.min_variance_pct` percent of genes by variance across training samples.
- Wraps the configured estimator in a `sklearn` `Pipeline` (step name `"clf"`) and performs hyperparameter search (`GridSearchCV` or `RandomizedSearchCV`) as configured per model.
- Special handling: `LinearSVC` is wrapped in `CalibratedClassifierCV` to enable probability outputs for ROC-AUC scoring.
- Writes per-fold outputs: `metrics.json`, `predictions.csv`, `feature_importances.csv`, `model.pkl`.

### `04_aggregate_metrics.py` — Aggregate and compare

- Collects all `metrics.json` files for a task, assembles a long-format DataFrame (one row per model × fold).
- Computes per-model mean ± std for every metric listed under `evaluation.metrics`.
- Writes `aggregated_metrics.csv` (per-fold rows plus summary rows tagged `fold = -1` for mean, `fold = -2` for std).
- Generates `model_comparison.png`: one panel per metric, scatter + mean ± SD error bars per model, sorted by `evaluation.primary_metric`, with chance-level reference lines.

### `05_interpret_report.py` — Interpret and report

- Collects all `feature_importances.csv` files for a task.
- Aggregates across folds using the strategy in `config.yaml[interpretation.aggregation]`:
  - `mean_rank`: ranks genes by |importance| within each fold, then averages ranks across folds. Robust to scale differences between model families.
  - `mean_importance`: averages raw importance scores (appropriate when scores are already on a comparable scale).
- Computes a **cross-model consensus ranking**: genes appearing as important across multiple models receive a combined score.
- Writes `gene_importance_report.csv` (columns: `gene`, `model`, `mean_importance`, `std_importance`, `mean_rank`, `n_folds_present`, `consensus_rank`, `n_models_present`).
- Generates `gene_importance_plot.png`: one panel per model plus a consensus panel, showing the top-N genes.

---

## Configuration (`config.yaml`)

All pipeline behaviour is controlled from `config.yaml`. Scripts contain no hardcoded task or model logic.

```yaml
data:
  counts_file:   "data/raw/tcga_luad_counts.csv"
  clinical_file: "data/raw/tcga_luad_clinical.csv"
  sample_id_col: "sample_id"
  output_dir:    "results"

cv:
  n_splits:    5
  random_seed: 42

preprocessing:
  max_gene_missing_frac:   0.20
  max_sample_missing_frac: 0.20

feature_selection:
  min_cpm:          1.0   # median CPM threshold (applied on training fold only)
  min_variance_pct: 10    # drop lowest-variance decile (applied on training fold only)

tasks:
  cancer_stage:
    label_col: "cancer_stage"
    models: [random_forest, xgboost]
    pos_label: 1                 # 1 = Late stage

  EGFR_mutation:
    label_col: "EGFR_mutation_status"
    models: [linear_svm, elasticnet_logreg]
    pos_label: 1

  KRAS_mutation:
    label_col: "KRAS_mutation_status"
    models: [linear_svm, elasticnet_logreg]
    pos_label: 1

models:
  random_forest:
    estimator_class: "sklearn.ensemble.RandomForestClassifier"
    search_strategy: random
    n_iter: 30
    scoring: "roc_auc"
    fixed_params: {n_jobs: 4, random_state: 42}
    param_grid:
      clf__n_estimators:     [200, 500, 1000]
      clf__max_depth:        [null, 5, 10, 20]
      clf__min_samples_leaf: [1, 3, 5, 10]
      clf__max_features:     ["sqrt", "log2", 0.1, 0.2]
      clf__class_weight:     ["balanced", null]
  # ... xgboost, linear_svm, elasticnet_logreg defined similarly

evaluation:
  metrics:
    - roc_auc
    - average_precision
    - balanced_accuracy
    - f1_weighted
    - matthews_corrcoef
  primary_metric: "roc_auc"

interpretation:
  top_n_genes:  50
  aggregation:  "mean_rank"
```

To add a new classification task, append an entry under `tasks` and ensure the label column exists in the clinical CSV. To add a new model, append an entry under `models` and reference it in the relevant task's `models` list. No script changes are required.

---

## Data

Raw data files are **not included** in this repository. Download from GDC:

1. **RNA-seq raw counts** (`tcga_luad_counts.csv`): TCGA-LUAD HTSeq raw counts from the GDC Data Portal (`https://portal.gdc.cancer.gov`). Select Project `TCGA-LUAD`, Data Category `Transcriptome Profiling`, Data Type `Gene Expression Quantification`, Workflow Type `HTSeq - Counts`. Export as a single merged matrix with genes as rows and `sample_id` values as column headers.

2. **Clinical metadata** (`tcga_luad_clinical.csv`): Clinical supplement from the GDC Data Portal for `TCGA-LUAD`. The file must contain at minimum the columns `sample_id`, `cancer_stage`, `EGFR_mutation_status`, and `KRAS_mutation_status`. Column names are configurable in `config.yaml`.

Place both files under `data/raw/` before running the pipeline.

---

## Installation

**Requirements:** Python ≥ 3.10, Conda or Mamba, Snakemake ≥ 7.

```bash
# Clone the repository
git clone https://github.com/<your-org>/tcga-luad-classification.git
cd tcga-luad-classification

# Create and activate the environment
conda env create -f envs/pipeline.yaml
conda activate luad-pipeline

# Verify Snakemake can parse the DAG
snakemake --dry-run
```

Core Python dependencies: `snakemake`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `pyyaml`.

---

## Running the Pipeline

### Dry run (check the DAG without executing)

```bash
snakemake --dry-run --cores 1
```

### Local execution

```bash
snakemake --cores 8
```

### HPC cluster (SLURM)

```bash
snakemake --profile profiles/slurm --jobs 200
```

The SLURM profile passes each `train_model` job as an independent SLURM job, making the most expensive step (hyperparameter search × folds) fully parallel across nodes.

### AWS

```bash
snakemake --profile profiles/aws --jobs 200
```

Compatible with Snakemake's native AWS Batch executor, Tibanna, and Kubernetes. No pipeline code changes are needed between execution environments.

### Running a single script standalone (for debugging)

Each script has a CLI entry point and can be run outside Snakemake:

```bash
# Step 1
python scripts/01_load_clean_data.py \
    --counts data/raw/tcga_luad_counts.csv \
    --clinical data/raw/tcga_luad_clinical.csv \
    --config config.yaml \
    --out-expression results/data/expression_clean.csv \
    --out-clinical results/data/clinical_clean.csv \
    --out-qc results/data/qc_report.json

# Step 2 (one task at a time)
python scripts/02_generate_cv_splits.py \
    --expression results/data/expression_clean.csv \
    --clinical results/data/clinical_clean.csv \
    --task cancer_stage \
    --config config.yaml \
    --out-splits results/splits/cancer_stage_splits.json

# Step 4 (aggregate after training)
python scripts/04_aggregate_metrics.py \
    --metrics-dir results/cancer_stage \
    --task cancer_stage \
    --config config.yaml \
    --out-table results/cancer_stage/aggregated_metrics.csv \
    --out-plot  results/cancer_stage/model_comparison.png

# Step 5
python scripts/05_interpret_report.py \
    --importances-dir results/cancer_stage \
    --task cancer_stage \
    --config config.yaml \
    --out-report results/cancer_stage/gene_importance_report.csv
```

---

## Outputs

After a full pipeline run, the `results/` directory has the following structure:

```
results/
├── data/
│   ├── expression_clean.csv          # Aligned, QC-filtered counts matrix
│   ├── clinical_clean.csv            # Standardised clinical metadata
│   └── qc_report.json                # Audit log of dropped genes/samples
├── splits/
│   ├── cancer_stage_splits.json      # CV fold assignments for stage task
│   ├── EGFR_mutation_splits.json
│   └── KRAS_mutation_splits.json
├── cancer_stage/
│   ├── random_forest/
│   │   └── fold{0..4}/
│   │       ├── metrics.json
│   │       ├── predictions.csv
│   │       ├── feature_importances.csv
│   │       └── model.pkl
│   ├── xgboost/
│   │   └── fold{0..4}/  ...
│   ├── aggregated_metrics.csv        # Per-fold + mean/std rows for all models
│   ├── model_comparison.png          # Multi-metric comparison figure
│   ├── gene_importance_report.csv    # Ranked gene list with consensus scores
│   └── gene_importance_plot.png      # Top-N genes per model + consensus panel
├── EGFR_mutation/  ...               # Same structure
├── KRAS_mutation/  ...               # Same structure
└── logs/                             # One log file per rule invocation
```

### Key output files

| File | Description |
|---|---|
| `data/qc_report.json` | Audit trail: genes/samples dropped, thresholds applied |
| `splits/<task>_splits.json` | Stratified fold indices (sample IDs) shared by all models |
| `<task>/aggregated_metrics.csv` | ROC-AUC, PR-AUC, balanced accuracy, F1, MCC per model × fold, plus mean ± std summary rows |
| `<task>/model_comparison.png` | Visual comparison of all models across all configured metrics |
| `<task>/gene_importance_report.csv` | Top genes ranked by mean_rank or mean_importance, with cross-model consensus scores |
| `<task>/gene_importance_plot.png` | Horizontal bar chart of top-N genes per model and consensus |

---

## Reproducibility

- The random seed (`config.yaml[cv.random_seed]`) controls both the `StratifiedKFold` split and all model random states.
- CV splits are generated once and stored as JSON before any model is trained; all models for a given task read the same split file.
- Fold-local feature selection is fit exclusively on training samples, preventing any leakage from test data into the feature set.
- The `qc_report.json` and per-fold log files provide a complete audit trail of every data transformation.

To exactly reproduce a run: fix the seed in `config.yaml`, use the same conda environment (`envs/pipeline.yaml`), and ensure the raw input files are identical (check MD5 checksums against the GDC manifest).

---

## Extending the Pipeline

**Add a new task** (e.g. predicting smoking status):

```yaml
# config.yaml
tasks:
  smoking_status:
    label_col: "smoking_history"   # must exist in clinical CSV
    models: [random_forest, elasticnet_logreg]
    pos_label: 1
```

**Add a new model** (e.g. LightGBM):

```yaml
# config.yaml
models:
  lightgbm:
    estimator_class: "lightgbm.LGBMClassifier"
    search_strategy: random
    n_iter: 30
    scoring: "roc_auc"
    fixed_params: {n_jobs: 4, random_state: 42, verbose: -1}
    param_grid:
      clf__n_estimators:  [100, 300, 500]
      clf__max_depth:     [3, 5, 7]
      clf__learning_rate: [0.01, 0.05, 0.1]
      clf__num_leaves:    [15, 31, 63]
```

Then reference `lightgbm` in any task's `models` list. The Snakefile and all scripts pick it up automatically.

---

## Citation

If you use this pipeline in your research, please cite the TCGA-LUAD dataset:

> Cancer Genome Atlas Research Network. (2014). Comprehensive molecular profiling of lung adenocarcinoma. *Nature*, 511(7511), 543–550. https://doi.org/10.1038/nature13385

---

## License

MIT License. See `LICENSE` for details.
