# =============================================================================
# Snakefile — TCGA-LUAD expression-based classification pipeline
#
# Two independent question families, both config-driven (see config.yaml):
#   1. cancer stage (Early vs Late)      -> random_forest, xgboost
#   2. EGFR / KRAS mutation status       -> linear_svm, elasticnet_logreg
#
# Pipeline stages (one script each, reused generically across task/model/fold):
#   01_load_clean_data.py     : missing-value handling, sample alignment      (once)
#   02_generate_cv_splits.py  : stratified CV split indices per task          (once per task)
#   03_train_model.py         : fold-local feature selection + train + eval   (per task x model x fold)
#   04_aggregate_metrics.py   : collect metrics across folds/models          (once per task)
#   05_interpret_report.py    : gene importance ranking across folds/models  (once per task)
#
# Designed to run unchanged on a laptop, an HPC cluster (snakemake --profile
# slurm ...) or AWS (snakemake --profile aws ... / Tibanna / k8s executor) —
# no absolute paths, no hardcoded task/model logic.
# =============================================================================

configfile: "config.yaml"

RESULTS = config["data"]["output_dir"]
TASKS = list(config["tasks"].keys())
N_FOLDS = config["cv"]["n_splits"]
FOLD_IDS = list(range(N_FOLDS))
ALL_MODELS = sorted({m for t in config["tasks"].values() for m in t["models"]})

wildcard_constraints:
    task = "|".join(TASKS),
    model = "|".join(ALL_MODELS),
    fold = r"\d+"


def models_for_task(task):
    return config["tasks"][task]["models"]


def train_outputs_for_task(wildcards):
    """All fold-level metrics files belonging to one task (all its models)."""
    task = wildcards.task
    return [
        f"{RESULTS}/{task}/{model}/fold{fold}/metrics.json"
        for model in models_for_task(task)
        for fold in FOLD_IDS
    ]


def importance_outputs_for_task(wildcards):
    task = wildcards.task
    return [
        f"{RESULTS}/{task}/{model}/fold{fold}/feature_importances.csv"
        for model in models_for_task(task)
        for fold in FOLD_IDS
    ]


rule all:
    input:
        [f"{RESULTS}/{task}/aggregated_metrics.csv" for task in TASKS],
        [f"{RESULTS}/{task}/model_comparison.png" for task in TASKS],
        [f"{RESULTS}/{task}/gene_importance_report.csv" for task in TASKS],


# -----------------------------------------------------------------------------
# 1) Load + clean (runs once for the whole project)
# -----------------------------------------------------------------------------
rule load_clean_data:
    input:
        counts=config["data"]["counts_file"],
        clinical=config["data"]["clinical_file"],
    output:
        expression=f"{RESULTS}/data/expression_clean.csv",
        clinical=f"{RESULTS}/data/clinical_clean.csv",
        qc_report=f"{RESULTS}/data/qc_report.json",
    log:
        f"{RESULTS}/logs/load_clean_data.log",
    script:
        "scripts/01_load_clean_data.py"


# -----------------------------------------------------------------------------
# 2) Generate CV splits — once per task, stratified, seed fixed in config.yaml
#    so every model for a given task is evaluated on identical folds.
# -----------------------------------------------------------------------------
rule generate_cv_splits:
    input:
        expression=rules.load_clean_data.output.expression,
        clinical=rules.load_clean_data.output.clinical,
    output:
        splits=f"{RESULTS}/splits/{{task}}_splits.json",
    params:
        task="{task}",
    log:
        f"{RESULTS}/logs/generate_cv_splits_{{task}}.log",
    script:
        "scripts/02_generate_cv_splits.py"


# -----------------------------------------------------------------------------
# 3) Train + evaluate one (task, model, fold) combination.
#    Low-expression / low-variance filtering happens INSIDE this script,
#    fit on the training fold only, to avoid leakage.
# -----------------------------------------------------------------------------
rule train_model:
    input:
        expression=rules.load_clean_data.output.expression,
        clinical=rules.load_clean_data.output.clinical,
        splits=f"{RESULTS}/splits/{{task}}_splits.json",
    output:
        metrics=f"{RESULTS}/{{task}}/{{model}}/fold{{fold}}/metrics.json",
        predictions=f"{RESULTS}/{{task}}/{{model}}/fold{{fold}}/predictions.csv",
        importances=f"{RESULTS}/{{task}}/{{model}}/fold{{fold}}/feature_importances.csv",
        model_pkl=f"{RESULTS}/{{task}}/{{model}}/fold{{fold}}/model.pkl",
    params:
        task="{task}",
        model="{model}",
        fold="{fold}",
    log:
        f"{RESULTS}/logs/train_{{task}}_{{model}}_fold{{fold}}.log",
    threads: 4
    script:
        "scripts/03_train_model.py"


# -----------------------------------------------------------------------------
# 4) Aggregate metrics + comparison plots across models/folds, per task.
# -----------------------------------------------------------------------------
rule aggregate_metrics:
    input:
        train_outputs_for_task,
    output:
        table=f"{RESULTS}/{{task}}/aggregated_metrics.csv",
        plot=f"{RESULTS}/{{task}}/model_comparison.png",
    params:
        task="{task}",
    script:
        "scripts/04_aggregate_metrics.py"


# -----------------------------------------------------------------------------
# 5) Interpretation: gene importance ranking across folds/models, per task.
# -----------------------------------------------------------------------------
rule interpret_report:
    input:
        importance_outputs_for_task,
    output:
        report=f"{RESULTS}/{{task}}/gene_importance_report.csv",
    params:
        task="{task}",
    script:
        "scripts/05_interpret_report.py"
