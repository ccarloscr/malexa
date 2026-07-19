# =============================================================================
# Snakefile — expression-based mutation-status classification pipeline
#
# Runs all 5 stages end-to-end (config-driven, see config.yaml):
#   01_load_clean_data.py     : missing-value handling, sample alignment      (once)
#   02_generate_cv_splits.py  : stratified CV split indices per task          (once per task)
#   03_train_model.py         : fold-local feature selection + train + eval   (per task x model x split)
#   04_aggregate_metrics.py   : aggregate per-fold metrics, produce summary + plot  (once per task)
#   05_interpret_report.py    : aggregate feature importances, gene ranking + plot  (once per task)
#
# Designed to run unchanged on a laptop, an HPC cluster (snakemake --profile
# slurm ...) or AWS (snakemake --profile aws ...) — no absolute paths, no
# hardcoded task/model logic.
#
# Expected layout (adjust the `script:` paths below if yours differs):
#   ./snakefile
#   ./config.yaml
#   ./scripts/01_load_clean_data.py
#   ./scripts/02_generate_cv_splits.py
#   ./scripts/03_train_model.py
#   ./scripts/04_aggregate_metrics.py
#   ./scripts/05_interpret_report.py
# =============================================================================

configfile: "config.yaml"

RESULTS = config["data"]["output_dir"]
TASKS = list(config["tasks"].keys())
ALL_MODELS = sorted({m for t in config["tasks"].values() for m in t["models"]})


def total_cv_splits(cv_config):
    """Mirror the split-count logic in 02_generate_cv_splits.py's generate_splits().

    - StratifiedKFold          -> n_splits
    - RepeatedStratifiedKFold  -> n_splits * n_repeats
    - StratifiedShuffleSplit   -> n_splits

    IMPORTANT: 03_train_model.py looks up a fold by matching the split's
    `split_idx` field (0 .. total_splits-1), not by `fold` (0 .. n_splits-1).
    With RepeatedStratifiedKFold, n_splits alone under-counts the real
    number of splits and silently drops every repeat past the first.
    """
    method = cv_config.get("method")
    n_splits = cv_config["n_splits"]
    if method == "RepeatedStratifiedKFold":
        return n_splits * cv_config.get("n_repeats", 1)
    return n_splits


SPLIT_IDS = list(range(total_cv_splits(config["cv"])))

wildcard_constraints:
    task = "|".join(TASKS),
    model = "|".join(ALL_MODELS),
    fold = r"\d+"


def models_for_task(task):
    return config["tasks"][task]["models"]



def train_outputs_for_task(wildcards):
    """All split-level metrics files belonging to one task (all its models)."""
    task = wildcards.task
    return [
        f"{RESULTS}/{task}/{model}/fold{split_idx}/metrics.json"
        for model in models_for_task(task)
        for split_idx in SPLIT_IDS
    ]


def interpret_inputs_for_task(wildcards):
    """All fold-level feature_importances files belonging to one task (all its models)."""
    task = wildcards.task
    return [
        f"{RESULTS}/{task}/{model}/fold{split_idx}/feature_importances.csv"
        for model in models_for_task(task)
        for split_idx in SPLIT_IDS
    ]


rule all:
    input:
        # Per-fold training outputs (rule train_model)
        [
            f"{RESULTS}/{task}/{model}/fold{split_idx}/metrics.json"
            for task in TASKS
            for model in models_for_task(task)
            for split_idx in SPLIT_IDS
        ],
        # Per-task aggregation and interpretation outputs (rules 04 and 05)
        [f"{RESULTS}/{task}/aggregated_metrics.csv" for task in TASKS],
        [f"{RESULTS}/{task}/model_comparison.png"   for task in TASKS],
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
    threads: 4
    resources:
        mem_mb=16000,
        time_min=60,
        partition="irbio01",
    script:
        "scripts/01_load_clean_data.py"


# -----------------------------------------------------------------------------
# 2) Generate CV splits — once per task, stratified, seed fixed in config.yaml
#    so every model for a given task is evaluated on identical splits.
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
    threads: 1
    resources:
        mem_mb=8000,
        time_min=30,
        partition="irbio01",
    script:
        "scripts/02_generate_cv_splits.py"


# -----------------------------------------------------------------------------
# 3) Train + evaluate one (task, model, split) combination.
#    Low-expression / low-variance filtering happens INSIDE this script,
#    fit on the training fold only, to avoid leakage.
#
#    The `fold` wildcard here is the split's `split_idx` (0 .. total_splits-1)
#    from the splits JSON, NOT necessarily 0 .. n_splits-1 — see
#    total_cv_splits() above for why that distinction matters under
#    RepeatedStratifiedKFold.
# -----------------------------------------------------------------------------
rule train_model:
    input:
        expression=rules.load_clean_data.output.expression,
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
    resources:
        mem_mb=8000,
        time_min=240,
        partition="irbio01",
    script:
        "scripts/03_train_model.py"

# -----------------------------------------------------------------------------
# 4) Aggregate per-fold metrics — once per task
# -----------------------------------------------------------------------------
rule aggregate_metrics:
    input:
        metrics=train_outputs_for_task,
    output:
        table=f"{RESULTS}/{{task}}/aggregated_metrics.csv",
        plot=f"{RESULTS}/{{task}}/model_comparison.png",
    params:
        task="{task}",
    log:
        f"{RESULTS}/logs/aggregate_metrics_{{task}}.log",
    threads: 1
    resources:
        mem_mb=4000,
        time_min=30,
        partition="irbio01",
    script:
        "scripts/04_aggregate_metrics.py"


# -----------------------------------------------------------------------------
# 5) Interpret + report feature importances — once per task
# -----------------------------------------------------------------------------
rule interpret_report:
    input:
        importances=interpret_inputs_for_task,
    output:
        report=f"{RESULTS}/{{task}}/gene_importance_report.csv",
    params:
        task="{task}",
    log:
        f"{RESULTS}/logs/interpret_report_{{task}}.log",
    threads: 1
    resources:
        mem_mb=4000,
        time_min=30,
        partition="irbio01",
    script:
        "scripts/05_interpret_report.py"
