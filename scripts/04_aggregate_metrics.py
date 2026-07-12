"""
04_aggregate_metrics.py

Aggregate per-fold metrics from all (model x fold) combinations for one task,
produce a tidy summary CSV, and generate a model-comparison figure.

What this script does:
  - Reads every metrics.json written by 03_train_model.py for the task.
  - Concatenates them into a long-format DataFrame (one row per fold x model).
  - Computes per-model summary statistics (mean ± std across folds) for every
    metric defined in config.yaml[evaluation][metrics].
  - Writes a CSV with both the per-fold rows AND summary rows (tagged with
    fold="mean" / "std") so downstream consumers can slice either way.
  - Generates a comparison figure:
      * One panel per metric (up to 5 configured metrics).
      * Box-/strip-plot of per-fold scores, with mean ± 1-SD error bars.
      * Models sorted by descending primary_metric mean.
      * Horizontal reference lines at chance level (0.5 for AUC-type metrics).

Input  (from Snakemake):
  snakemake.input  : list of metrics.json paths  [task x model x fold]

Output (from Snakemake):
  snakemake.output.table  : results/<task>/aggregated_metrics.csv
  snakemake.output.plot   : results/<task>/model_comparison.png

Parameters (from Snakemake):
  snakemake.params.task   : task name (string)
  snakemake.config        : full config dict

Standalone CLI:
  python 04_aggregate_metrics.py \\
      --metrics-dir results/<task> \\
      --task cancer_stage \\
      --config config.yaml \\
      --out-table results/<task>/aggregated_metrics.csv \\
      --out-plot  results/<task>/model_comparison.png
"""

import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(log_path=None):
    logger = logging.getLogger("aggregate_metrics")
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
# I/O helpers
# --------------------------------------------------------------------------- #
def _parse_model_fold_from_path(path: Path) -> tuple[str, int]:
    """Infer (model_name, fold_index) from the directory structure.

    Expected layout:
        results/<task>/<model>/fold<N>/metrics.json

    Falls back to scanning parent directory names for a pattern match so the
    function remains robust if the root results dir is renamed.
    """
    parts = path.parts
    # walk up from the file; look for fold<N> then the model name above it
    for i, part in enumerate(parts):
        m = re.fullmatch(r"fold(\d+)", part)
        if m:
            fold = int(m.group(1))
            model = parts[i - 1] if i >= 1 else "unknown"
            return model, fold
    raise ValueError(
        f"Cannot infer (model, fold) from path '{path}'. "
        "Expected structure: .../<model>/fold<N>/metrics.json"
    )


def load_all_metrics(metrics_paths: list[Path], metric_keys: list[str], logger) -> pd.DataFrame:
    """Read every metrics.json and return a long-format DataFrame.

    Columns: task, model, fold, <metric_1>, <metric_2>, ..., best_params
    """
    records = []
    for p in metrics_paths:
        p = Path(p)
        if not p.exists():
            logger.warning(f"metrics.json not found: {p}  — skipping.")
            continue

        try:
            with open(p) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Cannot parse {p}: {e}  — skipping.")
            continue

        model, fold = _parse_model_fold_from_path(p)

        row = {
            "task":        data.get("task", "unknown"),
            "model":       data.get("model", model),
            "fold":        data.get("fold",  fold),
            "n_train":     data.get("n_train", np.nan),
            "n_test":      data.get("n_test",  np.nan),
            "n_features_selected": data.get("n_features_selected", np.nan),
            "best_params": json.dumps(data.get("best_params", {})),
        }

        # pull requested metrics; warn if absent
        for key in metric_keys:
            val = data.get(key, np.nan)
            if np.isnan(float(val) if val is not None else float("nan")):
                logger.warning(f"Metric '{key}' missing in {p}")
            row[key] = val

        records.append(row)

    if not records:
        raise RuntimeError("No valid metrics.json files could be loaded.")

    df = pd.DataFrame(records)
    df["fold"] = df["fold"].astype(int)
    df = df.sort_values(["model", "fold"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def compute_summary(df: pd.DataFrame, metric_keys: list[str]) -> pd.DataFrame:
    """Append per-model mean and std rows to the per-fold DataFrame.

    Returns a combined DataFrame with fold values preserved and summary rows
    tagged: fold == -1 for mean, fold == -2 for std.
    """
    summary_rows = []
    for model, grp in df.groupby("model", sort=False):
        for stat, fold_tag in [("mean", -1), ("std", -2)]:
            row = {"task": grp["task"].iloc[0], "model": model, "fold": fold_tag}
            for key in metric_keys:
                vals = pd.to_numeric(grp[key], errors="coerce")
                row[key] = vals.mean() if stat == "mean" else vals.std(ddof=1)
            row["n_train"] = grp["n_train"].mean()
            row["n_test"]  = grp["n_test"].mean()
            row["n_features_selected"] = grp["n_features_selected"].mean()
            row["best_params"] = ""
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    combined   = pd.concat([df, summary_df], ignore_index=True)
    return combined


def print_summary_table(df: pd.DataFrame, metric_keys: list[str],
                        primary_metric: str, logger) -> None:
    """Log a human-readable per-model summary (mean ± std)."""
    mean_df = df[df["fold"] == -1].copy()
    std_df  = df[df["fold"] == -2].copy()

    mean_df = mean_df.set_index("model")
    std_df  = std_df.set_index("model")

    # sort models by primary metric mean (descending)
    if primary_metric in mean_df.columns:
        model_order = (
            mean_df[primary_metric]
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        model_order = mean_df.index.tolist()

    lines = [f"\n{'Model':30s}" + "".join(f"  {m:>22s}" for m in metric_keys)]
    lines.append("-" * (30 + 24 * len(metric_keys)))

    for model in model_order:
        cells = []
        for key in metric_keys:
            mu  = mean_df.loc[model, key] if key in mean_df.columns else np.nan
            sd  = std_df.loc[model,  key] if key in std_df.columns  else np.nan
            cells.append(f"{mu:.4f} ± {sd:.4f}")
        lines.append(f"{model:30s}" + "".join(f"  {c:>22s}" for c in cells))

    logger.info("Model comparison summary:" + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
# Chance-level reference for each metric type
_CHANCE_LEVEL = {
    "roc_auc":            0.5,
    "average_precision":  None,   # depends on class frequency; skip
    "balanced_accuracy":  0.5,
    "f1_weighted":        None,
    "matthews_corrcoef":  0.0,
}

# Nicer axis labels
_METRIC_LABELS = {
    "roc_auc":            "ROC-AUC",
    "average_precision":  "PR-AUC (Avg Precision)",
    "balanced_accuracy":  "Balanced Accuracy",
    "f1_weighted":        "F1 (weighted)",
    "matthews_corrcoef":  "Matthews CC",
}


def _model_color_map(models: list[str]) -> dict[str, str]:
    palette = plt.get_cmap("tab10")
    return {m: palette(i % 10) for i, m in enumerate(models)}


def plot_comparison(df: pd.DataFrame,
                    metric_keys: list[str],
                    primary_metric: str,
                    task_name: str,
                    out_path: Path,
                    logger) -> None:
    """Generate a multi-panel model comparison figure.

    Layout: one column per metric, with:
      - Individual fold scores as scatter points (jittered slightly).
      - Mean ± 1-SD error bar per model.
      - Optional chance-level reference line.
      - Models sorted by primary_metric mean (best first).
    """
    mean_df = df[df["fold"] == -1].set_index("model")
    std_df  = df[df["fold"] == -2].set_index("model")
    fold_df = df[df["fold"] >= 0]

    # sort models by primary metric descending
    if primary_metric in mean_df.columns:
        model_order = (
            mean_df[primary_metric]
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        model_order = sorted(mean_df.index.tolist())

    n_metrics = len(metric_keys)
    fig_width  = max(5 * n_metrics, 8)
    fig_height = max(4, 2.0 + 0.6 * len(model_order))

    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height),
                             squeeze=False)
    axes = axes[0]

    color_map = _model_color_map(model_order)
    y_positions = {m: i for i, m in enumerate(reversed(model_order))}

    rng = np.random.default_rng(seed=0)   # reproducible jitter

    for ax, metric in zip(axes, metric_keys):
        label = _METRIC_LABELS.get(metric, metric)

        for model in model_order:
            ypos   = y_positions[model]
            color  = color_map[model]

            # per-fold scatter (jittered on y)
            fold_vals = pd.to_numeric(
                fold_df.loc[fold_df["model"] == model, metric], errors="coerce"
            ).dropna()

            jitter = rng.uniform(-0.18, 0.18, size=len(fold_vals))
            ax.scatter(fold_vals, ypos + jitter,
                       color=color, alpha=0.65, s=28, zorder=3,
                       linewidths=0.4, edgecolors="white")

            # mean ± SD error bar
            if model in mean_df.index:
                mu = float(mean_df.loc[model, metric]) if metric in mean_df.columns else np.nan
                sd = float(std_df.loc[model,  metric]) if metric in std_df.columns  else np.nan
                ax.errorbar(mu, ypos,
                            xerr=sd if not np.isnan(sd) else 0,
                            fmt="D", color=color,
                            markersize=7, capsize=4, linewidth=1.8,
                            zorder=4, markeredgecolor="black",
                            markeredgewidth=0.6)

        # chance-level reference line
        chance = _CHANCE_LEVEL.get(metric)
        if chance is not None:
            ax.axvline(chance, color="grey", linestyle="--",
                       linewidth=0.9, alpha=0.7, label=f"chance ({chance})")
            ax.text(chance + 0.005, -0.55, f"chance={chance}",
                    fontsize=7, color="grey", va="bottom")

        # axes formatting
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(reversed(model_order)), fontsize=9)
        ax.set_xlabel(label, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        # highlight primary metric
        if metric == primary_metric:
            ax.set_title(f"★ {label}", fontsize=10, fontweight="bold")
        else:
            ax.set_title(label, fontsize=10)

    task_display = task_name.replace("_", " ").title()
    fig.suptitle(f"Model Comparison — {task_display}", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(metrics_paths: list,
         task_name: str,
         config: dict,
         out_table: str,
         out_plot: str,
         log_path: str = None) -> None:

    logger = get_logger(log_path)
    logger.info(f"Task: {task_name}")
    logger.info(f"Aggregating {len(metrics_paths)} metrics.json files ...")

    eval_cfg       = config.get("evaluation", {})
    metric_keys    = eval_cfg.get("metrics", ["roc_auc", "balanced_accuracy",
                                              "f1_weighted", "matthews_corrcoef"])
    primary_metric = eval_cfg.get("primary_metric", "roc_auc")

    # ensure primary metric is included even if not in list
    if primary_metric not in metric_keys:
        metric_keys = [primary_metric] + list(metric_keys)

    # ------------------------------------------------------------------ #
    # Load + aggregate
    # ------------------------------------------------------------------ #
    df = load_all_metrics([Path(p) for p in metrics_paths], metric_keys, logger)
    logger.info(
        f"Loaded metrics for {df['model'].nunique()} model(s) x "
        f"{df['fold'].nunique()} fold(s)."
    )

    combined = compute_summary(df, metric_keys)

    # ------------------------------------------------------------------ #
    # Save table
    # ------------------------------------------------------------------ #
    out_table = Path(out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_table, index=False)
    logger.info(f"Saved aggregated metrics table: {out_table}")

    # ------------------------------------------------------------------ #
    # Log summary
    # ------------------------------------------------------------------ #
    print_summary_table(combined, metric_keys, primary_metric, logger)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    plot_comparison(
        df=combined,
        metric_keys=metric_keys,
        primary_metric=primary_metric,
        task_name=task_name,
        out_path=Path(out_plot),
        logger=logger,
    )

    logger.info("Done.")


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    if "snakemake" in globals():
        main(
            metrics_paths=list(snakemake.input),
            task_name=snakemake.params.task,
            config=snakemake.config,
            out_table=snakemake.output.table,
            out_plot=snakemake.output.plot,
            log_path=snakemake.log[0] if snakemake.log else None,
        )

    else:
        import argparse
        import yaml

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--metrics-dir", required=True,
            help="Root results directory for the task (e.g. results/cancer_stage). "
                 "The script will glob for all metrics.json files recursively."
        )
        parser.add_argument("--task",      required=True,
                            help="Task name as defined in config.yaml")
        parser.add_argument("--config",    required=True,
                            help="Path to config.yaml")
        parser.add_argument("--out-table", required=True,
                            help="Output path for aggregated_metrics.csv")
        parser.add_argument("--out-plot",  required=True,
                            help="Output path for model_comparison.png")
        parser.add_argument("--log",       default=None)
        args = parser.parse_args()

        metrics_paths = sorted(
            Path(args.metrics_dir).rglob("metrics.json")
        )
        if not metrics_paths:
            raise FileNotFoundError(
                f"No metrics.json files found under '{args.metrics_dir}'"
            )

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        main(
            metrics_paths=metrics_paths,
            task_name=args.task,
            config=cfg,
            out_table=args.out_table,
            out_plot=args.out_plot,
            log_path=args.log,
        )
