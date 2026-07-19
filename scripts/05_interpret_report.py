"""
05_interpret_report.py

Aggregate per-fold feature importances across all models for one task,
rank genes, and produce a tidy gene-importance report CSV and figure.

What this script does:
  - Reads every feature_importances.csv written by 03_train_model.py for the
    task (all models x all folds).
  - Applies the aggregation strategy configured in
    config.yaml[interpretation][aggregation]:
      * "mean_rank"       — rank genes within each fold (1 = most important),
                            then average ranks across folds.  Robust to
                            different raw importance scales across models.
      * "mean_importance" — average raw importance scores across folds
                            (appropriate when scores are already comparable,
                            e.g. coefficients from the same model family).
  - Reports aggregated results:
      * Per-model ranking (top N genes per model).
      * Cross-model consensus ranking (genes appearing across multiple models
        receive a combined score).
  - Writes a CSV with columns:
      gene, model, mean_importance, std_importance, mean_rank,
      n_folds_present, consensus_rank
  - Generates a horizontal bar figure:
      * One panel per model + one consensus panel.
      * Top-N genes, coloured by mean importance (blue = positive, red =
        negative for signed coefficients; absolute value bar width).

Expected feature_importances.csv schema (from 03_train_model.py):
    gene, importance, fold, model, task
  OR the minimal schema:
    gene, importance
  (model/fold/task inferred from path when absent)

Input  (from Snakemake):
  snakemake.input  : list of feature_importances.csv paths

Output (from Snakemake):
  snakemake.output.report : results/<task>/gene_importance_report.csv

Parameters:
  snakemake.params.task   : task name
  snakemake.config        : full config dict

Standalone CLI:
  python 05_interpret_report.py \\
      --importances-dir results/<task> \\
      --task cancer_stage \\
      --config config.yaml \\
      --out-report results/<task>/gene_importance_report.csv \\
      --out-plot   results/<task>/gene_importance_plot.png
"""

import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(log_path=None):
    logger = logging.getLogger("interpret_report")
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
# I/O
# --------------------------------------------------------------------------- #
def _parse_model_fold_from_path(path: Path) -> tuple[str, int]:
    """Infer (model_name, fold_index) from directory structure.

    Expected layout:  .../<model>/fold<N>/feature_importances.csv
    """
    parts = path.parts
    for i, part in enumerate(parts):
        m = re.fullmatch(r"fold(\d+)", part)
        if m:
            fold  = int(m.group(1))
            model = parts[i - 1] if i >= 1 else "unknown"
            return model, fold
    raise ValueError(
        f"Cannot infer (model, fold) from path '{path}'. "
        "Expected structure: .../<model>/fold<N>/feature_importances.csv"
    )


def load_all_importances(importance_paths: list[Path], logger) -> pd.DataFrame:
    """Load all feature_importances CSVs and return a single long DataFrame.

    Guaranteed columns in output:
        gene, importance, model, fold, task
    """
    frames = []
    for p in importance_paths:
        p = Path(p)
        if not p.exists():
            logger.warning(f"feature_importances.csv not found: {p}  — skipping.")
            continue

        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.error(f"Cannot read {p}: {e}  — skipping.")
            continue

        # must have at least gene + importance
        if "gene" not in df.columns or "importance" not in df.columns:
            logger.error(
                f"{p}: expected columns ['gene', 'importance']; "
                f"found {list(df.columns)}  — skipping."
            )
            continue

        model_path, fold_path = _parse_model_fold_from_path(p)

        # fill in model/fold/task from path when missing from file
        if "model" not in df.columns:
            df["model"] = model_path
        if "fold" not in df.columns:
            df["fold"] = fold_path
        if "task" not in df.columns:
            df["task"] = "unknown"

        df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
        df = df.dropna(subset=["importance"])

        frames.append(df[["gene", "importance", "model", "fold", "task"]])

    if not frames:
        raise RuntimeError("No valid feature_importances.csv files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["fold"] = combined["fold"].astype(int)
    return combined


# --------------------------------------------------------------------------- #
# Ranking helpers
# --------------------------------------------------------------------------- #
def _rank_within_fold(group: pd.DataFrame) -> pd.Series:
    """Rank genes within one (model, fold) group by |importance|.

    Rank 1 = most important (largest absolute value).
    """
    return group["importance"].abs().rank(ascending=False, method="average")


def aggregate_by_mean_rank(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Aggregate importances by averaging per-fold ranks.

    Output columns per (gene, model):
        mean_rank, std_rank, mean_importance, std_importance, n_folds_present
    """
    logger.info("Aggregation strategy: mean_rank")

    df = df.copy()
    df["abs_rank"] = (
        df.groupby(["model", "fold"], group_keys=False)
          .apply(_rank_within_fold)
    )

    agg = (
        df.groupby(["gene", "model"])
          .agg(
              mean_rank        =("abs_rank",   "mean"),
              std_rank         =("abs_rank",   "std"),
              mean_importance  =("importance", "mean"),
              std_importance   =("importance", "std"),
              n_folds_present  =("fold",       "nunique"),
          )
          .reset_index()
    )
    # lower mean_rank = more important
    agg = agg.sort_values(["model", "mean_rank"])
    return agg


def aggregate_by_mean_importance(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Aggregate importances by averaging raw scores across folds.

    Output columns per (gene, model):
        mean_importance, std_importance, n_folds_present
        (mean_rank derived from mean_importance for compatibility)
    """
    logger.info("Aggregation strategy: mean_importance")

    agg = (
        df.groupby(["gene", "model"])
          .agg(
              mean_importance =("importance", "mean"),
              std_importance  =("importance", "std"),
              n_folds_present =("fold",       "nunique"),
          )
          .reset_index()
    )
    # derive mean_rank from |mean_importance| for consistency
    agg["mean_rank"] = (
        agg.groupby("model")["mean_importance"]
           .transform(lambda s: s.abs().rank(ascending=False, method="average"))
    )
    agg["std_rank"] = np.nan
    agg = agg.sort_values(["model", "mean_rank"])
    return agg


def compute_consensus_rank(agg_df: pd.DataFrame,
                           n_folds: int,
                           top_n: int,
                           logger) -> pd.DataFrame:
    """Compute a cross-model consensus ranking.

    Strategy:
      1. Take the top `top_n` genes per model (by mean_rank).
      2. For each gene, collect its mean_rank from every model it appeared in.
      3. Penalise genes that appear in fewer models by imputing the worst
         possible rank (max_rank + 1) for missing models.
      4. Consensus score = mean of (observed + imputed) ranks.

    Output: DataFrame with columns [gene, consensus_score, n_models_present,
    models_present, consensus_rank] sorted by consensus_rank.
    """
    models = agg_df["model"].unique().tolist()
    n_models = len(models)

    # worst rank = total genes in the reference model (use max observed)
    worst_rank = agg_df["mean_rank"].max() + 1

    # limit to top_n per model to keep the consensus focused
    top_per_model = (
        agg_df.sort_values("mean_rank")
              .groupby("model")
              .head(top_n)
    )
    candidate_genes = top_per_model["gene"].unique()

    records = []
    for gene in candidate_genes:
        gene_rows = agg_df[agg_df["gene"] == gene]
        models_with_gene = gene_rows["model"].tolist()
        ranks_observed   = gene_rows.set_index("model")["mean_rank"].to_dict()

        rank_values = []
        for m in models:
            rank_values.append(ranks_observed.get(m, worst_rank))

        records.append({
            "gene":              gene,
            "consensus_score":   np.mean(rank_values),
            "n_models_present":  len(models_with_gene),
            "models_present":    ",".join(sorted(models_with_gene)),
        })

    consensus_df = pd.DataFrame(records)
    consensus_df = consensus_df.sort_values("consensus_score")
    consensus_df["consensus_rank"] = range(1, len(consensus_df) + 1)

    logger.info(
        f"Consensus ranking: {len(consensus_df)} candidate genes across "
        f"{n_models} model(s)."
    )
    return consensus_df


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
_SIGNED_MODELS = {"elasticnet_logreg", "linear_svm"}


def _is_signed_model(model_name: str) -> bool:
    """Return True if the model produces signed importance scores (coefficients)."""
    return any(key in model_name.lower() for key in ("logreg", "svm", "linear"))


def _bar_colors_signed(values: pd.Series) -> list:
    """Map signed importance values to blue (positive) / red (negative)."""
    return ["steelblue" if v >= 0 else "tomato" for v in values]


def _bar_colors_unsigned(values: pd.Series) -> list:
    norm = plt.Normalize(values.min(), values.max())
    cmap = plt.get_cmap("viridis")
    return [cmap(norm(v)) for v in values]


def plot_gene_importance(agg_df: pd.DataFrame,
                         consensus_df: pd.DataFrame,
                         top_n: int,
                         task_name: str,
                         out_path: Path,
                         logger) -> None:
    """Horizontal bar figure: one panel per model + one consensus panel."""
    models  = sorted(agg_df["model"].unique().tolist())
    n_panels = len(models) + 1    # +1 for consensus
    fig_height = max(6, 0.35 * top_n + 2)

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6 * n_panels, fig_height),
                             squeeze=False)
    axes = axes[0]

    # ------------------------------------------------------------------ #
    # Per-model panels
    # ------------------------------------------------------------------ #
    for ax, model in zip(axes[:-1], models):
        model_df = (
            agg_df[agg_df["model"] == model]
            .sort_values("mean_rank")
            .head(top_n)
        )
        # reverse for horizontal bar (top gene at top)
        model_df = model_df.iloc[::-1]

        signed = _is_signed_model(model)
        if signed:
            bar_vals = model_df["mean_importance"]
            colors   = _bar_colors_signed(bar_vals)
            xlabel   = "Mean Coefficient"
        else:
            bar_vals = model_df["mean_importance"].abs()
            colors   = _bar_colors_unsigned(bar_vals)
            xlabel   = "Mean Importance"

        ax.barh(model_df["gene"], bar_vals, color=colors,
                xerr=model_df["std_importance"].fillna(0),
                error_kw=dict(elinewidth=0.7, capsize=2, alpha=0.6),
                height=0.7, edgecolor="white", linewidth=0.4)

        if signed:
            ax.axvline(0, color="black", linewidth=0.8)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(f"{model}\n(top {top_n})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # annotate fold coverage using axes-fraction x so placement is
        # stable regardless of bar direction or auto-scaled xlim
        for i, (_, row) in enumerate(model_df.iterrows()):
            ax.text(
                0.01, i,
                f" n={int(row['n_folds_present'])}",
                va="center", ha="left", fontsize=6, color="grey",
                transform=ax.get_yaxis_transform(),
            )

    # ------------------------------------------------------------------ #
    # Consensus panel
    # ------------------------------------------------------------------ #
    ax_cons = axes[-1]
    cons_top = consensus_df.head(top_n).iloc[::-1]

    # colour by n_models_present
    n_models_total = agg_df["model"].nunique()
    norm = mcolors.Normalize(vmin=1, vmax=n_models_total)
    cmap = plt.get_cmap("YlOrRd")
    colors_cons = [cmap(norm(v)) for v in cons_top["n_models_present"]]

    bar_widths = top_n - cons_top["consensus_score"].rank(method="first") + 1
    ax_cons.barh(cons_top["gene"], bar_widths / bar_widths.max(),
                 color=colors_cons, height=0.7,
                 edgecolor="white", linewidth=0.4)

    # add colorbar for n_models
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cons, shrink=0.6, pad=0.02)
    cbar.set_label("# models", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_cons.set_xlabel("Consensus score (relative)", fontsize=9)
    ax_cons.set_title(f"Cross-model Consensus\n(top {top_n})",
                      fontsize=10, fontweight="bold")
    ax_cons.tick_params(axis="y", labelsize=7)
    ax_cons.tick_params(axis="x", labelsize=8)
    ax_cons.set_xlim(0, 1.1)
    ax_cons.spines[["top", "right"]].set_visible(False)

    task_display = task_name.replace("_", " ").title()
    fig.suptitle(f"Gene Importance — {task_display}", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved gene importance plot: {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(importance_paths: list,
         task_name: str,
         config: dict,
         out_report: str,
         out_plot: str = None,
         log_path: str = None) -> None:

    logger = get_logger(log_path)
    logger.info(f"Task: {task_name}")
    logger.info(f"Processing {len(importance_paths)} feature_importances.csv files ...")

    interp_cfg   = config.get("interpretation", {})
    top_n        = int(interp_cfg.get("top_n_genes", 50))
    strategy     = interp_cfg.get("aggregation", "mean_rank")
    cv_cfg   = config.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    n_repeats = int(cv_cfg.get("n_repeats", 1)) if cv_cfg.get("method") == "RepeatedStratifiedKFold" else 1
    n_folds  = n_splits * n_repeats

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #
    df = load_all_importances([Path(p) for p in importance_paths], logger)
    logger.info(
        f"Loaded importances: {df['gene'].nunique()} unique genes, "
        f"{df['model'].nunique()} model(s), "
        f"{df['fold'].nunique()} fold(s)."
    )

    # fill missing task name
    df["task"] = df["task"].replace("unknown", task_name)

    # ------------------------------------------------------------------ #
    # Aggregate per (gene, model)
    # ------------------------------------------------------------------ #
    if strategy == "mean_rank":
        agg_df = aggregate_by_mean_rank(df, logger)
    elif strategy == "mean_importance":
        agg_df = aggregate_by_mean_importance(df, logger)
    else:
        logger.warning(
            f"Unknown aggregation strategy '{strategy}'; falling back to mean_rank."
        )
        agg_df = aggregate_by_mean_rank(df, logger)

    agg_df["task"] = task_name

    # ------------------------------------------------------------------ #
    # Consensus ranking across models
    # ------------------------------------------------------------------ #
    consensus_df = compute_consensus_rank(agg_df, n_folds, top_n, logger)

    # ------------------------------------------------------------------ #
    # Merge consensus info into main table
    # ------------------------------------------------------------------ #
    full_df = agg_df.merge(
        consensus_df[["gene", "consensus_score", "consensus_rank",
                       "n_models_present", "models_present"]],
        on="gene",
        how="left",
    )

    # Per-model rank within top_n
    full_df["top_n_rank"] = (
        full_df.groupby("model")["mean_rank"]
               .rank(ascending=True, method="first")
               .astype(int)
    )

    # Column order
    col_order = [
        "task", "gene", "model",
        "mean_importance", "std_importance",
        "mean_rank", "std_rank",
        "n_folds_present", "top_n_rank",
        "consensus_score", "consensus_rank",
        "n_models_present", "models_present",
    ]
    col_order = [c for c in col_order if c in full_df.columns]
    full_df   = full_df[col_order].sort_values(["model", "mean_rank"])

    # ------------------------------------------------------------------ #
    # Save report CSV
    # ------------------------------------------------------------------ #
    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_report, index=False, float_format="%.6f")
    logger.info(f"Saved gene importance report: {out_report}")

    # ------------------------------------------------------------------ #
    # Log top-N per model + consensus
    # ------------------------------------------------------------------ #
    for model in sorted(full_df["model"].unique()):
        top = full_df[full_df["model"] == model].head(min(top_n, 10))
        lines = [f"\n  Top genes for model '{model}':"]
        for _, row in top.iterrows():
            lines.append(
                f"    rank {int(row['top_n_rank']):3d}  {row['gene']:20s}  "
                f"mean_imp={row['mean_importance']:+.4f}  "
                f"mean_rank={row['mean_rank']:.1f}  "
                f"folds={int(row['n_folds_present'])}"
            )
        logger.info("\n".join(lines))

    cons_top10 = consensus_df.head(10)
    lines = ["\n  Cross-model consensus top genes:"]
    for _, row in cons_top10.iterrows():
        lines.append(
            f"    consensus_rank {int(row['consensus_rank']):3d}  "
            f"{row['gene']:20s}  "
            f"score={row['consensus_score']:.2f}  "
            f"n_models={int(row['n_models_present'])}  "
            f"({row['models_present']})"
        )
    logger.info("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Plot (optional — only if out_plot is provided)
    # ------------------------------------------------------------------ #
    if out_plot:
        plot_gene_importance(
            agg_df=agg_df,
            consensus_df=consensus_df,
            top_n=min(top_n, 30),   # cap at 30 for readability
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
        # The Snakefile only declares output.report; derive the plot path from it
        report_path = snakemake.output.report
        plot_path   = str(Path(report_path).with_name("gene_importance_plot.png"))

        main(
            importance_paths=list(snakemake.input),
            task_name=snakemake.params.task,
            config=snakemake.config,
            out_report=report_path,
            out_plot=plot_path,
            log_path=snakemake.log[0] if snakemake.log else None,
        )

    else:
        import argparse
        import yaml

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--importances-dir", required=True,
            help="Root results directory for the task. "
                 "The script will glob for all feature_importances.csv files recursively."
        )
        parser.add_argument("--task",       required=True,
                            help="Task name as defined in config.yaml")
        parser.add_argument("--config",     required=True,
                            help="Path to config.yaml")
        parser.add_argument("--out-report", required=True,
                            help="Output path for gene_importance_report.csv")
        parser.add_argument("--out-plot",   default=None,
                            help="Output path for gene_importance_plot.png "
                                 "(optional; derived from --out-report if omitted)")
        parser.add_argument("--log",        default=None)
        args = parser.parse_args()

        importance_paths = sorted(
            Path(args.importances_dir).rglob("feature_importances.csv")
        )
        if not importance_paths:
            raise FileNotFoundError(
                f"No feature_importances.csv files found under '{args.importances_dir}'"
            )

        out_plot = args.out_plot or str(
            Path(args.out_report).with_name("gene_importance_plot.png")
        )

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        main(
            importance_paths=importance_paths,
            task_name=args.task,
            config=cfg,
            out_report=args.out_report,
            out_plot=out_plot,
            log_path=args.log,
        )
