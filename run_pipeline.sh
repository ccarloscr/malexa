#!/bin/bash
# =============================================================================
# run_pipeline.sh
#
# Launches the Snakemake pipeline from the LOGIN node. Snakemake itself does
# very little work (it just submits sbatch jobs for each rule and polls their
# status via slurm-status.py) — the actual computation runs in separate SLURM
# jobs, one per rule invocation, using the resources set in the Snakefile.
#
# Run this with nohup/tmux/screen since it needs to stay alive for the whole
# pipeline duration to keep submitting and polling jobs:
#
#   tmux new -s malexa
#   ./run_pipeline.sh
#   # detach: Ctrl-b d   |   reattach later: tmux attach -t malexa
#
# or:
#
#   nohup ./run_pipeline.sh > logs/run_pipeline.log 2>&1 &
# =============================================================================
set -e

source /home/DDGcarlos/miniconda3/etc/profile.d/conda.sh
conda activate malexa

mkdir -p logs/slurm results/logs

snakemake \
    --profile profiles/slurm \
    --latency-wait 60 \
    --rerun-incomplete \
    --keep-going \
    "$@"

