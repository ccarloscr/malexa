#!/usr/bin/env python3
"""
slurm-status.py

Called by Snakemake as `cluster-status` with a SLURM job ID (the value
`sbatch --parsable` printed when the job was submitted). Must print exactly
one of: running / success / failed.

Uses `sacct` (job accounting) rather than relying on output files appearing,
so failed jobs are detected immediately instead of only after latency-wait
times out.
"""
import subprocess
import sys

jobid = sys.argv[1]

try:
    out = subprocess.check_output(
        ["sacct", "-j", jobid, "--format=State", "--noheader", "--parsable2"],
        text=True,
    )
except subprocess.CalledProcessError:
    # sacct itself failed (e.g. transient accounting DB hiccup) -> treat as
    # still running so Snakemake keeps polling instead of aborting the job.
    print("running")
    sys.exit(0)

lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
if not lines:
    print("running")
    sys.exit(0)

# First line corresponds to the main job step's state.
state = lines[0].split("|")[0]

RUNNING_STATES = {"PENDING", "CONFIGURING", "COMPLETING", "RUNNING", "RESIZING", "SUSPENDED"}
FAILED_STATES = {
    "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
}

if state == "COMPLETED":
    print("success")
elif state in RUNNING_STATES:
    print("running")
elif any(state.startswith(f) for f in FAILED_STATES):
    print("failed")
else:
    # Unknown/unclassified state -> keep polling rather than kill the run.
    print("running")
