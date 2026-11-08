#!/bin/bash
#SBATCH --job-name=hoi-eval-deprecated
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
set -euo pipefail
echo 'DEPRECATED: Separate evaluation W&B runs are disabled. Use cluster/hoi_unified_monitor.sbatch.sh (hoi_tables.unified_monitor) to write eval/* into its owning training run.' >&2
exit 2
