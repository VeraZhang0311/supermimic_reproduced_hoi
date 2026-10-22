#!/usr/bin/env bash
set -euo pipefail
r=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi
mkdir -p "$r/logs" "$r/provenance" "$r/tmp" "$r/.cache"
bash -n "$r/cluster/prepare.sbatch.sh"
test ! -e "$r/provenance/prepare-job.txt"
/opt/slurm/bin/sbatch --parsable --output="$r/logs/prepare-%j.out" "$r/cluster/prepare.sbatch.sh" > "$r/provenance/prepare-job.txt"
cat "$r/provenance/prepare-job.txt"
/mnt/projects/ll/tianzizh/SuperMimic_HOI/envs/tracking-py312/bin/python - <<'PY'
import wandb, inspect, json
api=wandb.Api()
print(json.dumps({'create_project_signature':str(inspect.signature(api.create_project)),'base_url':api.settings.get('base_url'),'viewer_entity':api.viewer.get('entity')}))
PY
