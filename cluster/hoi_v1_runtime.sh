#!/usr/bin/env bash
# PyTorch's CPU tensor transport creates AF_UNIX listeners underneath TMPDIR.
# Keep enough room for its random pymp/listener suffix within Linux's 108 bytes.
hoi_v1_short_tmpdir() {
  local base="${SLURM_TMPDIR:-/tmp}"
  if [[ ${#base} -gt 32 || ! -d "$base" ]]; then base=/tmp; fi
  TMPDIR=$(mktemp -d "${base%/}/smhoi-${SLURM_JOB_ID:-local}.XXXXXX")
  export TMPDIR
}
