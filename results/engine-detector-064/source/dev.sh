#!/usr/bin/env bash
# dev.sh CMD: sync the 064 source to the dev pod, then run CMD there in the oracle env.
N=watching-engine-064-dev; R=~/mpd-data/runpod/$N
rsync -a --exclude __pycache__ -e $R/rsh ~/mpd-data/scratch/watching-engine-064/ "$(cat $R/host):/workspace/e64/src/" || exit 1
$R/ssh "cd /workspace/e64 && export PATH=/workspace/pyenv/2020f9fd6d35/bin:\$PATH VIRTUAL_ENV=/workspace/pyenv/2020f9fd6d35 HF_HOME=/workspace/hf-home PYTHONPATH=/workspace/e64/src/src PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 && SNAP=\$(ls -d /workspace/hf-home/hub/models--Qwen--Qwen3-8B/snapshots/*) && $1"
