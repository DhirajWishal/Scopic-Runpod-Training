#!/usr/bin/env bash
# One-time setup on a fresh RunPod GPU Pod.
#
#   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
#   git clone <this repo> /workspace/sandbox && cd /workspace/sandbox
#   bash scripts/bootstrap.sh
#
# Pick a pod template that already has CUDA + PyTorch (e.g. "RunPod PyTorch 2.x").
# Starting from a bare Ubuntu image means installing the CUDA toolchain yourself,
# which is exactly the pain the serverless Dockerfile avoids by using
# lmsysorg/sglang as its base.

set -euo pipefail

# -e exit on error, -u error on unset vars, -o pipefail catch failures mid-pipe.
# Without these a failed pip install would scroll past and you'd debug a
# confusing ImportError later instead of the actual failure.

echo "==> GPU check"
# If this fails, you're on a CPU-only pod and nothing below will work.
nvidia-smi

echo "==> System deps"
# sgl_kernel's compiled extension links against libnuma.so.1, which the RunPod
# PyTorch templates don't ship (the serverless image gets it from lmsysorg/sglang).
# This lives here rather than in a one-off manual step because everything outside
# /workspace is wiped when the pod stops, so it must be reinstalled each session.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libnuma1

echo "==> Python deps"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==> Warming the model cache"
# Downloads the weights (~30 GB) to /workspace now, so the first real run isn't
# stalled behind it. /workspace is the persistent network volume -- the rest of
# the pod's disk is wiped when the pod is stopped, so caching anywhere else means
# re-downloading every session.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
python -c "
from huggingface_hub import snapshot_download
import os
model = os.environ.get('MODEL_ID', 'mistralai/Ministral-3-14B-Instruct-2512')
print(f'downloading {model} -> {os.environ[\"HF_HOME\"]}')
snapshot_download(model)
"

echo "==> Done. Run: bash scripts/run.sh"
