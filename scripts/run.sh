#!/usr/bin/env bash
# Run the interactive demo on a GPU Pod.

set -euo pipefail

# Load local overrides (MODEL_ID, MEM_FRACTION_STATIC, HF_TOKEN, ...) if present.
# `set -a` makes every following assignment an exported env var, so engine.py
# picks them up via os.environ.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# -u = unbuffered, so output streams to your SSH session in real time instead of
# appearing in bursts. Matters a lot for the streaming demo at the end.
python -u source/main.py
