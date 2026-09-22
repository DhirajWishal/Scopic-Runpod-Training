"""
Shared model setup. Imported by BOTH deployment paths:

  - source/main.py     -> interactive script you run on a GPU Pod over SSH
  - source/handler.py  -> the RunPod Serverless worker

Everything that is "how do I load and prompt this model" lives here, so the two
entrypoints can't drift apart. Everything tunable is read from an environment
variable, which means you can change the model or the memory budget from the
RunPod dashboard without rebuilding the Docker image.
"""

import os


# ---------------------------------------------------------------------------
# 1. Where model weights get cached (MUST run before importing HF / sglang)
# ---------------------------------------------------------------------------
def _resolve_cache_dir() -> str | None:
    """
    Pick a persistent directory for the Hugging Face cache.

    RunPod mounts a network volume at a DIFFERENT path depending on the product:
      - GPU Pod      -> /workspace
      - Serverless   -> /runpod-volume
    Everything else on the machine is ephemeral: it's wiped when the pod stops or
    the serverless worker is recycled, so a cache there means re-downloading
    Qwen2.5-0.5B (~1 GB) on every cold start.

    Returns None when neither exists (e.g. local dev, or a serverless endpoint
    with no volume attached). In that case we leave HF_HOME alone and HF falls
    back to ~/.cache/huggingface on the container disk -- which still works, it
    just re-downloads after each cold start.
    """
    for mount in ("/runpod-volume", "/workspace"):
        if os.path.isdir(mount):
            return os.path.join(mount, ".cache", "huggingface")
    return None


# `setdefault` = don't clobber it if the pod template or endpoint config already
# set HF_HOME explicitly. An explicit setting always wins over our guess.
_cache_dir = _resolve_cache_dir()
if _cache_dir:
    os.environ.setdefault("HF_HOME", _cache_dir)

# These imports are deliberately below the HF_HOME assignment, because both
# libraries read that variable at import time.
import sglang as sgl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# 2. Configuration (env vars, so they're changeable without a rebuild)
# ---------------------------------------------------------------------------
# A Hugging Face repo id, or a local path if you pre-fetched the weights.
# "-Instruct" matters: the base "Qwen/Qwen2.5-0.5B" is a raw completion model
# and will not follow chat turns.
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Ministral-3-14B-Instruct")

# Fraction of total GPU memory SGLang reserves up front for weights + KV cache.
# It grabs this as one big static pool instead of allocating per request.
# 0.5B in bf16 is only ~1 GB of weights, so on a 24 GB card 0.8 leaves a huge KV
# cache (= lots of concurrent requests). Lower it if you share the GPU.
MEM_FRACTION_STATIC = float(os.environ.get("MEM_FRACTION_STATIC", "0.8"))

# Hard cap on prompt + generated tokens. Qwen2.5-0.5B-Instruct natively supports
# 32k, but KV cache cost scales with this number -- capping it keeps memory
# predictable and lets more requests run concurrently.
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "4096"))

# "auto" reads torch_dtype from the model's config.json (bfloat16 for Qwen2.5).
# Set DTYPE=float16 only on older GPUs (e.g. T4) that lack bf16 support.
DTYPE = os.environ.get("DTYPE", "auto")

# The Engine defaults to "error" (silent). "info" shows the download progress,
# the memory pool size and per-request throughput -- useful in RunPod logs.
LOG_LEVEL = os.environ.get("SGLANG_LOG_LEVEL", "info")

# Defaults applied to any request that doesn't override them.
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 256,
}


# ---------------------------------------------------------------------------
# 3. Lazy singletons
# ---------------------------------------------------------------------------
# Loading weights onto the GPU takes 30-60s. A serverless worker handles many
# requests over its lifetime, so this must happen ONCE at worker startup, never
# per request. Module-level globals + a getter give us that for free: the first
# call pays the cost, every later call is a dict lookup.
_engine: sgl.Engine | None = None
_tokenizer = None


def get_engine() -> sgl.Engine:
    """Return the process-wide Engine, loading the model on first call."""
    global _engine
    if _engine is None:
        _engine = sgl.Engine(
            model_path=MODEL_ID,
            dtype=DTYPE,
            mem_fraction_static=MEM_FRACTION_STATIC,
            context_length=CONTEXT_LENGTH,
            # Qwen2.5 is natively supported by transformers, so no custom model
            # code needs executing. Keep False unless a repo genuinely requires it.
            trust_remote_code=False,
            log_level=LOG_LEVEL,
        )
    return _engine


def get_tokenizer():
    """
    Return the process-wide tokenizer.

    We only need it for the chat template -- it's a few small JSON files, not
    weights, and it lives on the CPU side.
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return _tokenizer


# ---------------------------------------------------------------------------
# 4. Prompt formatting
# ---------------------------------------------------------------------------
def to_prompt(messages: list[dict]) -> str:
    """
    Turn a list of chat messages into the single string the model actually sees.

    Instruct models are trained on a specific markup. Qwen2.5 uses ChatML:

        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant

    Rather than hand-writing that (easy to get subtly wrong), we let the
    tokenizer's built-in chat template do it. `add_generation_prompt=True`
    appends the trailing "<|im_start|>assistant\n" so the model knows it's its
    turn to speak.

    `tokenize=False` returns the formatted *string*, because Engine.generate()
    wants text and will tokenize it itself.
    """
    return get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def merge_sampling_params(overrides: dict | None) -> dict:
    """Layer caller-supplied sampling params on top of the defaults."""
    params = dict(DEFAULT_SAMPLING_PARAMS)
    if overrides:
        params.update(overrides)
    return params
