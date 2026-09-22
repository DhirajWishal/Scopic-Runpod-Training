# ---------------------------------------------------------------------------
# Serverless worker image for Qwen2.5-0.5B-Instruct on SGLang.
# ---------------------------------------------------------------------------
# We start from SGLang's own published image rather than a bare CUDA image.
# It already contains a matched torch / CUDA / flashinfer / sglang stack, which
# is the part that's painful to assemble by hand -- `pip install sglang[all]`
# into a generic image routinely picks incompatible wheel builds.
#
# Tag choice: SGLang 0.5.2 publishes cu126, cu128-b200 and cu129-*b200 variants.
# There is NO v0.5.2-cu124. cu126 is the general-purpose one and runs on RunPod's
# A4000 / 4090 / A100 / L40S fleet. Use a -b200 tag only if you target B200/GB200.
FROM lmsysorg/sglang:v0.5.2-cu126

# The base image may define its own ENTRYPOINT (it's built to launch SGLang's
# HTTP server). Clearing it guarantees our CMD below is what actually runs
# instead of being appended as arguments to the server launcher.
ENTRYPOINT []

# Unbuffered stdout, so print() and tracebacks reach the RunPod log viewer
# immediately rather than sitting in a pipe buffer when a worker is killed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Only the RunPod SDK is missing from the base image. Copied on its own first so
# Docker caches this layer and doesn't reinstall on every source edit.
COPY requirements-worker.txt /tmp/requirements-worker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-worker.txt

WORKDIR /app
COPY source ./source

# Weights are NOT baked in: the worker downloads Qwen2.5-0.5B (~1 GB) on first
# use, which keeps this image small and lets you switch models via the MODEL_ID
# env var without rebuilding. Attach a RunPod network volume to the endpoint and
# engine.py caches to /runpod-volume so only the very first cold start pays for it.
#
# If you later want faster cold starts, uncomment to bake the weights in:
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

# Python puts the script's own directory on sys.path, so source/handler.py can
# do `from engine import ...` without any packaging ceremony.
CMD ["python", "-u", "source/handler.py"]
