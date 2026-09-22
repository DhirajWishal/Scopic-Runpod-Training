# Qwen2.5-0.5B-Instruct on RunPod (SGLang)

A sandbox for running Qwen2.5-0.5B-Instruct with [SGLang](https://github.com/sgl-project/sglang),
deployable two ways:

| | GPU Pod | Serverless |
|---|---|---|
| What it is | A machine you rent by the hour and SSH into | A Docker image RunPod runs on demand |
| Billing | Per hour, while the pod exists | Per second of execution, scales to zero |
| Entrypoint | `source/main.py` | `source/handler.py` |
| Good for | Experimenting, debugging, notebooks | Serving an API to something else |

Both import `source/engine.py`, so model settings live in exactly one place and
the two paths can't drift apart.

```
source/
  engine.py     shared config, model loading, chat templating
  main.py       pod entrypoint: single / batch / streaming demo
  handler.py    serverless entrypoint: RunPod worker
Dockerfile              serverless image (FROM lmsysorg/sglang)
requirements.txt        pod + local deps
requirements-worker.txt extra deps for the image (just the runpod SDK)
scripts/bootstrap.sh    one-time pod setup
scripts/run.sh          run the demo on a pod
test_input.json         local handler test fixture
.env.example            all tunable settings
```

> **This will not run on Windows.** SGLang needs Linux + an NVIDIA GPU. The local
> `.venv` is for editing and linting only.

---

## Path A — GPU Pod

1. Create a pod from a **PyTorch + CUDA** template (RTX A4000 is plenty for 0.5B).
   Attach a network volume so it mounts at `/workspace`.
2. SSH in and set up:

   ```bash
   cd /workspace
   git clone <this-repo> sandbox && cd sandbox
   cp .env.example .env          # optional, edit as needed
   bash scripts/bootstrap.sh     # installs deps, pre-downloads the weights
   ```

3. Run it:

   ```bash
   bash scripts/run.sh
   ```

`bootstrap.sh` points `HF_HOME` at `/workspace`, the only directory that survives
a pod stop. Cache anywhere else and you re-download the model every session.

---

## Path B — Serverless

### 1. Build and push

```bash
docker build -t <dockerhub-user>/qwen-sglang:0.1.0 .
docker push <dockerhub-user>/qwen-sglang:0.1.0
```

Build on an x86 Linux machine, or pass `--platform linux/amd64` from an Apple
Silicon Mac — RunPod workers are x86 and an arm64 image fails at pull time.

### 2. Create the endpoint

In the RunPod console: **Serverless → New Endpoint → Custom Source (Docker image)**.

- **Image**: `<dockerhub-user>/qwen-sglang:0.1.0`
- **GPU**: 24 GB tier (A4000 / 4090 / L40S) is more than enough
- **Container disk**: 20 GB
- **Network volume**: attach one — it mounts at `/runpod-volume` and `engine.py`
  caches the weights there, so only the very first cold start pays the download
- **Env vars**: any from `.env.example` you want to override
- **Max workers / idle timeout**: start at 1 worker, 5s idle while testing

### 3. Call it

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

Response — note `output` is a **list**, because the handler is a generator
(see *Notes and gotchas*):

```json
{
  "output": [
    {
      "text": "A KV cache stores ...",
      "finish_reason": "stop",
      "prompt_tokens": 31,
      "completion_tokens": 64
    }
  ]
}
```

Set `"stream": true` in the input and use `/stream` for incremental chunks; the
list then contains one `{"text": "..."}` item per delta.

### Testing the handler without deploying

On any Linux GPU box (including a Pod), the RunPod SDK picks up `test_input.json`
automatically and runs one job locally:

```bash
python source/handler.py
```

---

## Settings

All read from environment variables by `source/engine.py` — set them in `.env`
on a pod, or in the endpoint's Environment Variables panel on serverless. No
rebuild needed either way.

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | HF repo id or local path |
| `MEM_FRACTION_STATIC` | `0.8` | GPU memory reserved for weights + KV cache. Lower on OOM |
| `CONTEXT_LENGTH` | `4096` | Model supports 32k; capping keeps KV cache small |
| `DTYPE` | `auto` | `float16` on T4-era GPUs without bf16 |
| `SGLANG_LOG_LEVEL` | `info` | `error` to quiet the logs |
| `MAX_CONCURRENCY` | `4` | Serverless only: concurrent jobs per worker |

---

## Notes and gotchas

**The handler must be an `async` generator.** Two separate constraints:

- `async`, because SGLang's sync `Engine.generate()` calls
  `loop.run_until_complete()` internally and RunPod invokes handlers from inside
  a running event loop — the sync API raises *"This event loop is already running"*.
- `yield` rather than `return`, **even for non-streaming replies**, because
  RunPod decides whether a worker streams by inspecting the handler function at
  startup (`inspect.isasyncgenfunction`), not by looking at what it returns. A
  plain `async def` that returns a generator is not detected as streaming and
  RunPod fails trying to serialise the generator object.

That's why every response is a list, even a single-item one.

**Don't batch manually in the handler.** On a pod, passing a list of prompts to
`generate()` batches them. On serverless, concurrent requests arrive as separate
jobs and SGLang's scheduler batches them for you — `MAX_CONCURRENCY` is what
lets enough of them be in flight for that to happen.

**Streaming chunks are cumulative.** Each chunk's `text` is the full output so
far, not just the new piece. Both entrypoints diff against what was already sent.

**Base image tag.** SGLang 0.5.20 publishes only one CUDA variant, `cu130`;
v0.5.19 is the last release carrying a `cu129` build. The tag must match the
`sglang` pin in `requirements.txt` so the Pod and Serverless paths run the same
runtime. If you change the SGLang version, check
[the tag list](https://hub.docker.com/r/lmsysorg/sglang/tags) rather than
assuming a tag exists.

**Cold starts.** First request to a scaled-to-zero endpoint pays image pull +
model load (~1-2 min without a warm volume). Set min workers to 1 if that
matters, or bake the weights into the image (commented snippet in the Dockerfile).
