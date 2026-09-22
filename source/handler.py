"""
RunPod Serverless worker for Qwen2.5-0.5B-Instruct on SGLang.

HOW SERVERLESS DIFFERS FROM A POD
---------------------------------
A Pod is a machine you rent by the hour and SSH into. Serverless is a Docker
image RunPod runs for you: it starts workers when requests arrive, and scales
back to zero when they stop. You are billed per second of execution.

The lifecycle of one worker:
  1. RunPod pulls the image and starts this file.
  2. Module-level code runs -> we load the model onto the GPU (the "cold start").
  3. runpod.serverless.start() blocks, polling RunPod for jobs.
  4. Each job calls handler() -- the model is ALREADY loaded, so this is fast.
  5. After an idle timeout the worker is killed and steps 1-2 repeat next time.

That's why the engine is built at import time and not inside handler(): paying
the 30-60s load once per worker instead of once per request is the whole game.

REQUEST FORMAT
--------------
POST to your endpoint's /runsync (blocking), /run (async job id) or /stream:

    {
      "input": {
        "messages": [
          {"role": "system", "content": "You are a concise assistant."},
          {"role": "user",   "content": "What is a KV cache?"}
        ],
        "sampling_params": {"temperature": 0.7, "max_new_tokens": 256},
        "stream": false
      }
    }

`messages` can be swapped for `prompt` (a raw string) if you want to bypass the
chat template and feed the model text directly.

RESPONSE SHAPE
--------------
`handler` is an async *generator* (see the note on its docstring), so RunPod
treats this worker as a streaming worker. With "return_aggregate_stream": True,
/run and /runsync collect everything yielded into a LIST:

    non-streaming  -> "output": [{"text": "...", "finish_reason": "stop", ...}]
    streaming      -> "output": [{"text": "A KV"}, {"text": " cache"}, ...]

/stream delivers those same items one at a time as they're produced.
"""

import os

import runpod

from engine import get_engine, merge_sampling_params, to_prompt

# How many jobs RunPod may hand a single worker at once -- see adjust_concurrency.
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))


# ---------------------------------------------------------------------------
# Cold start: load the model NOW, at import, before we accept any job.
# ---------------------------------------------------------------------------
# RunPod does not route work to this worker until the process is up and polling,
# so doing the slow thing here means the first real request sees a warm GPU.
print("[worker] loading model ...", flush=True)
ENGINE = get_engine()
print("[worker] model ready", flush=True)


def _build_prompt(job_input: dict) -> str:
    """Accept either a chat `messages` list or a raw `prompt` string."""
    if "messages" in job_input:
        messages = job_input["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("'messages' must be a non-empty list of chat turns")
        return to_prompt(messages)

    prompt = job_input.get("prompt")
    if not prompt:
        raise ValueError("request must include either 'messages' or 'prompt'")
    return prompt


async def handler(job):
    """
    Handle one job.

    WHY THIS IS AN `async def ... yield` GENERATOR, ALWAYS
    -----------------------------------------------------
    Two constraints force this shape:

    1. `async` is required. SGLang's synchronous `Engine.generate()` internally
       calls `loop.run_until_complete(...)`, and RunPod invokes handlers from
       inside an already-running asyncio event loop -- the sync API raises
       "This event loop is already running". So we use `async_generate`.

    2. `yield` is required *even for non-streaming replies*. RunPod decides
       whether a worker streams by inspecting the handler FUNCTION at startup
       (`inspect.isasyncgenfunction`), not by looking at what it returns. A
       plain `async def` that returns a generator is NOT detected as streaming;
       RunPod would try to serialise the generator object and fail. Making the
       handler itself a generator is the only shape that supports both modes.

    The cost is that every response is a list (see RESPONSE SHAPE above), even
    when it contains a single item.
    """
    job_input = job.get("input") or {}

    try:
        prompt = _build_prompt(job_input)
    except ValueError as exc:
        # A dict with an "error" key is RunPod's convention for a failed job.
        yield {"error": str(exc)}
        return

    sampling_params = merge_sampling_params(job_input.get("sampling_params"))

    if job_input.get("stream"):
        # SGLang gives CUMULATIVE text on each chunk (the full output so far),
        # but a stream consumer wants only the new piece, so we diff against
        # what we already sent. Getting this wrong is the classic "every chunk
        # repeats the whole answer" bug.
        sent = 0
        async for chunk in await ENGINE.async_generate(
            prompt, sampling_params, stream=True
        ):
            text = chunk["text"]
            delta = text[sent:]
            sent = len(text)
            if delta:
                yield {"text": delta}
        return

    output = await ENGINE.async_generate(prompt, sampling_params)
    meta = output["meta_info"]
    yield {
        "text": output["text"],
        # finish_reason tells you whether the model stopped on its own ("stop")
        # or was cut off by max_new_tokens ("length") -- worth surfacing.
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
    }


def adjust_concurrency(current_concurrency: int) -> int:
    """
    How many jobs RunPod may hand this ONE worker at the same time.

    Default is 1, which wastes the GPU: a 0.5B model on a 24 GB card can serve
    many requests at once, and SGLang's scheduler does continuous batching
    across whatever is in flight. Letting several jobs land concurrently is what
    turns that batching on -- you do NOT batch manually in a serverless handler.

    Raise MAX_CONCURRENCY if the GPU is idle under load; lower it if you see
    out-of-memory errors or latency spikes.
    """
    return MAX_CONCURRENCY


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": adjust_concurrency,
            # Makes streamed output also available from /run and /runsync by
            # aggregating everything yielded into one list, instead of only
            # being readable through /stream.
            "return_aggregate_stream": True,
        }
    )
