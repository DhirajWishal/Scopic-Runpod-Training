"""
Interactive SGLang demo for Qwen2.5-0.5B-Instruct -- the GPU Pod entrypoint.

HOW THIS RUNS
-------------
SGLang is a Linux + NVIDIA-GPU inference engine. It will NOT run on your Windows
machine -- the local .venv here is just for editing/linting. You run this on a
RunPod GPU Pod (RTX A4000 / 4090 / L40S all work fine for a 0.5B model):

    bash scripts/bootstrap.sh     # one-time: install deps on a fresh pod
    bash scripts/run.sh           # == python source/main.py

For the *serverless* deployment of the same model, see source/handler.py. Both
share source/engine.py, so the model config lives in exactly one place.

WHAT SGLang.Engine IS
---------------------
`sgl.Engine` is the "offline" / in-process API: it loads the model weights into
GPU memory inside THIS python process and you call `.generate()` directly.
No HTTP server, no network hop. (The alternative is `python -m sglang.launch_server`,
which exposes an OpenAI-compatible REST API -- better when something else needs to
call the model over the network.)

Under the hood the Engine spawns helper subprocesses:
  - TokenizerManager  -> text  -> token ids
  - Scheduler         -> batches requests, runs the forward pass on the GPU
  - DetokenizerManager-> token ids -> text
They talk to each other over ZeroMQ. You don't manage any of that; you just need
to know it's why startup takes ~30-60s and why we call `.shutdown()` at the end.
"""

from engine import MODEL_ID, get_engine, merge_sampling_params, to_prompt


def main():
    print(f"Loading {MODEL_ID} ...")

    # All the engine configuration (dtype, memory fraction, context length) now
    # lives in engine.py so the serverless worker uses identical settings.
    engine = get_engine()

    # Sampling knobs, passed per request rather than baked into the engine:
    #   temperature   0 = deterministic/greedy, higher = more random. 0.7 is a sane chat default.
    #   top_p         nucleus sampling: only consider tokens in the top 90% probability mass.
    #   max_new_tokens hard stop on the reply length (does NOT include the prompt).
    sampling_params = merge_sampling_params(
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 256,
        }
    )

    # ---- A. one request ---------------------------------------------------
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": "In two sentences, what is a GPU good at and why?"},
    ]
    prompt = to_prompt(messages)
    print(f"Prompt:\n{prompt}\n")

    # [Dhiraj] Generate the response from the model.
    outputs = engine.generate(prompt, sampling_params)

    # generate() returns a dict: {"text": "...", "meta_info": {...}}.
    # meta_info carries prompt_tokens / completion_tokens / finish_reason -- worth
    # printing while you tune, since finish_reason tells you whether the model
    # stopped naturally ("stop") or hit max_new_tokens ("length").
    print("\n--- single ---")
    print(outputs["text"].strip())

    # [Dhiraj] Show all the meata information given out by the model.
    print("\n--- meta_info ---")
    for keys in outputs["meta_info"]:
        print(f"\t{keys}: {outputs['meta_info'][keys]}")

    # ---- B. a batch -------------------------------------------------------
    # Pass a LIST of prompts and SGLang runs them through continuous batching:
    # they share one set of GPU forward passes instead of running one after
    # another. This is the main reason to use an inference engine over plain
    # transformers.generate(), and it's nearly free on a model this small.
    #
    # Note: in the serverless worker you do NOT do this manually -- concurrent
    # requests arrive separately and SGLang's scheduler batches them for you.
    questions = [
        "Name three uses for a vector database.",
        "What does 'quantization' mean for an LLM?",
        "Explain KV cache in one sentence.",
    ]
    batch_prompts = [to_prompt([{"role": "user", "content": q}]) for q in questions]
    batch_outputs = engine.generate(batch_prompts, sampling_params)

    print("\n--- batch ---")
    # Results come back in the same order as the prompts you sent.
    for question, output in zip(questions, batch_outputs):
        print(f"\nQ: {question}\nA: {output['text'].strip()}")

    # ---- C. streaming -----------------------------------------------------
    # stream=True returns a generator of partial results instead of one final dict.
    # Each chunk's "text" is the FULL text so far (cumulative), not just the new
    # piece -- so we slice off what we already printed to get the delta.
    print("\n--- streaming ---")
    stream_prompt = to_prompt(
        [{"role": "user", "content": "Count from 1 to 10 with a word for each."}]
    )
    printed = 0
    for chunk in engine.generate(stream_prompt, sampling_params, stream=True):
        text = chunk["text"]
        print(text[printed:], end="", flush=True)
        printed = len(text)
    print()

    # Frees GPU memory and kills the scheduler/detokenizer subprocesses. The Engine
    # also registers this via atexit, but calling it explicitly makes the lifecycle
    # obvious and avoids a hanging process if you run this in a notebook.
    engine.shutdown()


if __name__ == "__main__":
    main()
