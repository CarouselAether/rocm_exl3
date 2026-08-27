#!/usr/bin/env python3
"""Coherence smoke test: load a model, generate from a natural-language prompt
with ordinary sampling, print the text for eyeballing.

Not a metric -- per RDNA_NOTES.md, greedy loop-collapse probes are decoding
chaos, so this uses temperature sampling with a fixed seed and a real prompt,
and a human judges the output. The point is a fast per-model check after a
kernel or sync change: garbage here means broken kernels, coherent text here
means the stack end-to-end (load, prefill, decode, sampler) holds together.

    rocm_tools/gen_smoke.py -m /path/to/model [-n 150]

Exits via os._exit(): processes that load a model segfault in native teardown
after all work completes (see RDNA_NOTES.md).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import DefaultSampler

PROMPT = (
    "Q: Briefly explain why the sky is blue, then name three primary colors.\nA:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-n", "--new_tokens", type=int, default=150)
    args = ap.parse_args()

    label = os.path.basename(args.model_dir.rstrip("/"))
    print(f" -- loading {label}", flush=True)
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    ids = tokenizer.encode(PROMPT, add_bos=True)
    job = Job(
        input_ids=ids,
        max_new_tokens=args.new_tokens,
        sampler=DefaultSampler(),
        seed=1234,
    )
    generator.enqueue(job)
    text = []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                text.append(r.get("text", ""))

    print(f"----- {label} -----")
    print(PROMPT + "".join(text))
    print("----- end -----", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
