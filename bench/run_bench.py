#!/usr/bin/env python3
"""Phase 0 benchmark harness: pp512, pp2048, tg128 as exl3_server reports them.

Consistency with the server is the point (ENGINEER_QUESTIONS Q-3):
  - the model, cache and draft model load through model_init.init() with the server's
    own arguments (default -cs 65536, as serve_ds4f.sh), so chunk size, cache capacity
    and graph-capture geometry match;
  - speeds use the server's log_request() formulas on the same job fields:
      prefill = (prompt_tokens - cached_tokens) / time_prefill
      decode  = new_tokens / time_generate

Workloads, all at bsz 1:
  pp<N>   random-token prompt of N tokens, 1 new token (fresh ids each run: no prefix-cache hits)
  tg128   random 512-token prompt, then 128 new tokens (decode at ctx ~512-640); no EOS stop,
          so exactly 128 tokens are generated
  mtp     with --mtp: natural prompts, greedy, 128 tokens; reports tok/s and acceptance
  tglong  with --long N: one N-token prompt, then 64 new tokens (exercises the DSA indexer /
          top-k regime that tg128 never reaches, see CODE_SCAN "Notes for Phase 0")
  tg<M>@d<N>  with --tg_depth N [N ...]: decode at depth on NATURAL text (random ids ruin
          MTP / DFlash acceptance). Prompt = exactly N tokens: the model's chat template
          (user turn) around a wikitext-2 test slice + a short summarize instruction. Slices
          are token windows at fixed offsets (run r: offset 1000 + r * 8192 + N of the whole
          tokenized split, eval/ppl.py's loader), so models sharing a tokenizer see identical
          text; M = --depth_new new tokens (default 128), no EOS stop. Sampler: greedy in
          spec runs (--mtp / -dm), DefaultSampler with seed 1234 otherwise. Works in both
          passes; each run uses a different slice, so there are no prefix-cache hits.

Each workload: 1 discarded warmup + --runs timed runs, median reported. A sysfs sampler
thread records GPU clock / power / busy during every timed run; amd-smi snapshots are
taken before and after. Results go to bench/results/<commit>_<timestamp>.json.

    bench/run_bench.py -m ~/models/DeepSeek-V4-Flash-0731-exl3-2.04bpw
    bench/run_bench.py -m ... --mtp -ndt 2          # MTP pass (separate process)
    bench/run_bench.py -m ... --tg 0 --tg_depth 1024 2048            # decode at depth, plain
    bench/run_bench.py -m ... --mtp -ndt 2 --tg 0 --tg_depth 1024 2048
    bench/run_bench.py -m ~/models/Laguna-S-2.1-exl3-4.00bpw -dm ~/models/Laguna-S-2.1-DFlash -ndt 3 \
        --tg 0 --tg_depth 1024 2048                 # external drafter (DFlash), same model_init args as the server
    bench/run_bench.py -m ... --repo ../rocm_exl3_10 # measure another checkout

Exits via os._exit() (native teardown segfault after model load, see RDNA_NOTES).
"""

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_dir", required=True)
ap.add_argument("--repo", default=os.path.dirname(HERE), help="exllamav3 checkout to import (default: this repo)")
ap.add_argument("-cs", "--cache_size", type=int, default=65536, help="server default in serve_ds4f.sh")
ap.add_argument("--pp", type=int, nargs="*", default=[512, 2048])
ap.add_argument("--tg", type=int, default=128, help="0 skips the decode workload")
ap.add_argument("--tg_ctx", type=int, default=512)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--mtp", action="store_true", help="MTP pass instead of plain")
ap.add_argument("-dm", "--draft_model_dir", default=None,
                help="spec pass with a separate draft model (e.g. a DFlash drafter), via model_init's -dm as the server")
ap.add_argument("-ndt", "--num_draft_tokens", type=int, default=2)
ap.add_argument("--tg_depth", type=int, nargs="*", default=[],
                help="decode-at-depth workloads on natural text: prompt lengths (see the docstring)")
ap.add_argument("--depth_new", type=int, default=128, help="new tokens per --tg_depth run")
ap.add_argument("--long", type=int, default=0, help="also run tg64 after an N-token prompt")
ap.add_argument("--gen_chunk", type=int, default=None,
                help="Generator max_chunk_size (default: Generator's 2048, as the server uses). -chunk_size via --extra only sizes load-time buffers")
ap.add_argument("--regen", type=int, nargs="*", default=[],
                help="regeneration workloads: prompt lengths; each run primes a prompt, then re-sends it and times\n                the cached-prefix + tail prefill (latency ms), which is what a user feels on regenerate")
ap.add_argument("--label", default="")
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--ngram_lock", action="store_true",
                help="as the server's -ngl: n-gram table in RAM (-ngr) and mlock'ed (exllamav3/rocm_py/ngram_lock.py)")
ap.add_argument("--ngram_lock_max_gb", type=float, default=None,
                help="test hook: lock only the first N GiB of the table (skips the preflight), for boxes whose "
                     "RLIMIT_MEMLOCK cannot cover the whole table")
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra model_init args, e.g. --extra -cq 8")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.repo))

import torch  # noqa: E402
from exllamav3 import model_init, Generator, Job  # noqa: E402
from exllamav3.generator.sampler import ArgmaxSampler, DefaultSampler  # noqa: E402

CARD = "/sys/class/drm/card0/device"
PROMPTS = [
    "Q: Briefly explain why the sky is blue, then name three primary colors.\nA:",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\n",
    "Explain, step by step, how to compute the greatest common divisor of two integers, with an example.\n\n",
]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<{e}>"


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def hwmon(name):
    base = os.path.join(CARD, "hwmon")
    for h in os.listdir(base):
        v = read(os.path.join(base, h, name))
        if v is not None:
            return v
    return None


class Sampler(threading.Thread):
    """Samples GPU clock (MHz), power (W) and busy % from sysfs every 0.25 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop_ev = [], threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            f, p, b = hwmon("freq1_input"), hwmon("power1_input"), read(f"{CARD}/gpu_busy_percent")
            self.samples.append((
                int(f) / 1e6 if f else None,
                int(p) / 1e6 if p else None,
                int(b) if b else None))
            time.sleep(0.25)

    def stop(self):
        self.stop_ev.set()
        self.join()
        busy = [s for s in self.samples if s[2] is not None and s[2] > 50]
        def stat(i):
            v = [s[i] for s in busy if s[i] is not None]
            return {"min": min(v), "median": statistics.median(v), "max": max(v)} if v else None
        return {"n": len(self.samples), "n_busy": len(busy), "sclk_mhz": stat(0), "power_w": stat(1)}


def env_info(repo):
    git = lambda c: sh(f"git -C {repo} {c}")
    return {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "repo": os.path.abspath(repo),
        "commit": git("rev-parse --short HEAD"),
        "branch": git("rev-parse --abbrev-ref HEAD"),
        "dirty": bool(git("status --porcelain --untracked-files=no")),
        "torch": torch.__version__,
        "hip_runtime": torch.version.hip,
        "rocm_pip": sh(f"{sys.executable} -m pip show rocm 2>/dev/null | grep ^Version"),
        "device": torch.cuda.get_device_name(0),
        "perf_level": read(f"{CARD}/power_dpm_force_performance_level"),
        "cpu_boost": read("/sys/devices/system/cpu/cpufreq/boost"),
        "cpu_max_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "HIP_", "ROCM", "HSA_", "LD_"))},
        "argv": sys.argv,
    }


def run_job(generator, ids, n, sampler=None, seed=None):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=sampler, seed=seed)
    generator.enqueue(job)
    final, text = None, []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                text.append(r.get("text", ""))
                if r.get("eos", False):
                    final = r
    final["_text"] = "".join(text)
    return final


def rand_ids(tokenizer, n, rng):
    vocab = tokenizer.actual_vocab_size
    return torch.randint(int(vocab * 0.05), int(vocab * 0.95), (1, n), dtype=torch.long, generator=rng)


# server log_request() formulas
def pp_rate(f):
    return (f["prompt_tokens"] - f.get("cached_tokens", 0)) / f["time_prefill"]


def tg_rate(f):
    return f["new_tokens"] / f["time_generate"]


_WIKI = {}


def wiki_ids(tokenizer):
    """The whole wikitext-2 test split tokenized once (eval/ppl.py's loader and disk cache)."""
    if "ids" not in _WIKI:
        sys.path.insert(0, os.path.join(os.path.abspath(args.repo), "eval"))
        from ppl import get_dataset_text
        _WIKI["ids"] = tokenizer.encode(get_dataset_text({"dataset": "wiki2"}))
    return _WIKI["ids"]


DEPTH_INSTR = "\n\nSummarize the text above in a few paragraphs."
_MARK = "\u2063EXL3DEPTHMARK\u2063"


def depth_ids(tokenizer, n, r):
    """Exactly n tokens: chat-template head + wikitext slice (run r) + instruction + template tail."""
    try:
        text = tokenizer.hf_render_chat_template([{"role": "user", "content": _MARK}], add_generation_prompt=True)
        head, tail = text.split(_MARK)
    except Exception:
        head, tail = None, ""
    if head is not None:
        h = tokenizer.encode(head, encode_special_tokens=True)
    else:
        h = tokenizer.encode("", add_bos=True)
    t = tokenizer.encode(DEPTH_INSTR + tail, encode_special_tokens=True)
    k = n - h.shape[-1] - t.shape[-1]
    assert k > 0, f"depth {n} too short for the template ({h.shape[-1]} + {t.shape[-1]} tokens)"
    off = 1000 + r * 8192 + n   # depth in the offset: no shared prefix (prefix-cache hit) across depths
    w = wiki_ids(tokenizer)[:, off:off + k]
    assert w.shape[-1] == k, "wikitext split too short for this depth / run count"
    ids = torch.cat((h, w, t), dim=-1)
    assert ids.shape[-1] == n
    return ids


def regen_workload(n, runs, generator, tokenizer, rng):
    """Prime a fresh prompt (untimed), re-send it identically, and time the second prefill."""
    out = {"name": f"regen{n}", "runs": []}
    for r in range(runs + 1):
        ids = rand_ids(tokenizer, n, rng)
        run_job(generator, ids, 1)                       # prime: stores pages + recurrent stash
        smp = Sampler(); smp.start()
        f = run_job(generator, ids, 1)
        gpu = smp.stop()
        rec = {"rate": f["time_prefill"] * 1e3, "cached_tokens": f.get("cached_tokens", 0),
               "prompt_tokens": f["prompt_tokens"], "gpu": gpu}
        (out.setdefault("warmup", rec) if r == 0 else out["runs"].append(rec))
    lat = [x["rate"] for x in out["runs"]]
    out["median"] = statistics.median(lat)
    out["spread"] = (max(lat) - min(lat)) / out["median"]
    out["unit"] = "ms"
    cached = out["runs"][0]["cached_tokens"]
    print(f"  {out['name']:10} {out['median']:9.1f} ms   spread {out['spread']:5.1%}  cached {cached}/{n}", flush=True)
    return out


def workload(name, make_ids, n_new, rate, runs, generator, sampler=None):
    out = {"name": name, "runs": [], "cached_seen": 0}
    for r in range(runs + 1):
        ids = make_ids(r)
        smp = Sampler()
        smp.start()
        f = run_job(generator, ids, n_new, sampler=sampler, seed=1234 if sampler else None)
        gpu = smp.stop()
        out["cached_seen"] = max(out["cached_seen"], f.get("cached_tokens", 0))
        rec = {"rate": rate(f), "prompt_tokens": f["prompt_tokens"], "new_tokens": f["new_tokens"],
               "time_prefill": f["time_prefill"], "time_generate": f["time_generate"], "gpu": gpu}
        if "accepted_draft_tokens" in f:
            a, rj = f["accepted_draft_tokens"], f["rejected_draft_tokens"]
            rec["acceptance"] = a / max(a + rj, 1)
        if r == 0:
            out["warmup"] = rec
            out["sample_text"] = f["_text"][:600]   # warmup run's output, for eyeballing
            continue
        out["runs"].append(rec)
    rates = [x["rate"] for x in out["runs"]]
    out["median"] = statistics.median(rates)
    out["spread"] = (max(rates) - min(rates)) / out["median"]
    if out["runs"] and "acceptance" in out["runs"][0]:
        out["acceptance_median"] = statistics.median(x["acceptance"] for x in out["runs"])
    flag = "  <- spread > 5%" if out["spread"] > 0.05 else ""
    flag += "  <- PREFIX CACHE HIT" if out["cached_seen"] else ""
    acc = f"  acc {out['acceptance_median']:.0%}" if "acceptance_median" in out else ""
    print(f"  {name:10} {out['median']:9.2f} t/s  spread {out['spread']:5.1%}{acc}  "
          f"sclk {gpu['sclk_mhz']}{flag}", flush=True)
    return out


@torch.inference_mode()   # as server.py main()
def main():
    ia = ["-m", args.model_dir, "-cs", str(args.cache_size)]
    assert not (args.mtp and args.draft_model_dir), "--mtp and -dm are exclusive"
    spec = args.mtp or args.draft_model_dir is not None
    if args.mtp:
        ia += ["--mtp", "-ndt", str(args.num_draft_tokens)]
    elif args.draft_model_dir:
        ia += ["-dm", args.draft_model_dir, "-ndt", str(args.num_draft_tokens)]
    ia += args.extra
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True, add_sampling_args=False, add_draft_model_args=True,
                        default_autosplit_max_batch_size=4)   # server.py's defaults
    iargs = parser.parse_args(ia)
    lock_mod = None
    if args.ngram_lock:
        from exllamav3.rocm_py import ngram_lock as lock_mod
        iargs.ngram_ram = True
        if args.ngram_lock_max_gb is None:
            lock_mod.preflight(args.model_dir, args.cache_size)

    info = env_info(args.repo)
    info["amd_smi_before"] = sh("amd-smi metric -c -p -t -l --json")
    info["init_args"] = ia
    print(f" -- {info['commit']}{' (dirty)' if info['dirty'] else ''} on {info['branch']}, "
          f"torch {info['torch']}, HIP {info['hip_runtime']}, perf {info['perf_level']}", flush=True)

    t0 = time.perf_counter()
    model, config, cache, tokenizer, draft_model, _dc, draft_cache = model_init.init(iargs)
    info["load_s"] = time.perf_counter() - t0
    if lock_mod is not None:
        mx = None if args.ngram_lock_max_gb is None else int(args.ngram_lock_max_gb * 2**30)
        info["ngram_lock"] = lock_mod.lock_model(model, max_bytes=mx)
        print(f" -- {lock_mod.describe(info['ngram_lock'])}", flush=True)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=draft_model,
                    draft_cache=draft_cache, num_draft_tokens=iargs.num_draft_tokens,
                    **({"max_chunk_size": args.gen_chunk} if args.gen_chunk else {}))
    print(f" -- loaded in {info['load_s']:.0f}s; mtp_draft={getattr(gen, 'mtp_draft', None)} "
          f"dflash={getattr(gen, 'dflash_draft', None)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = ("_" + args.label) if args.label else ""
    kind = "_mtp" if args.mtp else ("_dm" if args.draft_model_dir else "")
    path = os.path.join(args.out, f"{info['commit']}_{stamp}{kind}{tag}.json")
    info["results"] = res = []

    def save():   # after every workload, so a killed run keeps what it finished
        with open(path, "w") as f:
            json.dump(info, f, indent=1)
            f.flush()
            os.fsync(f.fileno())

    rng = torch.Generator().manual_seed(1234)
    info["depth_prompt"] = {"source": "wikitext-2-raw-v1 test (eval/ppl.py get_dataset_text)",
                            "offsets": "1000 + run * 8192 + depth tokens", "instruction": DEPTH_INSTR,
                            "sampler": "greedy" if spec else "DefaultSampler seed 1234"}
    if not spec:
        for n in args.pp:
            res.append(workload(f"pp{n}", lambda r, n=n: rand_ids(tokenizer, n, rng), 1, pp_rate, args.runs, gen))
            save()
        if args.tg:
            res.append(workload(f"tg{args.tg}", lambda r: rand_ids(tokenizer, args.tg_ctx, rng), args.tg,
                                tg_rate, args.runs, gen))
            save()
        for n in args.regen:
            res.append(regen_workload(n, args.runs, gen, tokenizer, rng))
            save()
        if args.long:
            res.append(workload(f"tg64@{args.long}", lambda r: rand_ids(tokenizer, args.long, rng), 64,
                                tg_rate, 1, gen))
            save()
    else:
        enc = lambda r: tokenizer.encode(PROMPTS[r % len(PROMPTS)], add_bos=True)
        # plain greedy on the same prompts (no draft) for a like-for-like MTP ratio is the
        # plain pass's job; here: MTP greedy, then MTP with the default (sampling) sampler
        if args.tg:
            res.append(workload(f"mtp{args.num_draft_tokens}_greedy", enc, args.tg, tg_rate, args.runs, gen,
                                sampler=ArgmaxSampler()))
            save()
    for n in args.tg_depth:
        res.append(workload(f"tg{args.depth_new}@d{n}", lambda r, n=n: depth_ids(tokenizer, n, r),
                            args.depth_new, tg_rate, args.runs, gen,
                            sampler=ArgmaxSampler() if spec else DefaultSampler()))
        save()

    info["amd_smi_after"] = sh("amd-smi metric -c -p -t -l --json")
    info["complete"] = True
    save()
    print(" -- RESULT " + " | ".join(f"{r['name']}={r['median']:.2f}" for r in res) + f"\n -- wrote {path}",
          flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
