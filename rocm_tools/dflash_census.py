"""DFlash round census on Laguna (opt/dflash-drafter, 2026-09-30).

Loads Laguna-S-2.1 + a DFlash drafter as the server does, generates greedily, and reports per-round
wall time split into target verify, draft forward and draft sampling; with --time_calls, every
drafter Linear call by (key, m, K, N, format) with sync'd per-call time. --dump writes each round's
draft ids; --no_trunc restores the full 16-row block, so two dumps compare truncated vs full-block
drafts. --plain runs without a drafter (m = 1 target step). All timing is sync'd: a census, not a
throughput number (use bench/run_bench.py -dm for that).

  python rocm_tools/dflash_census.py --new 256 [--dm DIR] [--time_calls] [--dump f.json] [--no_trunc]
"""
import sys, os, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.modules.quant import fp16 as fp16mod

M = os.path.expanduser("~/models")
ap = argparse.ArgumentParser()
ap.add_argument("--new", type=int, default=96)
ap.add_argument("--time_calls", action="store_true")
ap.add_argument("--plain", action="store_true")
ap.add_argument("--check_slice", action="store_true")
ap.add_argument("--no_trunc", action="store_true")
ap.add_argument("--dump", default=None)
ap.add_argument("--dm", default=f"{M}/Laguna-S-2.1-DFlash")
a = ap.parse_args()

parser = argparse.ArgumentParser()
model_init.add_args(parser, cache=True, add_sampling_args=False, add_draft_model_args=True,
                    default_autosplit_max_batch_size=4)
iargs = parser.parse_args(["-m", f"{M}/Laguna-S-2.1-exl3-4.00bpw", "-cs", "32768",
                           "-dm", a.dm, "-ndt", "3"])
model, config, cache, tokenizer, draft_model, _dc, draft_cache = model_init.init(iargs)
gen = Generator(model=model, cache=cache, tokenizer=tokenizer, draft_model=None if a.plain else draft_model,
                draft_cache=None if a.plain else draft_cache, num_draft_tokens=iargs.num_draft_tokens)

draft_linears = set()
for top in draft_model.modules:
    for sub in top:
        if type(sub).__name__ == "Linear":
            draft_linears.add(id(sub))
print("draft LinearFP16 count:", len(draft_linears))

stats = collections.defaultdict(lambda: [0, 0.0])
active = [False]
from exllamav3.modules.linear import Linear as _Lin
orig = _Lin.forward
def fwd(self, x, params, out_dtype=None):
    if not active[0] or id(self) not in draft_linears:
        return orig(self, x, params, out_dtype)
    m = x.numel() // x.shape[-1]
    dt = out_dtype or self.out_dtype or torch.half
    key = (self.key, m, self.in_features, self.out_features, str(x.dtype), type(self.inner).__name__, str(dt))
    if a.time_calls:
        torch.cuda.synchronize(); t = time.perf_counter()
        y = orig(self, x, params, out_dtype)
        torch.cuda.synchronize(); stats[key][1] += time.perf_counter() - t
    else:
        y = orig(self, x, params, out_dtype)
    stats[key][0] += 1
    return y
_Lin.forward = fwd

dsteps = [0, 0.0]
dorig = draft_model.forward
def dfwd(*args, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    r = dorig(*args, **kw)
    torch.cuda.synchronize(); dsteps[1] += time.perf_counter() - t; dsteps[0] += 1
    return r
draft_model.forward = dfwd

tsteps = collections.defaultdict(lambda: [0, 0.0])
torig = model.forward
def tfwd(*args, **kw):
    ii = kw.get("input_ids", args[0] if args else None)
    torch.cuda.synchronize(); t = time.perf_counter()
    r = torig(*args, **kw)
    torch.cuda.synchronize()
    k = tuple(ii.shape) if ii is not None else None
    tsteps[k][1] += time.perf_counter() - t; tsteps[k][0] += 1
    return r
model.forward = tfwd
ssteps = [0, 0.0]
chk = [0, 0, None]
sorig = draft_model.sample_from_state
def sfwd(*args, **kw):
    torch.cuda.synchronize(); t = time.perf_counter()
    r = sorig(*args, **kw)
    torch.cuda.synchronize(); ssteps[1] += time.perf_counter() - t; ssteps[0] += 1
    if a.check_slice:
        st, pr = args[0], dict(args[1]); rows = pr.pop("draft_rows", None)
        full = sorig(st, pr)
        chk[0] += 1; chk[2] = (tuple(st.shape), tuple(r.shape), rows)
        if not torch.equal(full[..., :r.shape[-1]], r): chk[1] += 1
    return r
draft_model.sample_from_state = sfwd
drafts = []
_sf2 = draft_model.sample_from_state
def sfwd2(*args, **kw):
    r = _sf2(*args, **kw)
    if active[0]: drafts.append((tuple(args[0].shape), r[0, 1:4].tolist()))
    return r
draft_model.sample_from_state = sfwd2
if a.no_trunc:
    from exllamav3.architecture import dflash_laguna as _dl
    _pi = _dl.DFlashLagunaModel.prepare_inputs
    def _pi2(self, input_ids, params):
        r = _pi(self, input_ids, params); params.pop("draft_block_rows", None); return r
    _dl.DFlashLagunaModel.prepare_inputs = _pi2

prompt = "Write a detailed explanation of how a hash table handles collisions, with examples."
ids = tokenizer.encode(prompt, add_bos=True)
# warmup
gen.enqueue(Job(input_ids=ids, max_new_tokens=32, sampler=ArgmaxSampler()));
while gen.num_remaining_jobs(): gen.iterate()
stats.clear(); dsteps[:] = [0, 0.0]; tsteps.clear(); ssteps[:] = [0, 0.0]
active[0] = True
t = time.perf_counter()
gen.enqueue(Job(input_ids=ids, max_new_tokens=a.new, sampler=ArgmaxSampler()))
while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize(); wall = time.perf_counter() - t
active[0] = False

print(f"\nwall {wall*1e3:.0f} ms for {a.new} tokens; draft forwards {dsteps[0]}, {dsteps[1]*1e3:.1f} ms total, "
      f"{dsteps[1]/max(dsteps[0],1)*1e3:.2f} ms/step")
for k, v in sorted(tsteps.items(), key=lambda kv: str(kv[0])): print(f"target forward {k}: {v[0]} calls, {v[1]/v[0]*1e3:.2f} ms/call")
print(f"draft sample_from_state: {ssteps[0]} calls, {ssteps[1]/max(ssteps[0],1)*1e3:.2f} ms/call")
print(f"slice check: {chk[0]} rounds, {chk[1]} mismatches, shapes {chk[2]}")
tot = sum(v[1] for v in stats.values())
print(f"{'key':40} {'m':>4} {'K':>6} {'N':>6} {'x':>14} {'w':>14} {'y':>14} {'calls':>6} {'us/call':>8}")
for k, v in sorted(stats.items(), key=lambda kv: -kv[1][1] if a.time_calls else kv[0][0]):
    print(f"{str(k[0]):40} {k[1]:>4} {k[2]:>6} {k[3]:>6} {k[4]:>14} {k[5]:>14} {k[6]:>14} {v[0]:>6} "
          f"{v[1]/v[0]*1e6 if v[0] else 0:8.1f}")
print(f"sum linear time {tot*1e3:.1f} ms")

if a.dump:
    import json; json.dump({"drafts": drafts}, open(a.dump, "w"))
    print("dumped", len(drafts), "rounds, state shape", drafts[0][0] if drafts else None)
