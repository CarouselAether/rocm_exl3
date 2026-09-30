#!/usr/bin/env python3
"""Regression suite: record reference values on a known-good commit, then prove a branch regressed nothing.

Run through bench/regress.sh (sources exlproject/benv.sh, so $PY is the .venv10 interpreter):

    bench/regress.sh record [full|quick]          # write bench/regress_ref/ref.json (+ binaries in logs/regress_refs/<commit>/)
    bench/regress.sh check  [quick|full]          # compare against ref.json, PASS/FAIL table + markdown report
    bench/regress.sh check quick --allow-numerics ds4    # intentional numeric change on DS4: bitwise diff is not a failure
    bench/regress.sh check full --resume          # continue the latest interrupted check of this commit
    bench/regress.sh show                         # print the reference table
    bench/regress.sh report <run dir>             # rebuild the report of a finished / interrupted run

Selection: --models ds4 qwen ... and --only 'ds4/plain/pp*' (fnmatch globs over metric names) narrow a tier.
Metric names are <model>/<mode>/<workload>, e.g. ds4/plain/pp512, ds4/mtp2/tg128@d1024, laguna/dflash3/tg128@d2048,
ds4/plain/regen3000 (ms), ds4/bitwise/long, ds4/ppl/wiki2, suite/gates/gates, suite/isa/isa.

Every model step is its own process under exlproject/thermal_guard.py and a systemd scope
(MemoryMax=112G, MemorySwapMax=0), one at a time, after a cooldown / free-memory / no-other-GPU-process
check. Step results are written to <run dir>/state.json as they complete, so --resume continues after a
crash or a thermal kill. See bench/REGRESS.md for tiers, tolerances and re-recording.
"""

import argparse
import datetime
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = "/home/carousel/Desktop/exlproject"
GUARD = os.path.join(ROOT, "thermal_guard.py")
PY = sys.executable
REF_DIR = os.path.join(HERE, "regress_ref")
REF_JSON = os.path.join(REF_DIR, "ref.json")
REF_BIN = os.path.join(ROOT, "logs", "regress_refs")        # large binaries: NOT in git
RUNS = os.path.join(ROOT, "logs", "regress_runs")
REPORTS = os.path.join(HERE, "results", "regress")
SO = glob.glob(os.path.join(REPO, "exllamav3_ext.cpython-*.so"))
SO = SO[0] if SO else None
M = os.path.expanduser("~/models")

# Tolerances (fraction). plain/spec: tok/s, higher is better; regen: latency ms, lower is better; ppl: lower is better
TOL = {"plain": 0.02, "spec": 0.08, "regen": 0.03, "ppl": 0.005}
SPREAD_RERUN = 0.05        # record: a plain/regen metric with a spread above this is re-measured once
# Gate failures that are known flakes on this box (fail check only if a test outside this set fails)
KNOWN_FLAKES = {"test_dflash2.py::test_topk_cuda_matches_torch"}   # top-k tie order, 3/3 alone (dcc2f05 notes)
MEM_AVAIL_GIB = 96         # before each model load
COOL_C, COOL_MAX_S = 55.0, 600     # Tctl to wait for before each load (per-model cool= overrides)

MODELS = {   # order = run order
    "ds4": dict(path=f"{M}/DeepSeek-V4-Flash-0731-exl3-2.04bpw", cs=65536, spec="mtp2",
                ppl=(100, []), regen=[3000, 8000], label="DeepSeek-V4-Flash 2.04bpw"),
    "qwen": dict(path=f"{M}/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw", cs=65536, extra=["-ngr"], spec="mtp2",
                 ppl=(100, []), regen=[3000, 8000], noswap=True, label="Qwen3.8-Flash-Next 4bpw (-ngr)"),
    "mimo": dict(path=f"{M}/MiMo-V2.6-Flash-RL-exl3", cs=32768, spec="mtp2", ppl=(20, []),
                 label="MiMo-V2.6-Flash-RL 2.27bpw"),
    # cool=: start from near idle. GLM's load warmup pulls ~120 W and reached 100.1 C Tctl within 60 s
    # from a 51 C start (guard kill, 2026-09-29); Gemma peaked 97.5 C in the chart run
    "glm": dict(path=f"{M}/GLM-5.3-Flash-exl3-2.05bpw", cs=32768, spec="mtp2", ppl=(20, []), cool=40.0,
                label="GLM-5.3-Flash 2.05bpw"),
    "laguna": dict(path=f"{M}/Laguna-S-2.1-exl3-4.00bpw", cs=32768, spec="dflash3",
                   draft=f"{M}/Laguna-S-2.1-DFlash", label="Laguna-S-2.1 4bpw (+DFlash ndt 3)"),
    # Gemma needs BOS at position 0 for PPL: -gp (bench/run_ppl.sh)
    "gemma": dict(path=f"{M}/gemma-4-31b-it-exl3", cs=32768, ppl=(100, ["-gp"]), cool=40.0, label="Gemma-4-31B-it"),
}
TIMEOUT = {"plain": 2400, "spec": 1800, "bitwise": 1200, "ppl": 7200, "gates": 3600, "isa": 900}


# ----------------------------------------------------------------------------------------------- tiers

def tier_metrics(tier):
    out = []
    if tier == "quick":
        out += ["ds4/plain/pp512", "ds4/plain/pp2048", "ds4/plain/regen3000", "ds4/plain/tg128@d1024",
                "ds4/mtp2/tg128@d1024", "ds4/bitwise/short",
                "qwen/plain/pp512", "qwen/plain/pp2048", "qwen/plain/tg128@d1024", "qwen/bitwise/short",
                "suite/gates/gates", "suite/isa/isa"]
        return out
    for m, c in MODELS.items():
        out += [f"{m}/plain/pp512", f"{m}/plain/pp2048"]
        out += [f"{m}/plain/regen{n}" for n in c.get("regen", [])]
        out += [f"{m}/plain/tg128@d1024", f"{m}/plain/tg128@d2048"]
        if c.get("spec"):
            out += [f"{m}/{c['spec']}/tg128@d1024", f"{m}/{c['spec']}/tg128@d2048"]
        out += [f"{m}/bitwise/short", f"{m}/bitwise/long"]
        if c.get("ppl"):
            out += [f"{m}/ppl/wiki2"]
    return out + ["suite/gates/gates", "suite/isa/isa"]


def kind_of(metric):
    m, mode, wl = metric.split("/")
    if mode == "plain":
        return "regen" if wl.startswith("regen") else "plain"
    if mode in ("mtp2", "dflash3"):
        return "spec"
    return {"bitwise": "bitwise", "ppl": "ppl", "gates": "gates", "isa": "isa"}[mode]


def steps_of(metrics):
    """Group metrics into steps (model, mode), in run order."""
    steps, seen = [], {}
    order = list(MODELS) + ["suite"]
    modes = ["plain", "mtp2", "dflash3", "bitwise", "ppl", "gates", "isa"]
    for metric in sorted(metrics, key=lambda x: (order.index(x.split("/")[0]), modes.index(x.split("/")[1]))):
        sid = "/".join(metric.split("/")[:2])
        if sid not in seen:
            seen[sid] = []
            steps.append((sid, seen[sid]))
        seen[sid].append(metric)
    return steps


# ----------------------------------------------------------------------------------------------- system

def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"<{e}>"


def rd(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def git(c):
    return sh(f"git -C {REPO} {c}")


def commit():
    return git("rev-parse --short HEAD")


def dirty():
    """Tracked changes outside bench/ (bench/ holds this suite's own outputs)."""
    return bool(git("status --porcelain --untracked-files=no -- . ':!bench'"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tctl():
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(f"{d}/name") == "k10temp":
            v = rd(f"{d}/temp1_input")
            return int(v) / 1000 if v else None
    return None


def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def swap_on():
    return bool(sh("swapon --show --noheadings"))


def boost():
    return rd("/sys/devices/system/cpu/cpufreq/boost")


def kfd_users():
    """Other processes holding /dev/kfd (a GPU process still alive)."""
    me, out = os.getpid(), []
    for p in glob.glob("/proc/[0-9]*"):
        pid = int(os.path.basename(p))
        if pid == me:
            continue
        try:
            for fd in os.listdir(f"{p}/fd"):
                if os.readlink(f"{p}/fd/{fd}") == "/dev/kfd":
                    out.append(f"{pid}:{rd(f'{p}/comm')}")
                    break
        except OSError:
            continue
    return out


def stack_info():
    t = sh(f"{PY} -c \"import torch, json; print(json.dumps({{'torch': torch.__version__, 'hip': torch.version.hip}}))\"")
    try:
        t = json.loads(t.splitlines()[-1])
    except Exception:
        t = {"torch": t}
    so = {"path": SO, "sha256": sha256(SO) if SO else None,
          "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(SO)).isoformat(timespec="seconds") if SO else None}
    return {
        "commit": commit(), "branch": git("rev-parse --abbrev-ref HEAD"), "dirty": dirty(),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        **t, "rocm_pip": sh(f"{PY} -m pip show rocm 2>/dev/null | grep ^Version").replace("Version: ", ""),
        "kernel": os.uname().release,
        "cpu_boost": boost(), "swap": sh("swapon --show --noheadings") or "off",
        "perf_level": rd("/sys/class/drm/card0/device/power_dpm_force_performance_level"),
        "cpu_max_khz": rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "ext_so": so,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "HIP_", "HSA_", "ROCM", "PYTORCH_"))},
    }


def refuse(msg):
    print(f"\n !! REFUSING TO RUN: {msg}\n", flush=True)
    sys.exit(2)


def preflight_global(metrics):
    if boost() != "0":
        refuse(f"CPU boost is on (/sys/devices/system/cpu/cpufreq/boost = {boost()}). Boost + GPU load tripped "
               "THERMTRIP on 2026-09-26, and it changes the numbers. Turn it off: "
               "echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost")
    if any(MODELS.get(m.split("/")[0], {}).get("noswap") for m in metrics) and swap_on():
        refuse("swap is on and the selection includes Qwen3.8 -ngr (the n-gram table in RAM must not swap). "
               "Run `sudo swapoff -a` or drop qwen with --models.")
    if not SO:
        refuse(f"no built extension (exllamav3_ext*.so) in {REPO}")


def preflight_step(model):
    """Before every GPU process: boost off, swap off (Qwen), no other GPU process, free memory, cool CPU."""
    if boost() != "0":
        refuse("CPU boost turned on during the run")
    if MODELS.get(model, {}).get("noswap") and swap_on():
        refuse("swap turned on during the run (Qwen -ngr)")
    t0 = time.time()
    cool = MODELS.get(model, {}).get("cool", COOL_C)
    while True:
        users, avail, t = kfd_users(), mem_avail_gib(), tctl()
        waiting = []
        if users:
            waiting.append(f"GPU in use by {users}")
        if avail < MEM_AVAIL_GIB:
            waiting.append(f"MemAvailable {avail:.0f} GiB < {MEM_AVAIL_GIB}")
        if t is not None and t >= cool:
            waiting.append(f"Tctl {t:.1f} C >= {cool}")
        if not waiting:
            return {"mem_avail_gib": round(avail, 1), "tctl_start": t, "waited_s": round(time.time() - t0)}
        if time.time() - t0 > COOL_MAX_S:
            if users or avail < MEM_AVAIL_GIB:
                refuse("; ".join(waiting) + f" after {COOL_MAX_S}s")
            print(f"    (cooldown timeout: {'; '.join(waiting)}; starting anyway)", flush=True)
            return {"mem_avail_gib": round(avail, 1), "tctl_start": t, "waited_s": round(time.time() - t0)}
        time.sleep(5)


def gpu_cmd(cmd, timeout):
    return ["python3", GUARD, "--kill", "99.5", "--", "systemd-run", "--user", "--scope", "-q",
            "-p", "MemoryMax=112G", "-p", "MemorySwapMax=0", "--", "timeout", str(timeout)] + cmd


def run_logged(cmd, log, cwd=REPO):
    t0 = time.time()
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd).returncode
    txt = open(log, errors="replace").read()
    m = re.search(r"peak Tctl ([\d.]+) C", txt)
    return rc, time.time() - t0, (float(m.group(1)) if m else None), txt


def rc_note(rc):
    return {0: "", 99: "THERMAL KILL", 124: "TIMEOUT", 137: "KILLED (OOM?)"}.get(rc, f"rc {rc}")


# ----------------------------------------------------------------------------------------------- steps

def run_bench_step(model, mode, metrics, out, attempt, log):
    c = MODELS[model]
    wls = [x.split("/")[2] for x in metrics]
    pps = [w[2:] for w in wls if w.startswith("pp")]
    depths = [w.split("@d")[1] for w in wls if w.startswith("tg128@d")]
    regens = [w[5:] for w in wls if w.startswith("regen")]
    odir = os.path.join(out, "bench", f"{model}_{mode}_{attempt}")
    os.makedirs(odir, exist_ok=True)
    cmd = [PY, os.path.join(HERE, "run_bench.py"), "-m", c["path"], "-cs", str(c["cs"]), "--runs", "3",
           "--out", odir, "--label", f"{model}_{mode}", "--tg", "0",
           "--pp", *pps, "--tg_depth", *depths, "--regen", *regens]
    if mode == "mtp2":
        cmd += ["--mtp", "-ndt", "2"]
    elif mode == "dflash3":
        cmd += ["-dm", c["draft"], "-ndt", "3"]
    if c.get("extra"):
        cmd += ["--extra", *c["extra"]]
    rc, dt, peak, _ = run_logged(gpu_cmd(cmd, TIMEOUT["spec" if mode != "plain" else "plain"]), log)
    js = sorted(glob.glob(os.path.join(odir, "*.json")))
    res = {}
    if js:
        d = json.load(open(js[-1]))
        for r in d.get("results", []):
            name = f"{model}/{mode}/{r['name']}"
            if name in metrics:
                res[name] = {"value": r["median"], "spread": r["spread"],
                             "runs": [x["rate"] for x in r["runs"]],
                             "unit": r.get("unit", "tok/s"), "json": js[-1]}
                if "acceptance_median" in r:
                    res[name]["acceptance"] = r["acceptance_median"]
    for x in metrics:
        if x not in res:
            res[x] = {"error": rc_note(rc) or "missing from output", "log": log}
    return res, rc, dt, peak


def run_bitwise_step(model, metrics, out, attempt, log, save_dir=None, cmp_dir=None):
    c = MODELS[model]
    variants = [x.split("/")[2] for x in metrics]
    jpath = os.path.join(out, f"{model}_bitwise_{attempt}.json")
    cmd = [PY, os.path.join(HERE, "regress_bitwise.py"), "-m", c["path"], "--variants", *variants,
           "--tag", model, "--json", jpath] + (["--save", save_dir] if save_dir else ["--compare", cmp_dir])
    rc, dt, peak, _ = run_logged(gpu_cmd(cmd, TIMEOUT["bitwise"]), log)
    d = json.load(open(jpath)) if os.path.exists(jpath) else {}
    res = {}
    for v in variants:
        name = f"{model}/bitwise/{v}"
        res[name] = d[v] if v in d else {"error": rc_note(rc) or "missing from output", "log": log}
    return res, rc, dt, peak


def run_ppl_step(model, metrics, log):
    c = MODELS[model]
    rows, flags = c["ppl"]
    cmd = [PY, os.path.join(REPO, "eval", "ppl.py"), "-m", c["path"], "-r", str(rows), "-l", "2048", *flags]
    rc, dt, peak, txt = run_logged(gpu_cmd(cmd, TIMEOUT["ppl"]), log)
    m = re.findall(r"Perplexity: ([\d.]+)", txt)
    name = f"{model}/ppl/wiki2"
    r = {"value": float(m[-1]), "rows": rows, "length": 2048, "flags": flags} if m else \
        {"error": rc_note(rc) or "no Perplexity line", "log": log}
    return {name: r}, rc, dt, peak


def run_gates_step(out, log):
    gdir = os.path.join(out, "gates")
    cmd = ["bash", os.path.join(HERE, "run_gates.sh"), gdir]
    rc, dt, peak, txt = run_logged(gpu_cmd(cmd, TIMEOUT["gates"]), log)
    steps = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s+(\S+)\s+rc=(\d+)", txt, re.M)}
    pl = os.path.join(gdir, "pytest.log")
    ptxt = open(pl, errors="replace").read() if os.path.exists(pl) else ""
    failed = sorted(set(re.findall(r"^FAILED (\S+)", ptxt, re.M)) | set(re.findall(r"^ERROR (\S+)", ptxt, re.M)))
    summ = re.findall(r"^=*\s*(.*\d+ passed.*?)\s*=*$", ptxt, re.M)
    r = {"steps": steps, "pytest_failed": failed, "pytest_summary": summ[-1] if summ else None,
         "rc": rc, "logdir": gdir}
    if not steps or r["pytest_summary"] is None:
        r["error"] = rc_note(rc) or "gates did not complete"
    return {"suite/gates/gates": r}, rc, dt, peak


def run_isa_step(out, log, ref):
    ref_so = (ref or {}).get("isa_ref", {}).get("so")
    if not ref_so or not os.path.exists(ref_so):
        return {"suite/isa/isa": {"error": f"no reference .so ({ref_so})"}}, 1, 0.0, None
    work = os.path.join(out, "isa_work")
    os.makedirs(work, exist_ok=True)
    rc, dt, _, txt = run_logged([PY, os.path.join(REPO, "rocm_tools", "isa_diff.py"), ref_so, SO, "--work", work], log)
    m = re.search(r"common (\d+): (\d+) identical, (\d+) differ", txt)
    oo = re.search(r"only in old: (\d+)", txt)
    on = re.search(r"only in new: (\d+)", txt)
    if not m:
        return {"suite/isa/isa": {"error": f"isa_diff failed rc {rc}", "log": log}}, rc, dt, None
    r = {"common": int(m.group(1)), "identical": int(m.group(2)), "changed": int(m.group(3)),
         "only_old": int(oo.group(1)) if oo else None, "only_new": int(on.group(1)) if on else None,
         "same_so": sha256(ref_so) == sha256(SO), "log": log,
         "first_changed": re.findall(r"DIFF (.*?)\s+\(", txt)[:5]}
    return {"suite/isa/isa": r}, 0, dt, None


# ----------------------------------------------------------------------------------------------- compare

def better(kind, a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b) if kind in ("regen", "ppl") else max(a, b)


def judge(metric, cur, refm, allow_numerics=()):
    """-> (status, delta, note). status: PASS FAIL IMPROVED CHANGED INFO NOREF ERROR"""
    kind = kind_of(metric)
    model = metric.split("/")[0]
    if cur is None or "error" in cur:
        return "ERROR", None, (cur or {}).get("error", "not run")
    if kind == "isa":
        note = (f"{cur['changed']} of {cur['common']} existing functions changed, "
                f"+{cur['only_new']} new / -{cur['only_old']} removed" + (" (same .so as reference)" if cur.get("same_so") else ""))
        return "INFO", None, note
    if kind == "gates":
        allowed = KNOWN_FLAKES | set((refm or {}).get("pytest_failed", []))
        bad_steps = [k for k, v in cur["steps"].items() if v != 0 and k != "pytest"]
        new_fail = [t for t in cur["pytest_failed"] if t not in allowed]
        flakes = [t for t in cur["pytest_failed"] if t in allowed]
        note = f"{cur['pytest_summary']}" + (f"; known flake/ref failures: {', '.join(flakes)}" if flakes else "")
        if bad_steps or new_fail:
            return "FAIL", None, f"failed steps {bad_steps}, new test failures {new_fail}; " + note
        return "PASS", None, note
    if refm is None:
        return "NOREF", None, "no reference value"
    if kind == "bitwise":
        if refm.get("nondeterministic"):
            return "INFO", None, f"reference not reproducible run-to-run; now {cur['status']}"
        if cur["status"] == "identical":
            return "PASS", None, f"all {cur['steps']} steps bit-identical"
        note = (f"{cur.get('n_steps_differ')} of {cur.get('steps')} steps differ, first {cur.get('first_step')}, "
                f"max|d| {cur.get('max_abs_diff', 0):.3e}, tokens same {cur.get('tokens_identical')}")
        if model in allow_numerics:
            return "CHANGED", None, note + " (allowed by --allow-numerics)"
        return "FAIL", None, note
    ref, v = refm["value"], cur["value"]
    d = (v - ref) / ref
    tol = TOL[kind]
    note = ""
    if kind == "ppl":
        if d > tol:
            return "FAIL", d, "worse than +0.5%"
        if v != ref:
            return ("IMPROVED" if d < 0 else "PASS"), d, "changed" if v != ref else ""
        return "PASS", d, "identical"
    if "spread" in cur:
        note = f"spread {cur['spread']:.1%}"
    if "acceptance" in cur:
        note += f", acc {cur['acceptance']:.1%} (ref {refm.get('acceptance', 0):.1%})"
    if cur.get("rerun"):
        note += f", rerun: {' / '.join(f'{x:.2f}' for x in cur['rerun'])}"
    if kind == "regen":
        return ("FAIL" if d > tol else "IMPROVED" if d < -tol else "PASS"), d, note
    return ("FAIL" if d < -tol else "IMPROVED" if d > tol else "PASS"), d, note


def needs_rerun(metric, cur, refm, mode):
    kind = kind_of(metric)
    if kind not in ("plain", "spec", "regen"):
        return False
    if cur is None or "error" in cur:
        return True
    if mode == "record":
        return kind in ("plain", "regen") and cur["spread"] > SPREAD_RERUN
    return judge(metric, cur, refm)[0] == "FAIL"


def merge_rerun(metric, first, second, mode):
    """record: keep the lower-spread attempt; check: keep the better value (a drop must reproduce)."""
    if second is None or "error" in second:
        return first
    if first is None or "error" in first:
        return second
    kind = kind_of(metric)
    if mode == "record":
        keep = second if second["spread"] < first["spread"] else first
    else:
        keep = dict(first if better(kind, first["value"], second["value"]) == first["value"] else second)
    keep = dict(keep)
    keep["rerun"] = [first["value"], second["value"]]
    return keep


# ----------------------------------------------------------------------------------------------- driver

def load_ref():
    if os.path.exists(REF_JSON):
        return json.load(open(REF_JSON))
    return None


def save_state(out, st):
    tmp = os.path.join(out, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(out, "state.json"))


def find_resume(mode, arg):
    if arg and arg != "latest":
        return arg
    c = commit()
    cands = sorted(glob.glob(os.path.join(RUNS, f"{c}_{mode}_*")), key=os.path.getmtime)
    for d in reversed(cands):
        p = os.path.join(d, "state.json")
        if os.path.exists(p) and not json.load(open(p)).get("complete"):
            return d
    sys.exit(f" !! nothing to resume for {mode} on {c} in {RUNS}")


def execute(mode, metrics, out, st, ref, allow_numerics=(), rerun=True):
    """Run every not-yet-done step; state is saved after each step."""
    c = st["commit"]
    bin_dir = os.path.join(REF_BIN, c)
    for sid, ms in steps_of(metrics):
        if st["steps"].get(sid, {}).get("done"):
            print(f" -- {sid:16} done (resumed)", flush=True)
            continue
        model, smode = sid.split("/")
        info = {"t_start": datetime.datetime.now().isoformat(timespec="seconds"), "attempts": []}
        t0 = time.time()
        print(f" -- {sid:16} {', '.join(x.split('/')[2] for x in ms)}", flush=True)

        def attempt(sel, n):
            log = os.path.join(out, f"{sid.replace('/', '_')}_{n}.log")
            if smode not in ("isa",):
                info["attempts"].append({"pre": preflight_step(model)})
            else:
                info["attempts"].append({})
            if smode in ("plain", "mtp2", "dflash3"):
                r = run_bench_step(model, smode, sel, out, n, log)
            elif smode == "bitwise":
                if mode == "record":
                    r = run_bitwise_step(model, sel, out, n, log, save_dir=bin_dir)
                else:
                    cmp_dirs = {os.path.dirname(ref["metrics"][x]["file"]) for x in sel
                                if x in ref["metrics"] and "file" in ref["metrics"][x]}
                    r = run_bitwise_step(model, sel, out, n, log, cmp_dir=(cmp_dirs.pop() if cmp_dirs else bin_dir))
            elif smode == "ppl":
                r = run_ppl_step(model, sel, log)
            elif smode == "gates":
                r = run_gates_step(out, log)
            elif smode == "isa":
                if mode == "record":
                    os.makedirs(bin_dir, exist_ok=True)
                    dst = os.path.join(bin_dir, os.path.basename(SO))
                    shutil.copy2(SO, dst)
                    r = ({"suite/isa/isa": {"so": dst, "sha256": sha256(dst)}}, 0, 0.0, None)
                else:
                    r = run_isa_step(out, log, ref)
            res, rc, dt, peak = r
            info["attempts"][-1].update(rc=rc, seconds=round(dt), peak_tctl=peak, log=log)
            return res

        res = attempt(ms, 1)
        # record: bitwise determinism check in a second process (a reference that is not reproducible
        # run-to-run is useless as a bit-exact bar; it is then recorded as nondeterministic, report-only)
        if mode == "record" and smode == "bitwise" and all(res[x].get("status") == "saved" for x in ms):
            info["attempts"].append({"pre": preflight_step(model), "verify": True})
            v = run_bitwise_step(model, ms, out, 2, os.path.join(out, f"{sid.replace('/', '_')}_verify.log"),
                                 cmp_dir=bin_dir)
            for x in ms:
                res[x]["verify"] = v[0][x].get("status", v[0][x].get("error"))
                res[x]["nondeterministic"] = res[x]["verify"] != "identical"
                if res[x]["nondeterministic"]:
                    print(f"    !! {x}: second process {res[x]['verify']}: reference marked nondeterministic "
                          f"({v[0][x]})", flush=True)
        if rerun:
            again = [x for x in ms if needs_rerun(x, res.get(x), (ref or {}).get("metrics", {}).get(x), mode)]
            if again:
                print(f"    rerun: {', '.join(again)}", flush=True)
                res2 = attempt(again, 2)
                for x in again:
                    res[x] = merge_rerun(x, res.get(x), res2.get(x), mode)
        info["seconds"] = round(time.time() - t0)
        info["metrics"] = res
        info["done"] = True
        st["steps"][sid] = info
        save_state(out, st)
        for x in ms:
            print("    " + line_for(x, res.get(x), (ref or {}).get("metrics", {}).get(x), mode, allow_numerics), flush=True)
    st["complete"] = True
    st["t_end"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_state(out, st)


def fmt(metric, r):
    if r is None or "error" in r:
        return "—"
    k = kind_of(metric)
    if k in ("plain", "spec"):
        return f"{r['value']:.2f}"
    if k == "regen":
        return f"{r['value']:.1f} ms"
    if k == "ppl":
        return f"{r['value']:.6f}"
    if k == "bitwise":
        return r.get("status", "?") if "verify" not in r else ("saved" + (" (nondet)" if r.get("nondeterministic") else ""))
    if k == "gates":
        return "rc " + " ".join(f"{a}={b}" for a, b in r["steps"].items())
    if k == "isa":
        return f"{r.get('changed')} changed" if "changed" in r else "ref .so"
    return str(r)


def line_for(metric, cur, refm, mode, allow_numerics=()):
    if mode == "record":
        extra = ""
        if cur and "spread" in cur:
            extra = f"  spread {cur['spread']:.1%}" + (f"  acc {cur['acceptance']:.1%}" if "acceptance" in cur else "")
        if cur and "error" in cur:
            extra = f"  ERROR {cur['error']}"
        return f"{metric:32} {fmt(metric, cur):>14}{extra}"
    status, d, note = judge(metric, cur, refm, allow_numerics)
    ds = f"{d:+.2%}" if d is not None else ""
    return f"{metric:32} {fmt(metric, refm):>14} -> {fmt(metric, cur):>14} {ds:>8}  {status:8} {note}"


def tol_str(metric):
    k = kind_of(metric)
    return {"plain": "-2%", "spec": "-8%", "regen": "+3%", "ppl": "+0.5%", "bitwise": "bit-exact",
            "gates": "no new fail", "isa": "info"}[k]


def write_report(st, ref, allow_numerics=()):
    metrics = st["metrics"]
    rows, statuses = [], []
    for x in metrics:
        sid = "/".join(x.split("/")[:2])
        cur = st["steps"].get(sid, {}).get("metrics", {}).get(x)
        refm = (ref or {}).get("metrics", {}).get(x)
        s, d, note = judge(x, cur, refm, allow_numerics)
        statuses.append(s)
        model, mode, wl = x.split("/")
        rows.append(f"| {model} | {mode} | {wl} | {fmt(x, refm)} | {fmt(x, cur)} | "
                    f"{(f'{d:+.2%}' if d is not None else '')} | {tol_str(x)} | **{s}** | {note} |")
    ok = all(s in ("PASS", "IMPROVED", "INFO", "CHANGED") for s in statuses)
    s = st["stack"]
    secs = sum(v.get("seconds", 0) for v in st["steps"].values())
    ref_commits = sorted({(ref or {}).get("metrics", {}).get(x, {}).get("commit", "?") for x in metrics})
    lines = [
        f"# Regression check {s['commit']} ({st['tier']}) — {'PASS' if ok else 'FAIL'}", "",
        f"- **Commit:** `{s['commit']}` on `{s['branch']}`{' (DIRTY outside bench/)' if s['dirty'] else ''}; "
        f"reference commit(s): {', '.join(f'`{c}`' for c in ref_commits)}",
        f"- **Stack:** torch {s.get('torch')}, HIP {s.get('hip')}, rocm pip {s.get('rocm_pip')}, kernel {s.get('kernel')}",
        f"- **System:** CPU boost {s['cpu_boost']}, swap {s['swap']}, GPU perf level {s['perf_level']}, "
        f"cpu max {s['cpu_max_khz']} kHz",
        f"- **Extension:** `{os.path.basename(s['ext_so']['path'] or '')}` sha256 {str(s['ext_so']['sha256'])[:16]}…, "
        f"built {s['ext_so']['mtime']}",
        f"- **Env:** {s['env'] or 'no EXL3_/HIP_/HSA_ overrides'}",
        f"- **Started:** {st['t_start']}, steps took {secs / 60:.1f} min in total; run dir `{st['out']}`",
        f"- **Tolerances:** plain perf -2% (a drop must reproduce on an automatic rerun), spec (MTP/DFlash) -8%, "
        f"regen latency +3%, PPL +0.5%, decode logits bit-exact"
        + (f" (numerics allowed to change: {', '.join(allow_numerics)})" if allow_numerics else ""), "",
        "| Model | Mode | Metric | Reference | Current | Δ | Tol | Status | Notes |",
        "|---|---|---|---|---|---|---|---|---|", *rows, "",
        "## Step times", "", "| Step | Seconds | Peak Tctl | Attempts |", "|---|---|---|---|",
    ]
    for sid, v in st["steps"].items():
        peaks = [a.get("peak_tctl") for a in v.get("attempts", []) if a.get("peak_tctl")]
        lines.append(f"| {sid} | {v.get('seconds')} | {max(peaks) if peaks else ''} | {len(v.get('attempts', []))} |")
    counts = {k: statuses.count(k) for k in sorted(set(statuses))}
    lines += ["", f"**Result: {'PASS' if ok else 'FAIL'}** — {counts}", ""]
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"{s['commit']}_{st['stamp']}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return ok, path, counts


def write_ref(st, ref):
    """Merge this record run into ref.json (metrics not re-recorded are kept, each with its own commit)."""
    ref = ref or {"metrics": {}, "records": []}
    s = st["stack"]
    n = 0
    for sid, v in st["steps"].items():
        for x, r in v.get("metrics", {}).items():
            if r is None or "error" in r:
                print(f" !! {x}: not recorded ({(r or {}).get('error')})", flush=True)
                continue
            r = dict(r)
            if x == "suite/isa/isa":
                ref["isa_ref"] = {"so": r["so"], "sha256": r["sha256"], "commit": s["commit"]}
            r.update(commit=s["commit"], time=v["t_start"])
            ref["metrics"][x] = r
            n += 1
    if n == 0:
        return 0
    ref["stack"] = s
    ref["records"].append({"commit": s["commit"], "time": st["t_start"], "tier": st["tier"], "metrics": n,
                           "seconds": sum(v.get("seconds", 0) for v in st["steps"].values()),
                           "run_dir": st["out"]})
    ref["tolerances"] = TOL
    os.makedirs(REF_DIR, exist_ok=True)
    with open(REF_JSON, "w") as f:
        json.dump(ref, f, indent=1)
    with open(os.path.join(REF_DIR, "REF.md"), "w") as f:
        f.write(ref_table(ref))
    return n


def ref_table(ref):
    L = [f"# Regression reference values", "",
         f"Latest record: `{ref['stack']['commit']}`, torch {ref['stack'].get('torch')}, HIP {ref['stack'].get('hip')}, "
         f"boost {ref['stack']['cpu_boost']}, perf level {ref['stack']['perf_level']}. Generated by "
         f"`bench/regress.sh record`; do not edit by hand.", "",
         "| Model | Mode | Metric | Value | Spread | Acceptance | Commit |", "|---|---|---|---|---|---|---|"]
    order = tier_metrics("full")
    for x in sorted(ref["metrics"], key=lambda k: order.index(k) if k in order else 999):
        r = ref["metrics"][x]
        m, mode, wl = x.split("/")
        k = kind_of(x)
        val = fmt(x, r) if k not in ("bitwise", "gates", "isa") else (
            ("nondeterministic" if r.get("nondeterministic") else f"saved, reproducible ({r.get('prompt_tokens')} tok prompt)")
            if k == "bitwise" else (str(r.get("pytest_summary")) + (f"; failed {r['pytest_failed']}" if r.get("pytest_failed") else ""))
            if k == "gates" else f"ref .so {r.get('sha256', '')[:12]}")
        spread = f"{r['spread']:.1%}" if "spread" in r else ""
        acc = f"{r['acceptance']:.1%}" if "acceptance" in r else ""
        L.append(f"| {m} | {mode} | {wl} | {val} | {spread} | {acc} | {r.get('commit')} |")
    L += ["", "| Record | Tier | Metrics | Minutes |", "|---|---|---|---|"]
    for rec in ref["records"]:
        L.append(f"| `{rec['commit']}` {rec['time']} | {rec['tier']} | {rec['metrics']} | {rec['seconds'] / 60:.1f} |")
    return "\n".join(L) + "\n"


def select(tier, models, only):
    ms = tier_metrics(tier)
    if models:
        ms = [x for x in ms if x.split("/")[0] in models or (x.startswith("suite/") and "suite" in models)]
    if only:
        ms = [x for x in ms if any(fnmatch.fnmatch(x, g) for g in only)]
    if not ms:
        sys.exit(" !! empty selection")
    return ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "check", "show", "report"])
    ap.add_argument("tier", nargs="?", default=None, help="quick | full (record default full, check default quick); "
                                                          "for `report`: the run dir")
    ap.add_argument("--models", nargs="+", help=f"subset of {list(MODELS)} (+ 'suite' for gates / isa)")
    ap.add_argument("--only", nargs="+", help="metric-name globs, e.g. 'ds4/plain/pp512'")
    ap.add_argument("--allow-numerics", nargs="+", default=[], metavar="MODEL",
                    help="bitwise differences on these models are reported as CHANGED, not FAIL")
    ap.add_argument("--resume", nargs="?", const="latest", help="continue the latest incomplete run (or a given run dir)")
    ap.add_argument("--no-rerun", action="store_true", help="check: do not re-measure failing perf metrics")
    ap.add_argument("--allow-dirty", action="store_true", help="record: allow a tree with changes outside bench/")
    ap.add_argument("--label", default="", help="suffix for the run dir")
    ap.add_argument("--ref", default=None, help="alternate ref.json (testing the suite itself); REF.md goes next to it")
    a = ap.parse_args()
    global REF_JSON, REF_DIR
    if a.ref:
        REF_JSON, REF_DIR = os.path.abspath(a.ref), os.path.dirname(os.path.abspath(a.ref))
    ref = load_ref()

    if a.cmd == "show":
        if not ref:
            sys.exit(f" !! no reference at {REF_JSON}")
        print(ref_table(ref))
        return
    if a.cmd == "report":
        st = json.load(open(os.path.join(a.tier, "state.json")))
        ok, path, counts = write_report(st, ref, st.get("allow_numerics", []))
        print(f" -- {path}: {'PASS' if ok else 'FAIL'} {counts}")
        sys.exit(0 if ok else 1)

    mode = a.cmd
    tier = a.tier or ("full" if mode == "record" else "quick")
    if tier not in ("quick", "full"):
        sys.exit(f" !! unknown tier {tier}")
    if mode == "check" and not ref:
        refuse(f"no reference at {REF_JSON}; run `bench/regress.sh record` on the baseline commit first")

    if a.resume:
        out = find_resume(mode, a.resume)
        st = json.load(open(os.path.join(out, "state.json")))
        metrics = st["metrics"]
        print(f" -- resuming {out}: {sum(1 for v in st['steps'].values() if v.get('done'))} steps done", flush=True)
        if st["commit"] != commit():
            refuse(f"run dir is for {st['commit']}, HEAD is {commit()}")
    else:
        metrics = select(tier, a.models, a.only)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = os.path.join(RUNS, f"{commit()}_{mode}_{tier}_{stamp}" + (f"_{a.label}" if a.label else ""))
        os.makedirs(out, exist_ok=True)
        st = {"mode": mode, "tier": tier, "commit": commit(), "metrics": metrics, "steps": {}, "out": out,
              "stamp": stamp, "t_start": datetime.datetime.now().isoformat(timespec="seconds"),
              "allow_numerics": a.allow_numerics}
    preflight_global(metrics)
    if mode == "record":
        if dirty() and not a.allow_dirty:
            refuse("tree has uncommitted changes outside bench/; a reference must be a commit (--allow-dirty to override)")
        if [k for k in os.environ if k.startswith("EXL3_")]:
            refuse(f"EXL3_* overrides set in the environment: {[k for k in os.environ if k.startswith('EXL3_')]}")
    st["stack"] = stack_info()
    st.setdefault("allow_numerics", a.allow_numerics)
    save_state(out, st)
    s = st["stack"]
    print(f" -- regress {mode} {tier}: {len(metrics)} metrics in {len(steps_of(metrics))} steps; {s['commit']} "
          f"on {s['branch']}, torch {s.get('torch')}, HIP {s.get('hip')}, boost {s['cpu_boost']}, swap {s['swap']}, "
          f"perf {s['perf_level']}\n -- run dir {out}", flush=True)
    for k, v in s["env"].items():
        print(f"    env {k}={v}", flush=True)

    t0 = time.time()
    execute(mode, metrics, out, st, ref, st["allow_numerics"], rerun=not a.no_rerun)
    wall = time.time() - t0
    if mode == "record":
        n = write_ref(st, ref)
        print(f"\n -- recorded {n} metrics into {REF_JSON} (+ REF.md); binaries in {os.path.join(REF_BIN, st['commit'])}; "
              f"wall {wall / 60:.1f} min", flush=True)
        return
    ok, path, counts = write_report(st, ref, st["allow_numerics"])
    print(f"\n -- {'PASS' if ok else 'FAIL'} {counts}; wall {wall / 60:.1f} min\n -- report {path}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
