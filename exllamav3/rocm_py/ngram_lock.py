"""Lock an n-gram embedding table (PLE models, e.g. Qwen3.8-Flash-Next) in RAM.

Three ways to hold the table (tens of GB), picked at load time:

  default        stream rows from disk per forward (the page cache keeps hot rows; the
                 kernel may drop them under memory pressure, and a cold row is a disk read)
  -ngr           load the whole table into RAM (NGramEmbedding "*_ram" modes). Ordinary
                 anonymous memory: with swap on, the kernel may swap parts of it out, and a
                 swapped row costs a page-in inside the forward
  -ngl           -ngr, then mlock() the table's pages: they stay resident until the process
                 exits (never swapped, never reclaimed)

The lock is applied to the tensors NGramEmbedding already holds -- no second copy, no
change to how the table is read -- by calling mlock(2) on each tensor's address range.
Locking requires RLIMIT_MEMLOCK to cover the table (or CAP_IPC_LOCK); preflight() checks
that and the memory budget (weights + table + KV cache + headroom <= MemAvailable) BEFORE
the model is loaded, so a refusal costs seconds, not a load; postflight() re-checks the
headroom after the load, before the lock.

This is plain Linux (mlock / getrlimit), nothing ROCm-specific; it lives in rocm_py because
that is this port's home for Python-side additions, and model_init / ngram_embedding stay
upstream-identical. Callers: rocm_tools/exl3_server/server.py (-ngl), bench/run_bench.py
(--ngram_lock).
"""

from __future__ import annotations
import ctypes
import ctypes.util
import glob
import json
import os
import resource
import struct

CAP_IPC_LOCK = 14


class NGramLockError(RuntimeError):
    pass


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def table_bytes(model_dir: str) -> int:
    """Bytes of n-gram table tensors (.trellis / .weight, incl. shard_N) in model_dir's
    safetensors headers; 0 when the model has none. Reads headers only."""
    total = 0
    for fn in glob.glob(os.path.join(model_dir, "*.safetensors")):
        try:
            with open(fn, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k == "__metadata__" or "ngram_embedding" not in k:
                continue
            if k.endswith(".trellis") or k.endswith(".weight"):
                a, b = v["data_offsets"]
                total += b - a
    return total


def _has_cap_ipc_lock() -> bool:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    return bool(int(line.split()[1], 16) >> CAP_IPC_LOCK & 1)
    except OSError:
        pass
    return False


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def vm_locked(pid: int | str = "self") -> int:
    """VmLck of a process in bytes (/proc/<pid>/status)."""
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmLck:"):
                return int(line.split()[1]) * 1024
    return 0


def _how_to_raise(need: int) -> str:
    kb = -(-need // 1024) + 1024 * 1024      # table + 1 GiB headroom, in KiB
    user = os.environ.get("USER", "<user>")
    return (
        "Raise the locked-memory limit (RLIMIT_MEMLOCK) of the server's process, one of:\n"
        f"  - shell:   ulimit -l unlimited   (or ulimit -l {kb}) before starting the server; the\n"
        "             hard limit must allow it, see /etc/security/limits.conf:\n"
        f"             {user}  -  memlock  unlimited     (then log in again)\n"
        "  - systemd: LimitMEMLOCK=infinity in the service's [Service] section\n"
        "             (user sessions: DefaultLimitMEMLOCK= in /etc/systemd/user.conf / system.conf)\n"
        "  - running shell, as root: prlimit --pid $$ --memlock=unlimited:unlimited\n"
        "  - or give the interpreter CAP_IPC_LOCK: setcap cap_ipc_lock+ep <python binary>\n"
        "Or drop -ngl: -ngr keeps the table in (unlocked) RAM, the default streams it from disk."
    )


def model_bytes(model_dir: str) -> int:
    """Bytes of everything else the load puts in memory from the model's safetensors (all
    files but the n-gram table's). On unified-memory APUs (gfx1151) the "VRAM" weights are
    system RAM too, so they count against the same budget."""
    total = 0
    for fn in glob.glob(os.path.join(model_dir, "*.safetensors")):
        try:
            with open(fn, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hlen))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k == "__metadata__" or "ngram_embedding" in k:
                continue
            a, b = v["data_offsets"]
            total += b - a
    return total


def kv_cache_bytes(model_dir: str, cache_tokens: int) -> int:
    """Estimate of the fp16 KV cache for cache_tokens (full-attention layers only; recurrent
    and sliding layers are small next to it). 0 when the config can't be read."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        types = tc.get("layer_types") or ["full_attention"] * tc["num_hidden_layers"]
        n_full = sum(1 for t in types if t == "full_attention")
        kvh = tc.get("num_key_value_heads", tc.get("num_attention_heads"))
        hd = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
        return int(n_full * kvh * hd * 2 * 2 * cache_tokens)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return 0


def headroom_bytes() -> int:
    return int(float(os.environ.get("EXL3_NGRAM_LOCK_HEADROOM_GB", 8)) * 2**30)


def preflight(model_dir: str, cache_tokens: int = 0) -> int:
    """Check, before loading, that the model's n-gram table can be locked. Returns the table
    size in bytes. Raises NGramLockError with instructions when:
      - the model has no n-gram table,
      - RLIMIT_MEMLOCK (soft, raised to the hard limit if that suffices) cannot cover it and
        the process lacks CAP_IPC_LOCK,
      - the memory budget does not fit: model weights + table + KV cache estimate + headroom
        (EXL3_NGRAM_LOCK_HEADROOM_GB, default 8) must be <= MemAvailable. A locked table can
        never be reclaimed, so an over-committed box would OOM-kill instead of evicting."""
    need = table_bytes(model_dir)
    if need == 0:
        raise NGramLockError(f"-ngl: {model_dir} has no n-gram embedding table to lock")

    if not _has_cap_ipc_lock():
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        inf = resource.RLIM_INFINITY
        want = need + vm_locked() + (64 << 20)          # slack for other small locks
        if soft != inf and soft < want:
            if hard == inf or hard >= want:
                # An unprivileged process may raise its soft limit up to the hard limit
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
            else:
                lim = "unlimited" if hard == inf else _gib(hard)
                raise NGramLockError(
                    f"-ngl: the n-gram table is {_gib(need)}, but this process may lock at most "
                    f"{lim} (RLIMIT_MEMLOCK hard limit; soft {_gib(soft)}, no CAP_IPC_LOCK).\n"
                    + _how_to_raise(need))

    avail = _mem_available()
    if avail is not None:
        mb = model_bytes(model_dir)
        kv = kv_cache_bytes(model_dir, cache_tokens)
        hr = headroom_bytes()
        budget = mb + need + kv + hr
        if budget > avail:
            raise NGramLockError(
                f"-ngl: does not fit. Model weights {_gib(mb)} + n-gram table {_gib(need)} + KV cache "
                f"~{_gib(kv)} ({cache_tokens} tokens) + headroom {_gib(hr)} = {_gib(budget)}, but only "
                f"{_gib(avail)} is available (MemAvailable). The locked table can never be reclaimed, "
                "so this would end in the OOM killer. Free memory, lower -cs / use -cq, lower the "
                "headroom (EXL3_NGRAM_LOCK_HEADROOM_GB), or use -ngr / the default disk streaming.")
    return need


def postflight():
    """After the load and before locking: the table is already resident, so locking takes
    no new memory, but what is left must still cover the headroom."""
    avail = _mem_available()
    hr = headroom_bytes()
    if avail is not None and avail < hr:
        raise NGramLockError(
            f"-ngl: after loading only {_gib(avail)} of RAM is available, less than the "
            f"{_gib(hr)} headroom (EXL3_NGRAM_LOCK_HEADROOM_GB); not locking. Use -ngr or free memory.")


_libc = None


def _mlock(addr: int, size: int):
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno = True)
        _libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if _libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size)) != 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))


def _munlock(addr: int, size: int):
    if _libc is not None:
        _libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))


def ngram_modules(model) -> list:
    from ..modules.ngram_embedding import NGramEmbedding
    return [m for m in model if isinstance(m, NGramEmbedding)]


def lock_tensors(tensors, max_bytes: int | None = None) -> list[tuple[int, int]]:
    """mlock the address range of each CPU tensor (or the first max_bytes in total: a test
    hook for boxes whose limit cannot cover a whole table). Returns the (addr, size) ranges
    locked; on failure unlocks them and raises OSError."""
    done = []
    left = max_bytes
    try:
        for t in tensors:
            assert t.device.type == "cpu" and t.is_contiguous()
            size = t.numel() * t.element_size()
            if left is not None:
                size = min(size, left)
                left -= size
            if size <= 0:
                continue
            _mlock(t.data_ptr(), size)
            done.append((t.data_ptr(), size))
    except OSError:
        for a, s in done:
            _munlock(a, s)
        raise
    return done


def lock_model(model, max_bytes: int | None = None) -> dict:
    """mlock every RAM-resident n-gram table of a loaded model. The model must have been
    loaded with the table in RAM (-ngr). Returns a report dict; raises NGramLockError on any
    failure (after unlocking whatever this call locked)."""
    if max_bytes is None:
        postflight()
    mods = ngram_modules(model)
    if not mods:
        raise NGramLockError("-ngl: the model has no n-gram embedding module")
    before = vm_locked()
    locked = []
    total = 0
    for m in mods:
        if not m.tables:
            raise NGramLockError(
                f"-ngl: {m.key} is in '{m.mode}' mode (streaming from disk); the lock needs the "
                "table in RAM (-ngr)")
        try:
            r = lock_tensors(m.tables, None if max_bytes is None else max_bytes - total)
        except OSError as e:
            for a, s in locked:
                _munlock(a, s)
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            need = sum(t.numel() * t.element_size() for t in m.tables)
            raise NGramLockError(
                f"-ngl: mlock of {m.key} ({_gib(need)}) failed: {e}. RLIMIT_MEMLOCK soft "
                f"{'unlimited' if soft == resource.RLIM_INFINITY else _gib(soft)}.\n"
                + _how_to_raise(need)) from e
        locked += r
        total += sum(s for _, s in r)
    after = vm_locked()
    return {
        "tables": [m.key for m in mods],
        "bytes_locked": total,
        "vmlck_before": before,
        "vmlck_after": after,
        "ranges": len(locked),
    }


def describe(report: dict) -> str:
    return (f"n-gram table locked in RAM: {_gib(report['bytes_locked'])} in {report['ranges']} "
            f"range(s); VmLck {_gib(report['vmlck_before'])} -> {_gib(report['vmlck_after'])}")
