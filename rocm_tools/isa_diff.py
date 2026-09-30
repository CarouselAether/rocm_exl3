#!/usr/bin/env python3
"""Device-code A/B of two builds: which GPU functions changed, instruction for instruction.

The isolation check for changes that must leave existing kernels alone (e.g. new template
instantiations behind `if constexpr`): extract each .so's code object for one arch, disassemble,
split per function and compare. PC-relative symbol offsets (s_getpc_b64 + s_add_u32 literal) and
inter-function padding are masked -- they move whenever anything is added to the binary.

    python rocm_tools/isa_diff.py OLD.so NEW.so [--arch gfx1151] [--work DIR]

Prints: functions in both / identical / different, functions only in one build (by template
name), and the first differing functions. Exit 1 if any common function differs.
"""

import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile


def llvm_bin():
    import glob
    here = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "_rocm_sdk_devel", "lib", "llvm", "bin"))
    return (here or ["/opt/rocm/llvm/bin"])[0]


def extract(so, out, arch, B):
    subprocess.run([f"{B}/llvm-objcopy", "--dump-section", f".hip_fatbin={out}.fatbin", so, f"{out}.tmp"], check = True)
    lst = subprocess.run([f"{B}/clang-offload-bundler", "--list", "--type=o", f"--input={out}.fatbin"],
                         capture_output = True, text = True, check = True).stdout.split()
    tgt = [t for t in lst if t.endswith(arch)][0]
    subprocess.run([f"{B}/clang-offload-bundler", "--unbundle", "--type=o", f"--input={out}.fatbin",
                    f"--targets={tgt}", f"--output={out}.co"], check = True)
    dis = subprocess.run([f"{B}/llvm-objdump", "-d", "--no-show-raw-insn", "--no-leading-addr", f"{out}.co"],
                         capture_output = True, text = True, check = True).stdout
    for f in (f"{out}.fatbin", f"{out}.tmp", f"{out}.co"):
        os.remove(f)
    return dis


def funcs(text):
    fs, cur, body = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^(?:[0-9a-f]+ )?<(.+)>:$", line.strip())
        if m:
            if cur:
                fs[cur] = body
            cur, body = re.sub(r"\.intern\.[0-9a-f]+", "", m.group(1)), []
            continue
        if cur is None:
            continue
        l = re.sub(r"//.*$", "", line.strip()).strip()
        if not l or l.startswith("Disassembly") or l in ("v_illegal", "...", "s_code_end"):
            continue
        l = re.sub(r"<[^>]*>", "<>", l)
        if body and body[-1].startswith("s_getpc_b64") and l.startswith("s_add_u32"):
            l = re.sub(r"0x[0-9a-f]+|-?\d+$", "PCREL", l)
        body.append(l)
    if cur:
        fs[cur] = body
    return fs


def demangle(names):
    if not names:
        return []
    return subprocess.run(["c++filt"], input = "\n".join(names), capture_output = True, text = True).stdout.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--arch", default = "gfx1151")
    ap.add_argument("--work", default = None)
    a = ap.parse_args()
    B = llvm_bin()
    work = a.work or tempfile.mkdtemp(prefix = "isa_diff_")
    old = funcs(extract(a.old, os.path.join(work, "old"), a.arch, B))
    new = funcs(extract(a.new, os.path.join(work, "new"), a.arch, B))
    common = sorted(set(old) & set(new))
    diff = [f for f in common if old[f] != new[f]]
    print(f"{a.arch}: old {len(old)} functions, new {len(new)}; common {len(common)}: "
          f"{len(common) - len(diff)} identical, {len(diff)} differ")
    for label, names in (("only in old", sorted(set(old) - set(new))), ("only in new", sorted(set(new) - set(old)))):
        dm = demangle(names)
        by = collections.Counter(re.sub(r"<.*", "", n) for n in dm)
        print(f"{label}: {len(names)}" + (f"  {dict(by)}" if names else ""))
    for f, d in zip(diff[:20], demangle(diff[:20])):
        print(f"  DIFF {d[:140]}  ({len(old[f])} -> {len(new[f])} instructions)")
    print("ISA DIFF " + ("PASS (no common function changed)" if not diff else f"FAIL ({len(diff)} changed)"))
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
