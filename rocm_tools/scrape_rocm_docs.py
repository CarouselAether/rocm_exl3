#!/usr/bin/env python3
"""Scrape AMD ROCm/HIP documentation into rocm_docs/<name>/ as markdown.

Markdown rather than PDF/HTML so the whole corpus greps in one pass -- which is
how it actually gets used ("does HIP have __hmin2 for half2?").

Both sites are pinned to docs-7.2.4 to match the installed toolchain
(ROCm 7.2.4 / HIP 7.2.53211). Bump the URLs together with the toolchain.

    rocm_tools/scrape_rocm_docs.py hip       # HIP language/runtime reference
    rocm_tools/scrape_rocm_docs.py guide     # ROCm programming guide
    rocm_tools/scrape_rocm_docs.py all
"""

import argparse
import os
import re
import sys
import urllib.parse
import urllib.request

import html2text

SITES = {
    "hip": {
        "base": "https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.4",
        "pages": [
            "reference/cpp_language_extensions",
            "reference/hip_runtime_api_reference",
            "reference/deprecated_api_list",
            "reference/math_api",
            "how-to/hip_porting_guide",
            "how-to/performance_guidelines",
            "how-to/debugging",
            "understand/programming_model",
            "understand/hardware_implementation",
            "understand/amd_clr",
            "index",
        ],
    },
    "guide": {
        "base": "https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/docs-7.2.4",
        # Discovered from the index; install/ and genindex are not useful here.
        "discover": True,
        "skip": re.compile(r"^(install/|genindex|search|py-modindex)"),
    },
}


def fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def discover(base: str, skip: re.Pattern) -> list[str]:
    """Pull the page list off the site index."""
    html = fetch(base + "/") or ""
    hrefs = re.findall(r'href="([^"#?]+\.html)"', html)
    pages = []
    for h in hrefs:
        if h.startswith("http"):
            continue
        p = h[:-5]                       # strip .html
        if skip.match(p) or p in pages:
            continue
        pages.append(p)
    return pages


def extract_article(html: str) -> str:
    """Return the <article>/<main> body -- drops nav chrome and sidebars."""
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", html, re.S | re.I)
        if m:
            return m.group(1)
    return html


def scrape(name: str, spec: dict, out_root: str) -> tuple[int, int]:
    base = spec["base"]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    pages = spec.get("pages") or discover(base, spec.get("skip", re.compile(r"^$")))
    print(f"\n=== {name}: {len(pages)} pages from {base}")

    conv = html2text.HTML2Text()
    conv.ignore_images = True
    conv.ignore_links = True
    conv.body_width = 0          # no wrapping -- keeps code lines intact
    conv.unicode_snob = True

    ok = fail = 0
    for page in pages:
        html = fetch(f"{base}/{page}.html")
        if html is None:
            print(f"  {page:52s} FAIL")
            fail += 1
            continue
        text = conv.handle(extract_article(html)).strip()
        if len(text) < 400:
            print(f"  {page:52s} THIN {len(text)}b")
            fail += 1
            continue
        fname = page.replace("/", "_") + ".md"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(f"<!-- scraped from {base}/{page}.html -->\n\n{text}\n")
        print(f"  {page:52s} ok   {len(text):>7,}b")
        ok += 1
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?", default="all", choices=[*SITES, "all"])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "rocm_docs"))
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    targets = list(SITES) if args.site == "all" else [args.site]

    tot_ok = tot_fail = 0
    for name in targets:
        ok, fail = scrape(name, SITES[name], out_root)
        tot_ok += ok
        tot_fail += fail

    print(f"\n{tot_ok} pages written under {out_root}, {tot_fail} failed/thin")
    return 0 if tot_ok else 1


if __name__ == "__main__":
    sys.exit(main())
