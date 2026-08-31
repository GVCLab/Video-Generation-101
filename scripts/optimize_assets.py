#!/usr/bin/env python3
"""Convert diagram PNGs to WebP and repoint every Markdown reference.

The diagrams are 1672x941 AI-rendered images. As PNG they cost ~119 MB, which
every clone pays for. WebP q=95 reproduces them at higher fidelity than a
256-colour PNG at roughly a sixth of the size, and renders natively on GitHub
and in every current browser.

    python3 scripts/optimize_assets.py --dry-run
    python3 scripts/optimize_assets.py
"""
from __future__ import annotations

import argparse
import os
import re
import urllib.parse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
QUALITY = 95
LINK = re.compile(r"(!?\[[^\]\[]*\]\()([^)\s]+)(\))")
MD_ROOTS = ["docs", "resources", "sources"]


def markdown_files() -> list[Path]:
    out = [ROOT / "README.md"]
    for d in MD_ROOTS:
        out += sorted((ROOT / d).rglob("*.md"))
    return [p for p in out if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quality", type=int, default=QUALITY)
    args = ap.parse_args()

    pngs = sorted((ROOT / "assets").rglob("*.png"))
    before = after = 0
    mapping: dict[str, str] = {}

    for png in pngs:
        webp = png.with_suffix(".webp")
        im = Image.open(png)
        im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
        before += png.stat().st_size
        if args.dry_run:
            import io
            buf = io.BytesIO(); im.save(buf, "WEBP", quality=args.quality, method=6)
            after += buf.tell()
        else:
            im.save(webp, "WEBP", quality=args.quality, method=6)
            after += webp.stat().st_size
            png.unlink()
        mapping[str(png.relative_to(ROOT))] = str(webp.relative_to(ROOT))

    print(f"{len(pngs)} PNG -> WebP q{args.quality}: "
          f"{before/1048576:.1f} MB -> {after/1048576:.1f} MB "
          f"({after/max(before,1):.0%}, saved {(before-after)/1048576:.1f} MB)")

    if args.dry_run:
        return 0

    rewritten = 0
    for md in markdown_files():
        text = original = md.read_text(encoding="utf-8")

        def repl(m: re.Match) -> str:
            nonlocal rewritten
            pre, target, post = m.groups()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                return m.group(0)
            raw = urllib.parse.unquote(target.split("#")[0])
            if not raw.lower().endswith(".png"):
                return m.group(0)
            abs_old = (md.parent / raw).resolve()
            try:
                key = str(abs_old.relative_to(ROOT))
            except ValueError:
                return m.group(0)
            if key not in mapping:
                return m.group(0)
            rewritten += 1
            return f"{pre}{os.path.relpath(ROOT / mapping[key], md.parent)}{post}"

        text = LINK.sub(repl, text)
        if text != original:
            md.write_text(text, encoding="utf-8")
    print(f"repointed {rewritten} Markdown reference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
