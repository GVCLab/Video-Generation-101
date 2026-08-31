#!/usr/bin/env python3
"""Normalise asset filenames and repair relative links in Markdown.

Two problems this fixes, both of which render as broken images on GitHub:

1. Image generators write their prompt into the filename, producing paths with
   CJK characters, spaces and timestamps. Those break on case-insensitive and
   non-UTF-8 filesystems and are unusable as stable URLs.
2. Chapters under docs/ referenced `assets/...` when the assets live at the
   repository root, so the path only resolves from the root, not from the file
   doing the referencing.

The script is idempotent: run it after adding new diagrams.

    python3 scripts/fix_asset_links.py --dry-run
    python3 scripts/fix_asset_links.py
"""
from __future__ import annotations

import argparse
import os
import re
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
LINK = re.compile(r"(!?\[[^\]\[]*\]\()([^)\s]+)(\))")
MD_ROOTS = ["docs", "resources", "sources"]
IMAGE_EXT = {".png", ".webp", ".jpg", ".jpeg", ".gif", ".svg"}

# assets/zh/ never existed; the file lives under assets/diagrams/
MISPLACED = {"assets/zh/video-reasoning-roadmap.png":
             "assets/diagrams/video-reasoning-roadmap.png"}


def markdown_files() -> list[Path]:
    out = [ROOT / "README.md"]
    for d in MD_ROOTS:
        out += sorted((ROOT / d).rglob("*.md"))
    return [p for p in out if p.is_file()]


def normalise_names(dry: bool) -> dict[str, str]:
    """Rename generator-named files to stable slugs. Returns old->new (repo-relative)."""
    renames: dict[str, str] = {}

    # assets/imagegen-diagrams/NNN/ holds exactly one diagram
    for d in sorted(ASSETS.glob("imagegen-diagrams/[0-9][0-9][0-9]")):
        files = [f for f in d.iterdir() if f.is_file()]
        if len(files) == 1 and files[0].stem != "diagram":
            new = d / ("diagram" + files[0].suffix)
            renames[str(files[0].relative_to(ROOT))] = str(new.relative_to(ROOT))
            if not dry:
                os.rename(files[0], new)

    # assets/imagegen-diagrams/batches/<range>/ holds the multi-up source sheets
    for d in sorted(ASSETS.glob("imagegen-diagrams/batches/*")):
        if not d.is_dir():
            continue
        for i, f in enumerate(sorted(p for p in d.iterdir() if p.is_file()), 1):
            new = d / f"sheet-{i:02d}{f.suffix}"
            if f != new:
                renames[str(f.relative_to(ROOT))] = str(new.relative_to(ROOT))
                if not dry:
                    os.rename(f, new)
    return renames


def repair_links(renames: dict[str, str], dry: bool) -> tuple[int, list[str]]:
    fixed = 0
    unresolved: list[str] = []
    for md in markdown_files():
        text = original = md.read_text(encoding="utf-8")

        def repl(m: re.Match) -> str:
            nonlocal fixed
            pre, target, post = m.groups()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                return m.group(0)
            raw = urllib.parse.unquote(target.split("#")[0])
            if not raw or (md.parent / raw).exists():
                return m.group(0)
            if Path(raw).suffix.lower() not in IMAGE_EXT:
                return m.group(0)
            # re-anchor against the repository root, then apply known fixups
            norm = raw.lstrip("./")
            while norm.startswith("../"):
                norm = norm[3:]
            norm = MISPLACED.get(norm, norm)
            norm = renames.get(norm, norm)
            if not (ROOT / norm).exists():
                unresolved.append(f"{md.relative_to(ROOT)}: {target}")
                return m.group(0)
            fixed += 1
            return f"{pre}{os.path.relpath(ROOT / norm, md.parent)}{post}"

        text = LINK.sub(repl, text)
        if text != original and not dry:
            md.write_text(text, encoding="utf-8")
    return fixed, unresolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    renames = normalise_names(args.dry_run)
    print(f"{'would rename' if args.dry_run else 'renamed'} {len(renames)} asset file(s)")
    fixed, unresolved = repair_links(renames, args.dry_run)
    print(f"{'would repair' if args.dry_run else 'repaired'} {fixed} Markdown link(s)")
    for u in unresolved:
        print(f"  UNRESOLVED  {u}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
