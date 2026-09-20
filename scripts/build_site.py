#!/usr/bin/env python3
"""Stage the repository into a MkDocs docs_dir, then build the site.

MkDocs refuses to use the directory holding mkdocs.yml as docs_dir, but this
repo is read on GitHub as much as on the site, so its links are all written
relative to the repo root (docs/x.md, ../../assets/y.png). Copying the tree
into .mkdocs-build/docs preserves that shape exactly, so one set of links
works in both places and nothing has to be rewritten.

Files are hard-linked where the filesystem allows it, so staging ~240MB of
diagrams costs no extra disk and no copy time.

    python3 scripts/build_site.py            # build into _site/
    python3 scripts/build_site.py --serve    # live preview on :8000
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".mkdocs-build" / "docs"

# Everything the site needs, in repo-root-relative form.
INCLUDE = ["README.md", "CONTRIBUTING.md", "CITATION.cff", "LICENSE",
           "docs", "assets", "resources", "bibliography", "sources"]
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "_site",
             ".mkdocs-build"}


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    linked = copied = 0
    for name in INCLUDE:
        src = ROOT / name
        if not src.exists():
            print(f"  skip (missing): {name}", file=sys.stderr)
            continue
        if src.is_file():
            _place(src, STAGE / name)
            linked += 1
            continue
        for path in src.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts) or path.is_relative_to(ROOT / "docs" / "plans"):
                continue
            if path.is_dir():
                continue
            dst = STAGE / path.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if _place(path, dst):
                linked += 1
            else:
                copied += 1
    print(f"staged {linked} hard-linked + {copied} copied file(s) -> {STAGE}",
          file=sys.stderr)


def _place(src: Path, dst: Path) -> bool:
    # Historical research notes remain linkable, but do not dominate manual search.
    # Write a copy: changing a staged hard link would also modify the source file.
    if src.suffix == ".md" and src.is_relative_to(ROOT / "sources"):
        content = src.read_text(encoding="utf-8")
        if content.startswith("---\n"):
            import yaml
            _, front, body = content.split("---", 2)
            metadata = yaml.safe_load(front) or {}
            metadata["search"] = {"exclude": True}
            content = "---\n" + yaml.safe_dump(metadata, allow_unicode=True) + "---" + body
        else:
            content = "---\nsearch:\n  exclude: true\n---\n\n" + content
        dst.write_text(content, encoding="utf-8")
        return False
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    stage()
    cmd = ["mkdocs", "serve" if args.serve else "build",
           "-f", str(ROOT / "mkdocs.yml")]
    if args.strict:
        cmd.append("--strict")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
