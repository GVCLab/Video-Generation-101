#!/usr/bin/env python3
"""Verify in-chapter reference entries against authoritative sources.

The central bibliography (bibliography/registry.json) is already machine-built
from arXiv/Crossref, and CI runs lychee for dead links. Neither checks the
~2000 hand-written per-chapter reference entries, where the anchor text
(title), venue and year can be wrong even when the URL resolves fine.

This script checks exactly that:
  * arXiv links  -> title claimed in anchor text vs real arXiv title
                 -> year claimed in the line vs arXiv v1 announcement year
  * CVF / ICLR / NeurIPS / ECVA links -> venue+year encoded in the URL path
                                         vs venue+year claimed in the line

Usage:
    python3 scripts/verify_citations.py                # full run
    python3 scripts/verify_citations.py --offline      # URL-path checks only
    python3 scripts/verify_citations.py --json out.json
"""
from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["docs", "resources"]
SCAN_FILES = ["README.md"]

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_BATCH = 50
ARXIV_SLEEP = 3.0          # arXiv asks for >=3s between programmatic requests
TITLE_FLAG = 0.72          # similarity below this is reported

MD_LINK = re.compile(r"\[([^\]\[]+)\]\((https?://[^\s)]+)\)")
ARXIV_ABS = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")
YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")

# venue + year encoded in the URL path
URL_VENUE = [
    (re.compile(r"openaccess\.thecvf\.com/content/([A-Za-z]+)(\d{4})"), None),
    (re.compile(r"openaccess\.thecvf\.com/content_(\w+?)_(\d{4})"), None),
    (re.compile(r"proceedings\.iclr\.cc/paper_files/paper/(\d{4})"), "ICLR"),
    (re.compile(r"proceedings\.neurips\.cc/paper_files/paper/(\d{4})"), "NeurIPS"),
    (re.compile(r"papers\.nips\.cc/paper_files/paper/(\d{4})"), "NeurIPS"),
    (re.compile(r"ecva\.net/papers/eccv_(\d{4})"), "ECCV"),
]
VENUE_ALIASES = {
    "NEURIPS": "NEURIPS", "NIPS": "NEURIPS", "NEURAL": "NEURIPS",
    "CVPR": "CVPR", "ICCV": "ICCV", "WACV": "WACV", "ECCV": "ECCV",
    "ICLR": "ICLR", "ICML": "ICML", "AAAI": "AAAI", "SIGGRAPH": "SIGGRAPH",
    "CVPRW": "CVPR", "ICCVW": "ICCV",
}
VENUE_TOKEN = re.compile(
    r"\b(CVPR|ICCV|ECCV|WACV|ICLR|ICML|NeurIPS|NIPS|AAAI|SIGGRAPH|TPAMI|IJCV|TMLR|TVCG)\b",
    re.I,
)


@dataclass
class Finding:
    severity: str
    kind: str
    file: str
    line: int
    url: str
    claimed: str
    actual: str
    note: str = ""


@dataclass
class Ref:
    file: str
    line: int
    text: str          # anchor text
    url: str
    context: str       # the whole source line
    arxiv_id: str | None = None


def norm_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[‐-―]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def title_score(claimed: str, actual: str) -> float:
    a, b = norm_title(claimed), norm_title(actual)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # anchor text often prefixes the real title with a short model name
    # ("PAB: Real-Time ..." vs "Real-Time ..."), or truncates the subtitle.
    if a in b or b in a:
        return 0.97
    at, bt = set(a.split()), set(b.split())
    jacc = len(at & bt) / max(1, len(at | bt))
    return max(difflib.SequenceMatcher(None, a, b).ratio(), jacc)


def iter_markdown(include_sources: bool = False) -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS + (["sources"] if include_sources else []):
        p = ROOT / d
        if p.is_dir():
            out += sorted(f for f in p.rglob("*.md") if "plans" not in f.parts)
    for f in SCAN_FILES:
        p = ROOT / f
        if p.is_file():
            out.append(p)
    return out


class LinkParser(HTMLParser):
    """Collect HTML links too (timeline cards use HTML, not Markdown)."""
    def __init__(self):
        super().__init__()
        self.links = []
        self.active = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            url = dict(attrs).get("href", "")
            self.active = [self.getpos()[0], url, []]

    def handle_data(self, data):
        if self.active is not None:
            self.active[2].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.active is not None:
            line, url, text = self.active
            self.links.append((line, "".join(text).strip(), url))
            self.active = None


def collect_refs(include_sources: bool = False) -> list[Ref]:
    refs: list[Ref] = []
    for path in iter_markdown(include_sources):
        rel = str(path.relative_to(ROOT))
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        entries = [(n, text, url) for n, line in enumerate(lines, 1)
                   for text, url in MD_LINK.findall(line)]
        parser = LinkParser()
        parser.feed(content)
        entries.extend(parser.links)
        for lineno, text, url in entries:
            if not url.startswith(("https://", "http://")) or text.startswith("!") or "img.shields.io" in url:
                continue
            m = ARXIV_ABS.search(url)
            refs.append(Ref(rel, lineno, html.unescape(text.strip()), url,
                            lines[lineno - 1].strip(), m.group(1) if m else None))
    return refs


# Common method aliases used in timeline cards, tied to a verified paper identity.
TITLE_ALIASES = {
    "2506.09985": {"v jepa 2 v jepa 2 ac"},
    "2602.15031": {"editctrl paper"},
    "1605.07157": {"dna cdna stp"},
    "2212.09748": {"diffusion transformer"},
    "2304.08818": {"latent video diffusion"},
    "2501.03575": {"nvidia cosmos"},
}

def looks_like_title(text: str) -> bool:
    label = text.strip().lower()
    if len(label) <= 12 or label.startswith(("http", "github", "官方", "见", "arxiv", "paper", "abstract", "project", "论文", "预印本", "preprint", "first public preprint", "probing preprint", "workshop paper", "model card")):
        return False
    # Non-title navigation labels should not trigger metadata mismatch errors.
    if label in {"technical report", "official repository", "technical paper"}:
        return False
    return True


class ArxivUnavailable(RuntimeError):
    """The arXiv API could not be reached; absence of a record proves nothing."""


def arxiv_lookup(ids: list[str], verbose: bool = True) -> tuple[dict[str, dict], set[str]]:
    """Return (metadata_by_id, ids_actually_queried).

    Only ids in the second element carry a meaningful negative result. Ids from
    a failed batch are simply unknown and must not be reported as missing.
    """
    out: dict[str, dict] = {}
    queried: set[str] = set()
    failed_batches = 0
    total_batches = (len(ids) + ARXIV_BATCH - 1) // ARXIV_BATCH
    for i in range(0, len(ids), ARXIV_BATCH):
        chunk = ids[i:i + ARXIV_BATCH]
        params = urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}
        )
        if verbose:
            print(f"  arXiv batch {i // ARXIV_BATCH + 1}"
                  f"/{(len(ids) + ARXIV_BATCH - 1) // ARXIV_BATCH}"
                  f" ({len(chunk)} ids)", file=sys.stderr)
        try:
            with urllib.request.urlopen(f"{ARXIV_API}?{params}", timeout=90) as r:
                root = ET.fromstring(r.read())
        except Exception as exc:                      # network / API failure
            failed_batches += 1
            print(f"  ! batch failed: {exc}", file=sys.stderr)
            time.sleep(ARXIV_SLEEP)
            continue
        queried.update(chunk)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            eid = entry.findtext("a:id", "", ns)
            m = re.search(r"abs/(\d{4}\.\d{4,5})v(\d+)", eid)
            if not m:
                continue
            out[m.group(1)] = {
                "title": " ".join((entry.findtext("a:title", "", ns) or "").split()),
                "published": entry.findtext("a:published", "", ns),
                "authors": [a.findtext("a:name", "", ns)
                            for a in entry.findall("a:author", ns)],
            }
        time.sleep(ARXIV_SLEEP)
    if total_batches and failed_batches == total_batches:
        raise ArxivUnavailable(
            "every arXiv batch failed -- the API is unreachable from this host. "
            "Refusing to report missing records, which would all be false."
        )
    if failed_batches:
        print(f"  ! {failed_batches}/{total_batches} batches failed; "
              f"{len(ids) - len(queried)} ids left unverified", file=sys.stderr)
    return out, queried


def check_arxiv(refs: list[Ref], meta: dict[str, dict],
                queried: set[str]) -> list[Finding]:
    per_line: dict[tuple[str, int], int] = {}
    for r in refs:
        per_line[(r.file, r.line)] = per_line.get((r.file, r.line), 0) + 1
    findings: list[Finding] = []
    for r in refs:
        if not r.arxiv_id:
            continue
        info = meta.get(r.arxiv_id)
        if info is None:
            if r.arxiv_id not in queried:
                continue        # never successfully queried -- unknown, not absent
            findings.append(Finding(
                "ERROR", "arxiv-not-found", r.file, r.line, r.url,
                r.text, "(arXiv API returned nothing)",
                "id may be wrong, withdrawn, or not yet announced"))
            continue
        # title check -- only when the anchor text looks like a title
        if looks_like_title(r.text) and norm_title(r.text) not in TITLE_ALIASES.get(r.arxiv_id, set()):
            score = title_score(r.text, info["title"])
            if score < TITLE_FLAG:
                findings.append(Finding(
                    "ERROR", "title-mismatch", r.file, r.line, r.url,
                    r.text, info["title"], f"similarity={score:.2f}"))
        # year check -- compare arXiv v1 year against years stated on the line
        pub_year = info["published"][:4]
        stated = set(YEAR.findall(r.context)) if per_line[(r.file, r.line)] == 1 else set()
        if pub_year and stated and pub_year not in stated:
            # allow a claimed venue year later than the preprint year
            if all(int(y) < int(pub_year) for y in stated):
                findings.append(Finding(
                    "WARN", "year-before-preprint", r.file, r.line, r.url,
                    "/".join(sorted(stated)), f"arXiv v1 {pub_year}",
                    "line claims a year earlier than the first arXiv version"))
    return findings


def check_url_venue(refs: list[Ref]) -> list[Finding]:
    """Venue/year encoded in the URL vs venue/year claimed on the same line.

    Only applied to lines carrying exactly one reference link. Table rows list
    several papers from different venues side by side, so a line-level claim
    cannot be attributed to a specific link there.
    """
    per_line: dict[tuple[str, int], int] = {}
    for r in refs:
        per_line[(r.file, r.line)] = per_line.get((r.file, r.line), 0) + 1
    findings: list[Finding] = []
    for r in refs:
        if per_line[(r.file, r.line)] != 1:
            continue
        got_venue = got_year = None
        for pat, fixed in URL_VENUE:
            m = pat.search(r.url)
            if not m:
                continue
            if fixed:
                got_venue, got_year = fixed, m.group(1)
            else:
                got_venue, got_year = m.group(1), m.group(2)
            break
        if not got_venue:
            continue
        gv = VENUE_ALIASES.get(got_venue.upper(), got_venue.upper())
        # what does the surrounding line claim?
        claimed_venues = {VENUE_ALIASES.get(v.upper(), v.upper())
                          for v in VENUE_TOKEN.findall(r.context)}
        claimed_years = set(YEAR.findall(r.context))
        if claimed_venues and gv not in claimed_venues:
            findings.append(Finding(
                "ERROR", "venue-mismatch", r.file, r.line, r.url,
                "/".join(sorted(claimed_venues)), f"{gv} (from URL)"))
        if claimed_years and got_year not in claimed_years:
            findings.append(Finding(
                "ERROR", "venue-year-mismatch", r.file, r.line, r.url,
                "/".join(sorted(claimed_years)), f"{got_year} (from URL)"))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip arXiv API, run URL-path checks only")
    ap.add_argument("--json", type=Path, help="write findings as JSON")
    ap.add_argument("--coverage", type=Path, help="write query coverage as JSON")
    ap.add_argument("--metadata", type=Path, help="save retrieved arXiv metadata")
    ap.add_argument("--include-sources", action="store_true", help="also inspect historical source notes")
    ap.add_argument("--fail-on", default="ERROR", choices=["ERROR", "WARN", "NONE"])
    args = ap.parse_args()

    refs = collect_refs(args.include_sources)
    arxiv_ids = sorted({r.arxiv_id for r in refs if r.arxiv_id})
    print(f"scanned {len(set(r.file for r in refs))} files, "
          f"{len(refs)} links, {len(arxiv_ids)} unique arXiv ids",
          file=sys.stderr)

    findings = check_url_venue(refs)
    unverified = len(arxiv_ids)
    meta, queried = {}, set()
    if not args.offline:
        try:
            meta, queried = arxiv_lookup(arxiv_ids)
        except ArxivUnavailable as exc:
            print(f"\nFATAL: {exc}", file=sys.stderr)
            print("Re-run with --offline for URL-path checks only.", file=sys.stderr)
            # Continue to write findings and an explicit incomplete coverage record.
        findings += check_arxiv(refs, meta, queried)
        unverified = len(arxiv_ids) - len(queried)

    order = {"ERROR": 0, "WARN": 1}
    findings.sort(key=lambda f: (order.get(f.severity, 2), f.file, f.line))

    for f in findings:
        print(f"[{f.severity}] {f.kind}  {f.file}:{f.line}")
        print(f"    claimed: {f.claimed}")
        print(f"    actual : {f.actual}")
        if f.note:
            print(f"    note   : {f.note}")
        print(f"    url    : {f.url}")

    errs = sum(1 for f in findings if f.severity == "ERROR")
    warns = sum(1 for f in findings if f.severity == "WARN")
    print(f"\n{errs} error(s), {warns} warning(s)"
          + (f", {unverified} arXiv id(s) unverified" if unverified else ""),
          file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps([asdict(f) for f in findings],
                                        ensure_ascii=False, indent=2),
                             encoding="utf-8")

    coverage = {
        "mode": "offline" if args.offline else "online",
        "scope": "published-and-sources" if args.include_sources else "published",
        "files": len({r.file for r in refs}), "links": len(refs),
        "arxiv_requested": len(arxiv_ids), "arxiv_queried": len(queried),
        "arxiv_returned": len(meta), "arxiv_unverified": unverified,
        "online_complete": not args.offline and unverified == 0,
        "errors": errs, "warnings": warns,
    }
    if args.coverage:
        args.coverage.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.metadata:
        args.metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.offline and unverified:
        return 2
    if args.fail_on == "ERROR":
        return 1 if errs else 0
    if args.fail_on == "WARN":
        return 1 if (errs or warns) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
