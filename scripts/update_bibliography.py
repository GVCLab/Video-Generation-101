#!/usr/bin/env python3
"""Refresh citation metadata, GitHub stars, BibTeX, and the bibliography index.

The script uses only the Python standard library. Citation metadata comes from
Crossref and arXiv; GitHub stars are read from each public repository page so a
GitHub token is not required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "bibliography" / "registry.json"
METADATA_PATH = ROOT / "bibliography" / "metadata.json"
STARS_PATH = ROOT / "bibliography" / "github-stars.json"
BIBTEX_PATH = ROOT / "bibliography" / "references.bib"
INDEX_PATH = ROOT / "docs" / "bibliography.md"

USER_AGENT = "Video-Generation-101 bibliography updater (+https://github.com/GVCLab/Video-Generation-101)"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
STATUS_LABELS = {
    "official-code": "官方代码",
    "official-research-artifact": "官方研究产物",
    "official-related-code": "官方相关代码（非本论文实现）",
    "community-implementation": "社区实现",
    "official-project-page": "官方项目页（非模型代码）",
}


def request_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_registry(registry: dict[str, Any]) -> None:
    entries = registry.get("entries", [])
    if not entries:
        raise ValueError("registry contains no entries")

    keys: set[str] = set()
    for entry in entries:
        key = entry.get("citekey")
        if not key or key in keys:
            raise ValueError(f"missing or duplicate citekey: {key!r}")
        keys.add(key)

        source = entry.get("source", {})
        if source.get("kind") not in {"arxiv", "doi", "manual"}:
            raise ValueError(f"{key}: unsupported source kind")
        if source["kind"] == "manual" and "metadata" not in source:
            raise ValueError(f"{key}: manual source requires metadata")
        if source["kind"] != "manual" and not source.get("id"):
            raise ValueError(f"{key}: source identifier is missing")

        repository = entry.get("github")
        if repository and repository.get("status") not in STATUS_LABELS:
            raise ValueError(f"{key}: unsupported GitHub status")


def normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def crossref_metadata(doi: str) -> dict[str, Any]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    message = json.loads(request_bytes(url))["message"]
    type_map = {
        "journal-article": "article",
        "proceedings-article": "inproceedings",
        "book-chapter": "incollection",
        "book": "book",
    }
    authors = []
    for person in message.get("author", []):
        literal = person.get("name")
        if literal:
            authors.append(literal)
            continue
        given = person.get("given", "")
        family = person.get("family", "")
        name = normalize_space(f"{given} {family}")
        if name:
            authors.append(name)

    dates = message.get("published-print") or message.get("published-online") or message.get("issued", {})
    date_parts = dates.get("date-parts", [[""]])
    year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
    containers = message.get("container-title") or []
    entry_type = type_map.get(message.get("type"), "misc")
    metadata: dict[str, Any] = {
        "entry_type": entry_type,
        "title": normalize_space((message.get("title") or [""])[0]),
        "authors": " and ".join(authors),
        "year": year,
        "doi": doi,
        "url": f"https://doi.org/{doi}",
    }
    container_field = "booktitle" if entry_type == "inproceedings" else "journal"
    optional = {
        container_field: containers[0] if containers else "",
        "volume": message.get("volume", ""),
        "number": message.get("issue", ""),
        "pages": re.sub(r"(?<=\d)-(?=\d)", "--", message.get("page", "")),
        "publisher": message.get("publisher", ""),
    }
    metadata.update({key: normalize_space(str(value)) for key, value in optional.items() if value})
    return metadata


def arxiv_metadata_batch(ids: list[str]) -> dict[str, dict[str, Any]]:
    params = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    root = ET.fromstring(request_bytes(f"https://export.arxiv.org/api/query?{params}", timeout=60))
    results: dict[str, dict[str, Any]] = {}
    for item in root.findall("atom:entry", ATOM_NS):
        identifier = item.findtext("atom:id", "", ATOM_NS)
        match = re.search(r"/(\d{4}\.\d{4,5})(?:v\d+)?$", identifier)
        if not match:
            continue
        arxiv_id = match.group(1)
        authors = [
            normalize_space(author.findtext("atom:name", "", ATOM_NS))
            for author in item.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author and author != ":"]
        if authors and authors[0] == "NVIDIA":
            authors[0] = "{NVIDIA}"
        published = item.findtext("atom:published", "", ATOM_NS)
        primary = item.find("arxiv:primary_category", ATOM_NS)
        doi = normalize_space(item.findtext("arxiv:doi", "", ATOM_NS))
        journal_ref = normalize_space(item.findtext("arxiv:journal_ref", "", ATOM_NS))
        metadata: dict[str, Any] = {
            "entry_type": "article" if doi else "misc",
            "title": normalize_space(item.findtext("atom:title", "", ATOM_NS)),
            "authors": " and ".join(authors),
            "year": published[:4],
            "arxiv_id": arxiv_id,
            "primary_class": primary.attrib.get("term", "") if primary is not None else "",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }
        if doi:
            metadata["doi"] = doi
        if journal_ref:
            metadata["journal"] = journal_ref
        results[arxiv_id] = metadata
    return results


def refresh_metadata(registry: dict[str, Any]) -> dict[str, Any]:
    arxiv_ids = [
        entry["source"]["id"]
        for entry in registry["entries"]
        if entry["source"]["kind"] == "arxiv"
    ]
    arxiv_results = arxiv_metadata_batch(arxiv_ids) if arxiv_ids else {}
    output: dict[str, Any] = {}
    for entry in registry["entries"]:
        source = entry["source"]
        if source["kind"] == "manual":
            metadata = dict(source["metadata"])
        elif source["kind"] == "doi":
            metadata = crossref_metadata(source["id"])
        else:
            if source["id"] not in arxiv_results:
                raise RuntimeError(f"arXiv returned no metadata for {source['id']}")
            metadata = dict(arxiv_results[source["id"]])
        if source.get("author_override"):
            metadata["authors"] = source["author_override"]
        metadata["source_kind"] = source["kind"]
        metadata["source_id"] = source.get("id", metadata.get("url", ""))
        output[entry["citekey"]] = metadata
    return output


def github_stars(url: str) -> int:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            page = request_bytes(url).decode("utf-8", errors="replace")
            match = re.search(r'aria-label="([0-9,]+) users? starred this repository"', page)
            if not match:
                raise RuntimeError("star count was not present in the public repository page")
            return int(match.group(1).replace(",", ""))
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(str(last_error))


def refresh_stars(registry: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    urls = sorted({entry["github"]["url"] for entry in registry["entries"] if entry.get("github")})
    previous_repos = (previous or {}).get("repositories", {})
    counts: dict[str, int] = {}

    def fetch(url: str) -> tuple[str, int]:
        return url, github_stars(url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                _, count = future.result()
                counts[url] = count
                print(f"{url}: {count:,} stars")
            except Exception as exc:  # keep the last good snapshot on transient failures
                if url in previous_repos:
                    counts[url] = int(previous_repos[url]["stars"])
                    print(f"warning: {url}: {exc}; kept cached value", file=sys.stderr)
                else:
                    raise RuntimeError(f"failed to fetch {url}: {exc}") from exc

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    return {
        "as_of": now.date().isoformat(),
        "fetched_at": now.isoformat().replace("+00:00", "Z"),
        "repositories": {url: {"stars": counts[url]} for url in sorted(counts)},
    }


def bibtex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def bibtex_entry(entry: dict[str, Any], metadata: dict[str, Any], stars: dict[str, Any]) -> str:
    entry_type = metadata.get("entry_type", "misc")
    fields: list[tuple[str, str, bool]] = [
        ("title", metadata["title"], True),
        ("author", metadata["authors"], False),
        ("year", str(metadata["year"]), False),
    ]
    for key in ("journal", "booktitle", "volume", "number", "pages", "publisher", "howpublished"):
        if metadata.get(key):
            fields.append((key, str(metadata[key]), False))
    if metadata.get("doi"):
        fields.append(("doi", metadata["doi"], False))
    if metadata.get("arxiv_id"):
        fields.extend(
            [
                ("eprint", metadata["arxiv_id"], False),
                ("archiveprefix", "arXiv", False),
                ("primaryclass", metadata.get("primary_class", ""), False),
            ]
        )
    if metadata.get("url"):
        fields.append(("url", metadata["url"], False))

    repository = entry.get("github")
    if repository:
        url = repository["url"]
        snapshot = stars["repositories"][url]["stars"]
        fields.extend(
            [
                ("code_status", repository["status"], False),
                ("github", url, False),
                ("github_stars", str(snapshot), False),
                ("github_stars_as_of", stars["as_of"], False),
                ("github_note", repository.get("note", ""), False),
            ]
        )
    else:
        fields.extend(
            [
                ("code_status", "no-official-repository", False),
                ("github", "not-available", False),
                ("github_stars", "not-applicable", False),
            ]
        )

    lines = [f"@{entry_type}{{{entry['citekey']},"]
    for index, (name, value, protect_case) in enumerate(fields):
        if value == "":
            continue
        escaped = bibtex_escape(value)
        rendered = "{{" + escaped + "}}" if protect_case else "{" + escaped + "}"
        suffix = "," if index < len(fields) - 1 else ""
        lines.append(f"  {name} = {rendered}{suffix}")
    lines.append("}")
    return "\n".join(lines)


def render_bibtex(registry: dict[str, Any], metadata: dict[str, Any], stars: dict[str, Any]) -> str:
    header = (
        "% Generated by scripts/update_bibliography.py.\n"
        "% Custom fields: code_status, github, github_stars, github_stars_as_of, github_note.\n"
        "% Star counts are snapshots; run the updater to refresh them.\n\n"
    )
    entries = [bibtex_entry(entry, metadata[entry["citekey"]], stars) for entry in registry["entries"]]
    return header + "\n\n".join(entries) + "\n"


def markdown_escape(value: str) -> str:
    return html.escape(value, quote=False).replace("|", "\\|")


def render_index(registry: dict[str, Any], metadata: dict[str, Any], stars: dict[str, Any]) -> str:
    lines = [
        "<!-- Generated by scripts/update_bibliography.py. Edit bibliography/registry.json instead. -->",
        "",
        "# 论文引用、代码仓库与 GitHub Stars",
        "",
        f"本索引收录 **{len(registry['entries'])} 篇**在 `bibliography/registry.json` 中登记的核心论文与技术报告，用于提供标准 BibTeX 和官方代码状态。仓库正文引用的文献总数多于此，完整清单以各章文末的参考文献为准。论文元数据来自 Crossref、arXiv 或机构原始页面；GitHub Stars 是 **{stars['as_of']}** 的快照，不代表代码质量。",
        "",
        "- [完整 BibTeX](../bibliography/references.bib)",
        "- [引用与仓库登记表](../bibliography/registry.json)",
        "- [Star 原始快照](../bibliography/github-stars.json)",
        "- 刷新命令：`python scripts/update_bibliography.py --all`",
        "",
        "这里的“代码仓库”表示可公开访问的 GitHub 实现；是否属于 OSI 定义的开源软件、权重是否开放，以及可否商用，仍需逐项查看仓库许可证。",
        "",
        "仓库状态分为：**官方代码**（作者或机构维护）、**官方研究产物**（作者实验代码，但不是标准复现包）、**官方相关代码**（同团队的相关项目，不是该论文实现）、**社区实现**（第三方复现）与**官方项目页**（只有网页源码）。标记“未发现”的条目表示截至快照日未找到论文作者公开的 GitHub 仓库，并不等价于绝对不存在代码。",
        "",
        "| 章节 | Cite key | 论文 / 报告 | 年份 | GitHub 与可用性 | Stars |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]

    for entry in registry["entries"]:
        item = metadata[entry["citekey"]]
        title = markdown_escape(item["title"])
        paper = f"[{title}]({item['url']})" if item.get("url") else title
        repository = entry.get("github")
        if repository:
            url = repository["url"]
            slug = url.removeprefix("https://github.com/")
            label = STATUS_LABELS[repository["status"]]
            note = markdown_escape(repository.get("note", ""))
            availability = f"[{slug}]({url}) · **{label}**"
            if note:
                availability += f"；{note}"
            star_value = f"{stars['repositories'][url]['stars']:,}"
        else:
            availability = "**未发现官方 GitHub 仓库**"
            star_value = "—"
        lines.append(
            f"| {entry['section']} | `{entry['citekey']}` | {paper} | {item['year']} | {availability} | {star_value} |"
        )

    lines.extend(
        [
            "",
            "## 维护约定",
            "",
            "1. 新增阅读条目时，先在 `bibliography/registry.json` 登记元数据来源和仓库状态。",
            "2. 运行 `python scripts/update_bibliography.py --all`，同时刷新元数据、Stars、BibTeX 和本页。",
            "3. 若只想用本地快照重新生成输出，运行 `python scripts/update_bibliography.py --offline`。",
            "4. 官方仓库缺失时不要用社区复现冒充；如收录社区实现，必须使用 `community-implementation`。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="refresh metadata and GitHub stars, then regenerate outputs")
    group.add_argument("--metadata", action="store_true", help="refresh citation metadata and regenerate outputs")
    group.add_argument("--stars", action="store_true", help="refresh GitHub stars and regenerate outputs")
    group.add_argument("--offline", action="store_true", help="regenerate outputs from committed snapshots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_json(REGISTRY_PATH)
    validate_registry(registry)

    refresh_all = args.all or not any((args.metadata, args.stars, args.offline))
    metadata = load_json(METADATA_PATH, {})
    stars = load_json(STARS_PATH, {})

    if refresh_all or args.metadata:
        metadata = refresh_metadata(registry)
        write_json(METADATA_PATH, metadata)
    if refresh_all or args.stars:
        stars = refresh_stars(registry, stars)
        write_json(STARS_PATH, stars)

    expected_keys = {entry["citekey"] for entry in registry["entries"]}
    if set(metadata) != expected_keys:
        raise RuntimeError("metadata snapshot does not match registry; run with --metadata")
    expected_repos = {entry["github"]["url"] for entry in registry["entries"] if entry.get("github")}
    if set(stars.get("repositories", {})) != expected_repos:
        raise RuntimeError("star snapshot does not match registry; run with --stars")

    BIBTEX_PATH.write_text(render_bibtex(registry, metadata, stars), encoding="utf-8")
    INDEX_PATH.write_text(render_index(registry, metadata, stars), encoding="utf-8")
    print(f"wrote {BIBTEX_PATH.relative_to(ROOT)}")
    print(f"wrote {INDEX_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
