#!/usr/bin/env python3
"""Extract Mermaid figures, create Chinese ImageGen prompts, and replace blocks with images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MERMAID_BLOCK = re.compile(
    r"(?ms)^(?P<fence>```|~~~)mermaid\s*\n(?P<body>.*?)^(?P=fence)\s*$"
)
TITLE_LINE = re.compile(r"^\s*accTitle:\s*(.+?)\s*$", re.MULTILINE)
NODE_LABEL = re.compile(r'\[["\']?([^\]"\']+)["\']?\]')


def infer_title(body: str) -> str:
    """Return the accessible title or a short Chinese fallback title."""
    title = TITLE_LINE.search(body)
    if title:
        return title.group(1).strip()
    labels = [re.sub(r"<br\s*/?>", "", label, flags=re.I).strip() for label in NODE_LABEL.findall(body)]
    if len(labels) >= 2:
        return f"{labels[0]}到{labels[-1]}的流程"
    if labels:
        return f"{labels[0]}的结构示意"
    return "视频生成知识结构示意"


def image_prompt(title: str, body: str) -> str:
    """Make an ImageGen prompt focused on visual hierarchy rather than Mermaid syntax."""
    labels = [re.sub(r"<br\s*/?>", " ", label, flags=re.I).strip() for label in NODE_LABEL.findall(body)]
    concise_labels = "、".join(labels[:10]) if labels else "核心模块与关系"
    return (
        "创建一张用于中文学术教程的高质量横向信息图。主题："
        f"{title}。图中表达的关键模块包括：{concise_labels}。"
        "使用清晰的流程、分层或关系结构与箭头表达因果、数据流或能力边界；"
        "白色或浅灰背景，深蓝与青蓝为主色，少量紫色或橙色突出重点，简洁现代、可读性优先。"
        "所有可见文字必须为简体中文，文字尽量少且字号大；不要出现英文、代码、Mermaid 风格、"
        "水印、Logo、装饰性伪文本。画面要适合嵌入 Markdown 文档，16:9 横向构图。"
    )


def extract_manifest(root: Path, assets_root: Path) -> list[dict]:
    """Extract all Mermaid blocks under root in stable order."""
    manifest: list[dict] = []
    markdown_files = sorted(root.rglob("*.md"), key=lambda path: path.relative_to(root).as_posix())
    for markdown_path in markdown_files:
        relative_path = markdown_path.relative_to(root).as_posix()
        source = markdown_path.read_text(encoding="utf-8")
        for file_index, match in enumerate(MERMAID_BLOCK.finditer(source), start=1):
            figure_id = f"{len(manifest) + 1:03d}"
            body = match.group("body")
            title = infer_title(body)
            manifest.append(
                {
                    "id": figure_id,
                    "path": relative_path,
                    "file_index": file_index,
                    "title": title,
                    "prompt": image_prompt(title, body),
                    "image": f"{assets_root.relative_to(root).as_posix()}/{figure_id}",
                    "source": match.group(0),
                }
            )
    return manifest


def replace_blocks(root: Path, manifest: list[dict]) -> None:
    """Replace Mermaid blocks with Markdown image links based on manifest entries."""
    by_path: dict[str, list[dict]] = {}
    for entry in manifest:
        by_path.setdefault(entry["path"], []).append(entry)
    for relative_path, entries in by_path.items():
        markdown_path = root / relative_path
        source = markdown_path.read_text(encoding="utf-8")
        matches = list(MERMAID_BLOCK.finditer(source))
        if len(matches) != len(entries):
            raise ValueError(f"Mermaid block count changed in {relative_path}")
        rendered: list[str] = []
        cursor = 0
        for entry, match in zip(entries, matches):
            image_target = root / entry["image"]
            if image_target.is_file():
                image_path = image_target.relative_to(root).as_posix()
            else:
                image_files = sorted(
                    [path for path in image_target.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
                )
                if not image_files:
                    raise FileNotFoundError(f"Missing generated image for figure {entry['id']}: {image_target}")
                image_path = image_files[0].relative_to(root).as_posix()
            rendered.append(source[cursor : match.start()])
            rendered.append(f"![图 {entry['id']}：{entry['title']}]({image_path})")
            cursor = match.end()
        rendered.append(source[cursor:])
        markdown_path.write_text("".join(rendered), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace Mermaid blocks with ImageGen figures.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    if arguments.replace:
        manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        replace_blocks(arguments.root, manifest)
        return
    manifest = extract_manifest(arguments.root, arguments.assets)
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(manifest)} Mermaid figures to {arguments.manifest}")


if __name__ == "__main__":
    main()
