#!/usr/bin/env python3
"""Check published manual structure, rendered local links, and search boundaries.

Run after build_site.py. Does not claim to validate paper mechanisms or results.
"""
from __future__ import annotations
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag == 'a' and attrs.get('name'):
            self.ids.add(attrs['name'])
        for attr in ('href', 'src'):
            if attrs.get(attr):
                self.links.append(attrs[attr])


def source_issues(path: Path) -> list[str]:
    issues = []
    content = path.read_text(encoding='utf-8')
    fence = None
    h1 = 0
    for n, line in enumerate(content.splitlines(), 1):
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            char = match[1][0]
            fence = None if fence == char else char if fence is None else fence
            continue
        if fence:
            continue
        h1 += bool(re.match(r'^#\s', line))
        if re.search(r'\$`[^`]+`\$', line):
            issues.append(f'{path}:{n}: use portable inline math, not dollar-backtick syntax')
    if h1 != 1:
        issues.append(f'{path}: expected one H1, found {h1}')
    if path.name == 'datasets.md' and 'OpenVid-2M' in content:
        issues.append(f'{path}: removed unsupported dataset name has returned')
    return issues


def check_site(site: Path) -> tuple[list[str], int, int]:
    site = site.resolve()
    pages = {p.resolve(): Page(p.read_text(encoding='utf-8')) for p in site.rglob('*.html')}
    issues = []
    checked_pages = checked_links = 0
    for path, page in pages.items():
        rel = path.relative_to(site)
        if rel.parts[0] == 'sources' or 'plans' in rel.parts or rel.name == '404.html':
            continue
        checked_pages += 1
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path and not url.fragment:
                continue
            # The site is deployed under a project prefix. Relative links are required.
            dest = (site / unquote(url.path).lstrip('/')).resolve() if url.path.startswith('/') else (path.parent / unquote(url.path)).resolve() if url.path else path
            if dest.is_dir():
                dest = dest / 'index.html'
            checked_links += 1
            if not dest.exists():
                issues.append(f'{rel}: missing local target {link}')
            elif url.fragment and dest in pages and unquote(url.fragment) not in pages[dest].ids:
                issues.append(f'{rel}: missing anchor {link}')
    index = site / 'search/search_index.json'
    if not index.exists():
        issues.append('missing search index')
    else:
        for entry in json.loads(index.read_text())['docs']:
            location = entry['location']
            if location.startswith('sources/') or '/plans/' in location:
                issues.append(f'non-manual page in search: {location}')
    return sorted(set(issues)), checked_pages, checked_links


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT / '_site')
    args = parser.parse_args()
    sources = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md']
    sources += [p for folder in ('docs', 'resources') for p in (ROOT / folder).rglob('*.md') if 'plans' not in p.parts]
    issues = [issue for p in sources for issue in source_issues(p)]
    rendered, pages, links = check_site(args.site)
    issues.extend(rendered)
    for issue in issues:
        print(issue)
    print(f'{len(sources)} source files; {pages} rendered pages; {links} local links; {len(issues)} issue(s)')
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
