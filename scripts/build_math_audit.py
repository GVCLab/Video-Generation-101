#!/usr/bin/env python3
"""Build a local browser audit harness for every published page and math block.

After build_site.py, run with --output /tmp/vg101-math-audit and serve that
folder with python -m http.server. Open audit.html; results appear in the page.
The harness never changes the source or built site.
"""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class MathInventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.count = 0

    def handle_starttag(self, tag, attrs):
        if 'arithmatex' in dict(attrs).get('class', '').split():
            self.count += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, default=ROOT / '_site')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    site = args.site.resolve()
    pages = []
    for path in sorted(site.rglob('*.html')):
        rel = path.relative_to(site)
        if rel.parts[0] == 'assets' or rel.name == '404.html':
            continue
        parser = MathInventory()
        parser.feed(path.read_text(encoding='utf-8'))
        pages.append({'path': str(rel), 'expected': parser.count,
                      'scope': 'archive' if rel.parts[0] == 'sources' else 'manual'})
    args.output.mkdir(parents=True, exist_ok=True)
    link = args.output / 'site'
    if not link.exists():
        link.symlink_to(site, target_is_directory=True)
    elif link.resolve() != site:
        raise SystemExit('Output already points to another site; use a new directory')
    (args.output / 'inventory.json').write_text(json.dumps(pages, ensure_ascii=False))
    (args.output / 'audit.js').write_text((ROOT / 'scripts/math_render_audit.js').read_text())
    (args.output / 'audit.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>公式渲染检查</title><style>body{font:14px system-ui;margin:20px}iframe{height:720px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>
<h1>公式渲染检查</h1><p id="status">正在加载</p><pre id="summary"></pre><pre id="problems"></pre>
<iframe id="frame" title="被检查的页面"></iframe><pre id="results"></pre><script src="audit.js"></script></html>''')
    print(f'{len(pages)} pages, {sum(p["expected"] for p in pages)} math blocks; open audit.html in {args.output}')

if __name__ == '__main__':
    main()
