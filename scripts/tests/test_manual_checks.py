import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

citations = load('verify_citations')
manual = load('check_manual')
build = load('build_site')

class ManualChecks(unittest.TestCase):
    def test_html_titles_are_collected(self):
        parser = citations.LinkParser()
        parser.feed('<p>text</p>\n<a href="https://arxiv.org/abs/1603.04433">Out of <em>time</em></a>')
        self.assertEqual(parser.links, [(2, 'Out of time', 'https://arxiv.org/abs/1603.04433')])

    def test_wrong_paper_is_an_error(self):
        ref = citations.Ref('x.md', 1, 'Out of time: automated lip sync in the wild',
                            'https://arxiv.org/abs/1603.04433', '', '1603.04433')
        metadata = {'1603.04433': {'title': 'Determination of the total absorption peak in an electromagnetic calorimeter', 'published': '2016-03-14'}}
        findings = citations.check_arxiv([ref], metadata, {'1603.04433'})
        self.assertEqual(findings[0].kind, 'title-mismatch')

    def test_missing_and_unqueried_are_distinct(self):
        ref = citations.Ref('x.md', 1, 'Some Example Paper', '', '', '9999.00001')
        self.assertEqual(citations.check_arxiv([ref], {}, set()), [])
        self.assertEqual(citations.check_arxiv([ref], {}, {'9999.00001'})[0].kind, 'arxiv-not-found')

    def test_partial_online_query_cannot_pass(self):
        ref = citations.Ref('x.md', 1, 'Paper', '', '', '2401.00001')
        with patch.object(citations, 'collect_refs', return_value=[ref]), patch.object(citations, 'arxiv_lookup', return_value=({}, set())), patch.object(sys, 'argv', ['verify_citations.py', '--fail-on', 'NONE']):
            self.assertEqual(citations.main(), 2)

    def test_missing_anchor_and_search_leak_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'index.html').write_text('<a href="target.html#wrong">test</a>')
            (root / 'target.html').write_text('<h1 id="right">Title</h1>')
            (root / 'search').mkdir()
            (root / 'search/search_index.json').write_text(json.dumps({'docs': [{'location': 'sources/notes.html'}]}))
            issues, _, _ = manual.check_site(root)
            self.assertTrue(any('missing anchor' in i for i in issues))
            self.assertTrue(any('non-manual page' in i for i in issues))

    def test_archive_staging_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sources').mkdir()
            src = root / 'sources/notes.md'
            src.write_text('# Historical note\n')
            dst = root / 'staged.md'
            with patch.object(build, 'ROOT', root):
                self.assertFalse(build._place(src, dst))
            self.assertEqual(src.read_text(), '# Historical note\n')
            self.assertIn('exclude: true', dst.read_text())

if __name__ == '__main__':
    unittest.main()
