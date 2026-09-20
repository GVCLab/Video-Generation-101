"""Regression checks for HTML-sensitive TeX and adjacent prose."""
import unittest
from html.parser import HTMLParser
import markdown
from scripts.math_fences import format_math

class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []
    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
    def handle_data(self, data):
        self.text.append(data)

class MathRenderingTests(unittest.TestCase):
    def test_fenced_operators_preserve_text_and_prose(self):
        tex = r'\begin{aligned}p(y_i\mid y_{<i}) &= a & b > c\end{aligned}'
        html = markdown.markdown('```math\n' + tex + '\n```\n\nAfter equation.',
            extensions=['pymdownx.superfences'], extension_configs={
                'pymdownx.superfences': {'custom_fences': [
                    {'name':'math', 'class':'arithmatex', 'format':format_math}]}})
        parser = Parser()
        parser.feed(html)
        self.assertEqual(parser.tags, ['div', 'p'])
        self.assertIn(tex, ''.join(parser.text))
        self.assertTrue(html.endswith('<p>After equation.</p>'))

if __name__ == '__main__':
    unittest.main()
