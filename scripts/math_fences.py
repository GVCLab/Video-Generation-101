"""Escape TeX before embedding it in HTML, preserving operators such as <i."""
from html import escape
from pymdownx.arithmatex import arithmatex_fenced_format

_formatter = arithmatex_fenced_format(mode="generic", tag="div")


def format_math(source, language, class_name, options, md, **kwargs):
    return _formatter(escape(source, quote=False), language, class_name, options, md, **kwargs)
