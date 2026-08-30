import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "replace_mermaid_with_imagegen.py"


def load_module():
    spec = importlib.util.spec_from_file_location("replace_mermaid_with_imagegen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class MermaidImageReplacementTest(unittest.TestCase):
    def test_extracts_blocks_and_replaces_them_with_image_markdown(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            page = root / "page.md"
            assets = root / "assets" / "imagegen-diagrams" / "001"
            assets.mkdir(parents=True)
            image = assets / "diagram.png"
            image.write_bytes(b"png")
            page.write_text(
                "# 标题\n\n```mermaid\nflowchart LR\n A[输入] --> B[输出]\n```\n",
                encoding="utf-8",
            )
            manifest = module.extract_manifest(root, root / "assets" / "imagegen-diagrams")
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["id"], "001")
            manifest[0]["image"] = "assets/imagegen-diagrams/001/diagram.png"
            module.replace_blocks(root, manifest)
            rendered = page.read_text(encoding="utf-8")
            self.assertNotIn("mermaid", rendered)
            self.assertIn("![图 001：输入到输出的流程]", rendered)
            self.assertIn("assets/imagegen-diagrams/001/diagram.png", rendered)


if __name__ == "__main__":
    unittest.main()
