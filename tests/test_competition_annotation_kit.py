import io
import json
import re
import unittest
from collections import Counter
from contextlib import redirect_stdout

from scripts.competition_annotate import match_quote, show_source
from scripts.competition_annotation_kit import ARCH, launcher, revised_subset


class AnnotationKitTests(unittest.TestCase):
    def test_readable_subset_covers_every_original_unit(self):
        old = json.loads((ARCH / "v2/gold-subset.json").read_bytes())
        new = revised_subset(old)
        self.assertEqual(new, json.loads((ARCH / "v3/gold-subset.json").read_bytes()))
        restored = []
        for src in new:
            loc = src["locator"]
            if "sheet" in loc:
                header, *data = src["text"].splitlines()
                self.assertEqual(header.split(" | "), ["Variable", "xn1", "xn2", "xn3", "xn4", "xn5"])
                num = re.match(r"A(\d+):F\1$", loc["cell_range"])[1]
                for col, value in zip("ABCDEF", (data or [header])[0].split(" | ")):
                    if value:
                        restored.append((src["source_id"], col + num, value))
            elif "page" in loc:
                restored.append((src["source_id"], tuple(loc["bbox"]), src["text"]))
            else:
                lines = src["text"].splitlines()
                self.assertIn(len(lines), range(3, 9))
                self.assertEqual(len(lines), loc["end_line"] - loc["line"] + 1)
                restored.extend((src["source_id"], i, line) for i, line in enumerate(lines, loc["line"]))
        def key(src):
            loc = src["locator"]
            pos = tuple(loc["bbox"]) if "page" in loc else loc.get("cell", loc.get("line"))
            return src["source_id"], pos, src["text"]
        self.assertEqual(Counter(restored), Counter(map(key, old)))

    def test_quote_and_display_preserve_source_evidence(self):
        original = "A  convec- \r\ntion model: one-dimensional."
        match = match_quote(original, "convection   model")
        self.assertEqual(match["quote_normalized"], "convectionmodel")
        self.assertEqual(match["quote_raw"], "convec- \r\ntion model")
        self.assertEqual(original[slice(*match["quote_span"])], match["quote_raw"])
        self.assertIsNone(match_quote(original, "onedimensional"))
        self.assertIsNone(match_quote(original, "  "))
        stream = io.StringIO()
        with redirect_stdout(stream):
            show_source({"path": "paper.pdf", "locator": {"page": 3, "bbox": [1,2,3,4]}, "text": original})
        self.assertIn("convection model", stream.getvalue())
        self.assertIn("第 3 页", stream.getvalue())
        self.assertNotIn("bbox", stream.getvalue())

    def test_ascii_windows_launcher(self):
        bat = launcher("subset.json", "packet.json", "labels", "reviewer_a").decode("ascii")
        self.assertNotIn("chcp", bat.lower())
        self.assertNotIn("\r\r\n", bat)
        self.assertIn('%USERPROFILE%\\anaconda3\\python.exe', bat)
