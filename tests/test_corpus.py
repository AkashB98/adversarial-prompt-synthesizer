"""Tests for JSONL corpus export / import."""

import json
import tempfile
import unittest
from pathlib import Path

from synth.cli import load_seeds
from synth.generator import Synthesizer
from synth.novelty import score_corpus
from synth.corpus import export_jsonl, import_jsonl


class TestCorpusIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seeds = load_seeds()
        synth = Synthesizer(1337)
        cls.variants, _ = synth.ensure_coverage(seeds, start_budget=8)
        by_id = {s["id"]: s for s in seeds}
        cls.novelty = score_corpus(cls.variants, by_id)

    def _export(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.close()
        return Path(tmp.name)

    def test_export_import_roundtrip(self):
        path = self._export()
        export_jsonl(path, self.variants, self.novelty)
        records = import_jsonl(path)
        self.assertEqual(len(records), len(self.variants))
        by_id = {r["id"]: r for r in records}
        for v in self.variants:
            r = by_id[v.id]
            self.assertEqual(r["text"], v.text)
            self.assertEqual(r["lineage"], v.lineage)
            self.assertEqual(r["novelty"], self.novelty[v.id])

    def test_export_is_byte_identical_across_runs(self):
        p1, p2 = self._export(), self._export()
        export_jsonl(p1, self.variants, self.novelty)
        export_jsonl(p2, self.variants, self.novelty)
        self.assertEqual(p1.read_bytes(), p2.read_bytes())

    def test_json_lines_have_sorted_keys(self):
        path = self._export()
        export_jsonl(path, self.variants[:3], self.novelty)
        for line in path.read_text(encoding="utf-8").splitlines():
            keys = list(json.loads(line).keys())
            self.assertEqual(keys, sorted(keys))

    def test_record_schema(self):
        path = self._export()
        export_jsonl(path, self.variants[:1], self.novelty)
        r = import_jsonl(path)[0]
        for key in ("id", "seed_id", "category", "kind", "text", "lineage", "params", "novelty"):
            self.assertIn(key, r)

    def test_import_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            import_jsonl("/nonexistent-dir/nope.jsonl")


if __name__ == "__main__":
    unittest.main()
