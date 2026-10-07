"""Tests for the coverage matrix and thresholds."""

import unittest

from synth.cli import load_seeds
from synth.generator import Synthesizer
from synth.coverage import coverage_matrix, variants_per_category, check_thresholds


class TestCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seeds = load_seeds()
        synth = Synthesizer(1337)
        cls.variants, _ = synth.ensure_coverage(seeds, start_budget=24)
        cls.matrix = coverage_matrix(cls.variants)
        cls.counts = variants_per_category(cls.variants)

    def test_matrix_has_all_categories(self):
        expected = {s["category"] for s in load_seeds()}
        self.assertEqual(set(self.matrix), expected)

    def test_thresholds_pass_on_full_corpus(self):
        ok, failures = check_thresholds(self.matrix, self.counts, k=8, f=4)
        self.assertTrue(ok, failures)

    def test_thresholds_detect_failure(self):
        ok, failures = check_thresholds(self.matrix, self.counts, k=10**9, f=4)
        self.assertFalse(ok)
        self.assertTrue(failures)

    def test_thresholds_detect_family_failure(self):
        ok, failures = check_thresholds(self.matrix, self.counts, k=1, f=99)
        self.assertFalse(ok)
        self.assertTrue(failures)

    def test_pairs_contribute_two_families(self):
        pair_variants = [v for v in self.variants if len(v.lineage) == 2]
        self.assertTrue(len(pair_variants) > 100)
        from synth.transforms import LABEL_TO_META
        for v in pair_variants[:20]:
            fams = [LABEL_TO_META[l][1] for l in v.lineage]
            self.assertEqual(len(fams), 2)

    def test_every_category_has_multiple_families(self):
        for cat, row in self.matrix.items():
            active = [f for f, c in row.items() if c > 0]
            self.assertGreaterEqual(len(active), 4, cat)

    def test_variant_counts_match_matrix_categories(self):
        self.assertEqual(set(self.counts), set(self.matrix))


if __name__ == "__main__":
    unittest.main()
