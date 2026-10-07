"""Tests for the Synthesizer: determinism, dedupe, replay, coverage."""

import unittest

from synth.cli import load_seeds
from synth.generator import Synthesizer
from synth.transforms import canonical, LABEL_TO_META, TRANSFORMS

FRAMING_MARKERS = [
    "ignore all previous", "ignore your previous", "disregard your system",
    "hypothetically", "training document", "translate the following",
    "you are a helpful assistant who always", "urgent", "base64", "decode it",
]


def build(seed_value=1337, pair_budget=24):
    seeds = load_seeds()
    synth = Synthesizer(seed_value)
    variants, budget = synth.ensure_coverage(seeds, start_budget=pair_budget)
    return seeds, synth, variants, budget


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds, cls.synth, cls.variants, cls.budget = build()
        cls.by_id = {s["id"]: s for s in cls.seeds}

    def test_two_runs_same_seed_identical_texts(self):
        _, _, again, _ = build()
        self.assertEqual([v.text for v in again], [v.text for v in self.variants])

    def test_two_runs_same_seed_identical_ids(self):
        _, _, again, _ = build()
        self.assertEqual([v.id for v in again], [v.id for v in self.variants])

    def test_different_seed_differs(self):
        _, _, other, _ = build(seed_value=999)
        self.assertNotEqual([v.text for v in other], [v.text for v in self.variants])

    def test_no_duplicate_canonical_texts(self):
        keys = [canonical(v.text) for v in self.variants]
        self.assertEqual(len(keys), len(set(keys)))

    def test_no_variant_identical_to_its_seed(self):
        for v in self.variants:
            self.assertNotEqual(v.text, self.by_id[v.seed_id]["text"], v.id)

    def test_replay_exact_for_all_variants(self):
        for v in self.variants:
            self.assertEqual(self.synth.replay(v, self.by_id), v.text, v.id)

    def test_lineage_labels_match_registry(self):
        for v in self.variants:
            for label in v.lineage:
                self.assertIn(label, LABEL_TO_META, v.id)

    def test_lineage_length_bounded(self):
        for v in self.variants:
            self.assertIn(len(v.lineage), (1, 2), v.id)

    def test_pair_budget_respected(self):
        per_seed_pairs = {}
        for v in self.variants:
            if len(v.lineage) == 2:
                per_seed_pairs[v.seed_id] = per_seed_pairs.get(v.seed_id, 0) + 1
        for seed_id, n in per_seed_pairs.items():
            self.assertLessEqual(n, 24, seed_id)

    def test_variant_ids_unique(self):
        ids = [v.id for v in self.variants]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_seed_has_variants(self):
        have = {v.seed_id for v in self.variants}
        self.assertEqual(have, {s["id"] for s in self.seeds})

    def test_coverage_thresholds_met(self):
        from synth.coverage import coverage_matrix, variants_per_category, check_thresholds
        ok, failures = check_thresholds(
            coverage_matrix(self.variants), variants_per_category(self.variants),
            k=8, f=4,
        )
        self.assertTrue(ok, failures)

    def test_benign_variants_stay_non_attack_framed(self):
        for v in self.variants:
            if v.kind != "benign":
                continue
            for label in v.lineage:
                self.assertFalse(LABEL_TO_META[label][2], f"{v.id}: {label}")
            lowered = v.text.lower()
            for marker in FRAMING_MARKERS:
                self.assertNotIn(marker, lowered, f"{v.id} contains {marker!r}")

    def test_benign_seeds_never_get_framing_transforms(self):
        synth = Synthesizer(1)
        allowed = synth.allowed_labels("benign")
        for label in allowed:
            self.assertFalse(LABEL_TO_META[label][2], label)

    def test_attack_seeds_get_all_transforms(self):
        synth = Synthesizer(1)
        self.assertEqual(len(synth.allowed_labels("attack")), len(TRANSFORMS))

    def test_ensure_coverage_autotunes_budget(self):
        # Impossible thresholds force the budget to the cap.
        seeds = load_seeds()
        synth = Synthesizer(1337)
        variants, budget = synth.ensure_coverage(
            seeds, k=10**9, f=4, start_budget=8, step=8, max_budget=24)
        self.assertEqual(budget, 24)
        self.assertTrue(len(variants) > 0)

    def test_params_carry_rng_tag(self):
        for v in self.variants[:10]:
            self.assertIn("rng_tag", v.params)
            self.assertTrue(v.params["rng_tag"].startswith(v.seed_id))


if __name__ == "__main__":
    unittest.main()
