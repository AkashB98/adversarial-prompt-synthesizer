"""Novelty metrics (stdlib only).

Three per-variant distances vs its seed, all in [0, 1] where 0 means
identical:
  * char 3-gram novelty  = 1 - Jaccard(char 3-gram sets)
  * word-overlap novelty = 1 - Jaccard(word sets)
  * sequence novelty     = 1 - difflib.SequenceMatcher ratio

``variant.novelty`` is the minimum of the three: a variant only counts as
novel if it differs from its seed on every measure.

Also provides the corpus-wide pairwise minimum distance (char 3-gram),
i.e. how close the two most similar distinct variants are.
"""

from __future__ import annotations

import difflib
from typing import Dict, Tuple


def _norm(text: str) -> str:
    """Normalization for novelty measurement: lowercased but otherwise raw,
    so invisible-character and whitespace-only edits still register."""
    return text.lower()


def _words(text: str) -> list:
    return _norm(text).split()


def char_trigrams(text: str) -> frozenset:
    t = _norm(text)
    if len(t) < 3:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i + 3] for i in range(len(t) - 2))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def trigram_novelty(variant_text: str, seed_text: str) -> float:
    return round(1.0 - _jaccard(char_trigrams(variant_text), char_trigrams(seed_text)), 6)


def word_overlap_novelty(variant_text: str, seed_text: str) -> float:
    return round(
        1.0 - _jaccard(frozenset(_words(variant_text)), frozenset(_words(seed_text))), 6
    )


def sequence_novelty(variant_text: str, seed_text: str) -> float:
    ratio = difflib.SequenceMatcher(None, variant_text, seed_text).ratio()
    return round(1.0 - ratio, 6)


def novelty_vs_seed(variant_text: str, seed_text: str) -> float:
    """The variant's novelty score: min of the three distances vs its seed."""
    return round(
        min(
            trigram_novelty(variant_text, seed_text),
            word_overlap_novelty(variant_text, seed_text),
            sequence_novelty(variant_text, seed_text),
        ),
        6,
    )


def score_corpus(variants, seeds_by_id: Dict[str, Dict]) -> Dict[str, float]:
    """Map variant id -> novelty vs its seed."""
    return {
        v.id: novelty_vs_seed(v.text, seeds_by_id[v.seed_id]["text"]) for v in variants
    }


def pairwise_min_distance(variants) -> Tuple[float, Dict[str, float]]:
    """Corpus-wide pairwise minimum char-3-gram distance.

    Returns (global_min, {variant_id: min distance to any other variant}).
    Symmetric O(n^2) loop over precomputed trigram sets.
    """
    grams = [char_trigrams(v.text) for v in variants]
    n = len(variants)
    per_variant = {v.id: 1.0 for v in variants}
    global_min = 1.0
    for i in range(n):
        gi = grams[i]
        for j in range(i + 1, n):
            d = round(1.0 - _jaccard(gi, grams[j]), 6)
            if d < per_variant[variants[i].id]:
                per_variant[variants[i].id] = d
            if d < per_variant[variants[j].id]:
                per_variant[variants[j].id] = d
            if d < global_min:
                global_min = d
    return round(global_min, 6), {k: round(v, 6) for k, v in per_variant.items()}
