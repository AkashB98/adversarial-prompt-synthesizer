"""Variant corpus generator with lineage replay.

A ``Synthesizer`` expands each seed prompt into many variants by applying
every allowed single transform plus a deterministic sample of ordered
transform pairs. Every RNG draw is derived from ``(seed_value, seed_id,
variant_tag, step)`` so the whole corpus is byte-identical for a given
master seed, and ``replay()`` can re-derive the exact same draws from a
variant's lineage alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .transforms import TRANSFORMS, LABEL_TO_META, canonical
from .coverage import coverage_matrix, variants_per_category, check_thresholds


@dataclass
class Variant:
    id: str
    seed_id: str
    category: str
    kind: str  # "attack" | "benign"
    text: str
    lineage: List[str] = field(default_factory=list)
    params: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, novelty: float | None = None) -> Dict:
        d = {
            "id": self.id,
            "seed_id": self.seed_id,
            "category": self.category,
            "kind": self.kind,
            "text": self.text,
            "lineage": list(self.lineage),
            "params": dict(self.params),
        }
        if novelty is not None:
            d["novelty"] = round(float(novelty), 6)
        return d


class Synthesizer:
    """Deterministic seed -> variant corpus expander."""

    def __init__(self, seed_value: int = 1337):
        self.seed_value = seed_value

    # -- RNG ---------------------------------------------------------------
    def _rng(self, *parts) -> random.Random:
        tag = ":".join([str(self.seed_value)] + [str(p) for p in parts])
        return random.Random(tag)

    # -- transform selection -------------------------------------------------
    def allowed_labels(self, kind: str) -> List[str]:
        """Labels usable for a seed kind. Benign control seeds never get
        attack-framing transforms, so benign variants stay benign."""
        is_attack = kind == "attack"
        return [
            label for label, _, _, attack_framing in TRANSFORMS
            if is_attack or not attack_framing
        ]

    # -- chain application ---------------------------------------------------
    def _apply_chain(self, seed_text: str, labels: List[str], rng_tag: str) -> str:
        text = seed_text
        for step, label in enumerate(labels):
            fn = LABEL_TO_META[label][0]
            text, got = fn(text, self._rng(rng_tag, step))
            if got != label:
                raise AssertionError(f"label mismatch: expected {label}, got {got}")
        return text

    def replay(self, variant: Variant, seeds_by_id: Dict[str, Dict]) -> str:
        """Re-apply a variant's lineage from its seed. Must equal
        ``variant.text`` exactly."""
        seed = seeds_by_id[variant.seed_id]
        return self._apply_chain(seed["text"], variant.lineage, variant.params["rng_tag"])

    # -- generation ----------------------------------------------------------
    def generate(self, seeds: List[Dict], pair_budget: int = 24) -> List[Variant]:
        variants: List[Variant] = []
        seen: set = set()
        order = 0
        for seed in seeds:
            labels = self.allowed_labels(seed["kind"])
            chains: List[Tuple[str, List[str]]] = []
            for i, label in enumerate(labels):
                chains.append((f"s{i}", [label]))
            pair_labels = [(a, b) for a in labels for b in labels if a != b]
            sampler = self._rng("pairs", seed["id"])
            sampler.shuffle(pair_labels)
            for k, (a, b) in enumerate(pair_labels[:pair_budget]):
                chains.append((f"p{k}", [a, b]))
            for tag, chain in chains:
                rng_tag = f"{seed['id']}:{tag}"
                text = self._apply_chain(seed["text"], chain, rng_tag)
                if text == seed["text"]:
                    continue  # degenerate: transform left the seed unchanged
                key = canonical(text)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(
                    Variant(
                        id=f"{seed['id']}__v{order:04d}",
                        seed_id=seed["id"],
                        category=seed["category"],
                        kind=seed["kind"],
                        text=text,
                        lineage=list(chain),
                        params={"rng_tag": rng_tag},
                    )
                )
                order += 1
        return variants

    def ensure_coverage(
        self,
        seeds: List[Dict],
        k: int = 8,
        f: int = 4,
        start_budget: int = 24,
        step: int = 8,
        max_budget: int = 96,
    ) -> Tuple[List[Variant], int]:
        """Grow the pair budget until every category has >=K variants from
        >=F distinct transform families (or the budget cap is hit)."""
        budget = start_budget
        while True:
            variants = self.generate(seeds, pair_budget=budget)
            ok, _ = check_thresholds(
                coverage_matrix(variants), variants_per_category(variants), k=k, f=f
            )
            if ok or budget >= max_budget:
                return variants, budget
            budget += step
