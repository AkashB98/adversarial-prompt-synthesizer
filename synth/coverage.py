"""Coverage matrix: per category x transform-family application counts.

A category is covered when it has >= K variants drawn from >= F distinct
transform families. The generator auto-tunes its pair budget until these
thresholds hold.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .transforms import LABEL_TO_META


def coverage_matrix(variants) -> Dict[str, Dict[str, int]]:
    """{category: {family: application_count}}. Each transform application in
    a variant's lineage counts once (pairs contribute to two families)."""
    matrix: Dict[str, Dict[str, int]] = {}
    for v in variants:
        row = matrix.setdefault(v.category, {})
        for label in v.lineage:
            family = LABEL_TO_META[label][1]
            row[family] = row.get(family, 0) + 1
    return matrix


def variants_per_category(variants) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in variants:
        counts[v.category] = counts.get(v.category, 0) + 1
    return counts


def check_thresholds(
    matrix: Dict[str, Dict[str, int]],
    variant_counts: Dict[str, int],
    k: int = 8,
    f: int = 4,
) -> Tuple[bool, List[str]]:
    """Full threshold check: every category needs >= K variants AND those
    variants must come from >= F distinct transform families."""
    failures: List[str] = []
    for category in sorted(set(matrix) | set(variant_counts)):
        n_variants = variant_counts.get(category, 0)
        families = [fam for fam, c in matrix.get(category, {}).items() if c > 0]
        if n_variants < k:
            failures.append(f"{category}: {n_variants} variants, need {k}")
        if len(families) < f:
            failures.append(
                f"{category}: {len(families)} families with variants, need {f}"
            )
    return (len(failures) == 0), failures
