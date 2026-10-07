"""One-command end-to-end demo: seeds -> corpus -> novelty -> coverage.

Prints two sample variants per category with lineage + novelty, then
novelty/coverage summary tables. Asserts determinism, replay exactness and
coverage thresholds internally.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from synth.cli import build_corpus  # noqa: E402
from synth.coverage import coverage_matrix, variants_per_category, check_thresholds  # noqa: E402
from synth.transforms import LABEL_TO_META  # noqa: E402


def main() -> None:
    seeds, synth, variants, novelty, budget = build_corpus(1337, 24)
    seeds_by_id = {s["id"]: s for s in seeds}

    # --- internal assertions -------------------------------------------------
    again = synth.generate(seeds, pair_budget=budget)
    assert [v.text for v in again] == [v.text for v in variants], "not deterministic"
    for v in variants:
        assert synth.replay(v, seeds_by_id) == v.text, f"replay mismatch: {v.id}"
    matrix = coverage_matrix(variants)
    ok, failures = check_thresholds(matrix, variants_per_category(variants))
    assert ok, f"coverage failed: {failures}"

    # --- sample variants -----------------------------------------------------
    print("=" * 72)
    print("ADVERSARIAL PROMPT SYNTHESIZER — demo")
    print(f"{len(seeds)} seeds -> {len(variants)} variants (pair budget {budget})")
    print("=" * 72)
    shown: dict = {}
    for v in variants:
        if shown.get(v.category, 0) >= 2:
            continue
        shown[v.category] = shown.get(v.category, 0) + 1
        fams = [LABEL_TO_META[l][1] for l in v.lineage]
        print(f"\n[{v.category}] {v.id}  kind={v.kind}")
        print(f"  seed   : {seeds_by_id[v.seed_id]['text'][:90]}")
        print(f"  variant: {v.text[:160]}")
        print(f"  lineage: {' -> '.join(v.lineage)}  (families: {', '.join(fams)})")
        print(f"  novelty: {novelty[v.id]:.4f}")

    # --- novelty table -------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"{'category':28s} {'n':>5s} {'mean':>8s} {'min':>8s}")
    by_cat: dict = {}
    for v in variants:
        by_cat.setdefault(v.category, []).append(novelty[v.id])
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        print(f"{cat:28s} {len(vals):5d} {sum(vals)/len(vals):8.4f} {min(vals):8.4f}")

    # --- coverage table ------------------------------------------------------
    print("\n" + "-" * 72)
    fams = sorted({f for row in matrix.values() for f in row})
    print(f"{'category':28s}" + "".join(f"{f:>10s}" for f in fams))
    for cat in sorted(matrix):
        print(f"{cat:28s}" + "".join(f"{matrix[cat].get(f, 0):10d}" for f in fams))
    print(f"\ncoverage thresholds K=8 F=4: {'PASS' if ok else 'FAIL'}")
    print("demo OK: deterministic, replay-exact, coverage-satisfied")


if __name__ == "__main__":
    main()
