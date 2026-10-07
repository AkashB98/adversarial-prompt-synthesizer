"""Golden evals for the adversarial prompt synthesizer.

~22 deterministic checks over the generated corpus. Writes
``evals/eval_report.json`` with sorted keys and rounded floats so the file
is byte-identical across runs. Exit 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synth.cli import load_seeds  # noqa: E402
from synth.generator import Synthesizer  # noqa: E402
from synth.transforms import canonical, LABEL_TO_META, FAMILIES  # noqa: E402
from synth.novelty import score_corpus, pairwise_min_distance  # noqa: E402
from synth.coverage import coverage_matrix, variants_per_category, check_thresholds  # noqa: E402
from synth.corpus import export_jsonl, import_jsonl  # noqa: E402

SEED_VALUE = 1337
PAIR_BUDGET = 24

FRAMING_MARKERS = [
    "ignore all previous", "ignore your previous", "disregard your system",
    "hypothetically", "training document", "translate the following",
    "you are a helpful assistant who always", "urgent", "base64", "decode it",
]

# Patterns that would indicate a real secret leaked into the repo. The word
# "password" alone is NOT a secret (seeds use it fictionally).
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{10,}",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"xox[bap]-[A-Za-z0-9-]+",
    r"ghp_[A-Za-z0-9]{10,}",
    r"AKIA[0-9A-Z]{16}",
]


def _check(name, passed, detail=""):
    return {"name": name, "pass": bool(passed), "detail": str(detail)}


def run_evals():
    seeds = load_seeds()
    seeds_by_id = {s["id"]: s for s in seeds}
    synth = Synthesizer(SEED_VALUE)
    variants, budget = synth.ensure_coverage(seeds, start_budget=PAIR_BUDGET)
    novelty = score_corpus(variants, seeds_by_id)
    matrix = coverage_matrix(variants)
    counts = variants_per_category(variants)
    checks = []

    # 1. corpus size in expected range
    checks.append(_check("corpus_size_in_range", 800 <= len(variants) <= 1600,
                         f"n={len(variants)}"))

    # 2. every seed produced variants
    have = {v.seed_id for v in variants}
    checks.append(_check("all_seed_ids_present", have == set(seeds_by_id),
                         f"{len(have)}/{len(seeds_by_id)}"))

    # 3. all 9 categories present
    cats = {v.category for v in variants}
    checks.append(_check("all_categories_present", len(cats) == 9,
                         f"{len(cats)} categories"))

    # 4. per-category minimum count
    worst = min(counts.values())
    checks.append(_check("per_category_count_ge_24", worst >= 24,
                         f"min per-category n={worst}"))

    # 5. coverage thresholds K=8 F=4
    ok, failures = check_thresholds(matrix, counts, k=8, f=4)
    checks.append(_check("coverage_K8_F4", ok, "; ".join(failures)))

    # 6. lineage replay exact across the whole corpus
    bad = [v.id for v in variants if synth.replay(v, seeds_by_id) != v.text]
    checks.append(_check("replay_exact_all", not bad,
                         f"{len(bad)} mismatches" if bad else f"{len(variants)} ok"))

    # 7. determinism: two runs -> identical corpus md5
    def corpus_md5():
        v2, _ = synth.ensure_coverage(seeds, start_budget=PAIR_BUDGET)
        h = hashlib.md5()
        for v in v2:
            h.update(v.id.encode())
            h.update(b"\x00")
            h.update(v.text.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()
    md5_a, md5_b = corpus_md5(), corpus_md5()
    checks.append(_check("determinism_md5_twice", md5_a == md5_b, md5_a))

    # 8. export byte-identical twice
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.jsonl"
        p2 = Path(tmp) / "b.jsonl"
        export_jsonl(p1, variants, novelty)
        export_jsonl(p2, variants, novelty)
        same = p1.read_bytes() == p2.read_bytes()
    checks.append(_check("export_byte_identical_twice", same, ""))

    # 9. benign variants carry no attack framing markers
    bad = [v.id for v in variants if v.kind == "benign"
           and any(m in v.text.lower() for m in FRAMING_MARKERS)]
    checks.append(_check("benign_no_framing_markers", not bad,
                         f"{len(bad)} flagged" if bad else "clean"))

    # 10. benign lineage never uses attack-framing transforms
    bad = [v.id for v in variants if v.kind == "benign"
           and any(LABEL_TO_META[l][2] for l in v.lineage)]
    checks.append(_check("benign_lineage_no_attack_framing", not bad,
                         f"{len(bad)} flagged" if bad else "clean"))

    # 11. novelty scores in [0, 1]
    bad = [vid for vid, s in novelty.items() if not (0.0 <= s <= 1.0)]
    checks.append(_check("novelty_in_unit_interval", not bad,
                         f"{len(bad)} out of range" if bad else "ok"))

    # 12/13. per-category mean/min novelty above floors
    by_cat = {}
    for v in variants:
        by_cat.setdefault(v.category, []).append(novelty[v.id])
    mean_ok = all(sum(x) / len(x) >= 0.20 for x in by_cat.values())
    min_ok = all(min(x) >= 0.005 for x in by_cat.values())
    checks.append(_check("per_category_mean_novelty_ge_0.20", mean_ok,
                         f"lowest mean={min(sum(x)/len(x) for x in by_cat.values()):.4f}"))
    checks.append(_check("per_category_min_novelty_ge_0.005", min_ok,
                         f"lowest min={min(min(x) for x in by_cat.values()):.4f}"))

    # 14. >=90% of variants are meaningfully novel (>= 0.05)
    frac = sum(1 for s in novelty.values() if s >= 0.05) / len(novelty)
    checks.append(_check("fraction_novel_ge_90pct", frac >= 0.90,
                         f"{frac:.3f}"))

    # 15. pairwise min distance > 0 (no two variants textually identical)
    gmin, _ = pairwise_min_distance(variants)
    checks.append(_check("pairwise_min_distance_gt_0", gmin > 0, f"{gmin:.4f}"))

    # 16. no duplicate canonical texts
    keys = [canonical(v.text) for v in variants]
    checks.append(_check("dedup_no_duplicate_canonical", len(keys) == len(set(keys)),
                         f"n={len(keys)}"))

    # 17. variant ids unique
    ids = [v.id for v in variants]
    checks.append(_check("variant_ids_unique", len(ids) == len(set(ids)), ""))

    # 18. all lineage labels known
    bad = [v.id for v in variants
           if any(l not in LABEL_TO_META for l in v.lineage)]
    checks.append(_check("lineage_labels_all_known", not bad,
                         f"{len(bad)} unknown" if bad else "ok"))

    # 19. seeds JSON valid
    attacks = [s for s in seeds if s["kind"] == "attack"]
    benign = [s for s in seeds if s["kind"] == "benign"]
    seed_ids = [s["id"] for s in seeds]
    seeds_ok = (len(attacks) == 24 and len(benign) == 8
                and len(set(seed_ids)) == 32
                and all(set(s) == {"id", "kind", "category", "text"} for s in seeds)
                and len({s["category"] for s in attacks}) == 8)
    checks.append(_check("seeds_json_valid", seeds_ok,
                         f"{len(attacks)} attack / {len(benign)} benign"))

    # 20. no secrets anywhere in code, seeds, or corpus text
    blob_parts = []
    for path in sorted((ROOT / "synth").glob("*.py")):
        blob_parts.append(path.read_text(encoding="utf-8"))
    for path in sorted((ROOT / "seeds").glob("*.json")):
        blob_parts.append(path.read_text(encoding="utf-8"))
    blob_parts.extend(v.text for v in variants)
    blob = "\n".join(blob_parts)
    hits = [p for p in SECRET_PATTERNS if re.search(p, blob)]
    checks.append(_check("no_secrets_in_repo", not hits,
                         f"patterns hit: {hits}" if hits else "clean"))

    # 21. every transform family used at least once
    used = {LABEL_TO_META[l][1] for v in variants for l in v.lineage}
    checks.append(_check("all_families_used", set(FAMILIES) <= used,
                         f"missing: {set(FAMILIES) - used}" if set(FAMILIES) - used else "ok"))

    # 22. pair variants present in meaningful number
    n_pairs = sum(1 for v in variants if len(v.lineage) == 2)
    checks.append(_check("pair_variants_present", n_pairs >= 100,
                         f"n_pairs={n_pairs}"))

    metrics = {
        "n_variants": len(variants),
        "n_pairs": n_pairs,
        "pair_budget": budget,
        "mean_novelty": round(sum(novelty.values()) / len(novelty), 6),
        "min_novelty": round(min(novelty.values()), 6),
        "pairwise_min_distance": round(gmin, 6),
        "per_category_mean_novelty": {
            c: round(sum(x) / len(x), 6) for c, x in sorted(by_cat.items())
        },
    }
    report = {
        "evals_version": 1,
        "seed_value": SEED_VALUE,
        "n_seeds": len(seeds),
        "corpus_md5": md5_a,
        "metrics": metrics,
        "checks": checks,
        "passed": sum(1 for c in checks if c["pass"]),
        "total": len(checks),
    }

    # 23. the report itself serializes deterministically
    s1 = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False)
    s2 = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False)
    checks.append(_check("report_serialization_deterministic", s1 == s2, ""))
    report["passed"] = sum(1 for c in checks if c["pass"])
    report["total"] = len(checks)
    out_text = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False)
    (ROOT / "evals" / "eval_report.json").write_text(out_text + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run_evals()
    for c in report["checks"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']}"
              + (f" — {c['detail']}" if c["detail"] else ""))
    print(f"\n{report['passed']}/{report['total']} evals passed")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
