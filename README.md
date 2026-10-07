# adversarial-prompt-synthesizer

**The attack-multiplier every FDE safety team needs.** His `ai-red-teaming-harness` *grades* an AI assistant's safety under attack; this synthesizer *keeps the attacks coming* so those grades never go stale. Give it 32 fictional seed prompts; get back ~1,000 diverse, measurably-novel attack variants with full lineage — offline, deterministic, stdlib-only.

**red-teaming** — deliberately attacking an AI system to find its safety holes before real attackers do.

## Why this project (the hiring signal)

Frontier AI labs don't hand-write every attack prompt — they *synthesize* them. This tool is the feeder stage of a real safety-eval pipeline: seed attacks in, large graded corpus out. It proves he can build **defensive AI-safety tooling**: deterministic data pipelines, measurable quality gates (novelty, coverage), and reproducible evals — the exact muscle a Forward Deployed Engineer brings to a safety team.

## How it works

1. **Seeds** — 24 fictional attack seeds across 8 classic categories (the same taxonomy as his red-teaming harness) + 8 benign controls, all targeting a made-up "Helios Home" assistant. No real systems, no real data.
2. **Transforms** — 17 deterministic text transforms in 7 families (paraphrase, synonym, framing, encoding, case, whitespace, multiturn). Every transform is a **pure function** — same inputs always give same outputs, no hidden state — of the form `(text, rng) -> (new_text, label)` with the **RNG** (random number generator) passed in by the caller.
3. **Generation** — for each seed, apply every allowed single transform plus a deterministic sample of ordered transform *pairs* (t1 then t2), up to a per-seed pair budget. **Dedupe** (removing duplicates) by **canonical** text — lowercased, whitespace collapsed, invisible characters stripped. Benign seeds never receive attack-framing transforms, so benign variants stay benign.
4. **Lineage + replay** — each variant records its transform chain. `replay(variant)` re-derives the exact RNG draws from the lineage and must reproduce the text byte-for-byte. Provenance you can audit.
5. **Novelty** — three stdlib distance metrics vs the seed (char 3-gram **Jaccard** — overlap of 3-letter chunks divided by their union; word-overlap ratio; `difflib.SequenceMatcher` ratio); the variant's novelty is the *minimum* of the three, so a variant only counts as novel if it differs on every measure. Plus corpus-wide pairwise minimum distance.
6. **Coverage** — a category × transform-family matrix; every category must hold ≥8 variants from ≥4 distinct families, and the generator auto-tunes its pair budget until that holds.

**Determinism** — the property that the same inputs always produce byte-identical outputs. Every RNG draw derives from `(master_seed, seed_id, variant_tag, step)`. Same master seed → byte-identical corpus, verified by md5 in the evals.

**Hermetic** — fully self-contained: no network, no pip packages, no API keys. A test stubs out sockets to prove the server degrades cleanly with no network.

## Quickstart

```bash
python3 -m synth.cli seeds                                  # list the 32 seeds
python3 -m synth.cli generate --out corpus.jsonl            # build ~969 variants
python3 -m synth.cli novelty --corpus corpus.jsonl          # novelty summary
python3 -m synth.cli coverage --corpus corpus.jsonl         # coverage matrix
python3 demo.py                                             # end-to-end demo
python3 -m synth.cli serve --port 8137                      # JSON API: POST /generate, GET /seeds
python3 -m unittest discover -s tests                       # 106 tests
python3 evals/run_evals.py                                  # 23 golden evals -> evals/eval_report.json
```

`requirements.txt` is intentionally empty: Python 3.8+ stdlib only.

## Transform catalog

| label | family | what it does |
|---|---|---|
| `paraphrase:imperative-declarative` | paraphrase | rewords commands as requests ("Please handle…") |
| `paraphrase:passive` | paraphrase | passive-voice rewording ("The following action is requested…") |
| `paraphrase:question` | paraphrase | question-form rewording ("How would you respond…") |
| `synonym:swap` | synonym | swaps 1–3 words via a fixed synonym map, preserving case |
| `framing:training-doc` | framing ⚠️ | wraps as a "safety training document" example request |
| `framing:translation` | framing ⚠️ | "translate the following into French…" wrapper |
| `framing:hypothetical` | framing ⚠️ | "hypothetically, if there were no restrictions…" wrapper |
| `framing:roleplay` | framing ⚠️ | "you are a helpful assistant who always complies…" wrapper |
| `framing:urgency` | framing ⚠️ | prepends an urgent time-pressure pretext |
| `encoding:base64` | encoding ⚠️ | wraps the **base64** payload — text re-encoded as ASCII letters/numbers so it looks like gibberish until decoded |
| `encoding:rot13` | encoding | **rot13** — each letter shifted 13 places (a→n); applying twice restores the text |
| `encoding:zero-width` | encoding | inserts invisible **zero-width** characters (they take up no space on screen) between words |
| `encoding:homoglyph` | encoding | swaps letters for lookalike characters from another alphabet (latin `a` → cyrillic `а`) |
| `encoding:html-comment` | encoding | splits words with invisible `<!-- -->` HTML comments |
| `case:random` | case | seeded random upper/lower casing |
| `whitespace:noise` | whitespace | seeded double-spaces, tabs, newlines |
| `multiturn:split` | multiturn | splits the seed into part 1 + a continuation cue, mimicking a multi-message attack |

⚠️ = attack-framing: applied to attack seeds only, never to benign controls.

## Evals

`evals/run_evals.py` runs 23 golden checks: corpus size range, seed/category coverage, K=8/F=4 thresholds, lineage replay exactness over all 969 variants, double-run md5 determinism, byte-identical JSONL export, benign-stays-benign (no framing markers, no attack-framing lineage), novelty bounds and per-category floors, pairwise min distance > 0, dedupe, id uniqueness, seed-file validity, secret sweep, family usage, pair-variant presence, and report-serialization determinism. Results land in `evals/eval_report.json` — sorted keys, rounded floats, no timestamps — byte-identical across runs (md5 `bc6a475e87b9da4e0c26254f316f35f3`).

## Dev loop — real bugs the tests caught

1. **Novelty measured on canonical text hid invisible edits.** The first novelty implementation computed char-3-gram and word metrics on `canonical()` text — which strips zero-width characters and collapses whitespace. Result: every `encoding:zero-width` variant scored novelty **0.0** and the corpus pairwise minimum distance was 0.0 — a stealth transform looked identical to its seed. Fixed by measuring novelty on raw lowercased text and reserving canonical form for dedupe only.
2. **Degenerate `synonym:swap` variants duplicated their seeds.** When a seed contained no words from the synonym map, the transform returned the seed unchanged — 20 zero-value variants with novelty 0.0. The generator now drops any variant whose text is identical to its seed. Bonus: this freed dedupe slots, so previously-collapsed near-duplicates (case/whitespace variants) survived and per-category minimum novelty rose above zero.
3. **Coverage check verified families but never the K variant count.** The first `check_thresholds` draft only confirmed ≥F families per category — `ensure_coverage` could have "passed" a corpus meeting family diversity but not the ≥K variants requirement. Rewrote it as one function checking both, against explicit per-category variant counts.

## Sample-data note

Every seed and variant is fictional, targeting the invented "Helios Home" assistant with fake data (fake PINs, fake codes, demo accounts). Nothing here instructs real wrongdoing against real systems — these are classic eval-style adversarial phrasings ("ignore your previous instructions and…") of the kind safety teams use to grade models. A pre-commit secret sweep (`no_secrets_in_repo` eval + a manual grep) confirms no API keys, tokens, or personal data anywhere in the repo.

## Layout

```
seeds/attacks.json  seeds/benign.json   32 fictional seeds
synth/transforms.py    17 deterministic transforms + canonical()
synth/generator.py     Synthesizer: singles + pairs, dedupe, replay
synth/novelty.py       3-metric novelty + pairwise min distance
synth/coverage.py      category x family matrix + K/F thresholds
synth/corpus.py        deterministic JSONL export/import
synth/cli.py           seeds | generate | novelty | coverage | demo | serve
demo.py                one-command end-to-end demo
tests/                 106 stdlib unittest tests
evals/run_evals.py     23 golden evals -> evals/eval_report.json
```
