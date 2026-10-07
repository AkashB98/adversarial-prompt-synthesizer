"""Command-line interface for the adversarial prompt synthesizer.

Commands:
    seeds     list bundled seed prompts
    generate  build the variant corpus and write it as JSONL
    novelty   print per-category novelty summary for a corpus
    coverage  print the category x family coverage matrix
    demo      run the end-to-end demonstration
    serve     tiny JSON API: POST /generate, GET /seeds

Exit codes: 0 success, 1 check/threshold failure, 2 usage or runtime error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synth.transforms import TRANSFORMS, LABEL_TO_META, FAMILIES  # noqa: E402
from synth.generator import Synthesizer  # noqa: E402
from synth.novelty import score_corpus  # noqa: E402
from synth.coverage import coverage_matrix, variants_per_category, check_thresholds  # noqa: E402
from synth.corpus import export_jsonl, import_jsonl  # noqa: E402


def load_seeds() -> list:
    attacks = json.loads((ROOT / "seeds" / "attacks.json").read_text(encoding="utf-8"))
    benign = json.loads((ROOT / "seeds" / "benign.json").read_text(encoding="utf-8"))
    return attacks + benign


def build_corpus(seed_value: int, pair_budget: int):
    seeds = load_seeds()
    synth = Synthesizer(seed_value)
    variants, budget = synth.ensure_coverage(seeds, start_budget=pair_budget)
    seeds_by_id = {s["id"]: s for s in seeds}
    novelty = score_corpus(variants, seeds_by_id)
    return seeds, synth, variants, novelty, budget


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_seeds(args) -> int:
    seeds = load_seeds()
    attacks = [s for s in seeds if s["kind"] == "attack"]
    benign = [s for s in seeds if s["kind"] == "benign"]
    print(f"attack seeds : {len(attacks)} across "
          f"{len({s['category'] for s in attacks})} categories")
    print(f"benign seeds : {len(benign)}")
    print(f"transforms   : {len(TRANSFORMS)} across {len(FAMILIES)} families")
    for s in seeds:
        print(f"  [{s['kind']:6s}] {s['id']} ({s['category']})")
    return 0


def cmd_generate(args) -> int:
    seeds, synth, variants, novelty, budget = build_corpus(args.seed_value, args.pair_budget)
    out = Path(args.out)
    export_jsonl(out, variants, novelty)
    digest = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"seeds        : {len(seeds)}")
    print(f"variants     : {len(variants)}")
    print(f"pair budget  : {budget}")
    print(f"wrote        : {out}")
    print(f"md5          : {digest}")
    return 0


def cmd_novelty(args) -> int:
    records = import_jsonl(args.corpus)
    by_cat: dict = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["novelty"])
    print(f"{'category':32s} {'n':>5s} {'mean':>8s} {'min':>8s}")
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        print(f"{cat:32s} {len(vals):5d} {sum(vals)/len(vals):8.4f} {min(vals):8.4f}")
    return 0


def cmd_coverage(args) -> int:
    records = import_jsonl(args.corpus)
    matrix = coverage_matrix(records_as_variants(records))
    counts = variants_per_category(records_as_variants(records))
    families = sorted({f for row in matrix.values() for f in row})
    header = f"{'category':28s}" + "".join(f"{f:>10s}" for f in families)
    print(header)
    for cat in sorted(matrix):
        print(f"{cat:28s}" + "".join(f"{matrix[cat].get(f, 0):10d}" for f in families))
    ok, failures = check_thresholds(matrix, counts, k=args.k, f=args.f)
    for fail in failures:
        print("FAIL:", fail)
    print(f"thresholds K={args.k} F={args.f}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def records_as_variants(records):
    """Lightweight stand-ins exposing the attributes coverage needs."""
    from types import SimpleNamespace
    return [SimpleNamespace(category=r["category"], lineage=r["lineage"]) for r in records]


def cmd_demo(args) -> int:
    from demo import main as demo_main
    demo_main()
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/seeds":
            self._json({"seeds": load_seeds()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/generate":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON"}, 400)
            return
        text = req.get("text", "")
        labels = req.get("transforms", [])
        unknown = [l for l in labels if l not in LABEL_TO_META]
        if unknown:
            self._json({"error": f"unknown transforms: {unknown}"}, 400)
            return
        synth = Synthesizer(req.get("seed_value", 1337))
        out, applied = text, []
        for step, label in enumerate(labels):
            fn = LABEL_TO_META[label][0]
            out, got = fn(out, synth._rng("serve", step))
            applied.append(got)
        self._json({"text": out, "lineage": applied})

    def log_message(self, *a):
        pass


def build_server(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def cmd_serve(args) -> int:
    try:
        server = build_server(args.port)
    except OSError as exc:
        print(f"error: cannot bind 127.0.0.1:{args.port} ({exc})", file=sys.stderr)
        return 2
    print(f"serving on http://127.0.0.1:{server.server_address[1]} "
          f"(POST /generate, GET /seeds) — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synth", description="Adversarial prompt synthesizer")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("seeds", help="list bundled seed prompts")

    g = sub.add_parser("generate", help="build the variant corpus")
    g.add_argument("--seed-value", type=int, default=1337)
    g.add_argument("--pair-budget", type=int, default=24)
    g.add_argument("--out", default="corpus.jsonl")

    n = sub.add_parser("novelty", help="novelty summary for a corpus file")
    n.add_argument("--corpus", default="corpus.jsonl")

    c = sub.add_parser("coverage", help="coverage matrix for a corpus file")
    c.add_argument("--corpus", default="corpus.jsonl")
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--f", type=int, default=4)

    sub.add_parser("demo", help="run the end-to-end demonstration")

    s = sub.add_parser("serve", help="tiny JSON API on localhost")
    s.add_argument("--port", type=int, default=8137)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "seeds":
            return cmd_seeds(args)
        if args.command == "generate":
            return cmd_generate(args)
        if args.command == "novelty":
            return cmd_novelty(args)
        if args.command == "coverage":
            return cmd_coverage(args)
        if args.command == "demo":
            return cmd_demo(args)
        if args.command == "serve":
            return cmd_serve(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
