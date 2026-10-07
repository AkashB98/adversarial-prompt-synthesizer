"""JSONL corpus export / import with lineage + novelty scores.

Export is fully deterministic: fixed field order, sorted JSON keys,
rounded floats, UTF-8. Exporting the same corpus twice yields
byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def export_jsonl(path: str | Path, variants, novelty: Dict[str, float]) -> Path:
    path = Path(path)
    lines = []
    for v in variants:
        record = v.to_dict(novelty.get(v.id))
        lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def import_jsonl(path: str | Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
