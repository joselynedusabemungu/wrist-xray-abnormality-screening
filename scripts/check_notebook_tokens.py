from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATTERN = re.compile("(" + re.escape("{" + "{") + "|" + re.escape("<" + "<") + ")")


def scan(path: Path) -> list[str]:

    nb = json.loads(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        sources = ["".join(cell.get("source", []))]
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain") or ""
            sources.append("".join(text) if isinstance(text, list) else str(text))
        for chunk in sources:
            for m in PATTERN.finditer(chunk):
                start = max(m.start() - 30, 0)
                hits.append(
                    f"cell {i} ({cell['cell_type']}): ...{chunk[start:m.end() + 30]!r}..."
                )
    return hits


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: notebook not found: {path}", file=sys.stderr)
        return 1
    hits = scan(path)
    if hits:
        print(
            f"FAIL: {len(hits)} placeholder token(s) remain in {path}:", file=sys.stderr
        )
        for h in hits:
            print("  " + h, file=sys.stderr)
        return 1
    print(f"OK: no placeholder tokens in {path} (sources and outputs scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
