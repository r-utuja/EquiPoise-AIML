"""Handbook helper: check that every concept in docs/handbook/_inventory.md is explained.

For each inventory row, the concept name is split into key phrases (on commas, slashes,
semicolons, parentheses and "vs"); backticks and Markdown are stripped. A concept counts as
covered if one key phrase (3+ characters) appears verbatim, or all of its significant words
appear, in the handbook chapters 00-31
(the glossary and concept index are excluded, because the index repeats every concept by
construction). Prints the gaps and exits 1 if there are any.

Usage:
    python scripts/handbook/check_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "handbook" / "_inventory.md"
HANDBOOK = ROOT / "docs" / "HANDBOOK.md"


def chapters_text() -> str:
    text = HANDBOOK.read_text(encoding="utf-8")
    end = text.find("\n## 32. Glossary")
    return (text if end < 0 else text[:end]).lower()


def concepts() -> list[tuple[str, str]]:
    out = []
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Proposed handbook outline"):
            break
        if not line.startswith("| ") or line.startswith("| Concept"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and not set(cells[0]) <= set("-: "):
            out.append((cells[0], cells[3]))
    return out


STOP = {"with", "from", "into", "only", "used", "uses", "every", "each", "their", "this",
         "that", "test", "per", "the", "and", "for", "vs"}


def phrases(concept: str) -> list[str]:
    s = concept.replace("`", "").replace("*", "").replace('"', "")
    parts = re.split(r"[,/;()]| vs | and |: |→", s)
    return [p.strip().lower() for p in parts if len(p.strip()) >= 3]


def covered(concept: str, text: str) -> bool:
    """A key phrase appears verbatim, or all its significant words appear in the text."""
    for p in phrases(concept):
        if p in text:
            return True
        words = [w for w in re.findall(r"[a-z0-9_.²ᵀ]+", p) if len(w) >= 4 and w not in STOP]
        if words and all(w in text for w in words):
            return True
    return False


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    text = chapters_text()
    rows = concepts()
    gaps = [(c, ch) for c, ch in rows if not covered(c, text)]
    print(f"{len(rows)} concepts checked; {len(rows) - len(gaps)} covered; {len(gaps)} gaps")
    for c, ch in gaps:
        print(f"  GAP: {c}  (planned chapter {ch})")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
