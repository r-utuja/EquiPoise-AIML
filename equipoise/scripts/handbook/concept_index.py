"""Handbook helper: build the concept index (chapter 33) from docs/handbook/_inventory.md.

Reads every inventory table row (Concept | Category | Where used | Chapter), sorts the
concepts alphabetically and prints a Markdown table whose chapter numbers link to the
handbook's chapter headings in docs/HANDBOOK.md.

Usage:
    python scripts/handbook/concept_index.py > concept_index.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "handbook" / "_inventory.md"
HANDBOOK = ROOT / "docs" / "HANDBOOK.md"


def anchor(title: str) -> str:
    a = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return a.replace(" ", "-")


def chapter_anchors() -> dict[str, str]:
    out = {}
    for line in HANDBOOK.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (\d\d)\. (.+)$", line)
        if m:
            out[m.group(1)] = anchor(f"{m.group(1)}. {m.group(2)}")
    return out


def rows() -> list[tuple[str, str, str]]:
    out = []
    in_outline = False
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Proposed handbook outline"):
            in_outline = True
        if in_outline or not line.startswith("| ") or line.startswith("| Concept"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or set(cells[0]) <= set("-: "):
            continue
        concept, _, where, chapters = cells
        out.append((concept, chapters, where))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    anchors = chapter_anchors()

    def link(chs: str) -> str:
        parts = re.findall(r"\d\d", chs)
        return ", ".join(f"[§{p}](#{anchors[p]})" if p in anchors else f"§{p}" for p in parts)

    print("| Concept | Chapter(s) | Where used |")
    print("|---|---|---|")
    for concept, chs, where in sorted(rows(), key=lambda r: r[0].lower().strip("`")):
        print(f"| {concept} | {link(chs)} | {where} |")


if __name__ == "__main__":
    main()
