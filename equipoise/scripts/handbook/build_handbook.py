"""Handbook helper: rebuild the table of contents of docs/HANDBOOK.md and check its links.

The handbook is a single Markdown file. This script:
  1. regenerates the linked table of contents (between the "## Table of contents" heading and
     the first "---" line) from the chapter headings "## NN. Title" and the Part headings;
  2. checks every internal link (#anchor) points at a heading that exists, using GitHub's
     anchor rules (lower-case, punctuation removed, spaces -> hyphens);
  3. checks every relative image/file link points at a file that exists;
  4. prints a word count per chapter.

Usage:
    python scripts/handbook/build_handbook.py           # rebuild TOC + report
    python scripts/handbook/build_handbook.py --check   # report only, do not write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDBOOK = ROOT / "docs" / "HANDBOOK.md"

PARTS = {  # first chapter number of each part -> part title
    0: "Part A: The big picture",
    5: "Part B: Foundations from scratch",
    8: "Part C: The pipeline, module by module",
    20: "Part D: How it's built and how it runs",
    25: "Part E: Results and honesty",
    29: "Part F: Report and viva pack",
}
CHAPTER = re.compile(r"^## (\d\d)\. (.+)$")


def anchor(title: str) -> str:
    """GitHub-style heading anchor."""
    a = title.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def strip_code(text: str) -> str:
    """Text with fenced code blocks blanked out (headings/links inside code don't count)."""
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            out.append("")
            continue
        out.append("" if fence else line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = HANDBOOK.read_text(encoding="utf-8")
    plain = strip_code(text)

    chapters = [(int(m.group(1)), m.group(2)) for line in plain.splitlines()
                if (m := CHAPTER.match(line))]
    toc = ["## Table of contents", ""]
    for num, title in chapters:
        if num in PARTS:
            toc += ["", f"*{PARTS[num]}*", ""]
        toc.append(f"- [{num:02d}. {title}](#{anchor(f'{num:02d}. {title}')})")
    toc_text = "\n".join(toc).replace("\n\n\n", "\n\n") + "\n"
    start = text.index("## Table of contents")
    end = text.index("\n---\n", start)
    new = text[:start] + toc_text + text[end:]
    if not args.check and new != text:
        HANDBOOK.write_text(new, encoding="utf-8")
        text, plain = new, strip_code(new)

    # anchors that exist
    anchors = {anchor(re.sub(r"^#+ ", "", line)) for line in plain.splitlines()
               if line.startswith("#")}
    problems = []
    for m in re.finditer(r"\]\((#[^)]+)\)", plain):
        if m.group(1)[1:] not in anchors:
            problems.append(f"broken anchor {m.group(1)}")
    for m in re.finditer(r"\]\(((?!https?:|#|mailto:)[^)\s]+)\)", plain):
        target = (HANDBOOK.parent / m.group(1).split("#")[0]).resolve()
        if not target.exists():
            problems.append(f"missing file {m.group(1)}")

    # words per chapter
    print(f"{'chapter':<60} words")
    pieces = re.split(r"(?m)^(?=## \d\d\. )", text)
    total = 0
    for p in pieces:
        m = CHAPTER.match(p.splitlines()[0]) if p.strip() else None
        n = len(p.split())
        total += n
        print(f"{(m.group(0)[3:] if m else '(front matter)'):<60} {n:>6}")
    print(f"{'TOTAL':<60} {total:>6}  (~{total / 300:.0f} pages at 300 words/page)")
    for pr in problems:
        print("PROBLEM:", pr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
