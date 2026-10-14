"""Handbook helper: list imports and top-level functions/classes of every tracked Python file.

Writes a plain-text dump (libraries with the names used from them, then one block per file
with its module docstring and each function/class with the first line of its docstring) to
the path given as the first argument, or to `<temp dir>/equipoise_scan_out.txt`. Read-only.

Usage:
    python scripts/handbook/scan_code.py [OUTPUT]
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tracked_py() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return [ROOT / p for p in out.stdout.split() if not p.startswith("scripts/handbook/")]


def describe(node: ast.AST) -> list[str]:
    """One line per top-level function/class (and per method of a class)."""
    lines = []
    for item in getattr(node, "body", []):
        if isinstance(item, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            first = " ".join((ast.get_docstring(item) or "").split())[:160]
            kind = "class" if isinstance(item, ast.ClassDef) else "def"
            lines.append(f"  {kind} {item.name} (L{item.lineno}): {first}")
            if isinstance(item, ast.ClassDef):
                lines += [f"     .{sub.name} (L{sub.lineno})" for sub in item.body
                          if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return lines


def main() -> None:
    lib_names: dict[str, set[str]] = defaultdict(set)
    lib_files: dict[str, set[str]] = defaultdict(set)
    lines: list[str] = []
    for path in tracked_py():
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    lib_files[a.name.split(".")[0]].add(rel)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for a in node.names:
                    lib_names[node.module.split(".")[0]].add(f"{node.module}.{a.name}")
                    lib_files[node.module.split(".")[0]].add(rel)
        doc = " ".join((ast.get_docstring(tree) or "").split())[:400]
        lines += [f"\n=== {rel}\n  DOC: {doc}", *describe(tree)]
    default = Path(tempfile.gettempdir()) / "equipoise_scan_out.txt"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    stdlib = set(sys.stdlib_module_names)
    with out.open("w", encoding="utf-8") as f:
        f.write("##### LIBRARIES\n")
        for lib in sorted(lib_names.keys() | lib_files.keys()):
            tag = "stdlib" if lib in stdlib else "third-party/local"
            f.write(f"\n[{lib}] ({tag}) files: {', '.join(sorted(lib_files[lib]))}\n")
            f.writelines(f"    {n}\n" for n in sorted(lib_names[lib]))
        f.write("\n##### FILES\n" + "\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
