"""Handbook helper: map every imported name to the file:function where it is used.

Writes one line per name (`library.name <TAB> file:function; ...`) to the path given as the
first argument, or to `<temp dir>/equipoise_usage_out.tsv`. Read-only: it parses the tracked
Python files with `ast` and changes nothing. Used to build docs/handbook/_inventory.md.

Usage:
    python scripts/handbook/scan_usage.py [OUTPUT]
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = ("equipoise", "app.", "__future__", "typing", "tests.")


def import_names(tree: ast.AST) -> dict[str, str]:
    """Local name -> fully qualified imported name."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                names[a.asname or top] = a.name if a.asname else top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                names[a.asname or a.name] = f"{node.module}.{a.name}"
    return names


def collect(tree: ast.AST, rel: str, names: dict[str, str],
            usage: dict[str, set[str]]) -> None:
    """Record `name -> file:scope` for every use of an imported name."""
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, scope = stack.pop()
        for child in ast.iter_child_nodes(node):
            s = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                s = f"{scope}.{child.name}" if scope else child.name
            where = f"{rel}:{s or '<module>'}"
            if isinstance(child, ast.Attribute):
                chain, cur = [], child
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in names:
                    usage[names[cur.id] + "." + ".".join(reversed(chain))].add(where)
            elif (isinstance(child, ast.Name) and child.id in names
                  and isinstance(child.ctx, ast.Load)):
                usage[names[child.id]].add(where)
            stack.append((child, s))


def main() -> None:
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    usage: dict[str, set[str]] = defaultdict(set)
    for rel in files:
        if rel.startswith("scripts/handbook/"):
            continue
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        collect(tree, rel, import_names(tree), usage)
    default = Path(tempfile.gettempdir()) / "equipoise_usage_out.tsv"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    with out.open("w", encoding="utf-8") as f:
        for k in sorted(usage):
            if not k.startswith(SKIP):
                f.write(f"{k}\t{'; '.join(sorted(usage[k]))}\n")
    print(out)


if __name__ == "__main__":
    main()
