"""Main-flow wording: no technical terms on the pages a clinician sees.

Scans every string in `app/copy.py` and every string literal in the main-flow page files and
the modules that build their text. Technical terms belong on "Behind the model" only.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
MAIN_FLOW_FILES = [APP / "views" / f for f in ("home.py", "suggest.py", "price_check.py",
                                               "progress.py", "about.py")]
MAIN_FLOW_FILES += [APP / f for f in ("copy.py", "presenters.py", "charts.py", "pdf.py",
                                      "widgets.py")]

# (pattern, case-sensitive?)
BANNED = [
    (r"counterfactual", False), (r"\bCATE\b", True), (r"\bITE\b", True),
    (r"treatment[ -]effect", False), (r"heterogene", False), (r"\bSHAP\b", False),
    (r"fuzzy", False), (r"defuzzif", False), (r"membership", False), (r"propensit", False),
    (r"\bIPW\b", False), (r"policy value", False), (r"\bPEHE\b", False), (r"bootstrap", False),
    (r"confidence interval", False), (r"p-value", False), (r"\bMCID\b", False),
    (r"meta[- ]learner", False), (r"\b(S|T|X|DR)[- ]learner", False),
    (r"causal forest", False), (r"TARNet", False), (r"\bHMM\b", False), (r"Viterbi", False),
    (r"\bARIMA\b", False), (r"\bequipoise\b", True), (r"\bEQUIPOISE\b", True),
    (r"\bETDRS\b", False), (r"\bCST\b", True), (r"baseline", False),
]


def _strings_in_module_values(obj, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings_in_module_values(v, out)
    elif isinstance(obj, list | tuple | set):
        for v in obj:
            _strings_in_module_values(v, out)


def _copy_strings() -> list[str]:
    from app import copy as C

    out: list[str] = []
    for name in dir(C):
        if not name.startswith("_") and name.isupper():
            _strings_in_module_values(getattr(C, name), out)
    return out


def _literals(path: Path) -> list[str]:
    """String literals that are not dict keys / subscripts / docstrings (user-facing text)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            skip.add(id(node.slice))
        if isinstance(node, ast.Dict):
            skip.update(id(k) for k in node.keys if k is not None)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop", "setdefault") and node.args
                and isinstance(node.args[0], ast.Constant)):
            skip.add(id(node.args[0]))       # dict keys of API responses / session state
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


def _hits(texts: list[str]) -> list[tuple[str, str]]:
    found = []
    for t in texts:
        for pat, cs in BANNED:
            if re.search(pat, t, 0 if cs else re.IGNORECASE):
                found.append((pat, t))
    return found


def test_copy_has_no_banned_words() -> None:
    strings = _copy_strings()
    assert len(strings) > 50
    assert _hits(strings) == []


@pytest.mark.parametrize("path", [p for p in MAIN_FLOW_FILES if p.is_file()],
                         ids=lambda p: p.name)
def test_main_flow_files_have_no_banned_words(path: Path) -> None:
    assert _hits(_literals(path)) == []


def test_banned_list_catches_jargon() -> None:
    assert _hits(["Equipoise at MCID 5", "SHAP waterfall", "95% bootstrap interval",
                  "the equipoise flag", "baseline vision", "CST 400"])
    assert not _hits(["Equipoise", "No clear winner", "retinal thickness"])
