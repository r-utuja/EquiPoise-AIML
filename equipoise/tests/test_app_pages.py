"""Smoke tests for the Streamlit app with streamlit.testing.v1.AppTest.

Runs against the light synthetic bundle from conftest.py: every page renders without an
exception, and each "Try an example" preset produces a headline and three drug cards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import copy as C

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
MAIN_PAGES = ["views/home.py", "views/suggest.py", "views/price_check.py",
              "views/progress.py", "views/about.py", "views/behind.py"]


@pytest.fixture()
def app_test(light, bundle, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from equipoise.decision.recommend import default_bundle_path

    monkeypatch.setenv("EQUIPOISE_BUNDLE", str(default_bundle_path(light, synthetic=True)))
    monkeypatch.setenv("EQUIPOISE_REQUEST_LOG", "off")
    monkeypatch.delenv("API_URL", raising=False)
    return lambda: AppTest.from_file(str(APP), default_timeout=180).run()


def texts(at) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def click(at, label: str):
    return next(b for b in at.button if b.label == label).click().run()


@pytest.mark.parametrize("page", MAIN_PAGES)
def test_every_page_renders(app_test, page: str) -> None:
    at = app_test()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    assert C.NOTICE in texts(at)


@pytest.mark.parametrize("example", ["poorer", "moderate", "good"])
def test_example_gives_headline_and_three_cards(app_test, example: str) -> None:
    at = app_test()
    at.switch_page("views/suggest.py").run()
    at.button(key=f"ex_{example}").click().run()
    assert not at.exception
    assert C.EXAMPLE_LOADED.format(name=C.EXAMPLE_NAMES[example]) in texts(at)
    at = click(at, C.BTN_COMPARE)
    assert not at.exception, at.exception
    assert at.session_state["step"] == 2
    page = texts(at)
    headlines = [m.value for m in at.markdown if 'class="eq-headline"' in m.value]
    assert len(headlines) == 1
    assert any(h in headlines[0] for h in ("No clear winner", "is likely to help"))
    for name in ("Avastin (bevacizumab)", "Lucentis (ranibizumab)", "Eylea (aflibercept)"):
        assert page.count(name) >= 1
    assert sum(page.count(s) for s in ("injections in the first year",)) == 3
    assert C.UNCERTAINTY in page


def test_step3_why_and_pdf(app_test) -> None:
    at = app_test()
    at.switch_page("views/suggest.py").run()
    at.button(key="ex_poorer").click().run()
    at = click(at, C.BTN_COMPARE)
    at = click(at, C.BTN_SEE_WHY)
    assert not at.exception, at.exception
    assert at.session_state["step"] == 3
    page = texts(at)
    for title in (C.WHAT_MATTERED_TITLE, C.HOW_DECIDED_TITLE, C.WORTH_TITLE):
        assert title in page
    assert C.COSTS_HIDDEN_NOTE in page            # DEMO prices: no cost figures anywhere
    assert "₹" not in page
    assert any(b.label == C.BTN_BACK_COMPARISON for b in at.button)


def test_price_check_empty_state_then_session_prices(app_test) -> None:
    at = app_test()
    at.switch_page("views/suggest.py").run()
    at.button(key="ex_moderate").click().run()
    at = click(at, C.BTN_COMPARE)
    at.switch_page("views/price_check.py").run()
    assert not at.exception, at.exception
    assert C.COSTS_HIDDEN_NOTE in [i.value for i in at.info]      # nothing from DEMO prices
    assert "₹" not in texts(at).replace("(₹)", "")
    at.session_state["session_prices"] = {
        "per_injection": {"Bevacizumab": 2_000.0, "Ranibizumab": 15_000.0,
                          "Aflibercept": 30_000.0}, "laser": 5_000.0}
    at.session_state["value_per_line"] = 50_000.0
    at.run()
    assert not at.exception, at.exception
    page = texts(at)
    assert C.PRICES_SESSION in page
    assert "per injection" in page and "₹" in page
    assert len(at.slider) == 1


def test_price_check_form_submit(app_test) -> None:
    """Typing prices and pressing the button stores them (the form key must not clash)."""
    at = app_test()
    at.switch_page("views/suggest.py").run()
    at.button(key="ex_moderate").click().run()
    at = click(at, C.BTN_COMPARE)
    at.switch_page("views/price_check.py").run()
    for box, value in zip(at.number_input, (2_000.0, 15_000.0, 30_000.0, 5_000.0),
                          strict=True):
        box.set_value(value)
    at = click(at, C.BTN_USE_PRICES)
    assert not at.exception, at.exception
    assert at.session_state["session_prices"]["laser"] == 5_000.0
    assert C.PRICES_SESSION in texts(at)


def test_progress_check_forecast(app_test) -> None:
    at = app_test()
    at.switch_page("views/progress.py").run()
    at = click(at, C.PROGRESS_BTN)
    assert not at.exception, at.exception
    if C.PROGRESS_UNAVAILABLE in [i.value for i in at.info]:
        return                     # this light test bundle has no early-response model
    page = texts(at)
    assert "On this path, expected vision at 1 year is about" in page
    assert "The dashed line shows the expected path to 1 year." in page


def test_about_page_uses_the_replication_table(app_test) -> None:
    at = app_test()
    at.switch_page("views/about.py").run()
    page = texts(at)
    assert C.ABOUT_FOUND_TITLE in [s.value for s in at.subheader]
    assert "After 1 year, people gained about" in page or C.ABOUT_FOUND_MISSING in page


def test_behind_the_model_keeps_all_technical_content(app_test) -> None:
    """Everything technical removed from the main flow is still reachable here."""
    src = (APP.parent / "views" / "behind.py").read_text(encoding="utf-8")
    for needle in ("figures.point_range", "figures.strength_gauge", "figures.waterfall",
                   "figures.breakeven_curve", "figures.early_trajectory", "rule_trace",
                   "estimates_table", "held_out_patients", "m2_table1",
                   "m2_replication_vs_published", "m2_replication_tests", "m3_pca_map",
                   "m3_cluster_effects", "m5_regression_comparison", "m5_confusion_gain15",
                   "m6_semisynthetic_pehe", "m6_policy_values", "m6_calibration_aflibercept",
                   "m6_rediscovery", "m8_fuzzy_membership", "m8_fuzzy_rules",
                   "m8_newton_vs_brentq", "m8_heldout_no_clear_winner", "m9_shap_beeswarm",
                   "m9_subgroups", "m7_hmm", "m7_early_response", "mlflow ui", "dvc repro",
                   "GLOSSARY"):
        assert needle in src, needle
    at = app_test()
    at.switch_page("views/suggest.py").run()
    at.button(key="ex_good").click().run()
    at = click(at, C.BTN_COMPARE)
    at.switch_page("views/behind.py").run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 7
    assert any("Glossary" in m.value for m in at.markdown)
    assert at.code and "equipoise" in at.code[0].value.lower()    # full rule trace
