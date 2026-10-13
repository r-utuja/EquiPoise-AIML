"""Behind the model: technical details for reviewers and the curious.

Everything technical that used to be in the main flow lives here: the full per-patient
output (table, point-range chart, fuzzy gauge, rule trace, SHAP waterfall, break-even
curve, early-response chart) and the pipeline's aggregate results from reports/. Technical
terms are allowed on this page only; the glossary maps them to the words used elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import presenters as P
from app import services
from equipoise.serving import figures

SS = st.session_state
CFG = services.CFG

GLOSSARY = [
    ("Counterfactual outcome / CATE / ITE", "how much each medicine is likely to help this "
     "patient; how much more it's likely to help"),
    ("Treatment effect (drug − bevacizumab)", "how much more it's likely to help than Avastin"),
    ("95% bootstrap interval of the effect", "likely range (\"compared with Avastin: likely "
     "between X and Y lines\")"),
    ("P(benefit > MCID)", "how sure we are (confidence words, thresholds in config `ui:`)"),
    ("Equipoise (no drug's interval lower bound ≥ MCID)", "no clear winner"),
    ("MCID = 5 ETDRS letters", "1 line on the eye chart"),
    ("ETDRS letters", "lines on the eye chart (1 line = 5 letters), Snellen 20/x"),
    ("Baseline", "today / at the start"),
    ("CST (central subfield thickness)", "retinal thickness from the OCT report"),
    ("Fuzzy recommendation strength (0–100)", "Weak / Moderate / Strong case for paying more"),
    ("SHAP contributions (Eylea − Avastin effect)", "what mattered most"),
    ("Expert-system rule trace", "how we decided"),
    ("Break-even price (Newton-Raphson)", "\"is the extra cost worth it?\""),
    ("Willingness to pay per unit of U(letters)", "how much one extra line is worth (₹)"),
    ("Early-response model (weeks 4/8/12)", "progress check"),
]
TABS = ["Data", "Models", "Treatment-effect evaluation", "Decision logic", "Explainability",
        "Time series & HMM", "MLOps"]


@st.cache_data(show_spinner=False)
def report_dirs() -> tuple[str, str, bool]:
    tables = CFG.paths.tables_dir
    figs = CFG.paths.figures_dir
    if (tables / "m2_table1.md").is_file():
        return str(tables), str(figs), False
    base = tables.parent / "synthetic"
    return str(base / "tables"), str(base / "figures"), True


OPEN_TABLES = {"m2_replication_vs_published", "m8_heldout_no_clear_winner"}


def table(name: str, title: str | None = None) -> None:
    """A report table in a collapsed expander (key results open), so tabs stay scannable."""
    md = Path(report_dirs()[0]) / f"{name}.md"
    with st.expander(f"{title or name} · {name}", expanded=name in OPEN_TABLES):
        st.markdown(md.read_text(encoding="utf-8") if md.is_file() else "_not built_")


def figure(name: str, caption: str) -> None:
    path = Path(report_dirs()[1]) / f"{name}.png"
    if path.is_file():
        st.image(str(path), caption=f"{caption} · {name}.png", width="stretch")


def figures_grid(items: list[tuple[str, str]]) -> None:
    cols = st.columns(2)
    for i, (name, cap) in enumerate(items):
        with cols[i % 2]:
            figure(name, cap)


def show(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def current() -> dict[str, Any] | None:
    """This session's patient and prediction (with SHAP if it has been computed)."""
    feats = SS.get(services.PATIENT)
    if feats is None:
        return None
    return services.cached_prediction(feats, explain=True) or services.cached_prediction(feats)


def estimates_table(pred: dict[str, Any]) -> pd.DataFrame:
    unit = "units" if pred["prices_are_demo"] else "₹"
    rows = []
    for e in pred["estimates"]:
        rows.append({
            "Drug": e["drug"], "Letters (point)": round(e["letters"], 1),
            "Effect vs reference (95% interval)": "reference" if e["benefit_low"] is None
            else f"{e['benefit_vs_reference']:+.1f} ({e['benefit_low']:+.1f} to "
                 f"{e['benefit_high']:+.1f})",
            "P(benefit > MCID)": None if e["p_benefit_gt_mcid"] is None
            else round(e["p_benefit_gt_mcid"], 2),
            "Injections yr 1": round(e["injections_yr1"], 1),
            "P(laser yr 1)": round(e["p_laser_yr1"], 2),
            f"Cost ({unit})": round(e["cost"], 1),
            f"{unit} per extra letter": None if e["cost_per_extra_letter"] is None
            else round(e["cost_per_extra_letter"], 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page
st.title("Behind the model")
st.caption("Technical details — for reviewers and the curious. The main pages use plain "
           "words for everything below.")
services.notice()
tables_dir, _, synthetic_reports = report_dirs()
if synthetic_reports:
    st.warning("Real report assets not found: showing synthetic-data results.")

with st.container(border=True):
    st.markdown("**Glossary: technical term → words used in the app**")
    st.markdown("| Technical term | In the app |\n|---|---|\n" + "\n".join(
        f"| {a} | {b} |" for a, b in GLOSSARY))

summary = Path(tables_dir) / "m8_heldout_no_clear_winner.csv"
if summary.is_file():
    s = pd.read_csv(summary).set_index("quantity")["value"]
    n = int(s["n_heldout"])
    eq5 = int(s.get("n_no_clear_winner_mcid_5", -1))
    eq3 = int(s.get("n_no_clear_winner_mcid_3", -1))
    st.info(f"**Key finding.** On the held-out test set, the decision layer calls **{eq5} of "
            f"{n}** patients *no clear winner* (equipoise) at MCID 5 letters and **{eq3} of {n}** "
            f"at MCID 3. Max P(benefit > MCID) is {s['p_gt_mcid_max']:.2f}; the median width of "
            f"the effect interval is {s['interval_width_letters_median']:.1f} letters. With about "
            "220 patients per arm, Protocol T supports the published baseline-vision subgroup "
            "finding but is too small to single out individual patients who clearly need the "
            "costlier drug. (`m8_heldout_no_clear_winner`, written by "
            "`scripts/pick_example_patients.py`)")

pred = current()
health = services.health() if services.model_ready() else None
tabs = st.tabs(TABS)

with tabs[0]:   # Data
    table("m2_table1", "Table 1: baseline balance by arm")
    table("m2_replication_vs_published", "Replication vs the published trial")
    table("m2_replication_tests", "Hypothesis tests")
    figures_grid([("m2_replication_means", "Mean letters gained per arm (95% CI)"),
                  ("m2_outcome_by_arm", "Outcome distributions by arm"),
                  ("m3_pca_map", "PCA patient map (exploratory)"),
                  ("m3_cluster_effects", "Treatment effect within clusters (exploratory)")])
    st.markdown("**Held-out test patients** (not used for training; observed outcome shown "
                "for context only)")
    if health is not None:
        from equipoise.serving.service import held_out_patients

        tp = held_out_patients(CFG, bool(health["synthetic_bundle"]))
        if tp.empty:
            st.caption("No local data found to load test patients from.")
        else:
            y = CFG.primary_outcome
            labels = [f"#{i}: received {r.treatment}, observed {r[y]:+.0f} letters"
                      if pd.notna(r[y]) else f"#{i}: received {r.treatment}, no week-52 vision"
                      for i, r in tp.iterrows()]
            i = st.selectbox("Held-out patient", range(len(tp)), format_func=labels.__getitem__)
            if st.button("Load into Get a suggestion"):
                from equipoise.decision.recommend import FEATURES
                from equipoise.serving.examples import feature_dict

                SS[services.PATIENT] = services.clean_features(feature_dict(tp.iloc[i], FEATURES))
                SS[services.VA_EXACT] = True
                SS.pop(services.EXAMPLE, None)
                services.clear_widgets()
                services.go_to_wizard(step=1)

with tabs[1]:   # Models
    table("m5_regression_comparison", "M5 outcome models: week-52 letters")
    table("m5_classification_gain15", "M5 classification: gain ≥ 15 letters")
    table("m5_classification_laser_yr1", "M5 classification: laser in year 1")
    table("m5_count_comparison", "M5 count models: injections in year 1")
    figures_grid([("m5_regression_pred_vs_obs", "Predicted vs observed letters"),
                  ("m5_roc_pr_curves", "ROC and PR curves"),
                  ("m5_confusion_gain15", "Confusion matrix: gain ≥ 15 letters"),
                  ("m5_confusion_laser_yr1", "Confusion matrix: laser in year 1")])

with tabs[2]:   # Treatment-effect evaluation
    if pred is not None:
        st.markdown("**This patient (current session)**")
        st.caption(f"Model: {pred['model']} · prices: {pred['prices_label']} · equipoise at "
                   f"MCID {pred['mcid_letters']:g}: {pred['equipoise']} · sensitivity: "
                   f"{pred['equipoise_at_sensitivity']} · published < 69-letter rule suggests "
                   f"{pred['published_rule_suggests']}")
        show(figures.point_range(pred))
        st.dataframe(estimates_table(pred), hide_index=True, width="stretch")
        if pred.get("imputed_features"):
            st.caption("Imputed: " + ", ".join(f.replace("X_", "")
                                               for f in pred["imputed_features"]))
    table("m8_heldout_no_clear_winner", "Held-out patients: no clear winner (equipoise)")
    table("m6_learner_selection", "M6 learner selection (semi-synthetic PEHE)")
    table("m6_policy_values", "Policy values on held-out patients (IPW, bootstrap 95% CI)")
    table("m6_rediscovery", "Rediscovery check")
    table("m6_uncertainty_summary", "Per-patient interval widths")
    figures_grid([("m6_semisynthetic_pehe", "Semi-synthetic error (PEHE) by method"),
                  ("m6_policy_values", "Policy values with ranges"),
                  ("m6_calibration_aflibercept", "Calibration: aflibercept"),
                  ("m6_calibration_ranibizumab", "Calibration: ranibizumab"),
                  ("m6_effect_distribution", "Distribution of predicted effects")])

with tabs[3]:   # Decision logic
    if pred is not None:
        st.markdown("**This patient: full rule-engine trace (expert system)**")
        st.code("\n".join(pred["rule_trace"]), language=None)
        if pred["fuzzy_strength"]:
            show(figures.strength_gauge(pred["fuzzy_strength"]))
        if health is not None:
            try:
                be = services.breakeven(SS[services.PATIENT], "Aflibercept")
            except Exception:   # noqa: BLE001 - optional figure
                be = None
            if be and be["curve"]:
                show(figures.breakeven_curve(be))
                res = be.get("result") or {}
                st.caption(f"Break-even price {res.get('break_even_price')}, break-even benefit "
                           f"{res.get('break_even_benefit_letters')} letters "
                           f"({be['prices_label']}). {res.get('message', '')}")
    st.markdown("**Break-even derivation.** Net benefit of a drug = WTP · U(letters) − expected "
                "cost, with U(letters) = 1 − exp(−k·letters), k = "
                f"{CFG.decision['value_of_vision_k']} (concave, saturating). The break-even price "
                "is the dearer drug's price per injection at which its net benefit equals "
                "bevacizumab's; the break-even benefit is the extra letters at which it would. "
                "Both roots are found with Newton-Raphson (finite-difference derivative) and "
                "checked against scipy's brentq (`m8_newton_vs_brentq`). In the app, the user "
                "enters the value of one extra line V (₹); the app sets WTP = V / (1 − "
                "exp(−5k)), i.e. the first line gained (0 → 5 letters) is worth V.")
    table("m8_fuzzy_rules", "Fuzzy rules")
    table("m8_newton_vs_brentq", "Newton-Raphson vs brentq")
    table("m8_population_summary", "Decision layer on held-out patients (DEMO prices)")
    figures_grid([("m8_fuzzy_membership", "Fuzzy membership functions"),
                  ("m8_fuzzy_worked_example", "Fuzzy inference, worked example"),
                  ("m8_break_even", "Break-even curve, typical patient"),
                  ("m8_strength_vs_benefit", "Strength vs predicted benefit")])

with tabs[4]:   # Explainability
    if pred is not None:
        if pred.get("explanation"):
            st.markdown("**This patient: SHAP waterfall (aflibercept − bevacizumab effect)**")
            show(figures.waterfall(pred["explanation"]))
            st.caption(pred["explanation"]["note"])
        else:
            st.caption("Open step 3 of Get a suggestion to compute this patient's SHAP values.")
    table("m9_subgroups", "Subgroup checks")
    figures_grid([("m9_shap_importance", "Mean |SHAP| importance"),
                  ("m9_shap_beeswarm", "SHAP beeswarm"),
                  ("m9_shap_waterfall_typical", "SHAP waterfall, typical patient"),
                  ("m9_subgroups", "Subgroup performance")])

with tabs[5]:   # Time series & HMM
    payload = SS.get("progress_payload")
    if payload and health is not None:
        try:
            er = services.early_response(payload)
            st.markdown("**This patient: early-response forecast**")
            show(figures.early_trajectory(er))
            st.caption(er["note"])
        except Exception:   # noqa: BLE001 - optional figure
            pass
    table("m7_early_response", "Early-response model (weeks 4/8/12)")
    table("m7_hmm_states", "HMM states")
    table("m7_arima_forecast", "ARIMA forecast")
    figures_grid([("m7_early_response", "Early response"),
                  ("m7_mean_trajectories", "Mean trajectories by arm"),
                  ("m7_hmm", "HMM + Viterbi states"),
                  ("m7_time_series", "Arm-mean time series")])

with tabs[6]:   # MLOps
    if health is not None:
        st.markdown(f"- **Model:** {health['model']}\n- **Mode:** {services.client().mode}\n"
                    f"- **Synthetic bundle:** {health['synthetic_bundle']}\n"
                    f"- **Training data hash:** `{health['training_data_hash']}`\n"
                    f"- **Prices:** {health['prices_label']}\n"
                    f"- **SHAP assets:** {health['shap_available']} · **early-response model:** "
                    f"{health['early_response_available']}")
    st.markdown("- **MLflow:** `mlflow ui --backend-store-uri sqlite:///mlflow.db`\n"
                "- **DVC:** `dvc repro` (pipeline in `dvc.yaml`, params from "
                "`config/default.yaml`)\n"
                "- **CI:** `.github/workflows/ci.yml` (ruff, pytest on synthetic data, Docker "
                "builds)\n- **Report assets index:** `reports/REPORT_ASSETS.md`")
    table("m11_tech_stack", "Tech stack")
    table("m11_experiments", "Experiments")
    st.caption(f"Snellen conversion used by the app: letters ≈ 85 + 50·log10(Snellen "
               f"fraction), clamped to the trial range "
               f"{CFG.va_eligibility_letters[0]}–{CFG.va_eligibility_letters[1]}. "
               f"Lines are rounded to {P.num(float(services.UI['lines_step']))}.")
