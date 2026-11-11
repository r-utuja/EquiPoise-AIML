"""Model comparison (for reviewers): cluster-then-look-up model vs the causal model.

Reads the files written by `python -m equipoise unsupervised`; nothing is computed here.
Held-out test patients only. Shown for reviewers, not part of the clinician flow.
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from app import services
from equipoise.config import for_data_source

CFG = services.CFG
TABLE, PER_PATIENT = "m6_unsupervised_comparison.csv", "m6_unsupervised_heldout.csv"
PRED = "Predicting week-52 letters (test)"

st.title("Model comparison")
services.notice()


def locations() -> tuple[object, object]:
    """Real results unless $EQUIPOISE_SYNTHETIC is set or only synthetic ones exist."""
    real = CFG.paths.tables_dir / TABLE
    if os.environ.get("EQUIPOISE_SYNTHETIC") or not real.is_file():
        c = for_data_source(CFG, True).paths
        return c.tables_dir, c.derived_dir
    return CFG.paths.tables_dir, CFG.paths.derived_dir


tables_dir, derived_dir = locations()
path = tables_dir / TABLE
if not path.is_file():
    st.info("No comparison results yet. Run `python -m equipoise unsupervised` "
            "(add `--synthetic` for the demo data), then reload this page.")
    st.stop()
if "synthetic" in path.parts:
    st.warning("Showing results on synthetic demo data, not the trial data.")

df = pd.read_csv(path)
if not {"method", "unit", "better"} <= set(df.columns):
    st.info("These results are from an older run. Run `python -m equipoise unsupervised` again.")
    st.stop()
st.caption("Two ways to estimate how much each drug helps: a supervised causal model, and a "
           "simple one that groups patients without looking at outcomes and then reads off "
           "each group's average results by drug. Held-out test patients only.")


def val(section: str, metric: str, model_part: str) -> float | None:
    m = df[(df.section.str.startswith(section)) & (df.metric.str.startswith(metric))
           & df.model.str.contains(model_part, regex=False)]
    return float(m.value.iloc[0]) if len(m) and pd.notna(m.value.iloc[0]) else None


SUP, UNS, BASE = "Supervised", "Unsupervised", "Baseline"
# tab: (section prefix, [(measure label, metric prefix)], {method: model-name prefix})
TABS = {
    "1. Predict vision change": (
        "Predicting", [("Error, RMSE (letters)", "RMSE"), ("Fit, R²", "R2")],
        {SUP: "Lasso", UNS: "Cluster x arm", BASE: "Arm-mean"},
        "How close is each model to the letters a patient really gained at one year?"),
    "2. Classify a gain of 15+ letters": (
        "Classifying", [("Accuracy", "Accuracy"), ("Precision", "Precision"),
                        ("Recall", "Recall"), ("Specificity", "Specificity"), ("F1", "F1"),
                        ("AUC", "AUC")],
        {SUP: "Logistic regression", UNS: "Cluster x arm", BASE: "Arm-rate"},
        "A yes/no question: will this patient gain 15 or more letters?"),
    "3. Estimate drug benefit": (
        "Semi-synthetic", [("Error, PEHE (letters)", "PEHE"),
                           ("Average bias (letters)", "Average-effect bias")],
        {SUP: "Causal forest", UNS: "Cluster x arm", BASE: "Constant effect"},
        "Tested on simulated patients whose true benefit is known (average over 3 scenarios)."),
    "4. Treatment decisions": (
        "Policy", [("Letters gained if followed (AIPW)", "AIPW")],
        {SUP: "Causal: most letters", UNS: "Cluster: most letters", BASE: "All Bevacizumab"},
        "If doctors followed each model's pick, how many letters would patients gain?"),
}


def table_for(section: str, measures: list, models: dict) -> pd.DataFrame:
    rows = []
    for label, met in measures:
        d = df[df.section.str.startswith(section) & df.metric.str.startswith(met)]
        if d.empty:
            continue
        better = d.better.iloc[0]
        vals = {}
        for method, prefix in models.items():
            v = d[d.model.str.startswith(prefix)].value
            if len(v):
                vals[method] = float(v.abs().mean() if "bias" in label else v.mean())
            else:
                vals[method] = None
        a, b = vals[SUP], vals[UNS]
        if a is None or b is None or abs(a - b) < 1e-9:
            win = "Tie"
        else:
            win = SUP if (a < b) == (better in ("lower", "closer to 0")) else UNS
        rows.append({"Measure": label, "Good is": "lower" if better != "higher" else "higher",
                     SUP: a, UNS: b, BASE: vals[BASE], "Winner": win})
    return pd.DataFrame(rows)


tables = {name: table_for(sec, ms, mod) for name, (sec, ms, mod, _) in TABS.items()}
allrows = pd.concat(tables.values())
wins = allrows.Winner.value_counts()

st.subheader("At a glance")
c1, c2, c3 = st.columns(3)
c1.metric("Supervised wins", f"{wins.get(SUP, 0)} of {len(allrows)}")
c2.metric("Unsupervised wins", f"{wins.get(UNS, 0)} of {len(allrows)}")
c3.metric("Patients with a clear best drug",
          f"{100 - (val('Verdicts', 'Share', 'Cluster') or 0):.0f}%",
          help="Share of test patients for whom a drug is clearly better by at least one "
               "chart line (5 letters). Same answer for both models.")
st.caption("Wins are counted across the measures in the tabs below. "
           "**Supervised** learns from each patient's outcome. **Unsupervised** groups "
           "patients by their starting health only, then averages results per group. "
           "**Baseline** is a simple reference with no real learning. Held-out test patients.")

st.subheader("Results by question")
for tab, (name, tbl) in zip(st.tabs(list(tables)), tables.items(), strict=True):
    with tab:
        st.caption(TABS[name][3])
        st.dataframe(
            tbl, hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                           for c in (SUP, UNS, BASE)})
        long = tbl.melt(id_vars=["Measure"], value_vars=[SUP, UNS, BASE], var_name="Method",
                        value_name="Value").dropna()
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("Method:N", title=None, axis=alt.Axis(labels=False, ticks=False),
                    sort=[SUP, UNS, BASE]),
            y=alt.Y("Value:Q", title=None),
            color=alt.Color("Method:N", sort=[SUP, UNS, BASE]),
            column=alt.Column("Measure:N", title=None, header=alt.Header(labelLimit=160)),
            tooltip=["Measure", "Method", alt.Tooltip("Value:Q", format=".3f")]
        ).resolve_scale(y="independent").properties(width=90, height=150), width="content")

st.subheader("Do the two models agree on decisions?")
agree = df[df.section.str.startswith("Verdicts")][["metric", "model", "value", "unit", "note"]]
st.dataframe(agree.rename(columns=str.title), hide_index=True, width="stretch")

with st.expander("Full results table (every number, for reference)"):
    cols = [c for c in ["section", "metric", "model", "method", "value", "unit", "better",
                        "note"] if c in df]
    st.dataframe(df[cols].rename(columns=str.title), hide_index=True, width="stretch")

pp = derived_dir / PER_PATIENT
st.subheader("Aflibercept benefit, patient by patient")
if pp.is_file():
    p = pd.read_csv(pp)
    if "cf_effect_Aflibercept" in p:
        lim = [float(min(p.cf_effect_Aflibercept.min(), p.effect_Aflibercept.min())) - 1,
               float(max(p.cf_effect_Aflibercept.max(), p.effect_Aflibercept.max())) + 1]
        pts = alt.Chart(p).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("cf_effect_Aflibercept:Q", title="Causal model (letters vs bevacizumab)",
                    scale=alt.Scale(domain=lim)),
            y=alt.Y("effect_Aflibercept:Q", title="Cluster model (letters vs bevacizumab)",
                    scale=alt.Scale(domain=lim)),
            color=alt.Color("cluster:N", title="Cluster"), tooltip=["cluster"])
        diag = alt.Chart(pd.DataFrame({"v": lim})).mark_line(color="gray", strokeDash=[4, 4]
                                                             ).encode(x="v:Q", y="v:Q")
        st.altair_chart(pts + diag, width="stretch")
    else:
        st.info("The causal model's per-patient file was missing when the stage ran.")
else:
    st.info("Per-patient estimates not found. Run `python -m equipoise unsupervised`.")

rmse_c, rmse_l = val(PRED, "RMSE", "Cluster"), val(PRED, "RMSE", "Lasso")
eq_c, eq_f = val("Verdicts", "Share", "Cluster"), val("Verdicts", "Share", "Causal")
agree, rho = val("Verdicts", "Verdict agreement", "Cluster vs"), val("Verdicts", "Spearman", "vs")
parts = []
if None not in (rmse_c, rmse_l):
    parts.append(f"For predicting a patient's vision change at one year, the grouping model is "
                 f"off by about {rmse_c:.1f} letters against {rmse_l:.1f} for the supervised "
                 f"Lasso model.")
if None not in (eq_c, eq_f):
    parts.append(f"The grouping model leaves {eq_c:.0f}% of test patients with no clearly "
                 f"better drug, the causal model {eq_f:.0f}%")
    if agree is not None:
        parts[-1] += f", and they give the same call for {agree:.0f}% of patients"
    parts[-1] += "."
if rho is not None:
    parts.append(f"Their patient-by-patient estimates of the aflibercept benefit are "
                 f"correlated at {rho:.2f} (rank correlation).")
parts.append("Both are weak at telling patients apart: the group averages come from a few "
             "dozen people each, so read this as a check on the causal model, not as a "
             "second opinion.")
st.subheader("In plain words")
st.write(" ".join(parts))
