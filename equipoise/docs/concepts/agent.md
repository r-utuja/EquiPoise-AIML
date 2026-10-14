# The Equipoise agent (PEAS)

`equipoise.decision.agent.EquipoiseAgent` frames the decision layer as an intelligent agent with
three steps: `perceive(patient)` reads and checks a patient's baseline profile, `decide()` runs
the models, cost model, fuzzy system and expert system (through `recommend`), and `explain()`
returns the recommendation with its reasoning trace.

| PEAS | Equipoise |
|---|---|
| **Performance measure** | Expected letters gained at week 52 under the recommended drug, traded off against cost (letters - lambda x cost); judged on held-out patients by IPW / AIPW policy value. Secondary: the recommendation is honest about uncertainty (equipoise flagged when the benefit interval includes values below the MCID). |
| **Environment** | One patient at a time with diabetic macular edema, described by baseline features (demographics, diabetes, general health, eye findings, eye history); drug prices and willingness to pay set in `config/costs.yaml` (or DEMO units). Partially observable (true individual effects are never seen), stochastic, episodic (patients are independent), static, discrete actions. |
| **Actuators** | The recommendation returned through the API response and shown in the app: recommended drug (or "discuss"), equipoise flag, fuzzy strength 0-100, predicted letters / injections / laser / cost per drug, break-even price and benefit, reasoning trace and warnings. |
| **Sensors** | The patient form (Streamlit) or the `POST /predict` request fields: baseline features validated against the schema; optional price override. |

**Agent type.** A model-based reflex agent with no memory between patients: the "model" is
the learned effect and outcome models plus the trial facts encoded in the rule base; the
"reflex" is the rule base mapping those estimates to an action. It does not learn online.

**Why this matters for the viva.** The agent never claims more than the data support: when
the predicted benefit's interval includes values below the MCID it recommends the cheapest
drug and says why, and it always shows the published subgroup rule next to its own estimate.

Decision support for a college project, not clinical guidance.

**What it showed.** On the 132 held-out patients the agent recommended bevacizumab for every
patient: every benefit interval included values below the MCID (5 letters, and also the
3-letter sensitivity threshold), so each patient was in equipoise
(`reports/tables/m8_population_summary.md`). The published < 69-letter rule, shown alongside,
would have given aflibercept to half of them; the agent reports that disagreement rather than
hiding it.
