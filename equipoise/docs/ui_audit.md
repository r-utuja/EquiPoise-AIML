# UI audit and redesign proposal

Status: **proposal, waiting for approval.** Nothing in `app/` has been changed yet.

Audited on 2026-10-10 against the real-data bundle (`models/equipoise_bundle.joblib`,
causal forest), DEMO prices (`config/costs.yaml` is still all `null`), Streamlit 1.65.

---

## 1. What a first-time user sees today

The app is one Streamlit script with a radio-button sidebar and six pages:
1 Patient · 2 Counterfactual results · 3 Break-even · 4 Trial explorer · 5 Early response ·
6 OCT screening.

The main problems for a non-ML user (eye doctor or counsellor):

1. **Too many inputs, up front.** Page 1 has 33 fields in five boxes, labelled with
   variable names: `Central subfield thickness, machine units (um) (cst_um)`. Nothing shows
   which fields matter.
2. **Vision is asked in ETDRS letters.** Clinicians in India think in Snellen (6/18, 20/50).
3. **No clear flow.** After "Estimate outcomes" the result is on a different page, and you
   have to find it in the sidebar. If you switch pages during the ~10 s calculation, the
   result is lost with no message. (This happened during the audit.)
4. **The answer is buried.** The results page has four coloured boxes (red, yellow, green,
   yellow) before the chart. The one sentence that answers the question
   ("Equipoise at MCID 5 letters … Recommendation: Bevacizumab") uses two jargon terms.
5. **Jargon everywhere.** Counterfactual, MCID, 95% bootstrap interval, P(benefit > MCID),
   fuzzy recommendation strength, expert system, SHAP, Newton-Raphson, brentq, CV RMSE.
6. **A raw 8-column table** carries the per-drug numbers. Avastin's "Effect vs reference"
   cell says "reference", and "P(benefit > MCID)" says "None".
7. **Cost is in "units".** DEMO prices are flagged three times on one page.
8. **The disclaimer is shown four times** on the results page: the red banner, the trace,
   the warnings list and the footer.
9. **Two charts compete.** The point-range chart and the fuzzy gauge sit side by side.
   The gauge (strength 17/100 for both drugs) means nothing without the method.
10. **Break-even is in relative units and value-of-vision units.** "Extra letters needed to
    justify the extra cost: 16.97", and a mention of `U = 1 - exp(-k·letters)`.

## 2. Things the redesign has to work around (data facts, not opinions)

These come from running the model, not from the code. They shape the design.

| Fact | Evidence | Consequence for the UI |
|---|---|---|
| **Every held-out patient is "no clear winner"** at both 5 and 3 letters. | `reports/tables/m8_population_summary.md` (132/132); confirmed by scoring all 132 held-out patients | In practice the "clear winner" headline never appears with the real model. The design still supports it (and the synthetic bundle may trigger it), but **there is no honest "expensive drug clearly helps more" example patient.** See §5 and Decision D1. |
| P(Eylea benefit > 1 line) across held-out patients: median 0.04, max 0.52 | held-out scoring | The confidence wording thresholds need to work in this range. See D2. |
| The fuzzy strength for the dearer drug is 17/100 for most patients and 36 at most. | held-out scoring | "Strong / Moderate / Weak suggestion" will almost always read "Weak". See D3. |
| **Avastin has no likely range.** `letters_low/high` are `None` for the reference drug. The bootstrap covers *differences* from Avastin only. | `DrugEstimate` in `decision/recommend.py` | The Avastin card and chart can't show a range unless we invent one (we won't). See D4. |
| All prices are `null`. The DEMO set is 1 : 5 : 10 (Avastin : Lucentis : Eylea), with willingness-to-pay = 200. | `config/costs.yaml` | "About 8× the cost of Avastin" comes from **made-up ratios**. The note must say so. Break-even also depends on the made-up willingness-to-pay. |
| One prediction takes about **2.4 s** without the SHAP explanation, and about **6 s** with it. | timed in-process, real bundle | "About a second" is not reachable without changing the model. See §8 for how the design hides the wait. |
| CLAUDE.md requires the app to **always show the published "< 69 letters" rule next to the model**, and MCID 3 as a sensitivity check. | CLAUDE.md "Weak per-patient signal" | Kept in the main flow in plain words (wireframe S2, S3). |
| Retinal thickness is in **machine units**. Spectralis and Cirrus read thicker than Stratus. | README §7.5 | The OCT machine picker stays next to retinal thickness in the essential inputs. |

## 3. Inventory: every element, and what happens to it

Decisions: **Keep** (main flow, same idea) · **Simplify** (main flow, rewritten) ·
**Move** (to "Behind the model") · **Remove** (not needed anywhere).

### Global (every page)

| Element | Type | Decision | Reason |
|---|---|---|---|
| Red "Research prototype, not for clinical use" banner (`st.error`) | text | Simplify | Keep the notice, but show it once as one compact line under the page title. Red reads as an error. |
| Sidebar title + "Cost-aware treatment choice … (DRCR Protocol T)" | text | Simplify | The app name stays. The tagline moves to Home. |
| Sidebar radio "Page" (6 items) | nav | Simplify | Replace with `st.navigation`: 5 main pages, plus a "For reviewers" section. |
| Sidebar "Model: Causal forest (CausalForestDML)" | text | Move | Model name is for reviewers only. |
| Sidebar "Mode: local bundle" | text | Move | Infrastructure detail. Behind the model → MLOps tab. |
| Sidebar synthetic-bundle warning | callout | Keep | Important when the demo model is served. Rewritten as: "Demo model: trained on made-up data, not the trial." |
| Sidebar "Prices" expander (custom prices + willingness-to-pay inputs) | inputs | Simplify | Moves to Price check as one optional "Use your own prices" section. Willingness-to-pay moves to Behind the model, because it is a value judgement. |
| Footer: DRCR attribution (`st.caption`) | text | Keep | Required. Stays in the footer, unchanged. |

### Page 1 · Patient

| Element | Type | Decision | Reason |
|---|---|---|---|
| Header "Patient: baseline profile" | text | Simplify | Becomes "Step 1 of 3 · Patient details". |
| "Only baseline vision is required; … imputed from training patients." | text | Simplify | "We've used typical values for anything you leave blank." |
| "Load a held-out test patient" expander (selectbox of 132 patients: "received X, observed +N letters") | input | Simplify + Move | The main flow gets three named "Try an example" buttons. The full held-out picker (with the observed outcome and the drug actually received) moves to Behind the model, because the observed outcome isn't a prediction. |
| "Reset to typical training patient" button | button | Simplify | Becomes "Start over", which clears everything. |
| 5 bordered groups, 33 inputs labelled `Description (X_name)` | inputs | Simplify | 8 essential inputs (§4) with plain labels and tooltips. The other 24 go in one collapsed "More details (optional)" section, prefilled with typical values. |
| Vision as a number 24–78 "letters" | input | Simplify | Snellen dropdown 20/32 … 20/320, plus an "Enter exact letter score" toggle. |
| "unknown (imputed)" placeholders / `unknown` options | input | Simplify | Prefilled typical values. "Not known" is still an option in the optional section. |
| `X_va_below_69` | input | Remove (already hidden) | Derived from vision. |
| "Estimate outcomes under each drug" (primary) | button | Simplify | "Compare the medicines". It goes straight to step 2, so you never have to find the result page. |
| "Done: open 2 · Counterfactual results" | text | Remove | No longer needed, because the wizard advances by itself. |
| Validation error "Input rejected: <pydantic text>" | text | Simplify | Plain message per field, e.g. "Retinal thickness should be between 150 and 1200 µm." |

### Page 2 · Counterfactual results

| Element | Type | Decision | Reason |
|---|---|---|---|
| Header "Counterfactual results" | text | Simplify | "Step 2 of 3 · The comparison". |
| DEMO-prices warning box | callout | Simplify | One small note under the cards: "Prices not set — showing relative cost from example price ratios. Ask your team to add current prices." |
| Green/blue box "Equipoise at MCID 5 letters … Recommendation: Bevacizumab (reason)" | callout | Simplify | Becomes the **headline card**: one sentence (§6). |
| Caption "Published Protocol T subgroup rule suggests: X. Sensitivity: MCID 3: equipoise." | text | Simplify | Split. The trial finding stays on step 2 as one plain line, which CLAUDE.md requires. The lower-bar check moves to step 3 "How we decided". |
| Yellow "Per-patient differences … weakly estimated … read the intervals" | callout | Simplify | Folded into the confidence phrase plus one plain sentence: "Differences between patients are hard to estimate from one trial, so treat these as rough guides." |
| `point_range` chart (matplotlib, title "Counterfactual vision under each drug (bar = 95% bootstrap interval)") | chart | Simplify + Move | Main flow gets a new Altair chart: lines gained, soft band, no title, direct labels. The original moves to Behind the model → Treatment-effect evaluation. |
| `strength_gauge` chart (fuzzy strength 0–100) | chart | Move | Becomes a strength word. The gauge goes to Behind the model → Decision logic. |
| Dataframe: Letters (point), Effect vs reference (95% interval), P(benefit > MCID), Injections yr 1, P(laser yr 1), Cost, Cost per extra letter | table | Simplify + Move | Main flow: three cards with lines, range, injections and cost (§6). The full table goes to Behind the model, including laser probability and cost per extra letter. |
| Caption "Imputed: age, bmi, …" | text | Simplify | "We used typical values for: blood pressure, kidney test, …" as a small line under the inputs summary. |
| "Reasoning trace (expert system)" code block | text | Simplify + Move | Step 3 gets 2–4 plain bullets. The full numbered trace with rule names goes to Behind the model → Decision logic. |
| "Explanation (SHAP)" waterfall + note | chart | Simplify + Move | Step 3 gets the top 3 factors as sentences. The waterfall goes to Behind the model → Explainability. |
| Bullet list of `warnings` (disclaimer, DEMO, weak signal) | text | Remove | These duplicate the notice, the price note and the confidence phrase. |

### Page 3 · Break-even

| Element | Type | Decision | Reason |
|---|---|---|---|
| "Dearer drug" selectbox | input | Simplify | "Compare Avastin with: Eylea / Lucentis" (segmented control). |
| DEMO warning | callout | Simplify | Same single price note as step 2. |
| Price slider (relative units 0–30) | input | Keep | One slider: "Price of one Eylea injection". In ₹ when prices are set, otherwise "× Avastin's price". |
| `breakeven_curve` chart (net-benefit difference vs price) | chart | Move | "Net benefit in value units" needs the method to make sense. Moves to Behind the model → Decision logic. |
| Metric "Break-even price per injection 2.71" | metric | Simplify | One sentence: "If one Eylea injection costs more than about 2.7× Avastin's price, it stops being worth the extra cost for this patient." |
| Metric "Extra letters needed to justify the extra cost 16.97" | metric | Move | Useful to reviewers; confusing next to the price sentence. |
| Caption "value-of-vision function U = 1 - exp(-k·letters). Newton-Raphson, verified against brentq" | text | Move | Behind the model → Decision logic, as the break-even derivation. |
| "Willingness to pay is not set …" info | text | Simplify | "The price check needs a value placed on vision. Ask your team to set it." |

### Page 4 · Trial explorer

| Element | Type | Decision | Reason |
|---|---|---|---|
| Header + caption about `reports/` | text | Move | Whole page goes to Behind the model → Data. |
| Table 1 (baseline balance by arm) | table | Move | Behind the model → Data. |
| Replication vs the published trial | table | Move + Simplify | Behind the model → Data has the table. About the trial gets the three headline numbers in words. |
| Hypothesis tests | table | Move | Behind the model → Data. |
| Figures: mean letters per arm (95% CI), outcome distributions, PCA map, cluster effects | images | Move | Behind the model → Data (PCA and clusters labelled exploratory). |

### Page 5 · Early response

| Element | Type | Decision | Reason |
|---|---|---|---|
| Header "Early response (after week 12)" | text | Simplify | Page renamed "Progress check". |
| Intro "post-treatment data … not used to choose the drug" | text | Simplify | "This answers a different question: how is a patient on treatment doing so far? It doesn't change the suggestion." |
| "Drug the patient is receiving" selectbox | input | Keep | Brand + generic names. |
| Week 4/8/12 vision number inputs (letters, default = baseline) | input | Simplify | Snellen dropdowns, "Vision at the 1-month / 2-month / 3-month visit", with the exact-score toggle. |
| "Forecast week 52" button | button | Simplify | "Show expected progress". |
| Metric "Forecast change at week 52: +2.6 letters" | metric | Simplify | "On this path, expected vision at 1 year is about 20/40 (≈ 72 letters), about ½ line better than today." |
| "Typical error ± 8.1 letters (CV RMSE)" | text | Simplify | "This could easily be 1½ lines better or worse." |
| `early_trajectory` chart | chart | Simplify + Move | Main flow gets a new small Altair line in chart lines, with a band for the typical error. The original goes to Behind the model → Time series. |
| Note "Uses vision measured after treatment started…" | text | Remove | Duplicates the simplified intro. |

### Page 6 · OCT screening

| Element | Type | Decision | Reason |
|---|---|---|---|
| Whole page (uploader, example picker, input / preprocessed / Grad-CAM images, probability table) | page | Keep as its own page | The module exists (`models/oct/` built). Title: "OCT screening (demo)". Label: "Demo — separate image dataset, not connected to the suggestion". It sits in the "For reviewers" nav section and isn't held to the main-flow language rules. |

### Content that exists in `reports/` but is not in the app today

Behind the model will also surface these existing figures and tables, so the examiners have
one place to look: M5 metrics and confusion matrices, M6 semi-synthetic PEHE, policy values,
calibration, rediscovery, fuzzy membership and rules, SHAP beeswarm and importance,
subgroups, HMM and time series, MLOps info. Nothing new is computed; they are read from
`reports/tables/*.md` and `reports/figures/*.png`, with a synthetic fallback as today.

## 4. Essential inputs (step 1)

Ranked by mean |SHAP| of the Eylea-vs-Avastin estimate (`reports/tables/m9_shap_importance_aflibercept.md`)
plus clinical relevance:

| # | Shown as | Field | Mean SHAP rank | Why it's in |
|---|---|---|---|---|
| 1 | **Vision in the treated eye today** (eye-chart dropdown) | `X_va_letters` | 2 | Required by the model, and the published subgroup rule depends on it. |
| 2 | **Retinal thickness from OCT report (µm)** | `X_cst_um` | 1 | Top driver. |
| 3 | **OCT machine** (Spectralis / Cirrus / Stratus) | `X_oct_machine` | 22 | Retinal thickness depends on the machine; without this, #2 is misread. |
| 4 | **HbA1c (%)** | `X_hba1c` | 5 | Routinely known. |
| 5 | **Years with diabetes** | `X_diabetes_duration_yrs` | 6 | Routinely known. |
| 6 | **Sex** | `X_female` | 4 | Routinely known. |
| 7 | **Age** | `X_age` | 10 | Clinically obvious. |
| 8 | **Blood pressure** (top / bottom number, one row) | `X_sbp`, `X_dbp` | 8, 7 | Routinely measured. |

**Optional ("More details"), prefilled with typical training values**, in three short
groups: *Eye* (other eye's vision, fluid under the retina, epiretinal membrane, eye
pressure, lens replaced, DR severity ×2), *Past eye treatment* (any DME treatment, earlier
anti-VEGF, focal laser, scatter laser), *General health* (kidney test, BMI, insulin, diabetes
type, hypertension, heart disease, heart attack, stroke, kidney disease, smoker, race,
Hispanic).

*Kidney test* (`X_log_uacr`, SHAP rank 3) is optional because many clinics won't have a
urine albumin-to-creatinine ratio to hand. The form asks for the ratio itself and converts it
with `log(1 + x)`, the same transform the build script uses
(`scripts/build_protocolT_table.py:99`). **The unit of DRCR's `AlbCreatRatio` needs
confirming (mg/g assumed).** See D6.

All SHAP magnitudes are small (top feature 0.64 letters on average), so "matters most"
means most among small effects. The copy in step 3 says "made a small difference".

### Vision dropdown

`letters ≈ 85 + 50 × log10(Snellen fraction)`, clamped to the trial range 24–78:

| Snellen (feet) | 20/32 | 20/40 | 20/50 | 20/63 | 20/80 | 20/100 | 20/125 | 20/160 | 20/200 | 20/250 | 20/320 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Metric (6 m) | 6/9.5 | 6/12 | 6/15 | 6/19 | 6/24 | 6/30 | 6/38 | 6/48 | 6/60 | 6/75 | 6/95 |
| Letters | 75 | 70 | 65 | 60 | 55 | 50 | 45 | 40 | 35 | 30 | 25 |

The published rule's cut-off (< 69 letters, i.e. 20/50 or worse) falls between 20/40 and
20/50 as it should. The dropdown shows both the feet and metric notation ("20/50 (6/15)"),
since Indian clinics often use 6/x. When an example patient has an exact score (e.g. 57),
the toggle switches to "exact" automatically, so the stored value is never rounded.

## 5. Example patients

Picked from the held-out test set by scoring all 132 patients once (see §2). The real
model calls **all 132 "no clear winner"**, so the three presets are chosen to span the
range of "how sure we are that Eylea helps more":

| Button label | Pick rule | Profile (rounded) | Eylea − Avastin (likely range) | P(Eylea > 1 line) | Confidence word (D2) | Trial finding |
|---|---|---|---|---|---|---|
| **Patient with poorer vision** | highest P | about 20/100, very thick retina (~750 µm) | +6.2 letters (2.2 to 8.6) | 0.52 | We're not sure | favours Eylea |
| **Patient in between** | P closest to halfway between the other two (0.26) | about 20/63, thick retina (~550 µm) | +5.2 letters (2.1 to 6.7) | 0.24 | This is a close call | favours Eylea |
| **Patient with good vision** | P = 0 with vision ≥ 20/40, thinnest retina | about 20/32, ~300 µm | +2.1 letters (−0.9 to 3.8) | 0.00 | We're fairly confident | no advantage |

*Approved (2026-10-10):* button names "Poorer / Moderate / Good starting vision"; confidence
words "fairly confident" (≥ 0.8) / "somewhat confident" (≥ 0.6) / "close call"; costs hidden
until real prices are set; strength words on step 3 only. The kidney-test unit is mg/g,
stated in the DRCR data dictionary (`TReadMe.docx`, table TUrine). The "good" row above is
the one `scripts/pick_example_patients.py` picks (an earlier draft of this table, written
by hand, listed a different patient).

Rounded on purpose: this doc goes into the repo, which holds aggregate results only. The
three cover all three confidence words, so the demo shows the full range.

How they get into the app without committing patient data: a small script
`scripts/pick_example_patients.py` scores the held-out set once and writes
`data/derived/app_examples.json`. That folder is never committed, as CLAUDE.md requires.
The app reads that file if present. Otherwise (Docker, CI, synthetic bundle) it falls back
to the same rules applied to the synthetic held-out set at start-up. Committing real
patient rows or IDs to `config/` would break the DRCR data terms.

Even the first preset will show **"No clear winner"**. That's the honest result. See D1.

## 6. Plain-language copy for the key messages

Drug names: brand first with the generic on first mention per screen, brand only after.
Avastin (bevacizumab) · Lucentis (ranibizumab) · Eylea (aflibercept).

**Headline — no clear winner** (`equipoise = true`; this must never name the dearer drug as the suggestion):
> No clear winner for this patient. All three are likely to work about equally well, so the
> lower-cost option, Avastin (bevacizumab), is a reasonable choice.

**Headline — clear winner, worth the cost** (`recommendation` = Eylea or Lucentis):
> Eylea (aflibercept) is likely to help this patient most — about 1 more line on the eye
> chart than Avastin.

**Headline — clear benefit but not worth the cost at these prices** (`recommendation = "Discuss"`, a third case the rule engine can return):
> Eylea (aflibercept) is likely to help a little more, but at these prices the extra cost may
> not be worth it. Talk it through with the patient.

**Confidence phrase** (one line under the headline, thresholds in `config/default.yaml`):
> We're fairly confident · This is a close call · We're not sure

**Trial finding line** (always shown, required by CLAUDE.md):
- Vision 20/50 or worse: "In the original trial, people who started at 20/50 or worse gained
  more with Eylea on average. For this patient, the difference is too uncertain to call."
- Vision 20/40 or better: "In the original trial, people who started at 20/40 or better
  did about equally well on all three."

**Uncertainty line:**
> These are estimates from one trial of 660 people. Treat them as a rough guide.

**Drug card** (three numbers at most):
```
Eylea (aflibercept)                [Suggested | Lowest cost]
About 2 lines better (≈ 12 letters)
likely between 2 and 2.5 lines
About 9 injections in the first year
About 8× the cost of Avastin          (or: about ₹1.2 lakh in the first year)
```

**Price note (prices null):**
> Prices not set — showing relative cost based on example price ratios. Ask your team to add current prices.

**Chart summary (accessibility):**
> Avastin: about 2 lines. Lucentis: about 2 lines (likely 2 to 2.5). Eylea: about 2.5 lines
> (likely 2 to 3). The ranges overlap, so none is clearly ahead.

**What mattered most** (top 3 SHAP contributions, direction plus feature, templated):
> - Thicker retina (520 µm) made Eylea's advantage a little larger.
> - Better starting vision (20/40) made Eylea's advantage a little smaller.
> - The kidney test result made a small difference.

**How we decided** (from the trace and flags, 2–4 bullets):
> - We estimated how much each medicine would help this patient after one year.
> - Eylea's extra benefit over Avastin could be anywhere from almost nothing to about 1 line.
>   That's not enough to be sure it helps more.
> - The same is true for Lucentis.
> - Even with a lower bar (about half a line), there's still no clear winner.

**Is the extra cost worth it?**
> If one Eylea injection costs more than about 2.7× Avastin's price, it stops being worth
> the extra cost for this patient. At the example price (10×), it is not worth it.
> *(Based on an example value placed on vision; your team can change it.)*

**Errors:**
- Validation: "Retinal thickness should be between 150 and 1200 µm. Check the OCT report."
- Both API and local model unavailable: "We couldn't load the comparison model. Ask your
  team to run the setup steps in the README, then reload this page."

**Onboarding tip** (dismissible, once per session):
> 1. Pick the patient's vision and fill in what you know. Leave the rest.
> 2. Press Compare to see how the three medicines stack up.
> 3. Press See why to understand the result.

## 7. Wireframes

Max content width about 900 px, centred. Nav lives in the sidebar
(`st.navigation`): **Home · Get a suggestion · Price check · Progress check · About the
trial**, then a section "For reviewers": **Behind the model · OCT screening (demo)**.

### Home
```
┌─────────────────────────────────────────────────────────────────────┐
│ Equipoise                                                           │
│ Research prototype, not for clinical use.                    (small)│
│                                                                     │
│ Compare three eye-injection medicines for diabetic macular edema,   │
│ see which is likely to work better for this patient, and what each  │
│ would cost.                                                         │
│                                                                     │
│ How it works                                                        │
│ ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│ │ ① Enter       │  │ ② See the     │  │ ③ Understand  │             │
│ │ patient       │  │ comparison    │  │ why           │             │
│ │ details       │  │               │  │               │             │
│ └───────────────┘  └───────────────┘  └───────────────┘             │
│                                                                     │
│            [  Get a suggestion  ]   (primary)                       │
│                                                                     │
│   Try an example patient  ·  About the trial          (small links) │
├─────────────────────────────────────────────────────────────────────┤
│ Data: DRCR Retina Network, Protocol T … (footer, small)             │
└─────────────────────────────────────────────────────────────────────┘
```

### Get a suggestion · Step 1 of 3 · Patient details
```
┌─────────────────────────────────────────────────────────────────────┐
│ Get a suggestion                                                    │
│ ●──────○──────○   Step 1 of 3 · Patient details                     │
│                                                                     │
│ ┌ How to use this ─────────────────────────────────────── [Got it] ┐│
│ │ 1. Pick the vision, fill in what you know. 2. Compare. 3. See why.││
│ └───────────────────────────────────────────────────────────────────┘│
│                                                                     │
│ Try an example:  [Patient with poorer vision] [In between] [Good vision]
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Vision in the treated eye today   [ 20/50 (6/15)          ▾ ] ⓘ │ │
│ │                                   ☐ Enter exact letter score    │ │
│ │ Retinal thickness (µm)  [ 430 ]ⓘ   OCT machine [Spectralis ▾]ⓘ  │ │
│ │ HbA1c (%)               [ 7.7 ]ⓘ   Years with diabetes [ 15 ]ⓘ  │ │
│ │ Age                     [ 60  ]ⓘ   Sex      ( Male | Female ) ⓘ │ │
│ │ Blood pressure          [ 138 ] / [ 83 ] mmHg                 ⓘ │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ ▸ More details (optional)                                           │
│   We've used typical values for anything you leave blank.           │
│                                                                     │
│                      [ Start over ]   [ Compare the medicines → ]   │
└─────────────────────────────────────────────────────────────────────┘
  Phone width: the two-column grid becomes one column; example buttons wrap.
```

### Get a suggestion · Step 2 of 3 · The comparison
```
┌─────────────────────────────────────────────────────────────────────┐
│ ○──────●──────○   Step 2 of 3 · The comparison                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ No clear winner for this patient. All three are likely to work  │ │
│ │ about equally well, so the lower-cost option, Avastin           │ │
│ │ (bevacizumab), is a reasonable choice.                          │ │
│ │ We're fairly confident · Weak case for paying more              │ │
│ │ In the original trial, people who started at 20/50 or worse …   │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐         │
│ │ Avastin         │ │ Lucentis        │ │ Eylea           │         │
│ │ (bevacizumab)   │ │ (ranibizumab)   │ │ (aflibercept)   │         │
│ │ [Suggested]     │ │                 │ │                 │         │
│ │ About 2 lines   │ │ About 2 lines   │ │ About 2.5 lines │         │
│ │ (≈ 9 letters)   │ │ (≈ 10 letters)  │ │ (≈ 12 letters)  │         │
│ │ compared to the │ │ likely 2–2.5    │ │ likely 2–3      │         │
│ │ others ¹        │ │ lines           │ │ lines           │         │
│ │ About 9 inj.    │ │ About 9 inj.    │ │ About 9 inj.    │         │
│ │ Lowest cost     │ │ ≈ 4.5× Avastin  │ │ ≈ 8.5× Avastin  │         │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘         │
│ Lines gained on the eye chart after 1 year                          │
│  Avastin   ●                                                        │
│  Lucentis  ░░░●░░░                                                  │
│  Eylea        ░░░░●░░░░                                             │
│            0    1    2    3 lines                                   │
│ Avastin: about 2 lines. Lucentis … The ranges overlap. (text summary)│
│ Prices not set — showing relative cost … (small)                    │
│ These are estimates from one trial of 660 people. (small)           │
│                                                                     │
│        [ ← Edit details ]  [ Start over ]  [ See why → ]            │
└─────────────────────────────────────────────────────────────────────┘
  ¹ wording depends on D4.
  Phone width: cards stack, headline first; chart full width.
```

### Get a suggestion · Step 3 of 3 · Why this result
```
┌─────────────────────────────────────────────────────────────────────┐
│ ○──────○──────●   Step 3 of 3 · Why this result                     │
│ No clear winner · Avastin is a reasonable choice      (recap line)  │
│ ┌ What mattered most ─────────────────────────────────────────────┐ │
│ │ • Thicker retina (520 µm) made Eylea's advantage a little larger.│ │
│ │ • Better starting vision (20/40) made it a little smaller.       │ │
│ │ • The kidney test result made a small difference.                │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ ┌ How we decided ─────────────────────────────────────────────────┐ │
│ │ • We estimated how much each medicine would help after a year.   │ │
│ │ • Eylea's extra benefit could be almost nothing to about 1 line. │ │
│ │ • Even with a lower bar (half a line), no clear winner.          │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ ┌ Is the extra cost worth it? ────────────────────────────────────┐ │
│ │ Compare Avastin with: ( Eylea | Lucentis )                       │ │
│ │ Price of one Eylea injection  ────●───────── 10× Avastin         │ │
│ │ If one Eylea injection costs more than about 2.7× Avastin's      │ │
│ │ price, it stops being worth the extra cost for this patient.     │ │
│ │ Based on an example value placed on vision. (small)              │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│ [ ⬇ Download summary (PDF) ]                                        │
│        [ ← Back to comparison ]  [ Start over ]                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Price check
Same "Is the extra cost worth it?" card as step 3, full width, plus an optional
"Use your own prices" section (three price-per-injection boxes and one laser box, in ₹).
If no patient has been entered: "Enter a patient first" and a button to step 1.

### Progress check
```
┌─────────────────────────────────────────────────────────────────────┐
│ Progress check                                                      │
│ This answers a different question from the suggestion: how is a     │
│ patient on treatment doing so far? It doesn't change the suggestion.│
│ Medicine the patient is on   ( Avastin | Lucentis | Eylea )         │
│ Vision today (from patient details): 20/50                          │
│ Vision at the 1-month visit  [ 20/40 ▾ ]                            │
│ Vision at the 2-month visit  [ 20/40 ▾ ]                            │
│ Vision at the 3-month visit  [ 20/32 ▾ ]   ☐ Enter exact scores     │
│ [ Show expected progress ]                                          │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ On this path, expected vision at 1 year is about 20/32           │ │
│ │ (≈ 76 letters), about 2 lines better than at the start.          │ │
│ │ This could easily be 1½ lines better or worse.                   │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│  small line chart: today → 1, 2, 3 months → 1 year (dashed + band)  │
│  text summary of the chart                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### About the trial
Four short sections, no tables:
1. **What Protocol T was** — 660 adults with diabetic macular edema, 88 US clinics,
   2012–2013. Each was given Avastin, Lucentis or Eylea at random and followed for 2 years.
2. **Why a randomized trial is fair** — "It's like tossing a coin to decide who gets which
   medicine. Because the coin doesn't care how sick anyone is, the three groups start out
   alike, so differences later come from the medicine." (Two sentences.)
3. **What it found** (from README §9.1, our replication) — on average after 1 year: Eylea
   about 2.5 lines, Lucentis about 2, Avastin about 2. In people who started at 20/50 or
   worse, Eylea did better (about 4 lines vs 2.5); at 20/40 or better, all three were similar.
4. **What this app can and can't tell you** — four bullets: one trial of 660 people in the
   US; per-patient differences are hard to pin down; Lucentis was given at a lower dose
   than is common in India; prices are inputs, not market data.

### Behind the model (for reviewers)
"Technical details — for reviewers and the curious." Glossary first (technical term →
word used in the app), then tabs:
**Data** · **Models** · **Treatment-effect evaluation** · **Decision logic** ·
**Explainability** · **Time series & HMM** · **MLOps**. Each tab shows the existing
`reports/` tables and figures for that module. The current patient's raw `Recommendation`
(full table, point-range chart, fuzzy gauge, full rule trace, SHAP waterfall, break-even
curve, early-response chart) appears at the top of the relevant tabs when a patient has
been entered. The held-out patient picker with observed outcomes lives in **Data**.

### OCT screening (demo)
Unchanged functionality. New title and label: "Demo — separate image dataset, not connected
to the suggestion". The probability table becomes a small bar chart, and raw numbers sit
under it.

## 8. Implementation notes

- **Presentation layer.** `app/presenters.py` turns the `Prediction` dict into frozen
  dataclasses (`HeadlineVM`, `DrugCardVM`, `ChartVM`, `WhyVM`, `PriceCheckVM`,
  `ProgressVM`). All rounding, Snellen conversion, lakh formatting, relative-cost text and
  threshold lookups live there, with unit tests. Page files only lay out view models.
- **Copy.** `app/copy.py` holds every main-flow string. A test scans it and the main-flow
  page files for the banned words.
- **Config.** New `ui:` block in `config/default.yaml`: confidence thresholds, strength
  thresholds, the lines rounding step, the Snellen list, the example-picker rules. Nothing is
  hardcoded in `app/`.
- **Speed.** Step 2 calls `predict(explain=False)`, about 2.4 s. Step 3 calls
  `predict(explain=True)` only when opened, about 6 s, with the spinner "Working out what
  mattered most…". Both are cached per patient (`st.cache_data` keyed on the inputs), so
  going back and forth, or clicking an example a second time, is instant. Price check
  calls `breakeven` once and moves the slider along the returned curve, with no new model
  call per slider move. The net-benefit curve is linear in price, so reading it is exact.
  These are the same numbers the API returns.
- **API fallback.** If `API_URL` is set and the API fails, the app quietly falls back to the
  local bundle. It only shows an error if both fail. This lives in a small wrapper in
  `app/`, not in `serving/client.py`, to keep the contract untouched.
- **Existing test.** `tests/test_api.py::test_streamlit_app_runs` clicks the last button on
  the old page 1 and checks `st.error` for the disclaimer. It has to be updated for the new
  flow, with the same intent: the app runs, the notice is present, and a prediction lands
  in session state. All other existing tests stay untouched.
- **PDF.** Proposal: `fpdf2` (pure Python, ~250 KB) in the `[app]` extra. Its core fonts lack
  "₹", so the PDF writes "Rs" unless a TTF font is available. `reportlab` is the
  alternative (larger, same font issue).
- **Screenshots.** Playwright is not installed. It needs `pip install playwright` plus a
  Chromium download (~150 MB) into `.venv`. See D7.
- **Theme.** `.streamlit/config.toml`: light base, one calm blue accent, neutral surface,
  base font 16 px. Drug colours come from the existing CVD-validated
  `equipoise.plotting.ARM_COLORS`, re-exported by `app/theme.py`, so app and report figures
  match. Every chart mark also carries a direct text label, so colour is never the only
  signal.

## 9. Decisions needed before implementing

**D1 — No "clear winner" example exists.** The real model calls every held-out patient
"no clear winner". Options:
(a) **Recommended:** use the three presets in §5, the closest real cases to the brief.
"Patient with poorer vision" is the one where Eylea is most likely to help more, but it
still honestly shows "No clear winner". The demo script explains why.
(b) Fake or hand-tune a patient to force a clear winner. *Not recommended: it misrepresents
the model.*
(c) Use the synthetic bundle for the demo, which has a planted effect. It may produce a
clear winner, but the numbers aren't from the trial.

**D2 — Confidence wording.** The model gives P(benefit > 1 line) for the dearer drug.
Proposal: measure the confidence *in the headline's claim*. For "clear winner" use P. For
"no clear winner" use 1 − P (the chance the extra benefit is under 1 line).
Thresholds: ≥ 0.80 "We're fairly confident", 0.60–0.80 "This is a close call", < 0.60 "We're
not sure". On held-out patients this gives mostly "fairly confident" that there's no clear
winner, and "not sure" for the poorer-vision example (P = 0.52 → 0.48). Is that the
reading you want? The alternative is to always show "We're not sure" when the trial
rule and the model disagree.

**D3 — Strength words.** Fuzzy strength is "how strongly the evidence favours paying
more" (0–100). It is about 17 for most patients, so it would almost always read "Weak". Proposal:
show it as "Weak / Moderate / Strong case for paying more". Thresholds 0–40 / 40–70 /
70–100 in config. The word "suggestion" stays reserved for the headline, so the no-clear-
winner screen doesn't say "Weak suggestion" next to "Avastin is a reasonable choice"
(which would read as weak support for Avastin). OK?

**D4 — Avastin's range.** The model gives no likely range for the reference drug. Options:
(a) **Recommended:** Avastin card and chart show the estimate with no range, plus the
caption "The other two are compared with Avastin". Lucentis and Eylea show their ranges.
(b) Show every card's range as "how much more than Avastin" (Avastin: "—"). This is more
honest about what the bootstrap measures, but changes the card's main number.

**D5 — Relative cost from made-up ratios.** With prices null, "about 8× the cost of
Avastin" comes from the DEMO 1 : 5 : 10 ratios. Proposal: show it, with the note "…based on
example price ratios". Or hide cost entirely until prices are set?

**D6 — Kidney test unit.** Confirm DRCR `AlbCreatRatio` is mg/g, or keep that field
labelled in the model's own scale and only in the optional section.

**D7 — Installs.** OK to add `fpdf2` to the `[app]` extra, and to install `playwright` +
Chromium into `.venv` for screenshots (dev only, not added to core deps)?
