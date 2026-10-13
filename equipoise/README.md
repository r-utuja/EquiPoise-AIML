# Equipoise

**An explainable, cost-aware treatment-choice assistant for diabetic macular edema, learned from a randomized trial.**

> ⚠️ Research prototype built for a college AIML mini-project. **Not for clinical use.**

## Quick start

```bash
pip install -e ".[dev,app]"            # Python 3.11, inside a virtual environment
dvc repro                              # build the patient table, the split and every model
streamlit run app/streamlit_app.py     # open http://localhost:8501
```

`dvc repro` needs the DRCR files in `data/raw/` (§6.2). Without them, run
`python -m equipoise all --synthetic` and start the app with `EQUIPOISE_SYNTHETIC=1` to use the
synthetic demo model. Optional: `python scripts/pick_example_patients.py` (about 5 minutes)
precomputes the three "Try an example" patients so they show instantly. How to use the app:
[docs/USER_GUIDE.md](docs/USER_GUIDE.md). Presenting it: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

For a patient with diabetic macular edema (DME), Equipoise estimates what their vision would be after one year under **each** of three anti-VEGF drugs. It also estimates how many injections each drug would need, whether rescue laser is likely, and what each option would cost in India. It then says plainly whether one drug is meaningfully better *for this patient*, or whether the patient is in **equipoise** (no real difference, so the cheapest option is reasonable).

---

## 1. The problem

- **DME** is fluid leaking into the centre of the retina in people with diabetes. It blurs central vision and is a leading cause of vision loss in working-age adults.
- It is treated with repeated injections into the eye of one of three drugs:

| Drug | Brand | Note |
|---|---|---|
| Aflibercept | Eylea | Most expensive |
| Ranibizumab | Lucentis (+ biosimilars) | Mid-priced |
| Bevacizumab | Avastin | Off-label, a fraction of the price |

- Doctors and patients in India face a real trade-off: pay more for a drug that *may* work better, or choose the cheaper one.
- The landmark **DRCR Protocol T** trial showed that aflibercept's advantage was concentrated in patients with worse baseline vision (20/50 or worse). Equipoise turns that population-level finding into a **per-patient, cost-aware estimate**.

## 2. Why a randomized trial makes this possible

Ordinary hospital data shows *which* patients got *which* drug, and that choice depends on the patient. A model trained on it learns correlations, not effects.

In Protocol T, the drug was assigned **at random**, so differences in outcomes between the arms are caused by the drug. That makes the counterfactual question "what would happen to *this* patient under drug B?" answerable from data.

## 3. What Equipoise does

**Input:** a patient's baseline profile (age, diabetes history, HbA1c, blood pressure, baseline vision, retinal thickness, eye history and so on). Optionally also:
- early follow-up vision readings (for the early-response check),
- an OCT scan image (for the screening demo).

**Output, per drug:**
- predicted letters gained at 1 year, with an uncertainty interval,
- predicted injections in year 1 and probability of rescue laser,
- expected cost in ₹ and cost per letter gained,
- a recommendation with a strength score from a fuzzy decision layer, plus a rule-based reasoning trace,
- an explanation (SHAP) of *why* the model predicts a benefit for this patient,
- a **break-even price**: the price at which the more expensive drug stops being worth it for this patient.

## 4. Pipeline

| Module | What it does | Key methods |
|---|---|---|
| **M0** Data integration | ~20 DRCR tables → one row per patient (660 × 47) | relational joins on `PtID`, study-eye selection |
| **M1** Preprocessing | Clean, impute, harmonize, encode | complete-case vs KNN vs MICE; OCT-machine CST harmonization; IQR / Isolation Forest / DBSCAN outliers; discretization |
| **M2** EDA + statistical inference | Prove randomization worked; replicate the trial | ANOVA, t-tests, chi-square, Mann-Whitney U, F-test, CIs, partial correlation |
| **M3** Patient representation | Map and group patients | PCA / SVD, k-means, hierarchical, DBSCAN, Apriori rules |
| **M4** Feature search | Feature selection as a search problem | filter / embedded / wrapper; hill climbing, beam, tabu, GA, PSO |
| **M5** Outcome models | Predict outcomes | linear / Ridge / Lasso, kNN, trees, RF, gradient boosting, logistic regression (also from scratch via Newton-Raphson), Naive Bayes |
| **M6** Counterfactual engine (core) | Per-patient treatment effects | S/T/X/DR-learners, causal forest, 3-head TARNet (SGD vs Adam vs RMSprop), Optuna vs PSO tuning, AutoGluon |
| **M7** Longitudinal | Vision trajectories over 2 years | spline interpolation, finite differences, HMM + Viterbi, AR/ARIMA, early-responder forecasting |
| **M8** Decision layer | Turn estimates into a choice | cost model, Newton-Raphson break-even, fuzzy inference, rule-based expert system, agent framing |
| **M9** Explain + fairness | Why, and for whom it works | SHAP, subgroup performance |
| **M10** OCT screening (optional) | "Is this DME?" from an OCT image | OpenCV, Ultralytics YOLO classification vs CNN transfer learning, Grad-CAM |
| **M11** Engineering | Make it reproducible and usable | Git/GitHub, MLflow, DVC, Docker, GitHub Actions CI, FastAPI, Streamlit, Evidently |

```
Raw DRCR tables ─► M0 ─► M1 ─► M2 ─► M3 ─► M4 ─► M5 ─► M6 ─► M8 ─► Clinician app
                                                       │            ▲
                                                       └─► M9 ──────┤
                           visit-level data ─► M7 ──────────────────┤
                         Kermany OCT images ─► M10 ─────────────────┘
                     M11 (Git · MLflow · DVC · Docker · CI · deploy) underneath everything
```

## 5. Repository layout

```
equipoise/
├── README.md
├── CLAUDE.md                    # working rules for AI coding assistants
├── pyproject.toml               # package + dependencies
├── requirements*.txt            # pinned core / automl / cv extras
├── config/
│   ├── default.yaml             # seeds, paths, splits, MCID, model settings
│   └── costs.yaml               # drug and laser prices (filled in by the team)
├── data/                        # NOT committed (DRCR terms); tracked with DVC
│   ├── raw/                     # "Data Tables - Text files" from DRCR
│   ├── protocolT_analysis.csv   # one row per patient (M0 output)
│   ├── protocolT_visits.csv     # one row per patient × visit (M7 input)
│   └── synthetic/               # fake data with the same schema (committed; used by tests/CI)
├── scripts/
│   ├── build_protocolT_table.py
│   ├── build_protocolT_visits.py
│   └── make_synthetic_data.py
├── src/equipoise/
│   ├── cli.py                   # python -m equipoise <stage>
│   ├── data/  preprocess/  stats/  represent/  search/  supervised/
│   ├── causal/  longitudinal/  decision/  explain/  vision/
│   └── tracking.py              # MLflow helpers
├── api/                         # FastAPI service
├── app/                         # Streamlit clinician app
├── models/                      # trained artifacts (not committed)
├── reports/
│   ├── figures/  tables/        # generated for the project report
│   └── REPORT_ASSETS.md         # which figure/table goes in which report section
├── docs/
│   ├── concepts/                # short viva-ready explainers per method
│   └── data_structures.md
├── notebooks/                   # narrative EDA notebooks
├── tests/
├── docker/                      # Dockerfiles + docker-compose.yml
├── .github/workflows/ci.yml
├── dvc.yaml  dvc.lock          # pipeline (params read from config/default.yaml)
└── .gitignore
```

## 6. Getting started

### 6.1 Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[automl]"   # optional: AutoGluon
pip install -e ".[cv]"       # optional: OCT screening module
```

Python 3.11. Works on Windows, macOS and Linux.

### 6.2 Get the data

1. Download the **Protocol T** public dataset from the DRCR Retina Network: https://public.jaeb.org/drcrnet
2. Unzip it and place the `Data Tables - Text files` folder under `data/raw/`.
3. Never commit anything in `data/` except `data/synthetic/`.

### 6.3 Build the tables

```bash
python scripts/build_protocolT_table.py "data/raw/Data Tables - Text files" data/protocolT_analysis.csv
python scripts/build_protocolT_visits.py "data/raw/Data Tables - Text files" data/protocolT_visits.csv
python -m equipoise split        # fixed 80/20 train/test split -> data/splits.json
# or all three at once, cached by DVC (venv activated):
dvc repro
```

Both scripts fail loudly if patient counts, arm sizes or value ranges don't match the trial.

### 6.4 Run the pipeline

```bash
python -m equipoise all          # every stage in order
# or stage by stage:
python -m equipoise preprocess
python -m equipoise stats
python -m equipoise represent
python -m equipoise supervised
python -m equipoise search
python -m equipoise causal
python -m equipoise longitudinal
python -m equipoise decision
python -m equipoise explain
python -m equipoise report-assets
python -m equipoise monitor      # Evidently drift report on logged API requests

dvc repro                        # same pipeline, cached and versioned by DVC
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments at http://localhost:5000
```

Add `--synthetic` to any command to run on the fake dataset (no DRCR data needed).

### 6.5 Run the app

```bash
uvicorn api.main:app --reload            # API at http://localhost:8000/docs
streamlit run app/streamlit_app.py       # app at http://localhost:8501
# or both together:
docker compose -f docker/docker-compose.yml up --build
```

The app has five plain-language pages for clinicians (Home, Get a suggestion, Price check,
Progress check, About the trial) and, under "For reviewers", **Behind the model** (all the
technical results) and the OCT screening demo. Wording lives in `app/copy.py`, view models in
`app/presenters.py`; the UI audit and design decisions are in `docs/ui_audit.md`. Screenshots:
`python scripts/ui_screenshots.py` (needs `pip install -e ".[ui-dev]"` and
`python -m playwright install chromium`).

### 6.6 Tests

```bash
pytest
```

Tests run on `data/synthetic/`, so CI works without the real data.

## 7. The data

### 7.1 The trial

| Item | Detail |
|---|---|
| Patients | 660 adults with diabetes at 88 US clinics, enrolled 2012–2013 |
| Eligibility | One eligible eye each, vision between 20/32 and 20/320 |
| Arms | Aflibercept 224 · Bevacizumab 218 · Ranibizumab 0.3 mg 218 |
| Dosing | Every 4 weeks at first, then given or deferred by a fixed protocol rule; laser allowed from week 24 if edema persisted |
| Follow-up | 2 years |
| Primary outcome | Change in vision at 1 year, in ETDRS letters (5 letters ≈ one chart line) |

### 7.2 Column rules

| Prefix | Meaning | Rule |
|---|---|---|
| `pt_id`, `treatment`, `study_eye` | identity + randomized arm | `treatment` is what counterfactuals are computed over |
| `X_` | baseline features (at or before randomization) | the **only** model inputs |
| `Y_` | outcomes (after randomization) | targets only, **never** inputs |
| `M_` | bookkeeping | missing-data flags, completion status |

### 7.3 Features (`X_`)

| Group | Columns |
|---|---|
| Demographics | `age`, `female`, `race`, `hispanic` |
| Diabetes | `diabetes_type`, `diabetes_duration_yrs`, `insulin`, `hba1c` |
| General health | `bmi`, `sbp`, `dbp`, `smoker_current`, `hypertension`, `prior_mi`, `cad`, `prior_stroke`, `renal_disease`, `log_uacr` |
| Eye | `va_letters`, `fellow_va_letters`, `cst_um`, `oct_machine`, `subretinal_fluid`, `epiretinal_membrane`, `iop`, `pseudophakic`, `dr_severity_clinical`, `dr_severity_rc` |
| Eye history | `prior_dme_trt`, `prior_antivegf`, `prior_focal_laser`, `prior_prp` |
| Published subgroup | `va_below_69` (vision 20/50 or worse) |

### 7.4 Outcomes (`Y_`)

- `va_change_wk52` (**primary**), `va_change_wk104`: letters gained
- `cst_change_wk52`, `cst_change_wk104`: change in retinal thickness (µm)
- `n_inj_yr1`, `n_inj_total`: study-eye injections (used by the cost model)
- `laser_yr1`, `laser_any`: needed rescue laser

### 7.5 Caveats

- `X_cst_um` is in each OCT machine's own units: Spectralis and Cirrus read thicker than Stratus. M1 harmonizes it; always keep `X_oct_machine` available.
- 44 patients have no 1-year vision measurement (`M_has_wk52 = 0`). The published analysis used multiple imputation; M1 compares complete-case, KNN and MICE.
- Visit-level data for M7 comes from the raw `TVAReVATest` table, not the analysis table.

## 8. How we evaluate counterfactual predictions

A patient's outcome under the drugs they did *not* receive is never observed, so ordinary accuracy cannot measure the core model. Equipoise uses three checks:

1. **Semi-synthetic benchmark.** Keep the real patients' features, simulate outcomes with a *known* treatment effect, and measure how well each learner recovers it (PEHE). One scenario has **no** heterogeneity, to catch models that invent it.
2. **Policy value on held-out patients.** Treatment probabilities are known from randomization, so inverse-probability weighting gives an unbiased estimate of "how well would patients do if they followed the model's choice?". This is compared against treat-everyone-with-X policies and the published "vision < 69 → aflibercept" rule.
3. **Rediscovery.** The model should find, unprompted, that aflibercept's benefit is concentrated in patients with worse baseline vision.

Standard regression and classification metrics (RMSE, MAE, R², accuracy, precision, recall, F1, ROC-AUC) are reported for the outcome models in M5.

## 9. Results

### 9.1 Replication of the published trial (complete cases)

| 1-year result | Published (Afl / Bev / Ran) | Our table |
|---|---|---|
| Mean letters gained | 13.3 / 9.7 / 11.2 | 13.3 / 9.4 / 11.2 |
| Baseline 20/50 or worse | 18.9 / 11.8 / 14.2 | 19.1 / 11.8 / 14.2 |
| Baseline 20/32–20/40 | 8.0 / 7.5 / 8.3 | 7.8 / 7.0 / 8.3 |
| Median injections | 9 / 10 / 10 | 9 / 10 / 10 |
| Laser by 1 year | 37% / 56% / 46% | 37% / 56% / 47% |
| CST change (µm) | −169 / −101 / −147 | −165 / −98 / −147 |

At 2 years our table gives:
- 12.9 / 9.6 / 12.2 letters gained (published, Wells et al., *Ophthalmology* 2016: about 12.8 / 10.0 / 12.3),
- a median of 15 / 16 / 15 total injections,
- 41% / 64% / 52% laser.

### 9.2 Model results

All numbers below come from `reports/tables/` (real data, fixed 80/20 split, seed 42).
The held-out test set was used once, for reporting.

**Outcome models (M5)**, predicting week-52 letters (`m5_regression_comparison.md`):

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Baseline: arm mean | 10.91 | 10.80 | 0.00 |
| OLS | 9.56 | 9.50 | 0.23 |
| Ridge | 9.43 | 9.45 | 0.23 |
| Lasso (best by CV) | 9.39 | 9.43 | 0.24 |
| Random forest | 9.71 | 9.61 | 0.21 |
| Gradient boosting | 9.90 | 10.2 | 0.10 |

- Gaining ≥ 15 letters: logistic regression, test AUC 0.81 (`m5_classification_gain15.md`).
- Laser in year 1 is barely predictable: test AUC 0.58, vs 0.57 for the arm-rate baseline.
- Year-1 injections: Poisson GLM, test MAE 1.73 vs 1.90 for the baseline.

**Semi-synthetic PEHE (M6)** in letters, mean of 10 replications (`m6_learner_selection.md`):

| Learner | Null | Published-like | Nonlinear | Found heterogeneity in null |
|---|---|---|---|---|
| Causal forest (chosen) | 1.80 | 2.80 | 2.44 | 0% |
| TARNet (RMSprop) | 4.37 | 5.28 | 4.64 | 90% |
| X-learner (GB) | 6.02 | 5.91 | 6.09 | 100% |
| T-learner (GB) | 7.54 | 7.38 | 7.50 | 100% |
| DR-learner (GB) | 9.08 | 9.36 | 9.33 | 100% |
| Constant effect (reference) | 1.19 | 2.88 | 2.51 | n/a |

**Policy value on held-out patients** in letters gained, IPW with bootstrap 95% CI
(`m6_policy_values.md`): all-bevacizumab 11.7 (8.1–17.1), all-aflibercept 14.9 (9.7–20.7),
all-ranibizumab 11.2 (7.6–15.0), published < 69-letter rule 15.7 (10.8–21.7), model
(most letters) 14.9 (9.9–21.0). The model policy gives everyone aflibercept, and all
intervals overlap, so no policy is distinguishable on 126 test patients.

**Rediscovery** (`m6_rediscovery.md`):
- The predicted aflibercept − bevacizumab effect is larger when baseline vision is below
  69 letters (3.76 vs 2.59 letters, held-out): passes.
- Baseline vision is the *second* SHAP driver, after CST: the top-driver check fails.

**Honest bottom line.** Per-patient benefit intervals are wide (median 6.4 letters for the
causal forest). Every held-out patient is in equipoise at MCID 5, and also at 3
(`m8_population_summary.md`). Protocol T supports the published baseline-vision subgroup
finding, but not finer per-patient drug choices.

**Early response (M7):** adding vision at weeks 4/8/12 lowers the week-52 test RMSE from
9.47 to 8.21 letters (`m7_early_response.md`).

**OCT screening (M10, optional, separate Kermany dataset, not Protocol T)**
(`m10_comparison.md`): on the official test set (968 scans, patients disjoint from
training) YOLO-cls reaches 98.8% accuracy and ResNet18 94.8%. Patient-grouped validation
accuracy is lower (86.4% and 88.7%), because the curated test set is easier than typical
scans.

## 10. Course coverage

| Area | Covered in |
|---|---|
| Data preprocessing, integration, reduction, transformation, discretization | M0, M1 |
| PCA, SVD | M3 |
| Regression, classification, kNN, trees, Bayesian, metrics, cross-validation | M5, M6 |
| Clustering, outliers, Apriori | M3, M1 |
| SGD, Adam, RMSprop, hyperparameter tuning, bio-inspired optimization | M6, M4 |
| Statistical analysis, feature selection (filter / embedded / wrapper) | M2, M4 |
| Time series, ARIMA | M7 |
| Hypothesis testing, ANOVA, chi-square, Mann-Whitney, F-test | M2 |
| Newton-Raphson, interpolation, splines, finite differences | M5, M8, M7 |
| HMM, forward algorithm, Viterbi | M7 |
| Intelligent agents, expert systems, fuzzy logic | M8 |
| GA, PSO | M4, M6 |
| Self-study: CV, deployment, Git, experiment tracking, AutoML, MLOps, search space, data structures | M10, M11, M4, `docs/data_structures.md` |

## 11. Limitations

- **Small sample.** 660 patients, about 220 per arm. Per-patient effects beyond baseline vision may be too weak to detect reliably, and the app says so when that is the case.
- **Population.** US patients, 2012–2013. Ranibizumab was given at 0.3 mg, a US dose; India commonly uses 0.5 mg.
- **Imaging module.** The OCT module uses a separate public dataset (Kermany et al.). Its images are not linked to Protocol T patients.
- **Prices.** Costs are user-editable inputs, not market data.
- **Not clinical advice.** This is decision-support research for a course project, not a medical device.

## 12. Team

- Students: *names*
- Guide: *name*
- MKSSS's Cummins College of Engineering for Women, Pune · Third Year B.Tech Computer Engineering · 2026–27

## 13. Data attribution (required)

> The source of the data is the DRCR Retina Network (FAIN UG1EY014231). Protocol T: A Comparative Effectiveness Study of Intravitreal Aflibercept, Bevacizumab and Ranibizumab for Diabetic Macular Edema [Data file]. Retrieved from https://public.jaeb.org/drcrnet. The analyses content and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the DRCR Retina Network.

Raw and derived DRCR data are not redistributed in this repository.
