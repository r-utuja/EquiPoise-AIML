# Handbook concept inventory

Built from a scan of every tracked Python file (`scripts/handbook/scan_code.py`,
`scripts/handbook/scan_usage.py`), the configs, `dvc.yaml`, `.github/workflows/ci.yml`,
`docker/`, `docs/`, `reports/REPORT_ASSETS.md`, `README.md` (§10 course coverage) and
`CLAUDE.md`. "Where used" names `file:function` (paths relative to the repo root;
`eq/` = `src/equipoise/`). Chapter numbers refer to the proposed outline at the end.

`scripts/handbook/check_coverage.py` (written in the final step) will search the chapters for
every name in the **Concept** column and report gaps.

## 1. Domain (medicine and the trial)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Diabetes mellitus, type 1 / type 2 | Domain | `X_diabetes_type`; `eq/data/schema.py:DIABETES_TYPES` | 02 |
| HbA1c | Domain | `X_hba1c`; `config preprocess.log_features`, `bands` | 02, 04 |
| Retina, macula, fovea | Domain | background for CST / OCT | 02 |
| Diabetic retinopathy (DR) and its severity scale | Domain | `X_dr_severity_clinical`, `X_dr_severity_rc`; `eq/preprocess/columns.py:ORDINAL` | 02, 04 |
| Diabetic macular edema (DME) | Domain | whole project; M10 class `DME` | 02 |
| Optical coherence tomography (OCT), B-scan | Domain | `X_oct_machine`; M10 images | 02, 18 |
| Central subfield thickness (CST, µm) | Domain | `X_cst_um`; `Y_cst_change_wk52` | 02, 04 |
| OCT machines: Stratus (time-domain) vs Cirrus / Spectralis (spectral-domain) | Domain | `eq/preprocess/harmonize.py:CSTHarmonizer`; `config preprocess.cst.offset_um` | 02, 09 |
| Reading centre, "Unacceptable" CST grade, machine switch | Domain | `scripts/build_cst_quality.py:main`; `eq/preprocess/cst_quality.py:apply_cst_quality` | 04, 09 |
| Visual acuity (VA), ETDRS letters, Snellen 20/x and 6/x, "lines" | Domain | `X_va_letters`; `app/presenters.py:snellen_to_letters`, `letters_to_lines`; `config ui.snellen` | 02 |
| Snellen ↔ letters formula: letters ≈ 85 + 50·log10(20/denominator) | Domain / Formula | `app/presenters.py:snellen_to_letters` | 02 |
| Trial eligibility range 24–78 letters (20/320–20/32) | Domain | `config trial.va_eligibility_letters` | 04 |
| "20/50 or worse" subgroup, `X_va_below_69` | Domain | `eq/decision/expert.py:default_rules` (published-subgroup); `X_va_below_69` | 02, 04, 14 |
| Fellow eye, study eye, OD / OS | Domain / Data | `scripts/build_protocolT_table.py:pick_eye`; `X_fellow_va_letters` | 04, 08 |
| VEGF and anti-VEGF drugs (aflibercept / Eylea, bevacizumab / Avastin, ranibizumab / Lucentis) | Domain | `config trial.arms`; `app/presenters.py:brand` | 02 |
| Off-label use, biosimilars, brand vs generic | Domain | README §1 table | 02 |
| Ranibizumab 0.3 mg (US) vs 0.5 mg (India) dose | Domain caveat | README §11 | 02, 04, 27 |
| Intravitreal injection, injection burden | Domain | `Y_n_inj_yr1`, `Y_n_inj_total`; `eq/decision/cost.py:expected_cost` | 02, 16 |
| Focal/grid laser, panretinal photocoagulation (PRP), rescue laser | Domain | `X_prior_focal_laser`, `X_prior_prp`, `Y_laser_yr1` | 02, 04 |
| Pseudophakia, epiretinal membrane, subretinal fluid, intraocular pressure | Domain | `X_pseudophakic`, `X_epiretinal_membrane`, `X_subretinal_fluid`, `X_iop` | 04 |
| Blood pressure (SBP/DBP), BMI, hypertension, MI, CAD, stroke, renal disease, UACR | Domain | `X_sbp`… `X_log_uacr` | 04 |
| CNV, drusen (Kermany classes) | Domain | `config vision.classes` | 18 |
| Cost of treatment in India; rupees, lakh, crore | Domain | `app/presenters.py:format_inr`; `config/costs.yaml` | 02, 16 |
| DRCR Retina Network, Protocol T design (arms 224/218/218, 2 years, retreatment rules) | Domain / Data | `config trial`; `Protocol_T/*.pdf` | 04 |
| Published Protocol T results (Wells 2015 NEJM, Wells 2016 Ophthalmology) | Domain | `config/replication_targets.yaml` | 04, 10, 30 |
| DRCR data attribution text | Data / Ethics | README §13; `eq/serving/service.py:ATTRIBUTION` | 04, 27, 30 |
| Clinical equipoise | Domain / Causal | `eq/decision/expert.py:default_rules` (equipoise-*, all-equipoise) | 03, 16 |
| MCID (minimal clinically important difference) = 5 letters; sensitivity 3 | Domain / Config | `config effects.mcid_letters`, `mcid_sensitivity_letters` | 03, 16 |

## 2. Causal inference

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Correlation vs causation | Causal | README §2 | 03 |
| Confounding, confounder | Causal | `eq/report/diagrams.py:causal_dag` | 03 |
| Randomization, randomized controlled trial (RCT) | Causal | `eq/causal/data.py:known_propensities` | 03 |
| Causal DAG (directed acyclic graph) | Causal / Data structure | `eq/report/diagrams.py:causal_dag` (networkx.DiGraph); `reports/figures/m11_causal_dag.png` | 03, 19 |
| Potential outcomes Y(a), counterfactual | Causal | `scripts/make_synthetic_data.py:potential_outcomes`; `eq/causal/tarnet.py:TARNetLearner.potential_outcomes` | 03 |
| Fundamental problem of causal inference | Causal | `eq/causal/semisynth.py` (docstring) | 03, 14 |
| Average treatment effect (ATE) | Causal | `eq/causal/learners.py:ConstantEffect` | 03, 14 |
| CATE / ITE (conditional / individual treatment effect) | Causal | `eq/causal/learners.py:CATEModel` | 03, 14 |
| Treatment effect heterogeneity, effect modifier | Causal | `eq/stats/inference.py:interaction_regression`; `config causal.heterogeneity_rule` | 03, 10, 14 |
| Reference arm (bevacizumab), comparison arms | Causal / Config | `config trial.reference_arm`; `eq/causal/data.py:arm_codes` | 03, 14 |
| Propensity score; known propensities (≈0.34/0.33/0.33) | Causal | `eq/causal/data.py:KnownPropensity`, `known_propensities` | 03, 14 |
| Intention-to-treat, complete-case analysis | Causal / Stats | `eq/preprocess/missing.py:compare_strategies` | 09, 10 |
| S-learner | Model | `eq/causal/learners.py:EconMLMeta._make` (econml SLearner) | 14 |
| T-learner | Model | `eq/causal/learners.py:EconMLMeta._make` (econml TLearner) | 14 |
| X-learner | Model | `eq/causal/learners.py:EconMLMeta._make` (econml XLearner) | 14 |
| DR-learner (doubly robust) | Model | `eq/causal/learners.py:EconMLMeta._make` (econml DRLearner) | 14 |
| Constant-effect model (difference in means yardstick) | Model | `eq/causal/learners.py:ConstantEffect` | 14 |
| Causal forest / honest forest / DML (CausalForestDML) | Model | `eq/causal/learners.py:CausalForest.fit`, `effect_interval`, `feature_importances` | 14 |
| Double machine learning (DML), residualization | Causal | `eq/causal/learners.py:CausalForest` | 14 |
| Cross-fitting | Causal / ML | `config causal.cv_folds`; `eq/explain/fairness.py:out_of_sample_predictions` | 14, 17 |
| TARNet (shared representation + per-arm heads) | Model | `eq/causal/tarnet.py:TARNet`, `train_tarnet`, `TARNetLearner` | 14 |
| Factual vs counterfactual loss | Causal / DL | `eq/causal/tarnet.py:train_tarnet` | 14 |
| AutoGluon T-learner | Model / AutoML | `eq/causal/automl.py:AutoGluonTLearner` | 14, 19 |
| Semi-synthetic benchmark (null / published-like / nonlinear scenarios) | Evaluation | `eq/causal/semisynth.py:fit_simulator`, `scenario_effects`, `simulate`, `run_benchmark`, `summarize` | 14 |
| PEHE (precision in estimating heterogeneous effects) | Metric | `eq/causal/semisynth.py:run_benchmark` | 14 |
| ATE bias, "found heterogeneity in null" (tertile gap > MCID) | Metric | `eq/causal/semisynth.py:tertile_gap`; `config causal.selection.max_abs_ate_bias` | 14 |
| Learner selection rule (constant-effect exclusion → bias filter → worst-case PEHE) | Method / Config | `eq/causal/run.py:select_learner`; `config causal.selection` | 14 |
| Policy, treatment policy, "treat all with X", published < 69 rule | Causal | `eq/causal/policy.py:evaluate_policies` | 14 |
| IPW (inverse probability weighting) policy value | Causal / Metric | `eq/causal/policy.py:ipw_value` | 14 |
| AIPW / doubly robust policy value | Causal / Metric | `eq/causal/policy.py:aipw_value` | 14 |
| Cost-penalized policy (λ letters per cost unit) | Causal / Config | `config causal.policy.lambda_letters_per_cost_unit`; `eq/causal/run.py:run` | 14 |
| Calibration by predicted-effect tertile | Evaluation | `eq/causal/run.py:_calibration`, `_tertile_plot` | 14 |
| Rediscovery check (effect larger when VA < 69; top SHAP driver) | Evaluation | `eq/causal/run.py:run`; `reports/tables/m6_rediscovery.csv` | 14 |
| Bootstrap within arm, per-patient intervals, P(benefit > MCID) | Stats / Causal | `eq/causal/run.py:_bootstrap`; `eq/causal/bundle.py:ModelBundle.predict` | 14 |
| Surrogate random forest for effect importance | Explainability | `eq/causal/run.py:_importance_rank` | 14 |
| M4 subset tested in M6 | Evaluation | `reports/tables/m6_m4_subset_effect.csv`; `eq/causal/data.py:prepare(features=…)` | 12, 14 |
| "No clear winner" for every held-out patient | Finding | `reports/tables/m8_heldout_no_clear_winner.csv`; `eq/serving/examples.py:heldout_summary` | 14, 16, 26 |

## 3. Data, schema and integration (M0)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Relational tables, primary key, join / merge on `PtID` | Data | `scripts/build_protocolT_table.py:main` | 08 |
| "One row per form" raw tables; reshaping to one row per patient | Data | `scripts/build_protocolT_table.py:main` | 04, 08 |
| Study-eye selection (`RandEye`, `pick_eye`) | Data | `scripts/build_protocolT_table.py:pick_eye` | 04, 08 |
| Yes/No → 0/1 mapping | Data | `scripts/build_protocolT_table.py:yes` | 08 |
| Days from randomization (relative dates) | Data | `scripts/build_protocolT_visits.py:visits_with_va` | 04, 08 |
| Visit key `ParentLoginVisitID`; OCT and treatment windows | Data / Config | `scripts/build_protocolT_visits.py:attach_oct`, `attach_treatments`; `config visits.*` | 08 |
| Post-randomization tables banned as features | Data / Leakage | `eq/features.py:POST_RANDOMIZATION_TABLES`; `tests/test_leakage.py` | 04, 05, 08 |
| Column roles: id / `X_` / `Y_` / `M_` / `E_` | Data | `eq/data/schema.py:ColumnSpec`; `eq/features.py` | 04 |
| Every column of `protocolT_analysis.csv` (47 columns) | Data | `eq/data/schema.py:ANALYSIS_COLUMNS`; `reports/tables/m11_feature_description.csv` | 04 |
| Visit table (patient × visit, long format) | Data | `scripts/build_protocolT_visits.py`; `eq/data/schema.py:VISITS_COLUMNS` | 04, 08, 15 |
| Wide vs long data format | Data | analysis vs visit table | 04, 08 |
| Schema validation, data types (dtype), nullability, allowed ranges | Data / Eng | `eq/data/schema.py:validate_analysis`, `validate_visits`, `numeric_casts`, `dtype_name` | 08 |
| Cross-check visit table vs analysis table | Data / Testing | `eq/data/crosscheck.py:cross_check`; `reports/tables/m0_visits_crosscheck.csv` | 08 |
| Synthetic data (planted effect 7.0 / 0.5 / 2.0, noise SD 9.0, dropout) | Data | `scripts/make_synthetic_data.py:*`; `config synthetic.*` | 04 |
| Synthetic ground truth `protocolT_truth.csv` | Data | `scripts/make_synthetic_data.py:potential_outcomes` | 04, 14 |
| Train/test split (80/20, stratified by arm, seed 42, `data/splits.json`) | ML | `eq/data/splits.py:make_split`, `create_split`, `get_split` | 05, 08 |
| Stratification | ML | `eq/data/splits.py:make_split`, `cv_folds` | 05 |
| Missing week-52 vision (44 patients), `M_has_wk52` | Data | `config trial.n_missing_wk52` | 04, 09 |
| Kermany OCT dataset (Kaggle `kermany2018`), balanced subset, official test split | Data | `scripts/download_oct_subset.py:choose`; `config vision.*` | 04, 18 |
| HTTP range requests / reading a zip's central directory | Eng | `scripts/download_oct_subset.py:RangeFile`, `fetch_member` | 18 |
| Patient-grouped split (GroupShuffleSplit) | ML | `eq/vision/data.py:assign_splits` | 18 |
| Data hash (SHA-256) as data version | Eng | `eq/tracking.py:file_hash` | 19 |

## 4. Preprocessing (M1)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Missing data: MCAR / MAR / MNAR | Stats | `eq/preprocess/missing.py` (docstring) | 09 |
| Missingness table | Preprocessing | `eq/preprocess/missing.py:missingness_table` | 09 |
| Median / mode imputation (SimpleImputer) | Preprocessing | `eq/preprocess/encode.py:make_imputer`, `build_preprocessor` | 09 |
| kNN imputation (KNNImputer) | Preprocessing | `eq/preprocess/encode.py:make_imputer`; `eq/preprocess/missing.py:compare_strategies` | 09 |
| MICE / IterativeImputer, multiple imputation (m = 20) | Preprocessing | `eq/preprocess/missing.py:compare_strategies`; `config preprocess.mice_imputations` | 09 |
| Rubin's rules (within/between variance, fraction of missing info) | Stats | `eq/preprocess/missing.py:rubin_pool` | 09 |
| Imputer scoring by masking observed values (normalized RMSE) | Evaluation | `eq/preprocess/missing.py:score_feature_imputers` | 09 |
| Log transform for skewed features (HbA1c, BMI); `log(1+UACR)` | Preprocessing | `eq/preprocess/encode.py:build_preprocessor` (FunctionTransformer np.log); `config preprocess.log_features` | 09 |
| Skewness | Stats | `config stats.skew_threshold`; `eq/stats/table1.py:table1` | 06, 10 |
| Standard scaling (z-score, StandardScaler) | Preprocessing | `eq/preprocess/encode.py:build_preprocessor` | 09 |
| Min-max scaling | Preprocessing | `eq/search/selection.py:rank_features` | 12 |
| One-hot encoding, `drop="first"`, rare-category pooling (`min_frequency`) | Preprocessing | `eq/preprocess/encode.py:build_preprocessor`; `config supervised.one_hot_min_frequency` | 09 |
| Sparse matrix (CSR) | Data structure | `eq/preprocess/encode.py:one_hot_sparse`; `docs/data_structures.md` | 09 |
| Ordinal encoding (DR severity) | Preprocessing | `eq/preprocess/encode.py:build_preprocessor`; `eq/preprocess/columns.py:ORDINAL` | 09 |
| Feature groups (nominal / ordinal / binary / log / continuous / CST) | Preprocessing | `eq/preprocess/columns.py:feature_groups` | 09 |
| CST harmonization: z-score within machine / published offset / equations | Preprocessing | `eq/preprocess/harmonize.py:CSTHarmonizer`, `harmonizer_from_config`; `config preprocess.cst.method` | 09 |
| Custom sklearn transformer (fit/transform, TransformerMixin) | Library | `eq/preprocess/harmonize.py:CSTHarmonizer` | 07, 09 |
| CST quality policy (keep vs missing; sensitivity analysis) | Preprocessing | `eq/preprocess/cst_quality.py:apply_cst_quality`; `config preprocess.cst.quality_policy` | 09 |
| Outlier detection: IQR rule (k = 1.5, ≥ 2 features) | Preprocessing | `eq/preprocess/outliers.py:iqr_flags` | 09 |
| Isolation Forest (contamination 0.05) | Preprocessing | `eq/preprocess/outliers.py:detect_outliers` | 09 |
| DBSCAN; eps from k-distance curve (95th percentile) | Preprocessing / Clustering | `eq/preprocess/outliers.py:detect_outliers`; `eq/represent/cluster.py:dbscan_labels` | 09, 11 |
| Outlier overlap table (flag, never drop) | Preprocessing | `eq/preprocess/outliers.py:overlap_table` | 09 |
| Discretization / binning (HbA1c, VA, age bands; right-open intervals) | Preprocessing | `eq/preprocess/discretize.py:discretize`; `config preprocess.bands` | 09 |
| "Gained ≥ 15 letters" item | Preprocessing | `eq/preprocess/discretize.py:gain_item`; `config preprocess.gain_threshold_letters` | 09, 11, 13 |
| `ColumnTransformer` and `Pipeline` | Library | `eq/preprocess/encode.py:build_preprocessor`; `eq/supervised/models.py:_pipe` | 07, 09 |
| Fit on training folds only | ML / Leakage | `tests/test_preprocess.py:test_imputer_fitted_on_train_only` | 05, 09 |

## 5. Statistics and inference (M2)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Mean, median, variance, SD, IQR | Stats | `eq/stats/table1.py:table1`; `eq/stats/replication.py:burden_table` | 06 |
| Table 1 (baseline characteristics by arm) | Stats | `eq/stats/table1.py:table1`, `n_significant` | 10 |
| Normal, t, chi-square, F distributions | Stats | `scipy.stats.t`, `f` in `eq/stats/*` | 06 |
| Standard error, confidence interval (t-based) | Stats | `eq/stats/replication.py:mean_ci` | 06, 10 |
| Hypothesis test, null/alternative, p-value, α = 0.05, one- vs two-tailed | Stats | `config stats.alpha` | 06, 10 |
| Type I / Type II error, power | Stats | `eq/stats/table1.py` (docstring on chance findings) | 06 |
| One-way ANOVA | Test | `eq/stats/replication.py:hypothesis_tests`; `eq/stats/table1.py:table1` | 10 |
| Kruskal–Wallis test | Test | `eq/stats/table1.py:table1` | 10 |
| Chi-square test of independence (contingency table) | Test | `eq/stats/table1.py:table1`; `eq/stats/replication.py:hypothesis_tests`; `eq/longitudinal/clusters.py:cluster_trajectories` | 10, 15 |
| Welch's t-test, Welch–Satterthwaite CI | Test | `eq/stats/replication.py:welch_ci` | 10 |
| Holm multiple-testing correction | Stats | `eq/stats/replication.py:hypothesis_tests` (multipletests) | 06, 10 |
| Mann–Whitney U test | Test | `eq/stats/replication.py:hypothesis_tests` | 10 |
| Levene's test, Bartlett's test, F-test for variances | Test | `eq/stats/inference.py:variance_tests` | 10 |
| Pearson correlation, correlation matrix / heatmap | Stats | `eq/stats/run.py:_corr_heatmap`; `reports/tables/m2_correlation_matrix.csv` | 06, 10 |
| Partial correlation (residual method) | Stats | `eq/stats/inference.py:partial_correlation` | 10 |
| Multiple correlation R, R², adjusted R², overall F | Stats | `eq/stats/inference.py:multiple_correlation` | 10 |
| Interaction regression `y ~ C(treatment) * modifier`, nested-model F-test | Stats | `eq/stats/inference.py:interaction_regression` (statsmodels ols, anova_lm) | 10 |
| Replication vs published targets (tolerance 0.2 letters) | Stats / Testing | `eq/stats/replication.py:versus_targets`; `config stats.replication_tolerance_letters` | 10 |
| CST change sensitivity analysis | Stats | `reports/tables/m2_cst_change_sensitivity.csv` | 10 |

## 6. Representation (M3)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Vectors, matrices, transpose, matrix multiply | Maths | throughout; `eq/supervised/scratch.py:normal_equations` | 06 |
| Eigenvalues / eigenvectors | Maths | PCA theory (`eq/represent/pca.py`) | 06, 11 |
| SVD (X = U S Vᵀ) | Maths / Algorithm | `eq/represent/pca.py:svd_pca` (np.linalg.svd) | 06, 11 |
| PCA, principal components, loadings, scores, explained variance, scree plot | Algorithm | `eq/represent/pca.py:svd_pca`, `compare_with_sklearn`; `eq/represent/run.py:_scree` | 11 |
| Sign ambiguity of components | Maths | `eq/represent/pca.py:svd_pca` | 11 |
| k-means, inertia / elbow | Clustering | `eq/represent/cluster.py:kmeans_scan` | 11 |
| Silhouette score | Metric | `eq/represent/cluster.py:kmeans_scan`; `eq/longitudinal/clusters.py` | 11 |
| Hierarchical clustering, Ward linkage, dendrogram | Clustering | `eq/represent/cluster.py:ward_labels`; `eq/represent/run.py:_dendrogram_plot` | 11 |
| Effects by cluster | Analysis | `eq/represent/cluster.py:effects_by_cluster` | 11 |
| Apriori, itemset, support, confidence, lift, baskets | Algorithm | `eq/represent/rules.py:baskets`, `mine_rules`; `config represent.apriori` | 11 |
| Unsupervised learning | ML | M3 | 05, 11 |

## 7. Feature search (M4)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Feature selection: filter / embedded / wrapper | ML | `eq/search/selection.py:rank_features` | 12 |
| Mutual information | Stats | `eq/search/selection.py:rank_features` (mutual_info_regression) | 12 |
| Chi-square filter on min-max scaled features | Stats | `eq/search/selection.py:rank_features` (chi2) | 12 |
| Lasso path | Algorithm | `eq/search/selection.py:lasso_path_entry` | 12 |
| Random forest importance; RFE (recursive feature elimination) | Algorithm | `eq/search/selection.py:rank_features` | 12 |
| Search space (2ⁿ binary masks), neighbourhood, local optimum, plateau | Search | `eq/search/algorithms.py` | 12 |
| Objective J(mask) = CV RMSE + penalty × #features | Search | `eq/search/objective.py:Evaluator`, `make_evaluator`; `config search.feature_penalty` | 12 |
| Closed-form Ridge (centred, unpenalized intercept) | Algorithm (scratch) | `eq/search/objective.py:ridge_fit_predict` | 12 |
| Evaluation budget, memoisation | Search / Eng | `eq/search/objective.py:Evaluator`, `BudgetExhausted`; `config search.budget` | 12 |
| Random search (yardstick) | Algorithm (scratch) | `eq/search/algorithms.py:random_search` | 12 |
| Hill climbing (steepest descent) with random restarts | Algorithm (scratch) | `eq/search/algorithms.py:hill_climbing`; `eq/search/run.py:_hill_climb_diagnostic` | 12 |
| Beam search; heap / priority queue (`heapq`) | Algorithm (scratch) / Data structure | `eq/search/algorithms.py:beam_search` | 12 |
| Tabu search, tabu tenure, aspiration; deque | Algorithm (scratch) / Data structure | `eq/search/algorithms.py:tabu_search` | 12 |
| Genetic algorithm: roulette selection, 1/2-point crossover, bit-flip mutation, elitism | Algorithm (scratch) | `eq/search/algorithms.py:genetic_algorithm`; `config search.ga` | 12 |
| Binary PSO (sigmoid transfer), inertia, c1/c2, v_max | Algorithm (scratch) | `eq/search/algorithms.py:binary_pso`; `config search.pso` | 12 |
| Continuous PSO (used for M6 tuning) | Algorithm (scratch) | `eq/search/algorithms.py:continuous_pso` | 12, 14 |
| Convergence curve (best-so-far) | Evaluation | `eq/search/run.py:_convergence_plot` | 12 |
| Bio-inspired optimization | Concept | GA, PSO | 12 |

## 8. Supervised models and metrics (M5)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Regression vs classification vs count outcomes | ML | `eq/supervised/models.py:*_models` | 05, 13 |
| Baseline model (arm mean, arm rate, DummyClassifier) | Model | `eq/supervised/scratch.py:ArmMeanRegressor`, `ArmRateClassifier` | 13 |
| Linear regression / OLS; normal equations | Model / Algorithm (scratch) | `eq/supervised/scratch.py:normal_equations`, `NormalEquationsRegressor`; `evaluate.py:normal_equations_check` | 06, 13 |
| Matrix inverse vs `solve`; condition number; rank / full-rank design | Maths | `eq/supervised/scratch.py:normal_equations`; `encode.py:build_preprocessor(full_rank)` | 06, 13 |
| Regularization: Ridge (L2), Lasso (L1), α grid (RidgeCV, LassoCV) | Model | `eq/supervised/models.py:regression_models`; `config supervised.ridge_alphas` | 13 |
| k-nearest neighbours (regressor / classifier); KD-tree, ball tree, brute force | Model / Data structure | `eq/supervised/models.py`; `evaluate.py:knn_algorithm_timing` | 13 |
| Decision tree (max_depth, min_samples_leaf) | Model | `eq/supervised/models.py:regression_models`, `classification_models` | 13 |
| Random forest, bagging | Model | `eq/supervised/models.py:regression_models` | 13 |
| Gradient boosting (learning rate, subsample); HistGradientBoosting (Poisson loss) | Model | `eq/supervised/models.py:regression_models`, `count_models` | 13 |
| Logistic regression, sigmoid, log-odds | Model | `eq/supervised/models.py:classification_models` | 13 |
| IRLS / Newton–Raphson logistic regression | Algorithm (scratch) | `eq/supervised/scratch.py:IRLSLogisticRegression`; `evaluate.py:irls_check` | 13 |
| Maximum likelihood, likelihood | Stats | IRLS, statsmodels Logit | 06, 13 |
| Naive Bayes (Gaussian, Bernoulli), Bayes' rule | Model | `eq/supervised/models.py:classification_models` | 06, 13 |
| Poisson regression (GLM), overdispersion check | Model | `eq/supervised/models.py:count_models`; `reports/tables/m5_count_dispersion.csv` | 13 |
| Class imbalance, `class_weight="balanced"` | ML | `eq/supervised/models.py:class_weight_variants` | 13 |
| Classification threshold 0.5 | Config | `config supervised.classification_threshold` | 13 |
| RMSE, MAE, R² | Metric | `eq/supervised/evaluate.py:regression_metrics` | 13 |
| Mean Poisson deviance | Metric | `eq/supervised/evaluate.py:count_metrics` | 13 |
| Confusion matrix, accuracy, precision, recall/sensitivity, specificity, F1 | Metric | `eq/supervised/evaluate.py:classification_metrics` | 13 |
| ROC curve, ROC-AUC, precision–recall curve, average precision | Metric | `eq/supervised/run.py:_curves`; `evaluate.py:classification_metrics` | 13 |
| k-fold cross-validation (5 folds, stratified by arm) | ML | `eq/data/splits.py:cv_folds`; `eq/supervised/evaluate.py:evaluate` | 05, 13 |
| Held-out test set used once | ML | `eq/supervised/evaluate.py:evaluate` | 05, 13 |
| `clone`, `check_is_fitted`, BaseEstimator/Mixins | Library | `eq/supervised/scratch.py`; `evaluate.py:evaluate` | 07 |

## 9. Neural networks and optimizers (M6 TARNet, M10)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Neuron, weights, bias, layer, fully connected (Linear) | DL | `eq/causal/tarnet.py:TARNet.__init__` | 14 |
| Activation function (ELU) | DL | `eq/causal/tarnet.py:TARNet.__init__` | 14 |
| Forward pass | DL | `eq/causal/tarnet.py:TARNet.forward` | 14 |
| Loss function: MSE; cross-entropy (M10) | DL | `eq/causal/tarnet.py:train_tarnet`; `eq/vision/resnet.py:train` | 14, 18 |
| Gradient, chain rule, backpropagation | Maths / DL | `train_tarnet` (`loss.backward()`) | 06, 14 |
| Gradient descent, SGD, momentum (0.9) | Optimization | `eq/causal/tarnet.py:_make_optimizer`; `config causal.tarnet.momentum` | 14 |
| RMSprop, Adam, AdamW | Optimization | `eq/causal/tarnet.py:_make_optimizer`; `eq/vision/resnet.py:train` | 14, 18 |
| Learning rate | Hyperparameter | `config causal.tarnet.learning_rates` | 05, 14 |
| Epoch, batch, mini-batch (64; manual seeded batching in TARNet, `DataLoader` in M10) | DL | `eq/causal/tarnet.py:train_tarnet`; `eq/vision/resnet.py:_loader`; `config causal.tarnet.batch_size` | 14, 18 |
| Early stopping (patience 25), validation split, best-epoch weights | DL | `eq/causal/tarnet.py:train_tarnet` | 14 |
| Dropout, weight decay (L2) | Regularization | `config causal.tarnet.dropout`, `weight_decay` | 14 |
| Learning curves (train/val loss) | Evaluation | `eq/causal/run.py:_tarnet_curves`; `reports/figures/m6_tarnet_optimizers.png` | 14 |
| Outcome standardization (mean, SD) | DL | `eq/causal/tarnet.py:train_tarnet` | 14 |
| Tensor (PyTorch) | Data structure | `eq/causal/tarnet.py` | 06, 14 |
| CNN, convolution, pooling, ResNet18, residual block | DL | `eq/vision/resnet.py:build_model` | 18 |
| Transfer learning, ImageNet pretraining, freezing the backbone, fine-tuning | DL | `eq/vision/resnet.py:train`; `config vision.resnet.freeze_backbone_epochs` | 18 |
| Data augmentation (random horizontal flip), normalization | DL | `eq/vision/resnet.py:transforms` | 18 |
| YOLO classification (yolo11n-cls) | Model | `eq/vision/yolo.py:train` | 18 |
| Softmax, class probabilities | DL | `eq/vision/infer.py:OctScreener.predict` | 18 |
| Grad-CAM | Explainability | `eq/vision/resnet.py:gradcam_overlay` | 18 |

## 10. Hyperparameter tuning and AutoML

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Hyperparameters vs parameters | ML | `config causal.base_learners`, `supervised.*` | 05 |
| Grid search | Tuning | `eq/causal/tuning.py:grid_points`, `compare_tuners` | 14 |
| Optuna, TPE (Tree-structured Parzen Estimator) sampler | Tuning | `eq/causal/tuning.py:compare_tuners` | 14 |
| PSO as a tuner | Tuning | `eq/causal/tuning.py:compare_tuners` → `continuous_pso` | 14 |
| Search box / unit-free encoding, log scale | Tuning | `eq/causal/tuning.py:decode`, `box` | 14 |
| Same budget comparison (30 trials) | Tuning | `config causal.tuning.trials` | 14 |
| AutoML, AutoGluon presets, stacking, leaderboard | AutoML | `eq/causal/automl.py:AutoGluonTLearner`; `reports/tables/m6_autogluon_*.csv` | 14, 19 |

## 11. Longitudinal / time series (M7)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Trajectory, irregular visits, weekly grid | Time series | `eq/longitudinal/tensor.py:build_tensor` | 15 |
| 3-D tensor (patients × weeks × measures) | Data structure | `eq/longitudinal/tensor.py:TrajectoryTensor` | 06, 15 |
| Linear interpolation (np.interp), no extrapolation | Numerical | `eq/longitudinal/tensor.py:interpolate` | 15 |
| Cubic spline | Numerical | `eq/longitudinal/tensor.py:interpolate` (CubicSpline) | 15 |
| Lagrange polynomial interpolation; Runge's phenomenon | Numerical (scratch) | `eq/longitudinal/tensor.py:lagrange_coefficients`, `lagrange_eval`; `config longitudinal.lagrange` | 15 |
| Finite differences: forward, backward, central; error order (h vs h²) | Numerical (scratch) | `eq/longitudinal/derivatives.py:*` | 06, 15 |
| Mean trajectories per arm with bands | Analysis | `eq/longitudinal/run.py:_trajectory_plot` | 15 |
| Stationarity, ADF test, differencing | Time series | `eq/longitudinal/timeseries.py:adf_table` | 15 |
| Moving average (window 3) | Time series | `eq/longitudinal/timeseries.py:moving_average` | 15 |
| AR, MA, ARMA, ARIMA(p,d,q); order selection by AIC; forecasting | Time series | `eq/longitudinal/timeseries.py:fit_arima_by_aic` | 15 |
| AIC | Stats | `fit_arima_by_aic`; `reports/tables/m7_arima_aic_grid.csv` | 06, 15 |
| Hidden Markov model: states, transition matrix, Gaussian emissions, initial distribution | Probabilistic model | `eq/longitudinal/hmm.py:fit_hmm` (hmmlearn GaussianHMM) | 15 |
| EM / Baum–Welch training | Algorithm | `fit_hmm` (hmmlearn) | 15 |
| Forward algorithm (log space, logsumexp) | Algorithm (scratch) | `eq/longitudinal/hmm.py:forward_loglik` | 15 |
| Viterbi decoding (dynamic programming, backtracking) | Algorithm (scratch) | `eq/longitudinal/hmm.py:viterbi` | 15 |
| State labelling (improving / active / stable) | Analysis | `eq/longitudinal/hmm.py:label_states` | 15 |
| Trajectory clustering (fast / slow / non-responders) | Clustering | `eq/longitudinal/clusters.py:cluster_trajectories` | 15 |
| Early-response model (`E_` features: weeks 4/8/12 + slope) | Model | `eq/longitudinal/early_response.py:early_features`, `make_model`, `compare` | 15 |
| Least-squares slope | Maths | `eq/longitudinal/early_response.py:early_features` | 15 |

## 12. Decision layer (M8)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Expected cost = injections × price + P(laser) × laser price | Decision | `eq/decision/cost.py:expected_cost` | 16 |
| Cost per extra letter (ICER) | Decision | `eq/decision/cost.py:cost_per_letter` | 16 |
| DEMO prices (relative units 1/5/10), null prices, price override | Config / Ethics | `eq/config.py:load_prices`, `Prices`; `config/costs.yaml` | 16, 27 |
| Value of vision U(L) = 1 − exp(−kL), concave utility, k = 0.05 | Decision | `eq/decision/breakeven.py:value_of_vision`; `config decision.value_of_vision_k` | 16 |
| Willingness to pay (WTP), net benefit | Decision | `eq/decision/breakeven.py:break_even` | 16 |
| Root finding; Newton–Raphson (from scratch) | Numerical (scratch) | `eq/decision/breakeven.py:newton`; `config decision.newton` | 06, 16 |
| Derivative by central finite difference inside Newton | Numerical | `eq/decision/breakeven.py:newton` (`fd_step`) | 16 |
| Brent's method (brentq), bracketing | Numerical | `eq/decision/breakeven.py:brent_root` | 16 |
| Break-even price, break-even benefit, break-even curve | Decision | `eq/decision/breakeven.py:break_even`; `eq/serving/service.py:Service.breakeven` | 16 |
| Fuzzy set, membership function (triangular, trapezoidal), linguistic variable, universe | Fuzzy | `eq/decision/fuzzy.py:build` | 16 |
| Fuzzification, rule firing (AND = min, OR = max, NOT = 1 − μ), clipping, aggregation (max), centroid defuzzification | Fuzzy | `eq/decision/fuzzy.py:build`, `strength`, `aggregated_output` | 16 |
| The 10 fuzzy rules | Fuzzy | `eq/decision/fuzzy.py:RULE_TEXT` | 16 |
| Expert system: working memory, facts, rules, forward chaining, agenda, conflict resolution (salience, specificity), refraction | AI | `eq/decision/expert.py:Engine.run`, `Rule`, `default_rules` | 16 |
| Equipoise rule: claim a clear benefit only if the whole 95% interval ≥ MCID | Decision | `eq/decision/expert.py:default_rules` (`clear-benefit-*`, `equipoise-*`) | 14, 16 |
| Intelligent agent, PEAS, perceive–decide–act | AI | `eq/decision/agent.py:EquipoiseAgent` | 16 |
| Typical patient (training medians / modes) | Privacy / Method | `eq/decision/run.py:typical_patient` | 16, 22 |
| `recommend()` (the one function the API and app call) | Engineering | `eq/decision/recommend.py:recommend`, `patient_frame` | 16, 22 |
| LRU cache | Engineering | `eq/decision/recommend.py:_cached_bundle`, `_cached_fuzzy` | 22 |

## 13. Explainability and fairness (M9)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Shapley values, SHAP, additivity, base value | Explainability | `eq/explain/shap_effects.py:explain_effect` | 17 |
| Permutation explainer, Independent masker, background data | Explainability | `eq/explain/shap_effects.py:explain_effect`; `config explain.shap_background` | 17 |
| Summing one-hot SHAP to raw features | Explainability | `eq/explain/shap_effects.py:raw_feature_shap` | 17 |
| Global importance (mean \|SHAP\|), beeswarm, waterfall | Explainability | `eq/explain/shap_effects.py:global_importance`; `eq/explain/run.py:_beeswarm`, `_waterfall` | 17 |
| k-means centroids as SHAP background (no patient rows served) | Privacy / Eng | `eq/serving/assets.py:build_serving_assets`; `config serving.shap_background_clusters` | 17, 22 |
| Fairness, subgroup performance (sex, race, diabetes type, VA < 69), min n = 40 | Fairness | `eq/explain/fairness.py:subgroup_report`; `config explain.subgroups` | 17, 27 |
| Out-of-fold (out-of-sample) predictions | ML | `eq/explain/fairness.py:out_of_sample_predictions` | 17 |

## 14. OCT screening (M10)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Digital image, pixel, grayscale, uint8 | CV | `eq/vision/preprocess.py:read_gray`, `to_gray` | 18 |
| Speckle noise; median filter; non-local means | CV | `eq/vision/preprocess.py:denoise` | 18 |
| Histogram equalization, CLAHE | CV | `eq/vision/preprocess.py:clahe` | 18 |
| Resizing (INTER_AREA, 224 × 224) | CV | `eq/vision/preprocess.py:preprocess` | 18 |
| Patient overlap / leakage between image splits | ML | `eq/vision/data.py:patient_overlap`; `reports/tables/m10_patient_overlap.csv` | 18 |
| Per-class precision/recall/F1, confusion matrix | Metric | `eq/vision/run.py:metrics_rows`, `_confusion_plot` | 18 |
| Gradio demo | Engineering | `app/oct_gradio.py` | 18, 19 |
| AGPL licence of Ultralytics; analytics switched off | Ethics / Eng | `eq/vision/yolo.py:_quiet_settings` | 18, 27 |

## 15. Engineering and MLOps (M11)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Git, commit, branch, conventional commit messages, GitHub | Eng | `git log` (19 commits) | 19, 20 |
| `.gitignore`, `.gitattributes`, `.dockerignore`, `.dvcignore` | Eng | repo root | 19, 21 |
| Virtual environment (`.venv`), pip, Python 3.11 | Eng | README §6.1 | 19, 24 |
| Packaging: `pyproject.toml`, extras (`dev`, `app`, `ui-dev`, `automl`, `cv`), editable install | Eng | `pyproject.toml` | 19 |
| Pinned requirements files | Eng | `scripts/freeze_requirements.py:main`; `requirements*.txt` | 19 |
| CLI with Typer (`python -m equipoise <stage> [--synthetic]`) | Eng | `eq/cli.py:app`, `run_stage`, `run_all`, `split`, `validate`, `promote` | 19 |
| YAML config, frozen dataclass `Config`, absolute paths, `EQUIPOISE_ROOT` | Eng | `eq/config.py:load_config`, `Config`, `project_root` | 07, 19 |
| Environment variables (`EQUIPOISE_SYNTHETIC`, `EQUIPOISE_BUNDLE`, `EQUIPOISE_REQUEST_LOG`, `API_URL`) | Eng | `eq/serving/service.py:resolve_bundle_path`, `resolve_log_path`; `eq/serving/client.py:make_client` | 19, 22 |
| Random seed, `seed_everything`, deterministic torch | Eng / ML | `eq/reproducibility.py:seed_everything`, `rng` | 05, 19 |
| Logging warnings (`warnings`) | Eng | various | 19 |
| pytest, fixtures, parametrize, `importorskip`, `approx`, monkeypatch | Testing | `tests/*` | 19 |
| Unit / integration / smoke tests; test on synthetic data | Testing | `tests/test_pipeline_light.py`, `tests/test_app_pages.py` | 19 |
| Ruff linter | Eng | `pyproject.toml [tool.ruff]`; CI | 19 |
| GitHub Actions CI (test job + docker build job) | Eng | `.github/workflows/ci.yml` | 19 |
| DVC: pipeline stages, deps/outs/params, `dvc.lock`, `dvc repro`, `data/raw.dvc` | MLOps | `dvc.yaml`, `dvc.lock` | 19 |
| Metrics JSON per stage | MLOps | `eq/tracking.py:write_metrics`; `reports/metrics/*.json` | 19 |
| MLflow: tracking URI (SQLite), experiments, runs, nested runs, params, metrics, artifacts, tags | MLOps | `eq/tracking.py:start_run`, `child_run`, `log_params`, `log_table`, `log_figure` | 19 |
| MLflow model registry, pyfunc model, alias `champion` | MLOps | `eq/causal/bundle.py:register_bundle`; `eq/registry.py:promote_bundle` | 19 |
| Model bundle; joblib serialization | Eng | `eq/causal/bundle.py:ModelBundle`, `save_bundle`, `load_bundle` | 14, 19 |
| REST API, HTTP methods (GET/POST), JSON, status codes (HTTPException) | Eng | `eq/serving/api.py:create_app` | 19, 22 |
| FastAPI endpoints `/health`, `/predict`, `/breakeven`, `/early-response`, `/metadata`; lifespan; OpenAPI docs | Eng | `eq/serving/api.py:create_app`; `api/main.py` | 19, 22 |
| ASGI server Uvicorn | Eng | `api/main.py` | 19 |
| Pydantic models, validation, `extra="forbid"`, `create_model`, `model_validator` | Eng | `eq/serving/schemas.py:PatientFeatures`, `PricesIn`, etc. | 19, 22 |
| Service layer, API client vs local client, fallback | Eng | `eq/serving/service.py:Service`; `eq/serving/client.py`; `app/services.py:FallbackClient` | 22 |
| Request log (JSON Lines), baseline features only | Eng / Privacy | `eq/serving/service.py:append_request_log` | 19, 27 |
| Thread lock | Eng | `eq/serving/service.py` (threading.Lock) | 22 |
| Data drift, Evidently DataDriftPreset, KS / Wasserstein / chi-square per feature, drift share 0.5 | MLOps | `eq/monitor.py:drift_report`, `run`; `scripts/simulate_requests.py` | 19 |
| Docker, image, container, Dockerfile, volume mount, docker compose, healthcheck | Eng | `docker/*` | 19 |
| Hugging Face Spaces image (synthetic demo) | Eng | `docker/Dockerfile.space`; `docs/deployment.md` | 19 |
| Streamlit: pages/navigation, session state, forms, caching (`cache_data`, `cache_resource`), theme | Eng / UI | `app/streamlit_app.py`; `app/services.py` | 19, 23 |
| Presenter / view-model pattern; copy file; banned-jargon test | Eng / UI | `app/presenters.py`; `app/copy.py`; `tests/test_copy.py` | 23 |
| Altair charts | Library | `app/charts.py` | 23 |
| PDF summary (fpdf2) | Eng | `app/pdf.py:summary_pdf` | 23 |
| Playwright screenshots | Eng | `scripts/ui_screenshots.py` | 19, 23 |
| matplotlib shared style, colour-vision-safe palette | Eng | `eq/plotting.py:apply_style`, `ARM_COLORS` | 07 |
| Report assets generation (flow diagram, tech stack, feature description, experiments table) | Eng | `eq/report/*`; `reports/REPORT_ASSETS.md` | 19, 29 |
| Self-study: data structures (heap, deque, CSR, tensor, KD/ball tree, DiGraph) | Data structures | `docs/data_structures.md` | 06, 12, 13, 15 |

## 16. Libraries (every third-party import)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| numpy | Library | everywhere (`linalg.solve`, `linalg.svd`, `interp`, `polyfit`, `random.default_rng`…) | 07 |
| pandas | Library | everywhere (`read_csv`, `merge`, `groupby`, `cut`, `crosstab`, `get_dummies`) | 07 |
| scipy (`stats`, `optimize.brentq`, `interpolate`, `cluster.hierarchy`, `sparse`, `special.logsumexp`) | Library | M1, M2, M3, M7, M8 | 07 |
| scikit-learn | Library | M1–M7, M9, M10 | 07 |
| statsmodels (`ols`, `anova_lm`, `OLS`, `Logit`, `multipletests`, `ARIMA`, `adfuller`) | Library | M1, M2, M5, M7 | 07 |
| mlxtend (`apriori`, `association_rules`) | Library | `eq/represent/rules.py:mine_rules` | 07, 11 |
| econml (`SLearner`, `TLearner`, `XLearner`, `DRLearner`, `CausalForestDML`) | Library | `eq/causal/learners.py` | 07, 14 |
| PyTorch (`nn`, `optim`, `DataLoader`) | Library | `eq/causal/tarnet.py`; `eq/vision/resnet.py` | 07, 14 |
| Optuna | Library | `eq/causal/tuning.py:compare_tuners` | 07, 14 |
| AutoGluon (`TabularPredictor`) | Library | `eq/causal/automl.py` | 07, 14 |
| hmmlearn (`GaussianHMM`) | Library | `eq/longitudinal/hmm.py` | 07, 15 |
| scikit-fuzzy (`trimf`, `trapmf`, `control`) | Library | `eq/decision/fuzzy.py` | 07, 16 |
| SHAP | Library | `eq/explain/*` | 07, 17 |
| networkx | Library | `eq/report/diagrams.py:causal_dag` | 07 |
| matplotlib | Library | every `run.py`; `eq/plotting.py` | 07 |
| Typer | Library | `eq/cli.py` | 07, 19 |
| PyYAML (`safe_load`) | Library | `eq/config.py:_read_yaml` | 07 |
| Pydantic | Library | `eq/serving/schemas.py`; `eq/decision/recommend.py` | 07, 19 |
| joblib | Library | `eq/causal/bundle.py`; `eq/serving/assets.py` | 07, 19 |
| MLflow | Library | `eq/tracking.py`; `eq/registry.py` | 07, 19 |
| FastAPI, Uvicorn, httpx (TestClient) | Library | `eq/serving/api.py`; `tests/test_api.py` | 07, 19 |
| requests | Library | `eq/serving/client.py:ApiClient`; `scripts/download_oct_subset.py` | 07 |
| Streamlit, Altair, fpdf2 | Library | `app/*` | 07, 23 |
| Evidently | Library | `eq/monitor.py` | 07, 19 |
| OpenCV (`cv2`), torchvision, Ultralytics, pytorch-grad-cam, Pillow, Gradio, kaggle | Library | `eq/vision/*`; `app/oct_gradio.py`; `scripts/download_oct_subset.py` | 07, 18 |
| Playwright | Library | `scripts/ui_screenshots.py` | 07, 19 |
| pytest, pytest-cov, ruff, DVC | Library / Tool | `tests/`, CI, `dvc.yaml` | 07, 19 |
| tabulate (DataFrame → Markdown tables) | Library | `eq/tracking.py:log_table` | 07 |
| packaging | Library | `scripts/freeze_requirements.py` | 07 |
| Standard library used: `pathlib`, `json`, `dataclasses`, `functools.lru_cache`, `itertools`, `collections` (`deque`, `Counter`, `defaultdict`), `heapq`, `hashlib`, `subprocess`, `threading`, `argparse`, `ast`, `zipfile`, `zlib`, `struct`, `concurrent.futures`, `contextlib`, `warnings`, `random`, `math`, `csv`, `datetime`, `importlib` | Library | various | 07 |

## 17. CLI stages and DVC stages

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| `python -m equipoise split` | CLI | `eq/cli.py:split` | 08, 19 |
| `python -m equipoise validate` | CLI | `eq/cli.py:validate` | 08, 19 |
| `python -m equipoise preprocess` | CLI | `eq/preprocess/run.py:run` | 09 |
| `python -m equipoise stats` | CLI | `eq/stats/run.py:run` | 10 |
| `python -m equipoise represent` | CLI | `eq/represent/run.py:run` | 11 |
| `python -m equipoise search` | CLI | `eq/search/run.py:run` | 12 |
| `python -m equipoise supervised` | CLI | `eq/supervised/run.py:run` | 13 |
| `python -m equipoise causal` | CLI | `eq/causal/run.py:run` | 14 |
| `python -m equipoise longitudinal` | CLI | `eq/longitudinal/run.py:run` | 15 |
| `python -m equipoise decision` | CLI | `eq/decision/run.py:run` | 16 |
| `python -m equipoise explain` | CLI | `eq/explain/run.py:run` | 17 |
| `python -m equipoise vision` (not in `all`) | CLI | `eq/vision/run.py:run` | 18 |
| `python -m equipoise report-assets` | CLI | `eq/report/run.py:run` | 19 |
| `python -m equipoise monitor` | CLI | `eq/monitor.py:run` | 19 |
| `python -m equipoise promote` | CLI | `eq/cli.py:promote` → `eq/registry.py:promote_bundle` | 19 |
| `python -m equipoise all` | CLI | `eq/cli.py:run_all` | 19, 24 |
| DVC stages `build_analysis`, `build_visits`, `build_cst_quality`, `split`, `synthetic`, the 10 pipeline stages, `vision` | MLOps | `dvc.yaml` | 19, 24 |
| Helper scripts: `make_synthetic_data`, `pick_example_patients`, `simulate_requests`, `ui_screenshots`, `freeze_requirements`, `download_oct_subset`, `build_*` | Eng | `scripts/*.py` | 21, 24 |

## 18. Config keys (`config/default.yaml`, `config/costs.yaml`, `config/replication_targets.yaml`)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| `seed` | Config | `eq/reproducibility.py:seed_everything` | 05, 19 |
| `paths.*` | Config | `eq/config.py:Paths` | 19, 21 |
| `split.test_fraction`, `split.stratify_by` | Config | `eq/data/splits.py:create_split` | 08 |
| `cv.n_folds` | Config | `eq/data/splits.py:cv_folds` | 05, 13 |
| `trial.*` (arms, sizes, reference arm, primary outcome, eligibility) | Config | `eq/data/schema.py`; `eq/causal/data.py` | 04 |
| `visits.*` (OCT / treatment windows, cross-check tolerance) | Config | `scripts/build_protocolT_visits.py`; `eq/data/crosscheck.py` | 08 |
| `effects.mcid_letters`, `effects.mcid_sensitivity_letters` | Config | `eq/decision/recommend.py:recommend` | 03, 16 |
| `preprocess.*` (imputer, knn_neighbors, mice_*, log_features, cst.*, outliers.*, bands, gain_threshold_letters) | Config | M1 modules | 09 |
| `stats.*` (ci_level, alpha, skew_threshold, regression_covariates, multiple_correlation_features, replication_tolerance_letters) | Config | M2 modules | 10 |
| `bootstrap.n_reps` | Config | `eq/causal/policy.py:bootstrap_ci` | 14 |
| `mlflow.*` | Config | `eq/tracking.py:tracking_uri` | 19 |
| `synthetic.*` | Config | `scripts/make_synthetic_data.py` | 04 |
| `represent.*` (n_components_plot, variance_target, k_range, dendrogram_truncate, apriori.*) | Config | M3 modules | 11 |
| `supervised.*` (knn, tree, random_forest, gradient_boosting, ridge_alphas, lasso_n_alphas, logistic_C, irls, classification_threshold, poisson_alpha, hist_gb_poisson, one_hot_min_frequency) | Config | M5 modules | 13 |
| `search.*` (ridge_alpha, feature_penalty, budget, seed_offsets, beam, tabu, ga, pso, filter_top_k, rfe_n_features, lasso_path_alphas) | Config | M4 modules | 12 |
| `causal.*` (cv_folds, base_learners, causal_forest, tarnet, semi_synthetic, heterogeneity_rule, selection, tuning, bootstrap_reps, bundle_bootstrap_models, policy, automl) | Config | M6 modules | 14 |
| `longitudinal.*` (grid, ts_step, arima_orders, ma_window, lagrange, hmm, early_weeks, early_ridge_alpha, trajectory_k) | Config | M7 modules | 15 |
| `decision.*` (value_of_vision_k, newton, fuzzy universes, rules.net_benefit_margin, example_patient) | Config | M8 modules | 16 |
| `explain.*` (shap_background, subgroup_min_n, subgroups) | Config | M9 modules | 17 |
| `serving.*` (env var names, request_timeout_s, shap_background_clusters, shap_top_features, breakeven_curve_*) | Config | `eq/serving/*` | 19, 22 |
| `ui.*` (letters_per_line, lines_step, confidence, may_benefit_p, strength, factor_size_letters, snellen, essential_features, examples, explain_on_compare) | Config | `app/presenters.py` | 23 |
| `monitor.*` (drift_share, min_requests, simulate.*) | Config | `eq/monitor.py`; `scripts/simulate_requests.py` | 19 |
| `vision.*` (dataset, classes, per_class_train, val_fraction, preprocess.*, resnet.*, yolo.*) | Config | M10 modules | 18 |
| `costs.yaml`: `price_per_injection_inr`, `laser_session_inr`, `willingness_to_pay_inr_per_utility`, `sources`, `demo.*` | Config | `eq/config.py:load_prices` | 16, 27 |
| `replication_targets.yaml` (published vs our_table) | Config | `eq/stats/replication.py:load_targets` | 10 |

## 19. Tests (23 files)

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| `test_leakage.py` (only X_ reaches models; no post-randomization tables) | Test | `tests/test_leakage.py` | 05, 19 |
| `test_schema.py`, `test_visits.py`, `test_splits.py` | Test | data layer | 08, 19 |
| `test_preprocess.py` | Test | M1 | 09 |
| `test_stats.py` (incl. real-data replication targets) | Test | M2 | 10 |
| `test_represent.py` | Test | M3 | 11 |
| `test_search.py` (algorithms find known optimum on toy problem) | Test | M4 | 12 |
| `test_supervised.py` (scratch vs library) | Test | M5 | 13 |
| `test_causal.py` (IPW unbiasedness, selection rule) | Test | M6 | 14 |
| `test_longitudinal.py` (forward/Viterbi match hmmlearn) | Test | M7 | 15 |
| `test_decision.py` (Newton vs brentq, fuzzy monotone, rule engine) | Test | M8 | 16 |
| `test_pipeline_light.py`, `conftest.py` (light synthetic end-to-end) | Test | M6–M9 | 19 |
| `test_api.py`, `test_monitor.py`, `test_report.py`, `test_config_and_tracking.py` | Test | M11 | 19 |
| `test_app_pages.py`, `test_presenters.py`, `test_copy.py`, `test_pdf.py`, `test_examples.py` | Test | UI | 23 |
| `test_vision.py` | Test | M10 | 18 |

## 20. UI pages

| Concept | Category | Where used (file:function) | Chapter |
|---|---|---|---|
| Home | UI | `app/views/home.py` | 23 |
| Get a suggestion: step 1 patient details, step 2 comparison, step 3 why | UI | `app/views/suggest.py:step1`, `step2`, `step3` | 22, 23 |
| Price check ("is the extra cost worth it?") | UI | `app/views/price_check.py`; `app/widgets.py:worth_it` | 23 |
| Progress check (early-response forecast) | UI | `app/views/progress.py` | 23 |
| About the trial | UI | `app/views/about.py` | 23 |
| Behind the model (for reviewers) | UI | `app/views/behind.py` | 23 |
| OCT screening demo page; Gradio app | UI | `app/views/oct.py`; `app/oct_gradio.py` | 18, 23 |
| Headline + confidence words ("fairly / somewhat confident", "close call") | UI | `app/presenters.py:headline`, `confidence_word` | 22, 23 |
| Drug cards, improvement chart, "what mattered most" factors, "how we decided" | UI | `app/presenters.py:cards`, `chart`, `factors`, `decided` | 23 |
| Printable PDF summary | UI | `app/pdf.py:summary_pdf` | 23 |
| Try-an-example patients (poorer / moderate / good) | UI | `eq/serving/examples.py`; `scripts/pick_example_patients.py` | 23, 27 |

## Proposed handbook outline (for approval)

Word estimates assume ~300 words per printed page. Module chapters (08–19) follow the fixed
order: story → inputs/outputs → concept cards → code walkthrough → run it yourself → what we
found → limitations → plan vs reality → check yourself.

| # | File | Title | Est. words |
|---|---|---|---|
| **Part A** | | **The big picture** | |
| 00 | `00_how_to_use.md` | How to use this handbook (reading paths, card format, team split by module) | 1,200 |
| 01 | `01_one_page.md` | Equipoise on one page (plain + technical, Mermaid pipeline) | 1,500 |
| 02 | `02_medical_story.md` | The medical story (diabetes → retina → DME → OCT/CST → vision tests → anti-VEGF → cost in India) | 4,500 |
| 03 | `03_causation.md` | The central idea: causation and counterfactuals | 4,500 |
| 04 | `04_data.md` | The data (Protocol T, raw DRCR tables, all 47 columns generated from schema + real data, visit table, synthetic data, Kermany, caveats, attribution) | 6,000 |
| **Part B** | | **Foundations from scratch** | |
| 05 | `05_ml_from_zero.md` | Machine learning from zero | 5,000 |
| 06 | `06_maths.md` | The maths you need (linear algebra, statistics, probability, calculus, inference) | 7,500 |
| 07 | `07_python_toolbox.md` | The Python toolbox (every library; Pipeline/ColumnTransformer; fit/transform/predict) | 6,000 |
| **Part C** | | **The pipeline, module by module** | |
| 08 | `08_m0_data_integration.md` | M0 Data integration (build scripts, schema, visit table, cross-check, split) | 4,000 |
| 09 | `09_m1_preprocessing.md` | M1 Preprocessing | 6,500 |
| 10 | `10_m2_statistics.md` | M2 EDA and statistical inference | 6,500 |
| 11 | `11_m3_representation.md` | M3 Patient representation (PCA/SVD, clustering, Apriori) | 5,000 |
| 12 | `12_m4_feature_search.md` | M4 Feature search (pseudocode + one worked iteration each on a 4-feature toy) | 7,000 |
| 13 | `13_m5_outcome_models.md` | M5 Outcome models and metrics | 7,000 |
| 14 | `14_m6_counterfactual_engine.md` | M6 Counterfactual engine (6-patient example for S/T/X/DR; forests; TARNet from scratch; optimizers; PEHE; IPW/AIPW; bootstrap; calibration; "no clear winner") | 12,000 |
| 15 | `15_m7_longitudinal.md` | M7 Longitudinal (interpolation, finite differences, ARIMA, 3-step Viterbi by hand, early response) | 7,000 |
| 16 | `16_m8_decision_layer.md` | M8 Decision layer (cost, Newton by hand, full fuzzy worked example with real MFs, rule trace for the typical patient, PEAS) | 7,000 |
| 17 | `17_m9_explainability_fairness.md` | M9 Explainability and fairness | 4,000 |
| 18 | `18_m10_oct_screening.md` | M10 OCT screening (optional module) | 5,000 |
| 19 | `19_m11_engineering.md` | M11 Engineering and MLOps (every tool: problem, our use, demo commands) | 8,000 |
| **Part D** | | **How it's built and how it runs** | |
| 20 | `20_how_it_was_built.md` | Timeline from git log (19 commits, 9–10 Oct 2026), phases 0–11 + UI redesign | 1,800 |
| 21 | `21_repo_tour.md` | Repo tour (every folder/file; generated vs hand-written) | 3,000 |
| 22 | `22_life_of_a_prediction.md` | Life of one prediction (typical patient, with intermediate values from a real run) | 4,000 |
| 23 | `23_the_app.md` | The app, screen by screen (with screenshots) | 3,500 |
| 24 | `24_running_everything.md` | Running everything from a fresh laptop (incl. Windows errors) | 3,000 |
| **Part E** | | **Results and honesty** | |
| 25 | `25_all_results.md` | All results in one place (each cited + "what this means") | 5,000 |
| 26 | `26_key_findings.md` | Key findings, ranked | 1,800 |
| 27 | `27_limitations_ethics.md` | Limitations, ethics and safety | 2,500 |
| 28 | `28_issues_found.md` | Appendix: issues found while reading the code (not fixed) | 1,000–2,000 |
| **Part F** | | **Report and viva pack** | |
| 29 | `29_report_source_pack.md` | Report source pack (15 template sections, bullet facts, ready tables) | 5,000 |
| 30 | `30_literature_references.md` | Literature survey table (6–10 verified papers) + IEEE references | 3,500 |
| 31 | `31_viva_questions.md` | Viva question bank (50–60 Q&A) | 6,000 |
| 32 | `32_glossary.md` | Glossary A–Z | 5,000 |
| 33 | `33_concept_index.md` | Concept index (concept → chapter → file) | 2,500 |
| | | **Total** | **≈ 180,000 words (≈ 600 pages)** |

Supporting files: `scripts/handbook/build_handbook.py` (concatenate → `docs/HANDBOOK.md` with
linked TOC), `scripts/handbook/check_coverage.py`, `scripts/handbook/make_figures.py` (only
small new illustrations, e.g. the 6-patient meta-learner example, fuzzy worked example
overlays if the existing figure is not enough), `scripts/handbook/trace_typical_patient.py`
(intermediate values for chapter 22), `scripts/handbook/column_table.py` (chapter 04 column
table from schema + real data, aggregates only).
