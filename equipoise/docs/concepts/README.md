# Concepts: viva preparation

One short file per method used in Equipoise. Each answers three questions: **what is it**,
**why is it used here**, and **what did it show** on the Protocol T data. Every number is
copied from a file in `reports/tables/` or `reports/metrics/` (named in each file). If you
rerun the pipeline with different settings, update the numbers from those files.

| File | Methods | Module |
|---|---|---|
| [missing_data.md](missing_data.md) | complete-case, KNN imputation, MICE + Rubin's rules | M1 |
| [preprocessing.md](preprocessing.md) | CST harmonization, outliers (IQR / Isolation Forest / DBSCAN), encoding, discretization | M1 |
| [hypothesis_tests.md](hypothesis_tests.md) | ANOVA, Welch t, Holm, chi-square, Mann-Whitney, F / Levene, partial and multiple correlation, interaction regression | M2 |
| [svd_pca.md](svd_pca.md) | PCA via SVD | M3 |
| [clustering_and_rules.md](clustering_and_rules.md) | k-means, Ward, DBSCAN, Apriori | M3 |
| [search_algorithms.md](search_algorithms.md) | hill climbing, beam, tabu | M4 |
| [genetic_algorithm.md](genetic_algorithm.md) | GA | M4 |
| [pso.md](pso.md) | binary and continuous PSO | M4, M6 |
| [supervised_models.md](supervised_models.md) | OLS / normal equations, Ridge, Lasso, kNN (KD/ball tree), trees, RF, GB, Naive Bayes, Poisson | M5 |
| [logistic_irls.md](logistic_irls.md) | logistic regression by Newton-Raphson (IRLS) | M5 |
| [meta_learners.md](meta_learners.md) | S-, T-, X-, DR-learners | M6 |
| [causal_forest.md](causal_forest.md) | causal forest (CausalForestDML) | M6 |
| [tarnet_optimizers.md](tarnet_optimizers.md) | TARNet, SGD vs Adam vs RMSprop | M6 |
| [hyperparameter_tuning.md](hyperparameter_tuning.md) | grid vs Optuna TPE vs PSO, AutoGluon | M6 |
| [semi_synthetic.md](semi_synthetic.md) | semi-synthetic benchmark, PEHE | M6 |
| [ipw_policy_value.md](ipw_policy_value.md) | IPW / AIPW policy value | M6 |
| [interpolation_splines.md](interpolation_splines.md) | linear, cubic spline, Lagrange | M7 |
| [finite_differences.md](finite_differences.md) | forward / backward / central differences | M7 |
| [arima.md](arima.md) | ADF, MA, AR / ARMA / ARIMA | M7 |
| [hmm_viterbi.md](hmm_viterbi.md) | Gaussian HMM, forward algorithm, Viterbi | M7 |
| [newton_raphson.md](newton_raphson.md) | break-even by Newton-Raphson | M8 |
| [fuzzy_system.md](fuzzy_system.md) | fuzzy inference | M8 |
| [expert_system.md](expert_system.md) | forward-chaining rule engine | M8 |
| [agent.md](agent.md) | PEAS agent | M8 |
| [shap.md](shap.md) | SHAP, subgroup checks | M9 |
| [oct_screening.md](oct_screening.md) | OpenCV, YOLO classification, ResNet18 transfer learning, Grad-CAM | M10 |
| [mlops.md](mlops.md) | Git, DVC, MLflow, Docker, CI, Evidently | M11 |
