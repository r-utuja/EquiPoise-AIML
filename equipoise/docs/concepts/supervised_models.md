# Supervised outcome models (M5)

**Models.** All are compared by 5-fold CV on the training split, then scored once on the
held-out test set:
- **OLS** solves min ‖y − Xβ‖². We solve it twice: with sklearn, and with the **normal
  equations** XᵀXβ = Xᵀy in numpy. They agree to 4.6e-14 (`m5_scratch_checks.md`).
- **Ridge** (L2) and **Lasso** (L1) add a penalty that shrinks coefficients; Lasso can set
  them to exactly zero.
- **kNN** averages the k = 15 nearest training patients. We compare three ways of finding
  neighbours: **KD-tree**, **ball tree** and **brute force**. All give identical predictions;
  in 44 dimensions brute force is fastest, because trees stop pruning well in high
  dimension (`m5_knn_algorithms.md`).
- **Decision tree**, **random forest** (bagged trees) and **gradient boosting** (trees fitted
  to residuals).
- **Gaussian / Bernoulli Naive Bayes** for classification.
- **Poisson GLM** and **Poisson-loss boosting** for injection counts.

**What it showed** (`m5_regression_comparison.md`). Predicting week-52 letters: the arm-mean
baseline has a test RMSE of 10.8. Lasso is best by CV (CV RMSE 9.39, test 9.43,
R² = 0.24). Flexible models do not beat the linear ones (random forest 9.61, boosting 10.2).
With about 490 training patients and mostly linear signal, simple models win.

For "gained ≥ 15 letters", logistic regression reaches a test AUC of 0.81
(`m5_classification_gain15.md`). With `class_weight="balanced"`, CV recall rises from 0.49
to 0.64 and specificity falls from 0.87 to 0.74 (`m5_class_weight_gain15.md`): the classic
trade-off. Rescue laser in year 1 is barely predictable (test AUC 0.58 vs 0.57 for the
arm-rate baseline). We report this plainly. For injections, the Poisson GLM beats the
arm-mean baseline (test MAE 1.73 vs 1.90; `m5_count_comparison.md`), and the cost model uses
it.
