# Logistic regression from scratch (Newton-Raphson / IRLS)

**What.** Logistic regression maximises the log-likelihood
ℓ(β) = Σ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)], with pᵢ = σ(xᵢᵀβ). Newton-Raphson updates
β ← β + (XᵀWX)⁻¹ Xᵀ(y − p), with W = diag(pᵢ(1 − pᵢ)). Each step is a weighted least-squares
fit, hence the name *iteratively reweighted least squares*.

**Why here.** It shows what `LogisticRegression.fit` actually does, and it reuses the same
Newton idea as the break-even solver in M8.

**What it showed** (`m5_scratch_checks.md`). For "gained ≥ 15 letters", the method
converges in 7 iterations, and the coefficients match statsmodels `Logit` to 1.3e-15. For
laser in year 1 it converges in 5 iterations. Its test AUC (0.81) equals sklearn's
(`m5_classification_gain15.md`).
