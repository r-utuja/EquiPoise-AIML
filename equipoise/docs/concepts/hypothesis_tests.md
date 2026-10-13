# Statistical inference (M2)

**What and why.** Before any machine learning we check two things. Did randomization work
(Table 1)? And can we reproduce the published trial? If either failed, nothing built later
could be trusted.

- **Table 1** compares 33 baseline features across arms. It uses one-way ANOVA (or
  Kruskal-Wallis when skewed) for numbers and chi-square for categories. No feature
  differed at α = 0.05 (`m2_stats.json`: 0 of 33), which is what randomization should give.
- **One-way ANOVA** tests whether the three arm means are equal. The F statistic is
  between-arm variance divided by within-arm variance. At 1 year F = 6.85, p = 0.0011
  (`m2_replication_tests.md`).
- **Welch t-tests** compare two arms without assuming equal variances. **Holm correction**
  controls the error rate across the three comparisons: sort the p-values and multiply the
  k-th smallest by (number of tests − k + 1). Aflibercept − bevacizumab = 3.95 letters
  (95% CI 1.72 to 6.19, Holm p = 0.0016). Ranibizumab − bevacizumab = 1.82 (Holm
  p = 0.081).
- **Subgroups.** With baseline vision below 69 letters, aflibercept − bevacizumab = 7.24
  letters (CI 3.92 to 10.6). At 69 or better it is 0.84 (CI −1.70 to 3.38). This is the
  published finding.
- **Mann-Whitney U** compares injection counts, which are skewed counts, so a rank test fits
  better than a t-test. Medians are 9 / 10 / 10 at year 1 (`m2_replication_burden.md`).
  **Chi-square** compares laser rates: 37% / 56% / 47%.
- **F-test and Levene** test whether outcome variances are equal. Levene (robust):
  p = 0.156. The F-test finds ranibizumab less variable than bevacizumab (ratio 0.67,
  p = 0.004; `m2_variance_tests.md`). Bartlett's test also rejects equality, but it is
  sensitive to non-normality.
- **Partial correlation** between vision gain and baseline vision, controlling for CST:
  r = −0.42 (`m2_correlations.md`). Patients who start worse gain more, and not just because
  of thickness. **Multiple correlation** of gain on 8 baseline features: R = 0.50
  (R² = 0.25).
- **Interaction regression** `va_change ~ treatment * va_letters + covariates`. The
  treatment × baseline-vision interaction is the classical heterogeneity test: F = 8.10,
  p = 0.0003 for continuous vision, and p = 0.005 for the < 69 split
  (`m2_interaction_tests.md`).
