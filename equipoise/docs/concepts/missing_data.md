# Missing data: complete-case, KNN, MICE

**What.** 44 of 660 patients have no week-52 vision. *Complete-case* analysis drops them.
*KNN imputation* fills each gap with the average of the 5 most similar patients. *MICE*
(multiple imputation by chained equations, sklearn `IterativeImputer`) models each
incomplete variable from the others, repeats the fill 20 times with randomness, analyses
each completed dataset, and pools the results with **Rubin's rules**: the pooled estimate is
the mean of the 20 estimates, and its variance is the within-imputation variance plus
(1 + 1/m) × the between-imputation variance.

**Why here.** The published trial used multiple imputation. A single imputation (KNN) treats
guessed values as if they were measured, so its standard errors are too small. MICE carries
that extra uncertainty into the confidence interval.

**What it showed** (`reports/tables/m1_missing_strategies.md`). The aflibercept −
bevacizumab difference barely moves across strategies: 3.95 (complete-case), 4.01 (KNN) and
3.83 letters (MICE, CI 1.72 to 5.95). Ranibizumab − bevacizumab is 1.82 / 2.00 / 1.97. The
fraction of missing information is small (0.07 and 0.04), so the conclusions do not depend
on how the 44 missing outcomes are handled. When imputing hidden *feature* values, MICE had
the lowest normalised RMSE (0.85, vs 0.96 for KNN and 1.01 for the median;
`m1_preprocess.json`).
