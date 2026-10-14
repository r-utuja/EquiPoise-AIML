# Meta-learners for treatment effects (S, T, X, DR)

**The question.** How many more letters would *this* patient gain on aflibercept (or
ranibizumab) than on bevacizumab? This is the conditional average treatment effect (CATE),
τ(x). Randomization makes it estimable. Propensities are the known arm shares (≈ 0.34 /
0.33 / 0.33) and are never modelled.

- **S-learner**: one model with the arm as an extra input; τ(x) is the difference between
  its predictions with the arm switched.
- **T-learner**: a separate outcome model per arm; τ(x) = μ_afl(x) − μ_bev(x).
- **X-learner**: impute each patient's individual effect with the *other* arm's model, then
  regress those imputed effects on x.
- **DR-learner**: build a doubly robust pseudo-outcome (outcome model plus an inverse-
  propensity-weighted residual) and regress it on x, with cross-fitting.

Each is fitted with gradient-boosting and Ridge base learners (`econml`).

**What it showed** (`m6_learner_selection.md`). In the semi-synthetic benchmark the
flexible learners were noise-dominated. In the **null** scenario (no true heterogeneity),
T-, X- and DR-learners "found" heterogeneity in 100% of replications, with PEHE 6–9
letters. The S-learner (Ridge) gives every patient the same effect by construction, so it
was excluded. With about 370 fitting patients, the meta-learners mostly fit noise. This is
why the causal forest was chosen; see `causal_forest.md`.
