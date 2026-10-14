# Causal forest (CausalForestDML)

**What.** A random forest whose splits are chosen to maximise differences in the
*treatment effect* between child nodes, not to predict the outcome. Each tree is "honest":
one half-sample chooses the splits and the other estimates the effects in the leaves. The
DML wrapper first removes what baseline features predict about the outcome, using
cross-fitted nuisance models. The forest gives per-patient effects with confidence
intervals. We use 500 trees and at least 10 patients per leaf.

**Why chosen.** The selection rule (config `causal.selection`), fixed before looking at
test outcomes, is:
1. drop learners whose effect cannot vary by patient;
2. drop learners with more than 1 letter of average-effect bias in any semi-synthetic
   scenario;
3. pick the lowest worst-case PEHE.

The causal forest wins with a worst-case PEHE of 2.80 letters. The next candidate, TARNet,
scores 5.28 (`m6_learner_selection.md`). In the null scenario it never invents
heterogeneity (0 of 10 replications).

**What it showed.**
- On held-out patients its predicted aflibercept advantage is larger when baseline vision
  is below 69 letters: 3.76 vs 2.59 letters (`m6_rediscovery.md`).
- Its top SHAP driver is CST, with baseline vision second.
- Its intervals are wide: median 6.4 letters wide for aflibercept
  (`m6_uncertainty_summary.md`). No held-out patient has P(benefit > MCID 5) above 0.8, so
  every held-out patient is called **equipoise**.

This is the honest per-patient finding: Protocol T is too small to separate individual
effects beyond the published baseline-vision subgroup.
