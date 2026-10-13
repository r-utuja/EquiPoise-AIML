# Policy value by IPW and AIPW

**What.** A *policy* assigns a drug to each patient. Its value is the mean outcome if every
patient followed it. On held-out patients we keep only those whose randomized arm matches
the policy, and weight each by 1 / P(arm). The probabilities are known from randomization,
so this **inverse probability weighting** (IPW) estimate is unbiased. **AIPW** adds an
outcome-model correction, which usually reduces variance and stays consistent if either
part is right. Confidence intervals come from 200 bootstrap resamples.

**What it showed** (`m6_policy_values.md`, letters gained, 126 held-out patients with
week-52 vision; IPW, 95% CI):

| Policy | IPW | AIPW |
|---|---|---|
| All bevacizumab | 11.7 (8.1 to 17.1) | 11.1 |
| All aflibercept | 14.9 (9.7 to 20.7) | 14.6 |
| All ranibizumab | 11.2 (7.6 to 15.0) | 11.7 |
| Published rule (< 69 → aflibercept) | 15.7 (10.8 to 21.7) | 14.2 |
| Model: most letters (causal forest) | 14.9 (9.9 to 21.0) | 14.6 |
| Cost-aware model (DEMO prices) | 14.6 (9.9 to 20.5) | 12.9 |

The model's "most letters" policy assigns aflibercept to everyone, so it equals
all-aflibercept. The intervals overlap heavily: with about 41–47 followers per policy, no
policy is distinguishable from another. The cost-aware policy uses DEMO prices, so its
value is illustrative only. A test on synthetic data with known truth confirms that the
IPW estimator is unbiased (`tests/test_causal.py`).
