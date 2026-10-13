# Semi-synthetic benchmark and PEHE

**What.** A patient's outcome under the drugs they did not get is never observed, so we
cannot score CATE models directly. Instead we keep the real baseline features and simulate
outcomes, Y(a) = f₀(X) + τₐ(X) + noise, with f₀ and the noise scale fitted to the real
data. Then we check how well each learner recovers the *known* τ. **PEHE** is the RMSE
between estimated and true per-patient effects, averaged over 10 replications.

There are three scenarios: (a) **null** (constant effects, no heterogeneity), (b)
**published-like** (aflibercept advantage concentrated in vision < 69), and (c)
**nonlinear** (a smooth step in vision × CST plus an HbA1c threshold).

**What it showed** (`m6_semisynthetic_pehe.md`). The causal forest has PEHE 1.80 / 2.80 /
2.44 across the three scenarios. T-, X- and DR-learners score 6–9 letters, and in the null
scenario they report heterogeneity in 100% of replications. The causal forest reports it
in 0%. In the published-like scenario the forest (2.80) is only slightly better than a
constant-effect model (2.88): at this sample size, even a correct learner recovers little
beyond the average effect.
