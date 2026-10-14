# Hyperparameter tuning: grid vs Optuna (TPE) vs PSO; AutoGluon

**What.** We tune the gradient-boosting base learner (learning rate, depth, leaf size,
number of trees) by CV RMSE on the training split only, with 30 trials per method.
- **Grid search** tries fixed combinations.
- **Optuna's TPE** models which regions have produced good trials and samples more from
  them.
- **PSO** (from M4, continuous version) moves a swarm through the space.

**What it showed** (`m6_tuning_summary.md`, `m6_tuning.png`). Best CV RMSE: grid 9.93,
Optuna 9.86, PSO 9.87, vs 10.2 for the default settings. Optuna and PSO both found the same
kind of setting: small learning rate, depth-1 trees, about 350–400 trees. The gain from
tuning (about 0.3 letters) is small compared with the outcome noise.

**AutoGluon** (`[automl]` extra) is used as the base learner of a T-learner. In one
replication of the published-like scenario its PEHE was 5.41 letters, vs 4.77 for the
tuned gradient-boosting T-learner (`m6_autogluon_comparison.md`). The leaderboard is in
`m6_autogluon_leaderboard.md`. AutoML did not beat a hand-tuned base learner on 370
patients.
