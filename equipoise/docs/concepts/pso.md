# Particle swarm optimisation (binary and continuous)

**What.** Each particle has a position and a velocity. Each step, the velocity is pulled
toward the particle's own best position (c1 = 1.5) and the swarm's best (c2 = 1.5), with
inertia 0.7. **Binary PSO** (M4) passes each velocity through a sigmoid to get the
probability that the bit is 1 (velocity clipped at ±4). **Continuous PSO** (M6) moves
directly in the hyperparameter space.

**Why here.** It is a bio-inspired optimiser from the syllabus. The same code is reused
for a discrete problem (feature masks) and a continuous one (tuning).

**What it showed.** In feature search, binary PSO reached 9.43 vs 9.39 for the others
(`m4_search_summary.md`): slightly worse at equal budget. In tuning the gradient-boosting
base learner, PSO reached a CV RMSE of 9.87, vs 9.86 for Optuna and 9.93 for grid search,
all with 30 trials (`m6_tuning_summary.md`). PSO is competitive, but not better here.
