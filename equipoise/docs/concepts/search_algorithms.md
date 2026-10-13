# Feature selection as search: hill climbing, beam, tabu

**The problem.** A state is a binary mask over 33 baseline features. The objective is the
5-fold CV RMSE of a Ridge model predicting week-52 letters, plus 0.02 letters per selected
feature (`config search`). Every algorithm gets the same budget (1,500 distinct
evaluations) and the same seeds (3 runs each).

- **Hill climbing** evaluates all single-bit flips and moves to the best one. It stops when
  no flip improves the score, i.e. at a local optimum.
- **Beam search** keeps the best *w* = 5 masks in a heap (`heapq`) and expands all of them
  each round, so it can follow several promising paths at once.
- **Tabu search** always moves to the best neighbour, even a worse one, but forbids undoing
  recent moves for 7 steps (the tabu list). The **aspiration** rule lifts the ban when a
  forbidden move would beat the best score seen so far.

**What it showed** (`m4_search_summary.md`, `m4_hill_climbing_restarts.md`). Hill climbing,
beam, tabu and the GA all reach the same best objective, 9.39. Using every feature scores
10.13 (`m4_search.json`). From 10 random starts, hill climbing stalled in 2 distinct local
optima; one start ended at 9.42 with a different 8-feature set. That is the local-optimum
trap that beam and tabu are designed to escape. The selected 7 features (always including
baseline vision and age) give a held-out RMSE of 9.34, vs 9.47 with all features
(`m4_selected_vs_all.md`): a small gain. Forcing this subset onto the M6 causal model
barely changes its policy (94% of held-out patients get the same assignment;
`m6_m4_subset_effect.md`), so M6 keeps all features.
