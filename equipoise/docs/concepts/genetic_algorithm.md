# Genetic algorithm

**What.** A population of 30 binary feature masks evolves:
- **roulette-wheel selection** picks parents with probability proportional to fitness,
- **crossover** (rate 0.9; half single-point, half two-point) swaps segments between two
  parents,
- **bit-flip mutation** flips each bit with probability 1 / 33,
- **elitism** copies the best 2 masks unchanged into the next generation.

It stops when the evaluation budget is spent or after 25 generations without improvement.

**Why here.** It explores many regions of the search space at once and needs no gradient.
This makes it a natural fit for a discrete feature-subset problem, and it can be compared
directly with hill climbing under the same budget.

**What it showed** (`m4_search_summary.md`). Best objective 9.39 (the same optimum as hill
climbing, beam and tabu), with 6.7 features on average. It needed more evaluations to get
there (median about 1,120, vs about 570 for hill climbing). On this smooth, small problem the
GA's diversity costs evaluations without finding anything better. That is an honest result,
not a failure.
