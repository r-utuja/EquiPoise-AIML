"""Feature-subset search algorithms, implemented from scratch on a binary mask.

All minimise the same budgeted objective (see objective.py) and stop when the evaluator
raises `BudgetExhausted`.

- Random search: sample random masks; the yardstick every method must beat.
- Hill climbing (steepest descent): from a random mask, move to the best single-bit flip
  while it improves; at a local optimum (no flip helps) restart from a new random mask.
  Recording each restart's end point shows it stalling in different local optima.
- Beam search: keep the `width` best subsets (a heap); expand each by adding one feature;
  stop when no child beats the best parent.
- Tabu search: always move to the best neighbour, even if worse, but forbid re-flipping a
  recently flipped bit for `tenure` moves (the tabu list) - unless that move would beat the
  best ever seen (aspiration). This lets it walk out of local optima.
- Genetic algorithm: a population of masks; parents chosen by roulette wheel (probability
  proportional to fitness), single- or two-point crossover, bit-flip mutation, the best
  `elite` copied unchanged; stops at the budget or after `patience` generations without
  improvement.
- Binary PSO: each particle has a real-valued velocity per bit, pulled towards its own best
  and the swarm's best; the probability that a bit is 1 is sigmoid(velocity).
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from equipoise.search.objective import BudgetExhausted, Evaluator


@dataclass
class SearchResult:
    """Best mask found, its objective, the best-so-far curve and algorithm diagnostics."""

    algorithm: str
    seed: int
    best_mask: np.ndarray
    best_value: float
    history: list[float]
    n_evals: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _result(name: str, seed: int, ev: Evaluator, **diag: Any) -> SearchResult:
    stopped = ev.stopped_by if ev.stopped_by != "running" else "algorithm converged"
    return SearchResult(name, seed, ev.best_mask.copy(), ev.best_value, list(ev.history),
                        ev.n_evals, {"stopped_by": stopped, **diag})


def _random_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < 0.5).astype(np.int8)


def _flip(mask: np.ndarray, i: int) -> np.ndarray:
    out = mask.copy()
    out[i] ^= 1
    return out


def random_search(ev: Evaluator, rng: np.random.Generator, **_: Any) -> SearchResult:
    try:
        while True:
            ev(_random_mask(ev.n_features, rng))
    except BudgetExhausted:
        pass
    return _result("Random search", 0, ev)


def hill_climbing(ev: Evaluator, rng: np.random.Generator, **_: Any) -> SearchResult:
    """Steepest-descent hill climbing with random restarts."""
    local_optima: list[float] = []
    try:
        while True:
            current = _random_mask(ev.n_features, rng)
            value = ev(current)
            while True:
                neighbours = [_flip(current, i) for i in range(ev.n_features)]
                values = [ev(m) for m in neighbours]
                j = int(np.argmin(values))
                if values[j] >= value:          # no flip improves: local optimum
                    local_optima.append(value)
                    break
                current, value = neighbours[j], values[j]
    except BudgetExhausted:
        pass
    return _result("Hill climbing", 0, ev, local_optima=local_optima,
                   restarts=len(local_optima))


def beam_search(ev: Evaluator, rng: np.random.Generator, width: int = 5, **_: Any
                ) -> SearchResult:
    """Forward beam search from the empty subset, keeping the best `width` with a heap."""
    n = ev.n_features
    start = np.zeros(n, dtype=np.int8)
    beam = [(ev(start), start.tobytes())]
    best = beam[0][0]
    try:
        while True:
            children: list[tuple[float, bytes]] = []
            seen: set[bytes] = set()
            for _, key in beam:
                parent = np.frombuffer(key, dtype=np.int8)
                for i in rng.permutation(np.flatnonzero(parent == 0)):
                    child = _flip(parent, int(i))
                    ck = child.tobytes()
                    if ck in seen:
                        continue
                    seen.add(ck)
                    item = (ev(child), ck)
                    if len(children) < width:
                        heapq.heappush(children, (-item[0], item[1]))   # max-heap of worst
                    elif -children[0][0] > item[0]:
                        heapq.heapreplace(children, (-item[0], item[1]))
            if not children:
                break
            beam = sorted((-v, k) for v, k in children)
            if beam[0][0] >= best:               # no child beats the best so far: stop
                break
            best = beam[0][0]
    except BudgetExhausted:
        pass
    return _result("Beam search", 0, ev, width=width)


def tabu_search(ev: Evaluator, rng: np.random.Generator, tenure: int = 7, **_: Any
                ) -> SearchResult:
    """Best-neighbour moves with a tabu list of recently flipped bits and aspiration."""
    current = _random_mask(ev.n_features, rng)
    ev(current)
    tabu: deque[int] = deque(maxlen=tenure)
    moves, aspirations = 0, 0
    try:
        while True:
            best_move, best_val, is_aspiration = None, np.inf, False
            for i in rng.permutation(ev.n_features):
                cand = _flip(current, int(i))
                val = ev(cand)
                allowed = int(i) not in tabu or val < ev.best_value   # aspiration criterion
                if allowed and val < best_val:
                    best_move, best_val = int(i), val
                    is_aspiration = int(i) in tabu
            if best_move is None:
                break
            aspirations += int(is_aspiration)
            current = _flip(current, best_move)
            tabu.append(best_move)
            moves += 1
    except BudgetExhausted:
        pass
    return _result("Tabu search", 0, ev, tenure=tenure, moves=moves, aspirations=aspirations)


def genetic_algorithm(ev: Evaluator, rng: np.random.Generator, population: int = 30,
                      crossover_rate: float = 0.9, two_point_share: float = 0.5,
                      elite: int = 2, patience_generations: int = 25, **_: Any
                      ) -> SearchResult:
    """Generational GA with roulette-wheel selection, 1/2-point crossover, bit-flip mutation
    (rate 1/n) and elitism."""
    n = ev.n_features
    mut = 1.0 / n
    pop = [_random_mask(n, rng) for _ in range(population)]
    generations, stale, best_seen = 0, 0, np.inf
    stop = "budget"
    try:
        while True:
            costs = np.array([ev(m) for m in pop])
            order = np.argsort(costs)
            if costs[order[0]] < best_seen - 1e-12:
                best_seen, stale = costs[order[0]], 0
            else:
                stale += 1
            if stale >= patience_generations:
                stop = f"no improvement for {patience_generations} generations"
                break
            fitness = costs.max() - costs + 1e-9          # lower cost -> higher fitness
            prob = fitness / fitness.sum()
            children = [pop[i].copy() for i in order[:elite]]
            while len(children) < population:
                a, b = (pop[i] for i in rng.choice(population, 2, p=prob))   # roulette wheel
                c1, c2 = a.copy(), b.copy()
                if rng.random() < crossover_rate:
                    if rng.random() < two_point_share:
                        i, j = sorted(rng.choice(np.arange(1, n), 2, replace=False))
                        c1[i:j], c2[i:j] = b[i:j], a[i:j]
                    else:
                        i = int(rng.integers(1, n))
                        c1[i:], c2[i:] = b[i:], a[i:]
                for c in (c1, c2):
                    flips = rng.random(n) < mut
                    c[flips] ^= 1
                    if len(children) < population:
                        children.append(c)
            pop = children
            generations += 1
    except BudgetExhausted:
        pass
    if stop != "budget":
        ev.stopped_by = stop
    return _result("Genetic algorithm", 0, ev, generations=generations)


def binary_pso(ev: Evaluator, rng: np.random.Generator, particles: int = 20,
               inertia: float = 0.7, c1: float = 1.5, c2: float = 1.5, v_max: float = 4.0,
               **_: Any) -> SearchResult:
    """Binary particle swarm optimisation with a sigmoid velocity-to-probability transfer."""
    n = ev.n_features
    x = (rng.random((particles, n)) < 0.5).astype(np.int8)
    v = rng.uniform(-1, 1, (particles, n))
    pbest = x.copy()
    pbest_val = np.full(particles, np.inf)
    iterations = 0
    try:
        while True:
            for k in range(particles):
                val = ev(x[k])
                if val < pbest_val[k]:
                    pbest_val[k], pbest[k] = val, x[k].copy()
            g = pbest[int(np.argmin(pbest_val))]
            r1, r2 = rng.random((particles, n)), rng.random((particles, n))
            v = inertia * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
            v = np.clip(v, -v_max, v_max)
            x = (rng.random((particles, n)) < 1 / (1 + np.exp(-v))).astype(np.int8)
            iterations += 1
    except BudgetExhausted:
        pass
    return _result("Binary PSO", 0, ev, iterations=iterations)


ALGORITHMS = {
    "Random search": (random_search, None),
    "Hill climbing": (hill_climbing, None),
    "Beam search": (beam_search, "beam"),
    "Tabu search": (tabu_search, "tabu"),
    "Genetic algorithm": (genetic_algorithm, "ga"),
    "Binary PSO": (binary_pso, "pso"),
}


def run_algorithm(name: str, base: Evaluator, seed: int, params: dict[str, Any]
                  ) -> SearchResult:
    """Run one algorithm on a fresh evaluator (same budget) with its own seeded RNG."""
    fn, key = ALGORITHMS[name]
    res = fn(base.fresh(), np.random.default_rng(seed), **(params.get(key, {}) if key else {}))
    res.algorithm, res.seed = name, seed
    return res


def continuous_pso(fn: Any, bounds: np.ndarray, rng: np.random.Generator, particles: int,
                   iterations: int, inertia: float = 0.7, c1: float = 1.5, c2: float = 1.5
                   ) -> tuple[np.ndarray, float, list[float]]:
    """PSO in a box (bounds: (d, 2) array of [low, high]); minimises fn(position).

    Same update as the binary version, but positions are real numbers moved by velocity and
    clipped to the box. Returns (best position, best value, best-so-far after each call).
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = len(lo)
    x = lo + rng.random((particles, d)) * (hi - lo)
    v = (rng.random((particles, d)) - 0.5) * (hi - lo) * 0.2
    pbest, pval = x.copy(), np.full(particles, np.inf)
    history: list[float] = []
    best = np.inf
    for _ in range(iterations):
        for k in range(particles):
            val = float(fn(x[k]))
            best = min(best, val)
            history.append(best)
            if val < pval[k]:
                pval[k], pbest[k] = val, x[k].copy()
        g = pbest[int(np.argmin(pval))]
        r1, r2 = rng.random((particles, d)), rng.random((particles, d))
        v = inertia * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, lo, hi)
    k = int(np.argmin(pval))
    return pbest[k], float(pval[k]), history
