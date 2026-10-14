# Data structures in Equipoise

Where each data structure is used, why it fits, and what it cost or saved. The numbers come
from `reports/tables/`.

## 3-D trajectory tensor (numpy `ndarray`)

`longitudinal/tensor.py` aligns every patient's irregular visits onto a weekly grid and
stacks them into one array of shape **(patients × weeks × measures) = (660, 105, 3)**. The
three measures are vision, CST and injected (`m7_tensor.md`). Cells outside a patient's
follow-up stay `NaN`; nothing is extrapolated. One array makes the common operations
one-liners: an arm-mean trajectory is `tensor[arm_mask, :, 0]` averaged over axis 0, a
finite difference is a slice difference along axis 1, and trajectory clustering flattens
axis 1 into feature vectors. 92% of vision cells are observed or interpolated.

## Sparse one-hot matrices (`scipy.sparse.csr_matrix`)

One-hot encoding turns 3 nominal features (race, diabetes type, OCT machine) into 14 columns
in which most entries are zero. CSR (compressed sparse row) stores only the non-zeros
(`data`), their column indices (`indices`) and where each row starts (`indptr`). On the 528
training patients, density is 0.21, and CSR takes 21 KB instead of 59 KB dense
(`m1_sparse_one_hot.md`). The modelling `ColumnTransformer` outputs dense arrays, because
the matrix is small and some learners need dense input; `preprocess.encode.one_hot_sparse`
shows the sparse representation.

## KD-tree vs ball tree in kNN

Brute-force kNN compares a query with every training point. A **KD-tree** splits space
along one coordinate at a time. A **ball tree** nests hyperspheres, and its triangle-
inequality pruning holds up better in more dimensions. All three give identical
neighbours (`m5_knn_algorithms.md`). In our 44 dimensions with 490 points, brute force was
fastest (fit 0.7 ms, query 4.8 ms), ball tree next (3.1 / 9.2 ms) and KD-tree slowest
(4.9 / 11.5 ms). This is the curse of dimensionality: tree pruning stops paying off when
d is large relative to log n.

## Heap in beam search (`heapq`)

`search/algorithms.py::beam_search` keeps the best *w* = 5 candidate subsets. A bounded heap
gives O(log w) insert and evict-the-worst (`heappush` / `heapreplace` on negated scores, so
the worst kept subset sits at the top). It does not need to sort every child each round.

## Tabu list (`collections.deque(maxlen=tenure)`)

Tabu search must remember the last 7 flipped bits and forget older ones automatically. A
`deque` with `maxlen` does exactly that: appending the 8th move drops the oldest, in O(1).
Membership checks (`i in tabu`) are O(tenure), which is trivial at 7.

## Causal DAG (`networkx.DiGraph`)

The assumed causal structure is a directed acyclic graph (`report/diagrams.py`, drawn to
`reports/figures/m11_causal_dag.png`):
- **randomization → drug arm**, with no arrow from baseline features or unmeasured factors
  into the arm;
- **drug arm → injections/laser → letters**, and **drug arm → letters**;
- **baseline features → letters**, and **unmeasured factors → baseline / letters**.

Because nothing points into the arm except randomization, the arm's effect on letters is
identified without adjustment. Baseline features add precision and personalisation.
Injections and laser are consequences of treatment and are never adjusted for. Storing the
DAG as a graph object also makes it easy to check, for example with `nx.is_directed_acyclic_graph`.

## Trees

- **Decision trees / random forests / gradient boosting** (M5): binary trees whose internal
  nodes test one feature against a threshold and whose leaves hold predictions.
- **Causal forest** (M6): the same structure, but each split maximises the difference in
  *treatment effect* between children, and the leaves hold effect estimates.
- **Ward dendrogram** (M3): a binary merge tree. The leaves are patients and the height of
  each internal node is the variance increase of that merge (`m3_dendrogram.png`).
- **KD/ball trees** (above): spatial search trees.
