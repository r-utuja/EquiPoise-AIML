# PCA via SVD

**What.** Centre the scaled feature matrix X and factor it as X = U Σ Vᵀ with
`numpy.linalg.svd`. The rows of Vᵀ are the principal directions. Σ²/(n − 1) gives the
variance along each direction, and U Σ gives each patient's coordinates (scores). This is
PCA without calling a PCA library.

**Why here.** It shows whether patients form natural groups and how many directions are
needed to describe them. Implementing it directly, then checking it against sklearn, is a
course requirement.

**What it showed.** Our SVD matches `sklearn.decomposition.PCA` to machine precision (max
score difference 3.6e-15; `m3_svd_vs_sklearn.md`). The first component explains only 15.7%
of the variance, and 12 components are needed to reach 80% (`m3_represent.json`). Baseline
features are spread over many weak directions, with no dominant axis. The 2-D map
(`m3_pca_map.png`) shows the arms completely mixed, as randomization implies.
