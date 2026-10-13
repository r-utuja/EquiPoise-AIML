# Clustering and association rules (exploratory)

**k-means** places k centres and repeatedly assigns each patient to the nearest centre,
then moves each centre to the mean of its patients. k is chosen by the elbow (inertia) and
the **silhouette** (how much closer a point is to its own cluster than to the next one, from
−1 to 1). **Ward hierarchical clustering** starts with every patient alone and merges the
pair of clusters whose merge increases within-cluster variance least; the dendrogram shows
the merge order. **DBSCAN** grows clusters from points with at least `min_samples`
neighbours within eps (eps read off the k-distance plot). Points in no dense region are
noise.

**What it showed** (`m3_represent.json`, `m3_cluster_effects.md`). The best k was 3, but the
silhouette is only 0.11, so the clusters are weak. DBSCAN finds a single dense cluster plus
3 noise points. Within the k-means cluster with the worst mean baseline vision (54.8
letters), aflibercept − bevacizumab = 8.3 letters (CI 2.9 to 13.8). In the other clusters
the difference is near zero or modest. The clusters rediscover the baseline-vision story,
but 14 comparisons were made and none was pre-specified, so this is **exploratory**.

**Apriori** finds itemsets that occur in at least `min_support` of patients and grows only
from frequent subsets (if A is rare, A + B is rarer). Rules A → B are ranked by
**confidence** P(B | A) and **lift** P(B | A) / P(B). We mine within each arm, with the
item "gained ≥ 15 letters". In the aflibercept arm, "vision 20/100 or worse → gained ≥ 15"
has confidence 0.82 and lift 2.09 (`m3_association_rules.md`). The highest-lift rules are
clinical co-occurrences (prior PRP with PDR). These are **associations, not causes**.
