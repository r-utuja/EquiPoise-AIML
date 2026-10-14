"""Association rules (Apriori) on discretized baseline items plus a "gained >= 15 letters" item.

Each patient becomes a basket of items such as `va=20/50-20/80`, `prior_antivegf` or
`gain>=15`. Apriori finds itemsets whose support (share of baskets containing them) is above
a threshold, using the fact that any subset of a frequent itemset is also frequent. Rules
A -> B are then scored by confidence P(B | A) and lift = confidence / P(B); lift > 1 means A
and B occur together more often than chance. Rules are mined separately within each arm.
They describe co-occurrence in this sample only: exploratory, not causal.
"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from equipoise.config import Config
from equipoise.preprocess.discretize import discretize, gain_item

GAIN_ITEM = "gain>=15"


def baskets(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One-hot item matrix (bool) for complete-case patients."""
    d = df.dropna(subset=[cfg.primary_outcome])
    bands = discretize(d, cfg)
    items = pd.DataFrame(index=d.index)
    for col in bands.columns:
        short = col.replace("X_", "").replace("_band", "")
        for level in bands[col].cat.categories:
            items[f"{short}={level}"] = bands[col] == level
    for col in cfg.represent["apriori"]["binary_items"]:
        items[col.replace("X_", "")] = d[col] == 1
    for level in d["X_dr_severity_clinical"].dropna().unique():
        items[f"DR={level}"] = d["X_dr_severity_clinical"] == level
    items[GAIN_ITEM] = gain_item(d, cfg) == 1
    return items.astype(bool)


def mine_rules(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Top rules by lift within each arm (plus top rules whose consequent is the gain item)."""
    a = cfg.represent["apriori"]
    out = []
    for arm in cfg.arms:
        b = baskets(df[df.treatment == arm], cfg)
        freq = apriori(b, min_support=float(a["min_support"]), use_colnames=True,
                       max_len=int(a["max_len"]))
        if freq.empty:
            continue
        rules = association_rules(freq, metric="confidence",
                                  min_threshold=float(a["min_confidence"]))
        if rules.empty:
            continue
        rules = rules.assign(
            antecedents=rules.antecedents.map(lambda s: " & ".join(sorted(s))),
            consequents=rules.consequents.map(lambda s: " & ".join(sorted(s))),
            arm=arm, n_baskets=len(b))
        cols = ["arm", "antecedents", "consequents", "support", "confidence", "lift",
                "n_baskets"]
        rules = rules[cols].sort_values(["lift", "support", "antecedents"],
                                        ascending=[False, False, True])
        top = rules.head(int(a["top_n"])).assign(kind="top lift")
        gain = rules[rules.consequents == GAIN_ITEM].head(int(a["top_n"])).assign(
            kind=f"consequent = {GAIN_ITEM}")
        out += [top, gain]
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return res.assign(note="exploratory; association, not causation")
