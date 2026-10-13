"""Pipeline flow diagram (README section 4) and the assumed causal DAG, drawn with
matplotlib / networkx so no system Graphviz install is needed.

The DAG states the assumption behind every effect estimate: the drug was assigned at random,
so no arrow enters treatment from the patient's characteristics (measured or not). Injection
counts and laser are consequences of treatment, so they are never adjusted for.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from equipoise.plotting import INK, INK_2, NEUTRAL, SERIES, SURFACE

# (key, label, x, y, colour index); README section 4 layout
_NODES = [
    ("raw", "Raw DRCR\ntables", 0.0, 2.0, None),
    ("m0", "M0\nintegration", 1.25, 2.0, 0), ("m1", "M1\npreprocess", 2.5, 2.0, 0),
    ("m2", "M2\nstatistics", 3.75, 2.0, 0), ("m3", "M3\nrepresent", 5.0, 2.0, 0),
    ("m4", "M4\nfeature search", 6.25, 2.0, 0), ("m5", "M5\noutcome models", 7.5, 2.0, 0),
    ("m6", "M6\ncounterfactual\nengine", 8.75, 2.0, 1), ("m8", "M8\ndecision layer", 10.0, 2.0, 2),
    ("app", "Clinician app\n+ API", 11.4, 2.0, 3),
    ("m9", "M9\nexplain + fairness", 10.0, 0.9, 2),
    ("visits", "Visit-level\ndata", 7.5, -0.2, None), ("m7", "M7\nlongitudinal", 8.75, -0.2, 2),
    ("oct", "Kermany OCT\nimages (separate)", 7.5, -1.3, None),
    ("m10", "M10 OCT screening\n(optional)", 8.75, -1.3, 4),
]
_EDGES = [("raw", "m0"), ("m0", "m1"), ("m1", "m2"), ("m2", "m3"), ("m3", "m4"), ("m4", "m5"),
          ("m5", "m6"), ("m6", "m8"), ("m8", "app"), ("m6", "m9"), ("m9", "app"),
          ("visits", "m7"), ("m7", "app"), ("oct", "m10"), ("m10", "app")]


def flow_diagram() -> plt.Figure:
    """Boxes-and-arrows view of modules M0-M11 (M10 dashed: optional, separate data)."""
    fig, ax = plt.subplots(figsize=(15, 6.2))
    w, h = 1.05, 0.72
    pos = {k: (x, y) for k, _, x, y, _ in _NODES}
    for key, label, x, y, ci in _NODES:
        face = SURFACE if ci is None else SERIES[ci]
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03",
                             fc=face, ec=INK_2 if ci is None else face, alpha=1.0 if ci is None
                             else 0.18, lw=1.4, ls="--" if key in ("m10", "oct") else "-")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color=INK)
    for a, b in _EDGES:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        if y1 == y2:
            start, end = (x1 + w / 2, y1), (x2 - w / 2, y2)
        elif x1 == x2:
            start, end = (x1, y1 + h / 2 if y2 > y1 else y1 - h / 2), \
                (x2, y2 - h / 2 if y2 > y1 else y2 + h / 2)
        elif b == "app":   # converge on the app box's bottom edge
            start, end = (x1 + w / 2, y1), (x2 - 0.3 + 0.15 * (y1 + 1.3), y2 - h / 2)
        else:
            start, end = (x1 + w / 2, y1), (x2 - w / 2, y2 + (0.15 if y2 > y1 else -0.15))
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     color=NEUTRAL, lw=1.2,
                                     ls="--" if "m10" in (a, b) else "-",
                                     connectionstyle="arc3,rad=0.0"))
    ax.add_patch(FancyBboxPatch((-0.55, -2.25), 12.55, 0.5, boxstyle="round,pad=0.02",
                                fc=SERIES[6], alpha=0.12, ec="none"))
    ax.text(5.7, -2.0, "M11 engineering underneath everything: Git · DVC · MLflow · Docker · "
            "GitHub Actions CI · FastAPI · Streamlit · Evidently", ha="center", va="center",
            fontsize=9, color=INK)
    ax.set_xlim(-0.7, 12.1)
    ax.set_ylim(-2.4, 2.6)
    ax.axis("off")
    ax.set_title("Equipoise pipeline: from randomized-trial tables to a cost-aware "
                 "recommendation", fontsize=12)
    fig.tight_layout()
    return fig


def causal_dag() -> plt.Figure:
    """Assumed DAG for Protocol T (randomization removes every arrow into treatment)."""
    g = nx.DiGraph()
    nodes = {
        "U": ("Unmeasured patient\nfactors", (0.0, 1.2)),
        "X": ("Baseline features X_\n(vision, CST, HbA1c, ...)", (0.0, 0.0)),
        "R": ("Randomization\n(known probabilities)", (1.6, 1.4)),
        "T": ("Drug arm\n(treatment)", (1.6, 0.0)),
        "N": ("Injections, laser\n(post-randomization)", (3.2, 0.9)),
        "Y": ("Letters gained\nat week 52 (Y_)", (3.2, -0.6)),
    }
    for k, (label, _) in nodes.items():
        g.add_node(k, label=label)
    solid = [("X", "Y"), ("T", "Y"), ("T", "N"), ("X", "N"), ("N", "Y"), ("R", "T")]
    dashed = [("U", "X"), ("U", "Y"), ("U", "N")]
    g.add_edges_from(solid + dashed)
    pos = {k: p for k, (_, p) in nodes.items()}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [SERIES[1] if k == "T" else SERIES[0] if k == "Y" else
              (NEUTRAL if k == "U" else SERIES[2]) for k in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_size=5200, node_color=colors, alpha=0.2, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=nx.get_node_attributes(g, "label"), font_size=8,
                            ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=solid, arrowstyle="-|>", arrowsize=16,
                           node_size=5200, edge_color=INK_2, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=dashed, arrowstyle="-|>", arrowsize=16,
                           node_size=5200, edge_color=NEUTRAL, style="dashed", ax=ax)
    ax.text(1.6, -1.25, "No arrow from X or U into the drug arm: randomization. Effects of T on Y "
            "are identified\nwithout adjustment; X_ improves precision and personalisation. "
            "Injections/laser are never adjusted for.", ha="center", fontsize=8.5, color=INK_2)
    ax.set_title("Assumed causal DAG (Protocol T)")
    ax.set_xlim(-0.9, 4.1)
    ax.set_ylim(-1.5, 1.9)
    ax.axis("off")
    fig.tight_layout()
    return fig
