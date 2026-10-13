"""Example-patient selection and the held-out "no clear winner" summary
(equipoise.serving.examples), on hand-made scores."""

from __future__ import annotations

import pandas as pd

from equipoise.serving import examples as ex


def scores() -> pd.DataFrame:
    rows = [  # row, va, cst, p
        (0, 75, 360, 0.00), (1, 75, 308, 0.00), (2, 48, 768, 0.52), (3, 59, 560, 0.24),
        (4, 51, 414, 0.24), (5, 69, 668, 0.28), (6, 55, 300, 0.04), (7, 36, 427, 0.52),
    ]
    df = pd.DataFrame(rows, columns=["row", "va", "cst", "p_gt_mcid"])
    df["effect"] = 3.0 + 5 * df.p_gt_mcid
    df["effect_low"], df["effect_high"] = df.effect - 2.5, df.effect + 2.5
    df["best_drug"] = "Aflibercept"
    df["equipoise"] = True
    df["equipoise_mcid_3"] = True
    df["recommendation"] = "Bevacizumab"
    return df


def test_pick_by_scores_spans_the_range() -> None:
    picks = ex.pick_by_scores(scores())
    # highest P, ties -> poorer vision (row 7 has va 36 < 48)
    assert picks["poorer"] == 7
    # P = 0 among vision >= 69 letters, thinnest retina
    assert picks["good"] == 1
    # P closest to halfway (0.26): rows 3, 4, 5 tie at 0.02; vision closest to the middle
    # of the other two (36 + 75) / 2 = 55.5 -> row 4 (51) vs row 3 (59): 3 is 3.5 away, 4 is 4.5
    assert picks["moderate"] == 3


def test_pick_by_features_fallback(synthetic: pd.DataFrame) -> None:
    picks = ex.pick_by_features(synthetic.head(60))
    d = synthetic.head(60).reset_index(drop=True)
    assert set(picks) == set(ex.EXAMPLE_IDS) and len(set(picks.values())) == 3
    assert d.X_va_letters[picks["poorer"]] < d.X_va_letters[picks["moderate"]] \
        <= d.X_va_letters[picks["good"]]


def test_heldout_summary_counts_no_clear_winner() -> None:
    s = scores()
    s.loc[0, "equipoise"] = False
    table = ex.heldout_summary(s, 5.0).set_index("quantity")["value"]
    assert table["n_heldout"] == 8
    assert table["n_no_clear_winner_mcid_5"] == 7
    assert table["n_no_clear_winner_mcid_3"] == 8
    assert table["n_p_gt_mcid_at_least_0.5"] == 2
    assert table["interval_width_letters_median"] == 5.0


def test_examples_file_round_trip(tmp_path) -> None:
    path = ex.write_examples(tmp_path / "x.json", "abc", "sig", {"good": {"features": {}}})
    data = ex.read_examples(path)
    assert data["bundle_hash"] == "abc" and data["examples"]["good"] == {"features": {}}
    assert ex.read_examples(tmp_path / "missing.json") is None
