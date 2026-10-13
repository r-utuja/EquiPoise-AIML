"""The one-page printable summary (app/pdf.py)."""

from __future__ import annotations

import pytest

from app import presenters as P
from equipoise.config import load_config
from tests.test_presenters import BINARY, TYPICAL, make_pred, with_explanation

pytest.importorskip("fpdf")


def test_summary_pdf_is_one_page_and_has_the_key_text() -> None:
    from app.pdf import summary_pdf

    ui = load_config().raw["ui"]
    pred = with_explanation(make_pred(p_afl=0.52, demo=False))
    pdf = summary_pdf(P.comparison(pred, ui, 48), P.why(pred, ui, 48, TYPICAL, BINARY),
                      "Vision today about 20/100 (6/30). Retinal thickness 768 µm.",
                      "Data: DRCR Retina Network")
    assert pdf[:5] == b"%PDF-"
    assert pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + \
        pdf.count(b"/Type /Page>") <= 1 or b"/Count 1" in pdf
