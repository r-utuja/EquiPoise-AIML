"""One-page printable patient summary (fpdf2).

Contains the headline, the three medicine cards, "what mattered most" and the disclaimer,
all taken from the same view models the screen shows. Uses the DejaVu Sans font that ships
with matplotlib, so the rupee sign prints without extra font files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from fpdf import FPDF

from app import copy as C
from app.presenters import ComparisonVM, WhyVM
from app.theme import drug_color

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def summary_pdf(cmp: ComparisonVM, why: WhyVM | None, patient_line: str,
                attribution: str) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    w = pdf.epw

    def text(s: str, size: float = 10, bold: bool = False, gap: float = 1.5,
             color: tuple[int, int, int] = (29, 29, 27)) -> None:
        pdf.set_font("DejaVu", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(w, size * 0.48, s, align="L")
        pdf.ln(gap)

    grey = (90, 90, 86)
    text(C.PDF_TITLE, 16, bold=True, gap=1)
    text(C.NOTICE, 9, color=grey, gap=4)
    text(C.PDF_PATIENT, 11, bold=True, gap=0.5)
    text(patient_line, 10, gap=4)

    h = cmp.headline
    text(C.PDF_RESULT, 11, bold=True, gap=0.5)
    text(h.text, 11.5, bold=True)
    text(h.confidence, 10)
    if h.secondary:
        text(h.secondary, 10)
    text(h.trial_finding, 9, color=grey)
    text(h.uncertainty, 9, color=grey, gap=4)

    text(C.PDF_MEDICINES, 11, bold=True, gap=1)
    col_w, top = (w - 8) / 3, pdf.get_y()
    bottom = top
    for i, card in enumerate(cmp.cards):
        x = pdf.l_margin + i * (col_w + 4)
        pdf.set_xy(x, top)
        pdf.set_fill_color(*_rgb(drug_color(card.drug)))
        pdf.rect(x, top, col_w, 1.6, style="F")
        pdf.set_xy(x, top + 3)
        lines = [(card.name, 10, True), (card.tag or "", 8.5, True),
                 (f"{card.improvement} ({card.improvement_letters})", 10, False),
                 (card.range_line, 8.5, False), (card.injections, 8.5, False),
                 (card.cost or "", 8.5, False)]
        for s, size, bold in lines:
            if not s:
                continue
            pdf.set_x(x)
            pdf.set_font("DejaVu", "B" if bold else "", size)
            pdf.set_text_color(29, 29, 27)
            pdf.multi_cell(col_w, size * 0.48, s, align="L")
            pdf.ln(0.8)
        bottom = max(bottom, pdf.get_y())
    pdf.set_xy(pdf.l_margin, bottom + 3)
    if cmp.cost_note:
        text(cmp.cost_note, 9, color=grey, gap=4)

    if why and why.factors:
        text(C.PDF_WHY, 11, bold=True, gap=0.5)
        if why.factors_intro:
            text(why.factors_intro, 9.5, gap=0.5)
        for f in why.factors:
            text(f"•  {f}", 9.5, gap=0.5)
        pdf.ln(3)

    pdf.set_y(max(pdf.get_y(), pdf.h - 42))
    pdf.set_draw_color(200, 200, 196)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + w, pdf.get_y())
    pdf.ln(2)
    text(C.PDF_FOOTER, 8, color=grey, gap=1)
    text(attribution, 7, color=grey, gap=0)
    return bytes(pdf.output())
