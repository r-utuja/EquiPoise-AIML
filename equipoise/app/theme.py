"""Colours and layout for the Streamlit app, defined once.

Drug colours are the report palette from `equipoise.plotting` (validated for colour-vision
deficiency), so app charts, cards and report figures always agree. Every chart that uses
them also carries direct text labels: colour is never the only signal.
"""

from __future__ import annotations

from equipoise.plotting import ARM_COLORS, NEUTRAL

DRUG_COLORS: dict[str, str] = dict(ARM_COLORS)
MUTED = NEUTRAL
ACCENT = "#1f6fb2"            # matches .streamlit/config.toml primaryColor
MAX_WIDTH_PX = 960            # keep lines readable on wide screens

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ max-width: {MAX_WIDTH_PX}px; padding-top: 2.5rem; }}
.eq-notice {{ font-size: 0.85rem; opacity: 0.75; margin: -0.5rem 0 1rem 0; }}
.stMarkdown p.eq-headline {{ font-size: 1.3rem !important; line-height: 1.45;
                            font-weight: 600; margin: 0 0 0.4rem 0; }}
.stMarkdown p.eq-big {{ font-size: 1.2rem !important; font-weight: 600; margin: 0; }}
.eq-swatch {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 0.2rem;
              margin-right: 0.45rem; vertical-align: -0.05rem; }}
.eq-step {{ font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.2rem; }}
</style>
"""


INK = {"light": "#1d1d1b", "dark": "#ececea"}   # chart text, matching config.toml


def ink(theme_type: str | None) -> str:
    return INK["dark"] if theme_type == "dark" else INK["light"]


def drug_color(drug: str) -> str:
    return DRUG_COLORS.get(drug, MUTED)


def swatch(drug: str) -> str:
    """Small coloured square placed before a drug name (decorative; the name carries meaning)."""
    style = f"background:{drug_color(drug)}"
    return f'<span class="eq-swatch" style="{style}" aria-hidden="true"></span>'
