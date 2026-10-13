"""Altair charts for the main flow: minimal styling, direct labels, no legend (the drug
colours match the cards), no gridlines, no title repeating the heading. Built from the view
models in `presenters.py`; every chart is paired with a one-sentence text summary on the page.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from app.presenters import ChartVM, ProgressVM
from app.theme import INK, drug_color

ROW_HEIGHT = 64


def improvement(vm: ChartVM, ink: str = INK["light"]) -> alt.LayerChart:
    """Expected lines gained per drug: a dot, a soft band for the likely range, a text label."""
    df = pd.DataFrame([{"drug": r.label, "lines": r.lines, "low": r.low, "high": r.high,
                        "text": r.text} for r in vm.rows])
    order = [r.label for r in vm.rows]
    top = max([r.lines for r in vm.rows] + [r.high for r in vm.rows if r.high is not None])
    bottom = min([0.0] + [r.lines for r in vm.rows] + [r.low for r in vm.rows
                                                         if r.low is not None])
    x_scale = alt.Scale(domain=[min(0.0, bottom - 0.25), top + 1.0], nice=False)
    y = alt.Y("drug:N", sort=order, title=None,
              axis=alt.Axis(labelFontSize=14, labelFontWeight="bold", ticks=False,
                            domain=False, labelPadding=10))
    color = alt.Color("drug:N", scale=alt.Scale(domain=order,
                                                range=[drug_color(r.drug) for r in vm.rows]),
                      legend=None)
    base = alt.Chart(df)
    band = base.transform_filter("isValid(datum.low)").mark_bar(
        height=20, opacity=0.22, cornerRadius=6).encode(x=alt.X("low:Q", scale=x_scale),
                                                        x2="high:Q", y=y, color=color)
    dot = base.mark_circle(size=220, opacity=1).encode(
        x=alt.X("lines:Q", scale=x_scale, title=vm.axis,
                axis=alt.Axis(tickMinStep=0.5, grid=False, titleFontSize=13, format="~g",
                              titlePadding=10, labelFontSize=12)),
        y=y, color=color, tooltip=[alt.Tooltip("text:N", title=" ")])
    label = base.mark_text(align="center", dy=-17, fontSize=13, color=ink).encode(
        x=alt.X("lines:Q", scale=x_scale), y=y, text="text:N")
    return (band + dot + label).properties(height=ROW_HEIGHT * len(order)) \
        .configure_view(stroke=None).configure_axis(grid=False)


def progress(vm: ProgressVM, ink: str = INK["light"]) -> alt.LayerChart:
    """Lines gained since the start: entered visits (solid) and the 1-year forecast (dashed,
    with a soft band for the typical error)."""
    obs = pd.DataFrame([{"t": p.label, "week": p.week, "lines": p.lines} for p in vm.observed])
    fc = pd.DataFrame([{"t": p.label, "week": p.week, "lines": p.lines}
                       for p in (vm.observed[-1], vm.forecast)])
    c = drug_color(vm.drug)
    x = alt.X("week:Q", title=None, scale=alt.Scale(domain=[0, vm.forecast.week + 2]),
              axis=alt.Axis(values=[p.week for p in (*vm.observed, vm.forecast)],
                            labelExpr=vm.label_expr, grid=False, labelFontSize=12))
    y = alt.Y("lines:Q", title=vm.axis, axis=alt.Axis(grid=False, tickMinStep=0.5,
                                                      titleFontSize=13))
    solid = alt.Chart(obs).mark_line(point=alt.OverlayMarkDef(size=80, color=c), color=c,
                                     strokeWidth=3).encode(x=x, y=y)
    dashed = alt.Chart(fc).mark_line(strokeDash=[6, 4], color=c, strokeWidth=2).encode(x=x, y=y)
    layers = [solid, dashed]
    if vm.error_lines is not None:
        band = pd.DataFrame([{"week": vm.forecast.week, "low": vm.forecast.lines - vm.error_lines,
                              "high": vm.forecast.lines + vm.error_lines}])
        layers.append(alt.Chart(band).mark_rule(strokeWidth=8, opacity=0.2, color=c)
                      .encode(x="week:Q", y="low:Q", y2="high:Q"))
    end = pd.DataFrame([{"week": vm.forecast.week, "lines": vm.forecast.lines,
                         "text": vm.forecast.text}])
    layers.append(alt.Chart(end).mark_point(shape="diamond", size=160, filled=True, color=c)
                  .encode(x="week:Q", y="lines:Q"))
    layers.append(alt.Chart(end).mark_text(align="right", dx=-12, dy=-16, fontSize=13, color=ink)
                  .encode(x="week:Q", y="lines:Q", text="text:N"))
    zero = alt.Chart(pd.DataFrame([{"z": 0}])).mark_rule(color="#8c8b86", strokeWidth=1,
                                                         opacity=0.6).encode(y="z:Q")
    return alt.layer(zero, *layers).properties(height=260).configure_view(stroke=None)
