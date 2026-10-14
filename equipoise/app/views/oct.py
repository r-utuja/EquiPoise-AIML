"""OCT screening demo (M10): a separate image dataset, not connected to the suggestion.

Kermany retinal OCT images are not linked to Protocol T patients and never feed the
treatment model. Shown for reviewers; the page only appears when the [cv] extra is installed
and the M10 models have been trained.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import services
from equipoise.vision.infer import NOT_PROTOCOL_T, OctScreener, example_images
from equipoise.vision.preprocess import read_gray

st.title("OCT screening (demo)")
services.notice()
st.warning(f"**Demo — separate image dataset, not connected to the suggestion.** "
           f"{NOT_PROTOCOL_T}")


@st.cache_resource(show_spinner="Loading the OCT models…")
def screener() -> OctScreener:
    return OctScreener.load(services.CFG)


up = st.file_uploader("Upload an OCT B-scan (JPEG/PNG)", type=["jpg", "jpeg", "png"])
examples = example_images(services.CFG)
choice = None
if examples:
    choice = st.selectbox("…or use a held-out test image", [None, *examples],
                          format_func=lambda p: "–" if p is None else f"{p.parent.name}: {p.name}")
img = None
if up is not None:
    import cv2

    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
elif choice is not None:
    img = read_gray(choice)

if img is not None:
    out = screener().predict(img)
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Input", width="stretch", clamp=True)
    c2.image(out["preprocessed"], caption="Preprocessed (median + CLAHE)", width="stretch",
             clamp=True)
    c3.image(out["gradcam"], caption="Grad-CAM (ResNet18)", width="stretch")
    probs = {"ResNet18": out["resnet"]}
    if out.get("yolo"):
        probs["YOLO-cls"] = out["yolo"]
    df = pd.DataFrame(probs).rename_axis("class").reset_index().melt(
        id_vars="class", var_name="model", value_name="probability")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
        y=alt.Y("class:N", title=None), color=alt.Color("model:N", title="Model"),
        yOffset="model:N").properties(height=220)
    st.altair_chart(chart, width="stretch")
    st.dataframe(pd.DataFrame(probs).round(3), width="stretch")
