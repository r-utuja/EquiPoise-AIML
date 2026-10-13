"""
Build per-patient CST quality flags from the DRCR reading-centre OCT grades.

Output: one row per patient, bookkeeping (M_) columns only, never model features:

  M_cst_flag_baseline / _wk52 / _wk104   1 if the reading centre rated the study-eye central
                                         subfield "Unacceptable" (off-fovea / plotting errors)
  M_oct_switch_wk52                      1 if the week-52 scan used a different OCT machine
                                         than baseline (raw CST change then mixes machines)

Used for the CST sensitivity analysis (config `preprocess.cst.quality_policy`). Grades are the
same rows M0 uses for X_cst_um and Y_cst_change_*: GradeType ST, study eye.

Usage:
    python scripts/build_cst_quality.py "<path to 'Data Tables - Text files'>" <output.csv>

Data source: DRCR Retina Network (FAIN UG1EY014231), Protocol T public dataset,
https://public.jaeb.org/drcrnet (full attribution in build_protocolT_table.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from build_protocolT_table import load  # M0 helper (same folder)

from equipoise.data.schema import CST_QUALITY_COLUMNS, N_PATIENTS

UNACCEPTABLE_PREFIX = "Unacceptable"
VISITS = {"baseline": "Baseline", "wk52": "52 week", "wk104": "104 week"}


def main(folder: Path, out: Path) -> None:
    trt = load(folder, "TTrtGroupUnmasked").rename(columns={"PtID": "pt_id"})
    eye = trt.set_index("pt_id")["RandEye"]
    oc = load(folder, "RCOCTGradingResult").rename(columns={"PtID": "pt_id"})
    oc = oc[(oc.GradeType == "ST") & (oc.OCTScanEye == oc.pt_id.map(eye))]
    oc = oc[oc.Visit.isin(VISITS.values())].drop_duplicates(["pt_id", "Visit"])
    bad = oc.CentralSubQlty.fillna("").str.startswith(UNACCEPTABLE_PREFIX)

    df = pd.DataFrame({"pt_id": sorted(eye.index)})
    for key, visit in VISITS.items():
        flagged = oc.loc[bad & (oc.Visit == visit), "pt_id"]
        df[f"M_cst_flag_{key}"] = df.pt_id.isin(flagged).astype(int)
    machine = oc.pivot(index="pt_id", columns="Visit", values="OCTMachine")
    switch = machine["52 week"].notna() & (machine["52 week"] != machine["Baseline"])
    df["M_oct_switch_wk52"] = df.pt_id.isin(switch[switch].index).astype(int)

    assert len(df) == N_PATIENTS and list(df.columns) == [c.name for c in CST_QUALITY_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {out}  " + "  ".join(f"{c}={int(df[c].sum())}" for c in df.columns[1:]))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
