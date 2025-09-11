#!/usr/bin/env python
"""
Compare species richness for IBM, ODE and PSD with one-way ANOVA
under two body-mass regimes.

Outputs for every regime
  • richness_<regime>.png     … time-series plot
  • anova_<regime>.txt        … summary values + ANOVA table
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from analysis import richness_timeseries   # <-- your helper

# ----------------------------------------------------------------------
THRESH   = 1e-3          # biomass cut-off for richness
REC_STEP = 1000          # model recording interval (time units)
METRIC   = "median"      # "mean" | "median" | user-defined
OUTDIR   = "anova_results"
os.makedirs(OUTDIR, exist_ok=True)

CASES = {
    "low":  {
        "folder": "body-mass 1e-11 --inv 1e-10 --S 500",
        "label": "M_b = 10⁻¹¹ kg"
    },
    "high": {
        "folder": "body-mass 1e-4 --inv 1e-10 --S 500 --seeds456",
        "label": "M_b = 10⁻⁴ kg"
    }
}

def summarise(ts: np.ndarray, how: str = METRIC) -> float:
    """Reduce a time-series to a single number."""
    if how == "mean":
        return float(np.mean(ts))
    if how == "median":
        return float(np.median(ts))
    raise ValueError(f"Unknown summary statistic {how!r}")

# ======================================================================
for tag, case in CASES.items():
    # ---- load model outputs ------------------------------------------
    path = os.path.join(case["folder"], "model_outputs.npz")
    dat  = np.load(path)

    rich_ibm, _ = richness_timeseries(dat["IBM"],  THRESH)
    rich_ode, _ = richness_timeseries(dat["ODE"],  THRESH)
    rich_psd, _ = richness_timeseries(dat["PSD2"], THRESH)

    # ---- trim to the same length -------------------------------------
    n = min(len(rich_ibm), len(rich_ode), len(rich_psd))
    rich_ibm, rich_ode, rich_psd = rich_ibm[:n], rich_ode[:n], rich_psd[:n]

    # ---- summary statistics (you can switch METRIC at the top) -------
    stat_ibm = summarise(rich_ibm)
    stat_ode = summarise(rich_ode)
    stat_psd = summarise(rich_psd)

    # ---- one-way ANOVA ----------------------------------------------
    f_val, p_val = f_oneway(rich_ibm, rich_ode, rich_psd)

    # ---- write text summary ------------------------------------------
    txtfile = os.path.join(OUTDIR, f"anova_{tag}.txt")
    with open(txtfile, "w") as fh:
        fh.write(f"Case:          {case['label']}\n")
        fh.write(f"Threshold:     {THRESH:g}\n")
        fh.write(f"Statistic:     {METRIC}\n\n")
        fh.write("Summary values\n")
        fh.write(f"  IBM : {stat_ibm:.3f}\n")
        fh.write(f"  ODE : {stat_ode:.3f}\n")
        fh.write(f"  PSD : {stat_psd:.3f}\n\n")
        fh.write("One-way ANOVA\n")
        fh.write(f"  F-value : {f_val:.3f}\n")
        fh.write(f"  p-value : {p_val:.3g}\n")

    # ---- quick progress message --------------------------------------
    print(f"{case['label']} → F={f_val:.3f}, p={p_val:.3g}")

    # ---- optional richness plot --------------------------------------
    t = np.arange(n) * REC_STEP
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, rich_ibm, label="IBM")
    ax.plot(t, rich_ode, label="ODE")
    ax.plot(t, rich_psd, label="PSD")
    ax.set_xlabel("time")
    ax.set_ylabel(f"species ≥ {THRESH:g}")
    ax.set_title(f"Richness vs time ({case['label']})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"richness_{tag}.png"), dpi=300)
    plt.close(fig)
    
    
    
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# 'vals' is concatenated vector, 'labels' is matching vector of "IBM"/"ODE"/"PSD"
df       = pd.DataFrame({"value": np.concatenate([rich_ibm, rich_ode, rich_psd]),
                         "group": np.repeat(["IBM","ODE","PSD"], [len(rich_ibm), len(rich_ode), len(rich_psd)])})
tukey    = pairwise_tukeyhsd(df["value"], df["group"])
print(tukey.summary())

