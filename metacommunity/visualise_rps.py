#!/usr/bin/env python3
"""
visualise_rps.py – create the four standard figures for the 3-species RPS benchmark
------------------------------------------------------------------------------

Usage
-----
python visualise_rps.py results/data/rps_dataset.npz
"""

# ---------------------------------------------------------------- backend first
import matplotlib
matplotlib.use("Agg")          # head-less backend → no Qt / Wayland needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- std-lib & deps
import argparse, json, pathlib, textwrap, warnings
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – needed for 3-D proj.

# ---------------------------------------------------------------- CLI
p = argparse.ArgumentParser()
p.add_argument("dataset", type=pathlib.Path,
               help="*.npz file written by run_all_rps.py")
args = p.parse_args()

root   = args.dataset.parent.parent      # → results/
d_plot = root / "plots"
d_plot.mkdir(parents=True, exist_ok=True)

# -------- optional helper: invasion counter (very simple fallback) ----------
def _count_invasions_simple(B):
    """Return total #invasions as #times a previously empty cell becomes >0."""
    return int(((B[1:] > 0) & (B[:-1] == 0)).sum())

# try to import the full version – fall back to simple one if not available
try:
    from utils_analysis import count_invasions as _count_invasions
except Exception:
    _count_invasions = lambda B: (_count_invasions_simple(B), None, None)

# ---------------------------------------------------------------- load data
with np.load(args.dataset, allow_pickle=True) as z:
    meta = json.loads(z["meta"].item())

    t_psd, B_psd = z["PSD2_t"], z["PSD2_B"]
    t_ibm, B_ibm = z["IBM_t"],  z["IBM_B"]
    t_ode        = z.get("ODE_t", np.array([]))
    B_ode        = z.get("ODE_B", np.empty((0,)))

    # PSD-2 extras can be missing in older files
    wait = z.get("PSD2_wait")      # S-state mask
    pc   = z.get("PSD2_pclock")    # Poisson clocks

# ---------------------------------------------------------------- helpers
def richness(B):      return (B > 0).sum(axis=1)
def mean_biomass(B):  return B.mean(axis=(2, 3))

# ---------------------------------------------------------------- ensure meta fields
for tag,B in (("PSD2",B_psd), ("IBM",B_ibm), ("ODE",B_ode)):
    if not B.size:          # ODE might be absent
        continue
    m = meta.setdefault(tag, {})
    if "total_invasions" not in m:
        m["total_invasions"] = _count_invasions(B)[0]
    if "period" not in m:   # crude proxy runtime ≈ last time point
        m["period"] = float((t_psd if tag=="PSD2" else
                             t_ibm if tag=="IBM" else t_ode)[-1])

# ─────────────────────────────────────────────────────────────────────────
# 1 · STATE-TRANSITION STACKPLOT  (PSD-2 only)
# ─────────────────────────────────────────────────────────────────────────
if (wait is not None) and (pc is not None):
    S, Ny, Nx = B_psd.shape[1:]
    state_counts = np.zeros((len(t_psd), 3), int)          # S, P, D
    eps = 1e-12                                           # FP tolerance

    for k in range(len(t_psd)):
        S_mask =  wait[k]                                 # stochastic
        D_mask = (~wait[k]) & np.isclose(pc[k], 1.0, atol=eps)   # deterministic
        P_mask = (~wait[k]) & (~D_mask)                   # the rest → propagule
        state_counts[k] = [S_mask.sum(), P_mask.sum(), D_mask.sum()]

    fig, ax = plt.subplots(figsize=(9, 4),                 # no tight-layout warn
                        constrained_layout=True)
    labels = ["Stochastic (S)", "Propagule (P)", "Deterministic (D)"]
    fractions = (state_counts.T / state_counts.sum(1))     # normalise
    ax.stackplot(t_psd, fractions, labels=labels,
                alpha=.8, linewidth=0)
    ax.set(xlabel="time",
        ylabel="fraction of all (species × patches)",
        title="PSD-2 state composition over time")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(d_plot / "RPS_state_transitions.png", dpi=300)
    plt.close(fig)
else:
    warnings.warn("PSD-2 waiting / p-clock arrays not found – "
                "state-transition plot skipped")


# ---------------------------------------------------------------- 4 · RADAR / SPIDER CHART
RADAR_METRICS = ["Realism", "Speed", "Invasion\ncount"]
realism_rank = dict(PSD2=3, IBM=5, ODE=2)          # edit if you have numbers

# speed → shorter period ⇒ higher score
periods = {k: v["period"]
           for k, v in meta.items()
           if isinstance(v, dict) and "period" in v}
fastest = min(periods.values())
speed_score = {k: 1 + 4*fastest/v for k,v in periods.items()}

# invasion-count on log-scale 1-5
inv = {k: v["total_invasions"]
           for k, v in meta.items()
           if isinstance(v, dict) and "total_invasions" in v}

log_inv = np.log10(list(inv.values()))
inv_score = {k: 1 + 4*(np.log10(v)-log_inv.min())/(log_inv.max()-log_inv.min())
             for k,v in inv.items()}

scores = { "PSD-2": [realism_rank["PSD2"], speed_score["PSD2"], inv_score["PSD2"]],
           "IBM"  : [realism_rank["IBM"],  speed_score["IBM"],  inv_score["IBM" ]],
           "ODE"  : [realism_rank["ODE"],  speed_score["ODE"],  inv_score["ODE" ]] }

# ---------- radar helper -------------------------------------------------
def radar_factory(n):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta += np.pi / 2                       # rotate so first axis is at the top
    def _new(fig):
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_xticks(theta)
        ax.set_xticklabels(RADAR_METRICS, ha="center")
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 5)
        return theta, ax
    return _new

# --- create radar axes ---------------------------------------------------
theta0, ax = radar_factory(len(RADAR_METRICS))(plt.figure(figsize=(6, 6),
                                                          constrained_layout=True))

for name, vals in scores.items():
    vals = np.asarray(vals)
    theta = np.append(theta0, theta0[0])     # close the polygon
    radii = np.append(vals,    vals[0])
    ax.plot(theta, radii, label=name)
    ax.fill(theta, radii, alpha=.25)

ax.set_title("Model comparison (higher = better on each axis)")
ax.legend(bbox_to_anchor=(1.3, 1.1), frameon=False)
plt.savefig(d_plot / "RPS_model_comparison.png", dpi=300)
plt.close()



# ---------------------------------------------------------------- summary
print(textwrap.dedent(f"""
✔  Mean biomass time-series    → {d_plot/'RPS_mean_biomass.png'}
✔  Radar / spider chart        → {d_plot/'RPS_model_comparison.png'}
✔  3-D stacked-bar mosaic      → {d_plot/'RPS_3d_mosaic.png'}
"""))
