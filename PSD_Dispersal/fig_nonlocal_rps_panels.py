# rps_nonlocal_make_all.py
# Regenerates ODE/IBM/PSD nonlocal time-series for full & half D_source
# Outputs 6 PNGs + 2 combined mosaics into figures/multipatch/nonlocal/

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time

# --- project imports (as in your repo) ---
import config
config.LONG_DISTANCE_PROB = 1.0  # pure non-local in all runs

from config import (
    NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS, MORTALITY_RATE, STEP_SIZE
)
from models_ode import ODEModel
from models_ibm import IBMModel
from models_psd2 import PSD2Model

# ---------- theory: Axel–PDF period using D_source (leave rate) ----------
def predicted_period_axel_pdf(D_source, body_mass, r, b, mu):
    # T = 2√3 π * body_mass * (mu + r - b r) / ((1 - b) * D_source * r^2)
    return (2.0*np.sqrt(3.0)*np.pi * body_mass * (mu + r - b*r)) / ((1.0 - b) * D_source * (r**2))

# ---------- helpers ----------
def _ensure_dir(path):
    p = pathlib.Path(path); p.mkdir(parents=True, exist_ok=True); return p

def _plot_timeseries(times, mean_ts, out_png):
    plt.figure(figsize=(8,4))
    for s in range(mean_ts.shape[1]):
        plt.plot(times, mean_ts[:, s], label=f"Species {s+1}")
    plt.xlabel("Time")
    plt.ylabel("Spatial mean biomass per patch")
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    # Save PDF version
    out_pdf = str(out_png).replace('.png', '.pdf')
    plt.savefig(out_pdf, dpi=300)
    plt.close()

# ---------- common RPS params ----------
S = 3
r_vec = np.ones(S)
r0   = 1.0
a, b = 1.7, 0.4
C = np.array([[1.0, a,   b],
              [b,   1.0, a],
              [a,   b,   1.0]], dtype=float)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

# run lengths chosen to show several cycles:
# small-amplitude T(full) ~ 2902; T(half) ~ 5804  -> simulate 18–20k time units
TMAX = 2000.0
RECORD = 10
N_STEPS_IBM = int(TMAX / STEP_SIZE)
REC_STEP_IBM = 10  # record every 10 IBM steps

# ---------- initial conditions (same bias across models) ----------
rng = np.random.default_rng(42)
flat_idx = np.arange(Ny * Nx)
chosen = rng.choice(flat_idx, size=(Ny*Nx)//2, replace=False)

def init_ode_psd():
    B0 = np.zeros((S, Ny, Nx), float)
    B0[:] = 2 * BODY_MASS
    B0[0].flat[chosen] += BODY_MASS
    return B0

def init_ibm():
    # counts -> will convert to biomass by * BODY_MASS before plotting
    N0 = np.full((S, Ny, Nx), 2, dtype=int)
    N0[0].flat[chosen] += 1
    return N0

# ---------- per-model runners ----------
def run_ode(D_source, out_png):
    # set leave rate
    base = config.DISPERSAL_RATE
    config.DISPERSAL_RATE = D_source
    model = ODEModel(r=r_vec, C=C, initial_B=init_ode_psd(),
                     tmax=TMAX, record_step=RECORD,
                     dispersal_type="propagule", seed=123)
    t, traj = model.run()  # (nrec, S, Ny, Nx)
    mean_ts = traj.mean(axis=(2,3))
    Tpred = predicted_period_axel_pdf(D_source, BODY_MASS, r0, b, MORTALITY_RATE)
    # title = f"ODE • 100% LDD • D_source={D_source:g}  |  T_pred≈{Tpred:.0f}"
    # title = f"ODE "
    _plot_timeseries(t, mean_ts, out_png)
    config.DISPERSAL_RATE = base

def run_psd(D_source, out_png):
    base = config.DISPERSAL_RATE
    config.DISPERSAL_RATE = D_source
    model = PSD2Model(r=r_vec, C=C, initial_B=init_ode_psd(),
                      tmax=TMAX, record_step=RECORD,
                      dispersal_type="propagule", seed=123)
    t, traj, *_ = model.run()  # traj (nrec, S, Ny, Nx)
    mean_ts = traj.mean(axis=(2,3))
    Tpred = predicted_period_axel_pdf(D_source, BODY_MASS, r0, b, MORTALITY_RATE)
    # title = f"PSD"
    _plot_timeseries(t, mean_ts, out_png)
    config.DISPERSAL_RATE = base

def run_ibm(D_source, out_png):
    base = config.DISPERSAL_RATE
    config.DISPERSAL_RATE = D_source
    model = IBMModel(r=r_vec, C=C, initial_N=init_ibm(),
                     nsteps=N_STEPS_IBM, record_step=REC_STEP_IBM,
                     dispersal_type="propagule", seed=123)
    traj = model.run()  # (nrec, S, Ny, Nx) in COUNTS
    # convert to biomass for plotting
    traj_biomass = traj.astype(float) * BODY_MASS
    mean_ts = traj_biomass.mean(axis=(2,3))
    t = np.arange(traj.shape[0]) * (REC_STEP_IBM * STEP_SIZE)
    Tpred = predicted_period_axel_pdf(D_source, BODY_MASS, r0, b, MORTALITY_RATE)
    # title = f"IBM"
    _plot_timeseries(t, mean_ts, out_png)
    config.DISPERSAL_RATE = base


def main():
    out_dir = _ensure_dir("figures/multipatch/nonlocal")

    # full D: D_source = BODY_MASS * 0.005  -> ~2902
    D_full = BODY_MASS * 0.005
    print(f"Running FULL D (D_source={D_full:g}) …")
    ode_full = "ode_timeseries_full.png"
    ibm_full = "ibm_timeseries_full.png"
    psd_full = "psd_timeseries_full.png"
    run_ode(D_full, out_dir / ode_full)
    run_ibm(D_full, out_dir / ibm_full)
    run_psd(D_full, out_dir / psd_full)

    half D: D_source = BODY_MASS * 0.0025 -> ~5804
    D_half = BODY_MASS * 0.0025
    print(f"Running HALF D (D_source={D_half:g}) …")
    ode_half = "ode_timeseries_half.png"
    ibm_half = "ibm_timeseries_half.png"
    psd_half = "psd_timeseries_half.png"
    run_ode(D_half, out_dir / ode_half)
    run_ibm(D_half, out_dir / ibm_half)
    run_psd(D_half, out_dir / psd_half)

    print("\nDone. Wrote:")
    print(f"  {out_dir/ode_full}")
    print(f"  {out_dir/ibm_full}")
    print(f"  {out_dir/psd_full}")
    
    print("\nDone. Wrote:")
    print(f"  {out_dir/ode_half}")
    # print(f"  {out_dir/ibm_half}")
    # print(f"  {out_dir/psd_half}")

if __name__ == "__main__":
    main()
