# ibm_rps_nonlocal.py

import numpy as np
import matplotlib.pyplot as plt

# ——— 1. Force 100% non‐local dispersal ———
import config
config.LONG_DISTANCE_PROB = 1.0

# ——— 2. Import the IBMModel (assumes you're in the same directory or installed) ———
from models_ibm import IBMModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS, STEP_SIZE  # grid size & units
from utils_analysis import count_invasions  # optional, for summary stats
from utils_vis import make_mosaic       # to build the mosaic at the end

# ——— 3. Set up 3‐species RPS parameters ———
S = 3
r = np.ones(S)           # intrinsic growth rates [1, 1, 1]
a, b = 1.7, 0.4
# RPS competition: species 0 dominates 1 by a, 1→2 by a, 2→0 by a, with off‐diagonals b the other way
C = np.array([
    [1.0, a,   b],
    [b,   1.0, a],
    [a,   b,   1.0]
], dtype=float)

# ——— 4. Grid size and recording parameters ———
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X
# For “well‐mixed” flavor, you can choose Ny=Nx=1 or small. 
# Usually RPS requires more than one patch to see spatial heterogeneity,
# but with 100% non‐local dispersal, patches will mix quickly. 
# We’ll keep the default grid (e.g. 5×5 or whatever your config uses).
# Record every 100 IBM steps, run for 10 000 steps in total.

nsteps = 2000
record_step = 1  # record every 1000 steps
# nrecords = nsteps // record_step  # number of records to keep
nrecords = nsteps // record_step

# ——— 5. Build an “initial condition” that differs by a small offset ———
# We want each species to start with slightly different biomasses on each patch,
# so that the amplitude of the ensuing slow oscillation depends on those offsets.

# Option A: start each species as 2 individuals (so biomass=2*BODY_MASS) everywhere,
# but add a small random offset to one species.
initial_N = np.zeros((S, Ny, Nx), dtype=int)

# Base: each species has 2 individuals on each patch (so biomass = 2*BODY_MASS everywhere)
for s in range(S):
    initial_N[s, :, :] = 2

# Now add a small bias to the first species in half the patches:
# (so species 0 has “2 or 3” individuals on random half; others stay at 2)
rng = np.random.default_rng(42)
flat_idx = np.arange(Ny * Nx)
chosen = rng.choice(flat_idx, size=(Ny*Nx)//2, replace=False)
initial_N[0].flat[chosen] += 1  # now some patches have 3 individuals of species 0

# That small bias in initial counts → slight differences in starting biomass (2 vs. 3) → we expect
# “slow, large‐amplitude” cycles whose amplitude depends on that initial bias.

# ——— 6. Instantiate the IBMModel ———
model = IBMModel(
    r=r,
    C=C,
    initial_N=initial_N,
    nsteps=nsteps,
    record_step=record_step,
    dispersal_type="propagule",  # “propagule” phase dispersal
    seed=123
)

# ——— 7. Run the IBM simulation ———
trajectory = model.run() 
# trajectory has shape (nrecords, S, Ny, Nx) and records biomass (= N * BODY_MASS) at each record.

# ——— 8. Post‐process: compute mean biomass time series ———
# “mean over patches” for each species at each recorded time
mean_ts = trajectory.mean(axis=(2, 3))  # shape = (nrecords, S)

# Turn record indices into actual “time in steps”
times = np.arange(record_step, nsteps + 1, record_step)  # length = nrecords

# ——— 9. Plot the mean biomass time series for each species ———
plt.figure(figsize=(8, 5))
for s in range(S):
    plt.plot(times, mean_ts[:, s], label=f"Species {s}")
plt.xlabel("Time (IBM steps)")
plt.ylabel("Mean biomass per patch")
plt.title("IBM RPS - half dispersal(D = 0.002) (100% non‐local)")
plt.legend()
plt.tight_layout()
plt.savefig("ibm_rps_nonlocal_timeseries_half.png", dpi=300)
plt.show()

# ——— 10. (Optional) Print total invasions and final diversities ———
n_total_invasions, inv_sp_px, rich_px = count_invasions(trajectory)
print(f"▶ Total invasions (IBM): {n_total_invasions}")
print("▶ Final richness per patch (number of distinct species ever invaded each patch):")
print(rich_px)

# ——— 11. Create a small mosaic of snapshots to show “spatial patterns” (if any) ———
# Choose 6 equally spaced time points out of our records
snapshot_times = np.linspace(times[0], times[-1], 6, dtype=int)
snapshot_idxs = [np.abs(times - tt).argmin() for tt in snapshot_times]
frames = [trajectory[i] for i in snapshot_idxs]  # each is (S, Ny, Nx)

# We need a colour table for S species; simplest: red, green, blue
colour_table = np.array([
    [1.0, 0.0, 0.0],  # species 0: red
    [0.0, 1.0, 0.0],  # species 1: green
    [0.0, 0.0, 1.0],  # species 2: blue
])

# ✓ Pass snapshot_times as numbers—make_mosaic will format with `:g` itself.
make_mosaic(
    frames,
    snapshot_times,                   # these are ints (e.g. [1000, 2000, …])
    colour_table,
    save_to="ibm_rps_nonlocal_mosaic_half.png",
    ncols=3,
    dpi=300
)
# The mosaic will show the spatial distribution of species at those times.
# This will save a 2×3 mosaic (6 frames) under “ibm_rps_nonlocal_mosaic.png”.

print("✔ IBM RPS (100% non‐local) complete. Outputs:")
print("  • ibm_rps_nonlocal_timeseries_half.png")
print("  • ibm_rps_nonlocal_mosaic_half.png")
