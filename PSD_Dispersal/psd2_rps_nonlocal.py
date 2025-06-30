# psd2_rps_nonlocal.py

import numpy as np
import matplotlib.pyplot as plt

# ——— 1. Force 100% non‐local dispersal ———
import config
config.LONG_DISTANCE_PROB = 1.0

# ——— 2. Import the PSD2Model ———
from models_psd2 import PSD2Model
from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS
from utils_analysis import count_invasions
from utils_vis import make_mosaic

# ——— 3. 3‐species RPS parameters ———
S = 3
r = np.ones(S)
a, b = 1.7, 0.4
C = np.array([
    [1.0, a,   b],
    [b,   1.0, a],
    [a,   b,   1.0]
], dtype=float)

# ——— 4. Grid and simulation parameters ———
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X
window_time = 20000.0    # total “time” for the PSD2 run
record_step = 1.0      # record every 10 time units
nrecords = int(window_time // record_step) + 1

# ——— 5. Build a similar “small offset” initial biomass ———
# Base biomass = BODY_MASS * 2 everywhere for each species
initial_B = np.zeros((S, Ny, Nx), dtype=float)
initial_B[:] = 2 * BODY_MASS

# Add +1*BODY_MASS to species 0 on half the patches
rng = np.random.default_rng(42)
flat_idx = np.arange(Ny * Nx)
chosen = rng.choice(flat_idx, size=(Ny*Nx)//2, replace=False)
initial_B[0].flat[chosen] += BODY_MASS

# ——— 6. Instantiate PSD2Model ———
model = PSD2Model(
    r=r,
    C=C,
    initial_B=initial_B,
    tmax=window_time,
    record_step=record_step,
    dispersal_type="propagule",
    seed=123
)

# ——— 7. Run the PSD2 simulation ———
t, traj, w_traj, pc_traj, g_traj, inv_traj, est_prob_traj = model.run()
# traj shape = (nrecords, S, Ny, Nx)

# ——— 8. Compute mean biomass time series ———
mean_ts = traj.mean(axis=(2, 3))  # shape (nrecords, S)

# ——— 9. Plot time series ———
plt.figure(figsize=(8, 5))
for s in range(S):
    plt.plot(t, mean_ts[:, s], label=f"Species {s}")
plt.xlabel("Time (PSD units)")
plt.ylabel("Mean biomass per patch")
plt.title("PSD (100% non‐local dispersal): 3‐species RPS\nInitial bias sets amplitude of slow oscillations")
plt.legend()
plt.tight_layout()
plt.savefig("psd_rps_nonlocal_timeseries_full.png", dpi=300)
plt.show()

# ——— 10. Invasion stats (optional) ———
n_total_inv, inv_sp_px, rich_px = count_invasions(traj)
print(f"▶ Total invasions (PSD): {n_total_inv}")

# ——— 11. Mosaic of snapshots (6 frames) ———
snapshot_times = np.linspace(t[0], t[-1], 6)
snapshot_idxs = [np.abs(t - tt).argmin() for tt in snapshot_times]
frames = [traj[i] for i in snapshot_idxs]

colour_table = np.array([
    [1.0, 0.0, 0.0],  # red
    [0.0, 1.0, 0.0],  # green
    [0.0, 0.0, 1.0],  # blue
])

make_mosaic(
    frames,
    snapshot_times,               # pass floats, not strings
    colour_table,
    save_to="psd_rps_fixed_mosaic.png",
    ncols=3,
    dpi=300
)

print("✔ PSD RPS (100% non‐local) complete. Outputs:")
print("  • psd_rps_nonlocal_timeseries.png")
print("  • psd_rps_nonlocal_mosaic.png")
