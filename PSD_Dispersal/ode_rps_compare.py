# ode_rps_compare.py

import numpy as np
import matplotlib.pyplot as plt

# ——— 1. Force 100% non‐local dispersal ———
import config
config.LONG_DISTANCE_PROB = 1.0

# ——— 2. Import ODEModel ———
from models_ode import ODEModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS
from utils_vis import make_mosaic    # for a quick spatial mosaic

# ——— 3. 3‐species RPS parameters & initial biomass ———
S = 3
r = np.ones(S)
a, b = 1.7, 0.4
C = np.array([
    [1.0, a,   b],
    [b,   1.0, a],
    [a,   b,   1.0]
], dtype=float)

Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X

# Same initial bias: base 2*BODY_MASS + small offset on species 0
initial_B = np.zeros((S, Ny, Nx))
initial_B[:] = 2 * BODY_MASS
rng = np.random.default_rng(42)
flat_idx = np.arange(Ny * Nx)
chosen = rng.choice(flat_idx, size=(Ny*Nx)//2, replace=False)
initial_B[0].flat[chosen] += BODY_MASS

# ——— 4. Simulation parameters ———
tmax = 2000.0
record_step = 10.0

# ——— 5. Instantiate and run ODEModel ———
model = ODEModel(
    r=r,
    C=C,
    initial_B=initial_B,
    tmax=tmax,
    record_step=record_step,
    dispersal_type="propagule",
    seed=123
)
times, traj = model.run()
# traj shape = (nrecords, S, Ny, Nx)

# ——— 6. Compute mean biomass time series ———
mean_ts = traj.mean(axis=(2, 3))  # (nrecords, S)

# ——— 7. Plot ODE time series ———
plt.figure(figsize=(8, 5))
for s in range(S):
    plt.plot(times, mean_ts[:, s], label=f"Species {s}")
plt.xlabel("Time (ODE units)")
plt.ylabel("Mean biomass per patch")
plt.title("ODE RPS - full dispersal(D = 5.000e-07) (100% non‐local)")
plt.legend()
plt.tight_layout()
plt.savefig("ode_rps_fixed_timeseries_full.png", dpi=300)
plt.show()

# ——— 8. (Optional) mosaic of snapshots ———
# Take 6 evenly spaced times from `times` (these are floats)
snapshot_times = np.linspace(times[0], times[-1], 6)
snapshot_idxs  = [np.abs(times - tt).argmin() for tt in snapshot_times]
frames         = [traj[i] for i in snapshot_idxs]  # each is (S, Ny, Nx)

colour_table = np.array([
    [1.0, 0.0, 0.0],  # species 0: red
    [0.0, 1.0, 0.0],  # species 1: green
    [0.0, 0.0, 1.0],  # species 2: blue
])

make_mosaic(
    frames,
    snapshot_times,               # pass floats, not strings
    colour_table,
    save_to="ode_rps_fixed_mosaic_full.png",
    ncols=3,
    dpi=300
)

print("✔ ODE RPS (100% non‐local) complete. Outputs:")
print("  • ode_rps_fixed_timeseries_full.png")
print("  • ode_rps_fixed_mosaic_full.png")
