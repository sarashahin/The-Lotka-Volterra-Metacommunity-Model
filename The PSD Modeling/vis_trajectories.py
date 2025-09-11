

#!/usr/bin/env python3
import os, numpy as np, matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────
NPZ_HIGH = "/Users/sarashahin/Documents/The PSD ModelingNew/body-mass 1e-4 --inv 1e-10 --S 500 --seeds456/model_outputs.npz"
NPZ_LOW  = "/Users/sarashahin/Documents/The PSD ModelingNew/body-mass 1e-11 --inv 1e-10 --S 500/model_outputs.npz"
OUTDIR   = "/Users/sarashahin/Documents/The PSD ModelingNew/viz"; os.makedirs(OUTDIR, exist_ok=True)

# ── typography: match main text ─────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,        # base
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ── plotting knobs ──────────────────────────────────────────────────────
FIGSIZE       = (14, 10)
GLOBAL_YMIN   = 0.0
GLOBAL_YMAX   = 1.0                     # global y-axis limits
YTICK_COUNT   = 6                               # evenly spaced ticks
SMOOTH_WIN    = 13                              # odd
HIGHLIGHT_N   = 4                             # number of highlighted species per panel
FG_LW, FG_A   = 1.2, 0.9                # foreground line width and alpha
BG_LW, BG_A   = 0.3, 0.70                # background line width and alpha
BG_COLOR      = (0.80, 0.80, 0.80) # light gray
LINE_LW, L_A  = 0.15, 0.40 # default line width and alpha
HIGH_BOLD_LW  = 0.70 # bold line width for high-mass regime
HIGH_BOLD_A   = 0.90 # bold alpha for high-mass regime
HILITE_COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'] 

# ── helpers ─────────────────────────────────────────────────────────────
def load_traj(path):
    d = np.load(path); out = {}
    for k in ("ODE","IBM","PSD2"):
        if k in d: out[k] = np.asarray(d[k])
        elif k == "PSD2" and "PSD" in d: out[k] = np.asarray(d["PSD"])
        else: raise KeyError(f"Missing key {k!r} in {path}")
    return out

def movavg_time_edge_safe(M, w):
    """Centered moving average with edge padding to avoid end dips."""
    M = np.asarray(M)
    if w is None or w <= 1: return M
    if w % 2 == 0: w += 1
    pad = w // 2
    k = np.ones(w, dtype=float) / w
    # reflect or edge-pad along time axis, then 'valid' conv → no zero-pad bias
    Mp = np.pad(M, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, Mp)

def choose_highlights_mixed(rawY, smY, k=4,
                            eps=3e-2, occ_ephem_max=0.08, occ_persist_min=0.80,
                            amp_quantile=0.70):
    occ = (rawY > eps).mean(axis=0)               # occupancy from RAW
    q01, q99 = np.quantile(smY, 0.01, axis=0), np.quantile(smY, 0.99, axis=0)
    amp = q99 - q01                               # amplitude from SMOOTHED
    amp_min = np.quantile(amp, amp_quantile)

    S = smY.shape[1]; k = min(k, S)
    n_ephem, n_pers = k // 2, k - (k // 2)
    ephem_idx = np.where((occ <= occ_ephem_max) & (amp >= amp_min))[0]
    pers_idx  = np.where((occ >= occ_persist_min) & (amp >= amp_min))[0]
    ephem_sorted = ephem_idx[np.argsort(amp[ephem_idx])[::-1]] if ephem_idx.size else np.array([], int)
    pers_sorted  = pers_idx[np.argsort(amp[pers_idx])[::-1]]   if pers_idx.size else np.array([], int)
    keep = np.unique(np.concatenate([ephem_sorted[:n_ephem], pers_sorted[:n_pers]]))
    if keep.size < k:
        rest = np.setdiff1d(np.arange(S), keep)
        keep = np.concatenate([keep, rest[np.argsort(amp[rest])[::-1]][:k-keep.size]])
    bg = np.setdiff1d(np.arange(S), keep)
    return keep, bg

def plot_all_colored(ax, traj, lw=None, alpha=None):
    ax.plot(traj, lw=(lw or LINE_LW), alpha=(alpha or L_A), solid_capstyle="round")

def plot_with_highlight(ax, traj, keep_idx, bg_idx, colors=None):
    if bg_idx.size:
        ax.plot(traj[:, bg_idx], color=BG_COLOR, lw=BG_LW, alpha=BG_A, zorder=1, solid_capstyle="round")
    for j, sp in enumerate(keep_idx):
        kw = {} if colors is None else {"color": colors[j % len(colors)]}
        ax.plot(traj[:, sp], lw=FG_LW, alpha=FG_A, zorder=3, solid_capstyle="round", **kw)

# ── load & pre-smooth (edge-safe) ───────────────────────────────────────
H = load_traj(NPZ_HIGH); L = load_traj(NPZ_LOW)
H_raw, L_raw = {m: np.asarray(H[m]) for m in ("ODE","IBM","PSD2")}, {m: np.asarray(L[m]) for m in ("ODE","IBM","PSD2")}
H_sm , L_sm  = {m: movavg_time_edge_safe(H_raw[m], SMOOTH_WIN) for m in H_raw}, {m: movavg_time_edge_safe(L_raw[m], SMOOTH_WIN) for m in L_raw}

# choose once from LOW-M ODE and reuse across LOW-M panels
keep_low, _ = choose_highlights_mixed(L_raw["ODE"], L_sm["ODE"], k=HIGHLIGHT_N)

# ── figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, sharex=True, sharey=True, constrained_layout=True)
titles     = {"high": r"$M = 10^{-4}$", "low": r"$M = 10^{-11}$"}   # ⑤ no subscript b
model_rows = ("ODE","IBM","PSD2")

for col, regime in enumerate(("high","low")):
    for row, model in enumerate(model_rows):
        ax = axes[row, col]

        if regime == "low":
            traj = L_sm[model]
            S = traj.shape[1]
            bg = np.setdiff1d(np.arange(S), keep_low)
            plot_with_highlight(ax, traj, keep_low, bg, colors=HILITE_COLORS)

        elif regime == "high" and model == "ODE":
            keep, bg = choose_highlights_mixed(H_raw["ODE"], H_sm["ODE"], k=HIGHLIGHT_N)
            plot_with_highlight(ax, H_sm["ODE"], keep, bg)

        elif regime == "high" and model in ("IBM","PSD2"):
            plot_all_colored(ax, H_sm[model], lw=HIGH_BOLD_LW, alpha=HIGH_BOLD_A)

        # axes cosmetics
        ax.set_ylim(GLOBAL_YMIN, GLOBAL_YMAX)
        ax.set_yticks(np.linspace(GLOBAL_YMIN, GLOBAL_YMAX, YTICK_COUNT))  # ③ evenly spaced
        if row == 0: ax.set_title(titles[regime], pad=6)
        if row == 2: ax.set_xlabel("Time")                                 # ① label = Time
        if col == 0: ax.set_ylabel("Biomass")
        ax.text(0.02, 0.92, "PSD" if model=="PSD2" else model, transform=ax.transAxes,
                fontsize=12, fontweight="bold")

out_png = os.path.join(OUTDIR, "trajectories_3x2_prlstyle.png")
fig.savefig(out_png, dpi=300); plt.close(fig)
print("Saved →", out_png)





