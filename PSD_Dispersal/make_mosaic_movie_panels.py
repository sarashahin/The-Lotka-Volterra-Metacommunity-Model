#!/usr/bin/env python3
# make_mosaic_movie_panels.py
#
# Movie-like mosaic panels with a clear colour legend.
# - Stable colour mapping across all frames (global top-K species)
# - Legend that explains colour ↔ species ID
# - Explicit or auto-selected times
#
# Expected keys (robust to variants):
#   IBM:  ("IBM_B", "IBM_t") or ("IBM", "t")
#   PSD2: ("PSD2_B", "PSD2_t") or ("PSD_B","PSD_t") or ("PSD","t")
#   ODE:  ("ODE_B", "ODE_t") or ("ODE","t")
#
# Output: a single PNG panel with titles "t=..." and a legend.

import os; os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, numpy as np, matplotlib.pyplot as plt

# ------------------------- loading helpers ------------------------------

def _get(arrdict, *names):
    for n in names:
        if n in arrdict: return arrdict[n]
    raise KeyError(f"None of {names} found in NPZ")

def load_model(npz_path: str, model: str):
    d = np.load(npz_path)
    model = model.upper()
    if model == "IBM":
        B = _get(d, "IBM_B", "IBM")
        t = _get(d, "IBM_t", "t")
    elif model in ("PSD2"):
        B = _get(d, "PSD2_B", "PSD2")
        t = _get(d, "PSD2_t", "t")
    elif model == "ODE":
        B = _get(d, "ODE_B", "ODE")
        t = _get(d, "ODE_t", "t")
    else:
        raise SystemExit(f"Unknown model {model}")
    B = np.asarray(B)     # shape (T,S,Ny,Nx) or (T, S, ...) assumed
    t = np.asarray(t).ravel()
    if B.ndim != 4:
        raise SystemExit(f"B must be (T,S,Ny,Nx), got {B.shape}")
    if B.shape[0] != t.shape[0]:
        raise SystemExit(f"T mismatch: B has {B.shape[0]} frames, t has {t.shape[0]}")
    return B, t

# ---------------------- stable palette / legend -------------------------

def pick_global_topk(B: np.ndarray, top_k: int):
    # B: (T,S,Ny,Nx). Sum over time and space for a stable global ranking
    totals = B.sum(axis=(0,2,3))          # (S,)
    order  = np.argsort(totals)[::-1]
    keep   = order[:min(top_k, B.shape[1])]
    keep   = np.sort(keep)  # keep in ascending order
    return keep, totals

def default_palette(K: int):
    # Bright, distinct RGB in [0,1]; repeats if K>len(list)
    base = np.array([
        [0.88, 0.15, 0.07],  # red
        [0.12, 0.47, 0.71],  # blue
        [0.17, 0.63, 0.17],  # green
        [0.84, 0.37, 0.00],  # orange
        [0.58, 0.40, 0.74],  # purple
        [0.55, 0.34, 0.29],  # brown
        [0.89, 0.47, 0.76],  # pink
        [0.50, 0.50, 0.50],  # (will be overridden by grey for "Other")
        [0.74, 0.74, 0.13],  # olive
        [0.09, 0.75, 0.81],  # cyan
    ])
    if K <= len(base): return base[:K]
    reps = int(np.ceil(K/len(base)))
    return np.vstack([base for _ in range(reps)])[:K]

def slice_to_rgb_fixedpalette(state: np.ndarray,
                              keep_ids: np.ndarray,
                              colour_table: np.ndarray,
                              empty_colour=(0,0,0),
                              other_colour=(0.5,0.5,0.5)) -> np.ndarray:
    # state: (S,Ny,Nx); keep_ids: species indices of global top-K
    S, Ny, Nx = state.shape
    idx_dom   = state.argmax(axis=0)              # dominant species per pixel
    occ       = state.sum(axis=0) > 0             # occupied pixels
    img       = np.zeros((Ny, Nx, 3), float) + empty_colour
    # map species ID -> palette slot (only for kept species)
    inv = {int(s): i for i, s in enumerate(keep_ids)}
    for s in keep_ids:
        mask = (idx_dom == s) & occ
        img[mask] = colour_table[inv[int(s)]]
    # other species (not in top-K)
    other_mask = occ & ~np.isin(idx_dom, keep_ids)
    img[other_mask] = other_colour
    return img

def add_legend(fig, keep_ids, colours, title="Colour coding"):
    # legend boxes explaining colour ↔ species
    handles = []
    for s, c in zip(keep_ids, colours):
        patch = plt.Line2D([0],[0], marker='s', linestyle='',
                           markersize=10, markerfacecolor=c, markeredgecolor='k')
        handles.append(patch)
    labels = [f"sp {int(s)}" for s in keep_ids]
    # add “Other” and “Empty”
    handles += [plt.Line2D([0],[0], marker='s', linestyle='', markersize=10,
                           markerfacecolor=(0.5,0.5,0.5), markeredgecolor='k'),
                plt.Line2D([0],[0], marker='s', linestyle='', markersize=10,
                           markerfacecolor=(0,0,0), markeredgecolor='k')]
    labels  += ["Other species", "Empty"]
    leg = fig.legend(handles, labels, loc="lower center",
                     ncol=min(6, len(labels)), frameon=True,
                     facecolor="white", edgecolor="black", title=title)
    leg.get_title().set_fontsize(10)

# -------------------------- time selection ------------------------------

def indices_for_times(t, wanted):
    wanted = np.asarray(wanted, float)
    idx = [int(np.argmin(np.abs(t - w))) for w in wanted]
    # enforce strict increase & uniqueness
    uniq = []
    last = -1
    for i in idx:
        if i != last:
            uniq.append(i); last = i
    return uniq

def auto_pick_times(B, t, nframes=8, target_change=0.2):
    """
    Greedy pick frames so successive frames change by ~target_change
    where 'change' = fraction of patches that switch dominant species.
    """
    T, S, Ny, Nx = B.shape
    # start from t=0 (or earliest)
    chosen = [0]
    dom = B.argmax(axis=1)  # (T, Ny, Nx)
    while len(chosen) < nframes:
        last = chosen[-1]
        best_j, best_diff = None, 1e9
        for j in range(last+1, T):
            diff = (dom[last] != dom[j]).mean()
            score = abs(diff - target_change)
            if score < best_diff:
                best_diff, best_j = score, j
        if best_j is None or best_j == last:
            break
        chosen.append(best_j)
    return chosen

# ------------------------------ plotting -------------------------------

def make_panel(B, t, keep_ids, colours, idx_list, out_path,
               ncols=4, dpi=200, suptitle=None):
    frames = B[idx_list]            # (n, S, Ny, Nx)
    times  = t[idx_list]
    n = len(idx_list)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6*ncols, 2.6*nrows),
                             squeeze=False)
    fig.patch.set_facecolor("black")
    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("black")

    for k, (snap, tt) in enumerate(zip(frames, times)):
        ax  = axes.flat[k]
        img = slice_to_rgb_fixedpalette(snap, keep_ids, colours)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"t={tt:g}", color="white", fontsize=11, pad=4)

    if suptitle:
        fig.suptitle(suptitle, color="white", fontsize=12, y=0.99)

    # legend at bottom on white background (readable on black panel)
    add_legend(fig, keep_ids, colours, title="Dominant species per patch")

    plt.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[save] → {out_path}")

# ------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to rps_dataset.npz")
    ap.add_argument("--model", required=True, choices=["IBM","PSD2","ODE"])
    ap.add_argument("--times", type=str, default=None,
                    help="comma-separated times, e.g. 0,25000,50000,...")
    ap.add_argument("--auto-times", type=int, default=None,
                    help="pick N frames automatically with gentle changes")
    ap.add_argument("--target-change", type=float, default=0.2,
                    help="fraction of pixels changing dominance between frames (auto)")
    ap.add_argument("--top-k", type=int, default=8, help="number of species shown distinctly")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    B, t = load_model(args.npz, args.model)
    keep_ids, _ = pick_global_topk(B, args.top_k)
    colours = default_palette(len(keep_ids))

    if args.times:
        wanted = [float(x) for x in args.times.split(",")]
        idx = indices_for_times(t, wanted)
    elif args.auto_times:
        idx = auto_pick_times(B, t, nframes=int(args.auto_times),
                              target_change=float(args.target_change))
    else:
        # fallback: first 8 frames
        idx = list(range(min(8, B.shape[0])))

    disp_name = "PSD" if args.model.upper() in {"PSD2"} else args.model.upper()
    title = f"{disp_name}: dominant-species mosaics (stable colours)"
    make_panel(B, t, keep_ids, colours, idx, args.out,
               ncols=args.ncols, suptitle=title)

if __name__ == "__main__":
    main()
