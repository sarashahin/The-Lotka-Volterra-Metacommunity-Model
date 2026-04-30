# utils_vis.py -----------------------------------------------------------
"""
Little helpers that turn a (S,Ny,Nx) biomass / abundance slice
into an RGB image and arrange several images into a mosaic.
"""
from __future__ import annotations
import os; os.environ["MPLBACKEND"] = "Agg" # stop Matplotlib from copying CuPy arrays back & forth
import numpy as np
import matplotlib.pyplot as plt


def slice_to_rgb(state: np.ndarray,
                 colour_table: np.ndarray, top_k=8,
                 empty_colour: tuple[float, float, float] = (0, 0, 0)
                 ) -> np.ndarray:
    """
    Convert one model slice to an RGB image.

    Parameters
    ----------
    state_3d : ndarray, shape (S, Ny, Nx)
    colour_table : ndarray, shape (S, 3), RGB in 0‑1 range
    empty_colour : RGB triple for empty patches

    Returns
    -------
    rgb : ndarray, shape (Ny, Nx, 3)
    """
    S, Ny, Nx = state.shape
    tot   = state.sum(axis=(1,2))
    order = np.argsort(tot)[::-1]          # most abundant first
    keep  = order[:top_k]
    idx   = state.argmax(axis=0)
    occupied = state.sum(axis=0) > 0
    rgb = np.zeros((Ny, Nx, 3), float) + empty_colour
    for rank, s in enumerate(keep):
        rgb[(idx == s) & occupied] = colour_table[rank]   # re‑use palette
    rgb[occupied & ~np.isin(idx, keep)] = (0.5, 0.5, 0.5) # grey
    return rgb


def make_mosaic(frames: list[np.ndarray],
                times: list[float],
                colour_table: np.ndarray,
                save_to: str,
                ncols: int = 4,
                dpi: int = 200) -> None:
    """
    Arrange frames in a grid and save png.

    Parameters
    ----------
    frames : list of (S,Ny,Nx) ndarrays
    times  : label for each frame
    colour_table : colour look‑up
    save_to : filename
    """
    n = len(frames)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.5 * ncols, 2.5 * nrows),
                             squeeze=False)

    # -------- black background ----------------------------------------
    fig.patch.set_facecolor("black")

    # blank axes in case n < nrows*ncols
    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("black")          # <-- axes background

    for k, (snap, tt) in enumerate(zip(frames, times)):
        ax = axes.flat[k]
        img = slice_to_rgb(snap, colour_table)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"t={tt:g}", color="white")
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_to, dpi=dpi)
    plt.close()
    print(f"[save] → {save_to}")
    plt.show()