############################################
# assembly_utils.py
############################################

from accelerator import np
from config import CONNECTANCE, INTERACTION_STRENGTH
from typing import Tuple  # Add this import

def draw_interactions(k: int, *, rng=np.random):
    """
    Return (row, col) – the interactions between ONE *new* species
    and the current γ=k residents.

       row[j] = effect of resident j  → new
       col[j] = effect of new        → resident j
    Non–zero entries occur with probability CONNECTANCE and have the
    *constant* magnitude INTERACTION_STRENGTH (Axel’s default).
    """
    mask = rng.random(k) < CONNECTANCE
    new_row = np.zeros(k)
    new_row[mask] = INTERACTION_STRENGTH         # constant value

    mask = rng.random(k) < CONNECTANCE
    new_col = np.zeros(k)
    new_col[mask] = INTERACTION_STRENGTH         # constant value
    return new_row, new_col


def expand_RC(r: np.ndarray, C: np.ndarray, r_new: float,
              row: np.ndarray, col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Append one species to (r,C) and keep C_ii = 1."""
    r_out = np.append(r, r_new)
    k = len(r)
    C_out = np.empty((k+1, k+1), float)
    C_out[:k, :k] = C
    C_out[k, :k]  = row          # new last row  (j → new)
    C_out[:k, k]  = col          # new last col  (new → j)
    C_out[k, k]   = 1.0
    return r_out, C_out

def prune_extinct(mask_alive: np.ndarray,
                  r: np.ndarray, C: np.ndarray, *states):
    """Remove species where mask_alive == False, returns pruned copies."""
    keep = np.where(mask_alive)[0]
    r2   = r[keep]
    C2   = C[np.ix_(keep, keep)]
    sts  = [s[keep] if s.shape[0]==len(mask_alive) else s for s in states]
    return (r2, C2, *sts, keep)
