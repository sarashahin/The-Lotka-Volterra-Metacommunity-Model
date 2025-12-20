import numpy as np

def random_colour_table(S: int, seed: int = 123) -> np.array:
    """Return an (S,3) array of RGB values (0‑1)."""
    rng = np.random.default_rng(seed)
    return rng.random((S, 3))
