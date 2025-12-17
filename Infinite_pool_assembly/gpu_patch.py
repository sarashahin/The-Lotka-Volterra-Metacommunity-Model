# gpu_patch.py -----------------------------------------------------------
import os

Generate an error if this file is used.
Use accelerator.py instead

if os.getenv("USE_GPU", "0") == "1":
    import cupy as xp

    # ── pull in the GPU versions explicitly ────────────────────────────
    import cupyx.scipy.fft    as _fft
    import cupyx.scipy.linalg as _linalg
    import cupyx.scipy.sparse as _sparse

    print("🔋  Using CuPy on GPU")

else:                           # fall back to CPU NumPy/SciPy stack
    import numpy  as xp
    import scipy.fft    as _fft
    import scipy.linalg as _linalg
    import scipy.sparse as _sparse

    print("🖥️   Using NumPy/SciPy on CPU")

# ── symbols your code expects ───────────────────────────────────────────
np     = xp
fft    = _fft
linalg = _linalg
sparse = _sparse
