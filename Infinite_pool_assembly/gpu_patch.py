# # gpu_patch.py ----------------------------------------------------------
# """
# Import this *before* anything else to replace the global `numpy` module
# with CuPy when USE_GPU=1 is set in the environment.
# Nothing happens on machines without a suitable GPU.
# """
# import os, importlib, sys
# import numpy as np 
# from  scipy.fft import rfft, rfftfreq

# if os.getenv("USE_GPU", "0") == "1":            # opt‑in switch
#     np_gpu = importlib.import_module("cupy")    # CuPy is the drop‑in
#     sys.modules["numpy"] = np_gpu               # hijack the name
#     print("[gpu_patch] ✔ NumPy -> CuPy (GPU) enabled")
# else:
#     print("[gpu_patch] ⏩ running on CPU – set USE_GPU=1 to enable CUDA")


# gpu_patch.py -----------------------------------------------------------
import os

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




# # gpu_patch.py  ───────────────────────────────────────────────
# """
# Import this *first* (e.g. in run_all_rps.py) to hijack the
# global NumPy/SciPy names so every later

#     import numpy as np
#     import scipy.linalg

# actually gives you CuPy / CuPy‑SciPy on the GPU.
# """
# # gpu_patch.py  --------------------------------------------------------
# import os, sys
# import numpy   as _np_real
# import cupy    as _cp
# import cupyx.scipy          as _cpx
# import cupyx.scipy.sparse as _cpx_sparse
# # import cupyx.scipy.fft, cupyx.scipy.linalg, cupyx.scipy.sparse

# # -- graft missing NumPy sub‑modules/types onto CuPy --------------------
# _cp.ma          = _np_real.ma          # masked arrays   (already present)
# _cp.char        = _np_real.char        # string ops
# _cp.datetime64  = _np_real.datetime64  # <<< NEW
# _cp.timedelta64 = _np_real.timedelta64 # <<< NEW
# # -----------------------------------------------------------------------

# sys.modules["numpy"] = _cp
# sys.modules["scipy"] = _cpx
# print("🔋  NumPy / SciPy are now CuPy (GPU)")

# # ------------- re‑export for `from gpu_patch import np, fft …` ---------
# # AFTER ──────────────────────────────────────────
# np     = _cp
# fft    = _cpx.fft                        # OK (exists)
# linalg = _cp                      # <-- use CuPy’s own linalg
# sparse = _cpx_sparse                    # OK
# __all__ = ["np", "fft", "linalg", "sparse"]
# print("🔋  NumPy is now CuPy (GPU) – SciPy left untouched")

# # allow implicit GPU → CPU copy when some lib calls np.asarray(obj)
# _cp.ndarray.__array__ = lambda self, dtype=None: _np_real.asarray(self.get(), dtype)

# # ── accept python‑lists in csr_matrix *without* recursion ─────────────
# _orig_csr = sparse.csr_matrix        # ← save once

# # ----------------------------------------------------------------------NEW
# from cupyx.scipy.sparse._csr import csr_matrix as _csr_class   # low‑level

# def _csr_safe(arg, *a, **kw):
#     """
#     Intercept *nested* Python lists (e.g. [[1, 0], [0, 2]]).  
#     Everything else is forwarded directly to CuPy’s real constructor.
#     """
#     if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (list, tuple)):
#         # ① build a tiny NumPy array on the CPU
#         _dense = _np_real.asarray(arg, dtype=_np_real.float64)
#         # ② create a CSR matrix on the GPU *without* cusparse.denseToSparse
#         rr, cc = _np_real.nonzero(_dense)              # CPU indices
#         data   = _dense[rr, cc]
#         coo    = _cpx_sparse.coo_matrix(               # GPU COO
#                     (_cp.asarray(data), (_cp.asarray(rr), _cp.asarray(cc))),
#                     shape=_dense.shape)
#         return coo.tocsr()                             # GPU CSR – done
#     # delegate – bypass our wrapper entirely
#     return _csr_class(arg, *a, **kw)

# sparse.csr_matrix = _csr_safe              # monkey‑patch once
# # ----------------------------------------------------------------------

# # ----------------------------------------------------------------------

# # gpu_patch.py  (after the np / fft / linalg exports)
# # --------------------------------------------------
# def _to_numpy(self, dtype=None):
#     """Allow implicit GPU -> CPU copy when a library calls np.asarray(obj)."""
#     import numpy as _np
#     return _np.asarray(self.get(), dtype=dtype)

# _cp.ndarray.__array__ = _to_numpy        # <─ PATCH









