# # gpu_patch.py ----------------------------------------------------------
# """
# Import this *before* anything else to replace the global `numpy` module
# with CuPy when USE_GPU=1 is set in the environment.
# Nothing happens on machines without a suitable GPU.
# """
# import os, importlib, sys

# if os.getenv("USE_GPU", "0") == "1":            # opt‑in switch
#     np_gpu = importlib.import_module("cupy")    # CuPy is the drop‑in
#     sys.modules["numpy"] = np_gpu               # hijack the name
#     print("[gpu_patch] ✔ NumPy -> CuPy (GPU) enabled")
# else:
#     print("[gpu_patch] ⏩ running on CPU – set USE_GPU=1 to enable CUDA")

"""
gpu_patch.py

Lightweight backend shim that exposes a consistent API whether CuPy (GPU) is available
or not. Exports:
  - np: cupy (if available) or numpy
  - fft: cupy.fft / cupyx.fft (if available) or scipy.fft / numpy.fft
  - sparse: cupyx.scipy.sparse (if available) or scipy.sparse
  - GPU_AVAILABLE: bool
  - asnumpy(x): convert array to numpy.ndarray
  - to_gpu(x): convert array to cupy array (raises if GPU not available)
  - asarray(x, dtype=None): backend-aware asarray
"""
__all__ = [
    "np",
    "fft",
    "sparse",
    "GPU_AVAILABLE",
    "asnumpy",
    "to_gpu",
    "asarray",
]

# Try to use CuPy + cupyx (GPU) first, fall back to NumPy + SciPy (CPU).
try:
    import cupy as _cp  # type: ignore
    # FFT: try cupyx.fft then cupy.fft
    try:
        from cupyx import fft as _fft_module  # type: ignore
    except Exception:
        try:
            import cupy.fft as _fft_module  # type: ignore
        except Exception:
            _fft_module = None

    # sparse: cupyx.scipy.sparse
    try:
        from cupyx.scipy import sparse as _sparse_module  # type: ignore
    except Exception:
        # Some cupy versions expose sparse under cupyx.scipy.sparse; if not present, set None
        _sparse_module = None

    # If critical submodules are missing, still treat as GPU available but warn on use.
    np = _cp
    fft = _fft_module
    sparse = _sparse_module
    GPU_AVAILABLE = True

    def asnumpy(x):
        """Return a numpy.ndarray from a backend array (moves data to host)."""
        return _cp.asnumpy(x) if hasattr(_cp, "asnumpy") else x.get()

    def to_gpu(x, dtype=None):
        """Move/convert x to a CuPy array. Raises if cupy isn't actually available."""
        return _cp.asarray(x, dtype=dtype) if _cp is not None else (_raise_no_gpu())

    def asarray(x, dtype=None):
        """Create an array on the active backend (CuPy)."""
        return _cp.asarray(x, dtype=dtype)

except Exception:
    # No cupy available — use numpy + scipy
    import numpy as _np  # type: ignore
    np = _np
    GPU_AVAILABLE = False

    # FFT: prefer scipy.fft if available, else numpy.fft
    try:
        from scipy import fft as _fft_module  # type: ignore
    except Exception:
        import numpy.fft as _fft_module  # type: ignore

    # Sparse: require scipy.sparse (used by code expecting sparse module).
    try:
        from scipy import sparse as _sparse_module  # type: ignore
    except Exception as e:
        # Provide a clearer error if SciPy sparse isn't installed.
        raise ImportError(
            "scipy.sparse is required when CuPy is not available. "
            "Install SciPy (e.g. pip install scipy) or enable CuPy."
        ) from e

    fft = _fft_module
    sparse = _sparse_module

    def asnumpy(x):
        """For NumPy backend, arrays are already numpy arrays."""
        return x

    def to_gpu(x, dtype=None):
        """No GPU available: advise user."""
        raise RuntimeError("No GPU/CuPy available on this system. to_gpu() is not supported.")

    def asarray(x, dtype=None):
        """Create a NumPy array."""
        return _np.asarray(x, dtype=dtype)


# Small internal helper used when to_gpu called while cupy import failed
def _raise_no_gpu():
    raise RuntimeError("CuPy/GPU backend is not available on this system.")


# Optional: expose some convenience aliases commonly used in codebases
# (so "from gpu_patch import np, sparse" or "from gpu_patch import np, fft" works).
# The module-level names above satisfy that.
