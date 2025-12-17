# accelerator.py -----------------------------------------------------------
import sys
import os

# CONFIGURATION
# Set to False to force standard CPU usage for debugging
ENABLE_GPU = True

# Initialize state
has_cuda = False
has_mps = False
backend_name = "CPU"

# 1. Detect Hardware
try:
    import torch
    if torch.cuda.is_available():
        has_cuda = True
        backend_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        has_mps = True
        backend_name = "Apple Metal (MPS)"
except ImportError:
    pass

# 2. Define the Shim Logic
if ENABLE_GPU and (has_cuda or has_mps):
    import torch
    
    # A. Set Global Default Device
    if has_cuda:
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("mps")

    # B. Define Helper to map 'axis' (SciPy) -> 'dim' (PyTorch)
    def _map_args(kwargs):
        if 'axis' in kwargs:
            kwargs['dim'] = kwargs.pop('axis')
        return kwargs

    # ── MODULE 1: np (The Array Shim) ──────────────────────────────────
    class RandomShim:
        def seed(self, seed_val):
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)
            print(f"🎲 Seed set to {seed_val} (on {backend_name})")
        def randn(self, *args, **kwargs): return torch.randn(*args, **kwargs)
        def rand(self, *args, **kwargs): return torch.rand(*args, **kwargs)
        def randint(self, low, high=None, size=None, **kwargs):
            if high is None: return torch.randint(0, low, size if size else (), **kwargs)
            return torch.randint(low, high, size if size else (), **kwargs)

    # Patch PyTorch to look like NumPy
    torch.random = RandomShim()
    torch.array = torch.tensor
    torch.concatenate = torch.cat
    torch.power = torch.pow
    torch.abs = torch.abs
    torch.min = torch.min
    torch.max = torch.max
    
    # EXPORT 1: np
    np = torch

    # ── MODULE 2: fft (The FFT Shim) ───────────────────────────────────
    class FFTShim:
        def fft(self, x, n=None, **kwargs):
            return torch.fft.fft(x, n=n, **_map_args(kwargs))
        def ifft(self, x, n=None, **kwargs):
            return torch.fft.ifft(x, n=n, **_map_args(kwargs))
        def fft2(self, x, s=None, **kwargs):
            return torch.fft.fft2(x, s=s, **_map_args(kwargs))
        def ifft2(self, x, s=None, **kwargs):
            return torch.fft.ifft2(x, s=s, **_map_args(kwargs))
        def rfft(self, x, n=None, **kwargs):
            return torch.fft.rfft(x, n=n, **_map_args(kwargs))
        def irfft(self, x, n=None, **kwargs):
            return torch.fft.irfft(x, n=n, **_map_args(kwargs))
        def fftfreq(self, n, d=1.0):
            # fftfreq is often needed on CPU for plotting, but PyTorch has it too
            return torch.fft.fftfreq(n, d=d)
            
    # EXPORT 2: fft
    fft = FFTShim()

    # ── MODULE 3: linalg (The Linear Algebra Shim) ─────────────────────
    class LinalgShim:
        def inv(self, x): return torch.linalg.inv(x)
        def eig(self, x): return torch.linalg.eig(x)
        def eigh(self, x): return torch.linalg.eigh(x)
        def solve(self, a, b): return torch.linalg.solve(a, b)
        def norm(self, x, **kwargs): return torch.linalg.norm(x, **_map_args(kwargs))
        def det(self, x): return torch.linalg.det(x)
        def svd(self, x, **kwargs): return torch.linalg.svd(x, **kwargs)
        
    # EXPORT 3: linalg
    linalg = LinalgShim()

    # ── MODULE 4: sparse (The Tricky One) ──────────────────────────────
    # WARNING: PyTorch Sparse != SciPy Sparse. 
    # Attempting to force PyTorch sparse tensors into legacy SciPy code 
    # usually crashes. 
    # STRATEGY: Fallback to CPU SciPy for sparse operations to ensure safety.
    import scipy.sparse as _scipy_sparse
    sparse = _scipy_sparse
    
    print(f"🔋  Using PyTorch Wrapper on {backend_name}")
    print(f"    (Note: 'sparse' module remains on CPU for compatibility)")

else:
    # ── FALLBACK: Standard CPU Stack ───────────────────────────────────
    import numpy
    import scipy.fft
    import scipy.linalg
    import scipy.sparse

    # EXPORT SYMBOLS
    np = numpy
    fft = scipy.fft
    linalg = scipy.linalg
    sparse = scipy.sparse

    print("🖥️   Using NumPy/SciPy on CPU")

# ── HELPER: To CPU ─────────────────────────────────────────────────────
# Use this when you need to plot or save data
def to_cpu(data):
    if hasattr(data, "cpu"):
        return data.detach().cpu().numpy()
    return data
