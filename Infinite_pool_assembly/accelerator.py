############################################
# accelerator.py
############################################
import os
import sys

# <FIX> Define locally to avoid circular import with config.py
ENABLE_GPU = False
# </FIX>

has_cuda = False
has_mps = False
backend_name = "CPU"

try:
    import torch
    if torch.cuda.is_available():
        has_cuda = True
        backend_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        has_mps = True
        backend_name = "Apple Metal (MPS)"
except ImportError:
    pass

if ENABLE_GPU and (has_cuda or has_mps):
    import torch
    import torch.nn.functional as F
    import numpy as _real_numpy 

    device = "cuda" if has_cuda else "mps"
    torch.set_default_device(device)

    def _as_tensor(x, dtype=None):
        if isinstance(x, _real_numpy.ndarray):
            return torch.from_numpy(x).to(device)
        if not isinstance(x, torch.Tensor):
            return torch.as_tensor(x, device=device)
        return x

    def _tensor_astype(self, dtype):
        if dtype is int: return self.to(torch.int64)
        if dtype is float: return self.to(torch.float32)
        if dtype is bool: return self.to(torch.bool)
        return self.to(dtype)

    def _tensor_copy(self): return self.clone()
        
    def _tensor_array(self, *args, **kwargs):
        dtype = kwargs.get('dtype', None)
        if dtype: 
            return self.detach().cpu().numpy().astype(dtype)
        return self.detach().cpu().numpy()

    @property
    def _tensor_flat(self): return self.view(-1)

    torch.Tensor.astype = _tensor_astype
    torch.Tensor.copy = _tensor_copy
    torch.Tensor.__array__ = _tensor_array 
    torch.Tensor.flat = _tensor_flat

    def _map_args(kwargs):
        if 'axis' in kwargs: kwargs['dim'] = kwargs.pop('axis')
        return kwargs

    class RandomShim:
        def seed(self, seed_val):
            torch.manual_seed(seed_val)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_val)
            print(f"🎲 Seed set to {seed_val} (on {backend_name})")

        def default_rng(self, seed=None):
            if seed is not None: self.seed(seed)
            return self

        def random(self, size=None, **kwargs):
            return torch.rand(size if size is not None else (), **kwargs)
        
        def rand(self, *args, **kwargs): return torch.rand(*args, **kwargs)
        def randn(self, *args, **kwargs): return torch.randn(*args, **kwargs)

        def randint(self, low, high=None, size=None, **kwargs):
            if high is None: return torch.randint(0, low, size if size else (), **kwargs)
            return torch.randint(low, high, size if size else (), **kwargs)
        
        def integers(self, low, high=None, size=None, **kwargs):
            return self.randint(low, high, size, **kwargs)
            
        def normal(self, loc=0.0, scale=1.0, size=None):
            if size is None: size = ()
            return torch.normal(mean=float(loc), std=float(scale), size=size)

        def poisson(self, lam, size=None):
            """
            Branchless Poisson for MPS/M4.
            Computes BOTH Normal Approx and Exact Inverse Transform for ALL elements,
            then selects the valid one on GPU. Wastes FLOPs to save Latency.
            """
            lam_t = _as_tensor(lam).float()
            if size is not None: lam_t = lam_t.expand(size)
            
            # 1. Normal Approximation (Always Valid, but inaccurate for small lam)
            # We use abs() to prevent NaN if lam < 0 during transient states
            sigma = torch.sqrt(torch.abs(lam_t)) 
            res_normal = torch.normal(lam_t, sigma).round().long()
            res_normal = torch.clamp(res_normal, min=0)

            # 2. Exact Inverse Transform Sampling (Vectorized)
            # We use a fixed expansion of K=128. This covers Lambda up to ~80 safely.
            # Our threshold is 30, so this is plenty.
            K_MAX = 128
            
            # (N, 1)
            lam_flat = lam_t.unsqueeze(-1) 
            # (1, 128)
            k = torch.arange(K_MAX, device=lam_t.device, dtype=torch.float32).unsqueeze(0)
            
            # Log PMF: k*ln(lam) - lam - ln(k!)
            # Add epsilon to lam to avoid log(0)
            log_pmf = (k * torch.log(lam_flat + 1e-30)) - lam_flat - torch.lgamma(k + 1.0)
            cdf = torch.cumsum(torch.exp(log_pmf), dim=-1)
            
            # Draw uniforms: (N, 1)
            u = torch.rand(lam_flat.shape, device=lam_t.device)
            # Count how many steps in CDF we passed
            res_exact = (cdf < u).sum(dim=-1).long()
            
            # 3. Branchless Selection
            # If Lambda > 30, use Normal. Else use Exact.
            # No CPU Sync.
            return torch.where(lam_t > 30.0, res_normal, res_exact)

        def binomial(self, n, p, size=None):
            """
            Branchless Binomial for MPS/M4.
            Computes Normal, Poisson, and Exact branches for ALL elements.
            """
            n_in = _as_tensor(n)
            p_in = _as_tensor(p)
            
            if size is not None:
                n_in = n_in.expand(size)
                p_in = p_in.expand(size)

            n_float = n_in.float()
            p_float = p_in.float()
            
            # Thresholds
            N_CUTOFF = 30.0
            LAM_CUTOFF = 30.0
            lam = n_float * p_float
            
            # --- Branch 1: Normal Approx (Large N, Large P) ---
            sigma = torch.sqrt(lam * (1.0 - p_float) + 1e-6)
            res_normal = torch.normal(lam, sigma).round().long()
            res_normal = torch.clamp(res_normal, min=0)
            res_normal = torch.min(res_normal, n_in) # Cap at N

            # --- Branch 2: Poisson Approx (Large N, Small P) ---
            # Reuse our branchless poisson from above
            res_poisson = self.poisson(lam)
            res_poisson = torch.min(res_poisson, n_in)

            # --- Branch 3: Exact Bernoulli (Small N) ---
            # We simulate 30 coin flips for EVERYONE.
            # If N > 30, this result is garbage (capped at 30), but masked out later.
            MAX_FLIPS = 30
            
            # Expand to (..., 30)
            # u: Random numbers [0,1]
            u_flips = torch.rand(n_float.shape + (MAX_FLIPS,), device=n_float.device)
            
            # Trial indices: (1, 30)
            idx = torch.arange(MAX_FLIPS, device=n_float.device).unsqueeze(0)
            
            # Active trials mask: only count flips where trial_index < n
            # (..., 1)
            n_broad = n_float.unsqueeze(-1)
            p_broad = p_float.unsqueeze(-1)
            
            active_mask = idx < n_broad
            success_mask = (u_flips < p_broad) & active_mask
            res_exact = success_mask.sum(dim=-1).long()

            # --- Branchless Selection ---
            # Condition 1: Use Exact if N <= 30
            # Condition 2: Use Poisson if N > 30 AND Lam <= 30
            # Condition 3: Otherwise Normal
            
            is_small_n = n_float <= N_CUTOFF
            is_rare    = (n_float > N_CUTOFF) & (lam <= LAM_CUTOFF)
            
            # Nested Where: 
            # If Small N -> Exact
            # Else If Rare -> Poisson
            # Else -> Normal
            final = torch.where(
                is_small_n, 
                res_exact, 
                torch.where(is_rare, res_poisson, res_normal)
            )
            return final.long()
        
        def choice(self, a, size=None, replace=True, p=None):
            if isinstance(a, int):
                if replace: return torch.randint(0, a, size if size else ())
                else: return torch.randperm(a)[:size] if size is not None else torch.randperm(a)[0]
            else:
                a_t = _as_tensor(a)
                if replace:
                    idx = torch.randint(0, len(a_t), size if size else ())
                    return a_t[idx]
                else:
                    return a_t[torch.randperm(len(a_t))[:size]] if size is not None else a_t[torch.randperm(len(a_t))[0]]

    class R_Shim:
        def __getitem__(self, item):
            if not isinstance(item, tuple): item = (item,)
            tensors = [_as_tensor(x) for x in item]
            return torch.cat(tensors, dim=0)

    class NumpyShim:
        def __init__(self):
            self.random = RandomShim()
            self.float32 = torch.float32
            self.float64 = torch.float32
            self.float16 = torch.float16
            self.int64 = torch.int64
            self.int32 = torch.int32
            self.int16 = torch.int16
            self.int8  = torch.int8
            self.uint8 = torch.uint8
            self.bool = torch.bool
            self.pi = torch.pi
            self.nan = float('nan')
            self.inf = float('inf')
            self.newaxis = None
            self.r_ = R_Shim()
            self.ndarray = torch.Tensor

        def _sanitize_dtype(self, dtype):
            if dtype is None: return None
            if dtype is float or dtype == 'float' or dtype == _real_numpy.float64:
                return torch.float32
            if dtype is int or dtype == 'int' or dtype == _real_numpy.int64:
                return torch.int64
            if dtype is bool or dtype == 'bool':
                return torch.bool
            return dtype

        def _to_numpy_dtype(self, dtype):
            if dtype == torch.float32 or dtype == torch.float64: return _real_numpy.float64
            if dtype == torch.float16: return _real_numpy.float16
            if dtype == torch.int64: return _real_numpy.int64
            if dtype == torch.int32: return _real_numpy.int32
            if dtype == torch.int16: return _real_numpy.int16
            if dtype == torch.bool: return bool
            return dtype

        def _parse_creation_args(self, args, kwargs):
            if len(args) > 1:
                potential_dtype = args[1]
                if isinstance(potential_dtype, (type, torch.dtype, str)):
                    kwargs['dtype'] = self._sanitize_dtype(potential_dtype)
                    args = (args[0],) + args[2:]
            if 'dtype' in kwargs:
                kwargs['dtype'] = self._sanitize_dtype(kwargs['dtype'])
            return args, kwargs
            
        def array(self, data, dtype=None, **kwargs):
            dtype = self._sanitize_dtype(dtype)
            if isinstance(data, torch.Tensor): return data.to(dtype) if dtype else data
            return torch.tensor(data, dtype=dtype, **kwargs)
            
        def asarray(self, data, dtype=None): return self.array(data, dtype=dtype)
        def arange(self, *args, **kwargs): return torch.arange(*args, **kwargs)
        def linspace(self, *args, **kwargs): return torch.linspace(*args, **kwargs)
        
        def zeros(self, *args, **kwargs):
            args, kwargs = self._parse_creation_args(args, kwargs)
            return torch.zeros(*args, **kwargs)
        def ones(self, *args, **kwargs):
            args, kwargs = self._parse_creation_args(args, kwargs)
            return torch.ones(*args, **kwargs)
        
        def zeros_like(self, a, dtype=None, **kwargs):
            dtype = self._sanitize_dtype(dtype)
            return torch.zeros_like(_as_tensor(a), dtype=dtype, **kwargs)
            
        def ones_like(self, a, dtype=None, **kwargs):
            dtype = self._sanitize_dtype(dtype)
            return torch.ones_like(_as_tensor(a), dtype=dtype, **kwargs)

        def empty(self, *args, **kwargs):
            args, kwargs = self._parse_creation_args(args, kwargs)
            return torch.zeros(*args, **kwargs)
        def eye(self, *args, **kwargs): return torch.eye(*args, **kwargs)
        def full(self, shape, fill_value, **kwargs): return torch.full(shape, fill_value, **kwargs)
        def full_like(self, input, fill_value, **kwargs): return torch.full_like(input, fill_value, **kwargs)
        
        def rint(self, x): return torch.round(_as_tensor(x))
        def pad(self, array, pad_width, mode='constant', constant_values=0):
            array = _as_tensor(array)
            torch_pad = []
            for (before, after) in reversed(pad_width): torch_pad.extend([before, after])
            return F.pad(array, torch_pad, mode=mode, value=constant_values)
        def broadcast_to(self, array, shape): return _as_tensor(array).expand(shape)
        def atleast_1d(self, *arys):
            res = []
            for ary in arys:
                t = _as_tensor(ary)
                if t.ndim == 0: t = t.view(1)
                res.append(t)
            if len(res) == 1: return res[0]
            return tuple(res)
            
        def clip(self, input, min=None, max=None):
            min_val = min if min is not None else -float('inf')
            max_val = max if max is not None else float('inf')
            return torch.clamp(_as_tensor(input), min=min_val, max=max_val)

        def concatenate(self, seq, axis=0):
            seq_t = [_as_tensor(x) for x in seq]
            return torch.cat(seq_t, dim=axis)
        def append(self, arr, values, axis=None):
            arr = _as_tensor(arr)
            values = _as_tensor(values)
            if axis is None: return torch.cat((arr.flatten(), values.flatten()))
            return torch.cat((arr, values), dim=axis)
        
        def exp(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.exp(x)
            return torch.exp(_as_tensor(x))
            
        def log(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.log(x)
            return torch.log(_as_tensor(x))
            
        def log1p(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.log1p(x)
            return torch.log1p(_as_tensor(x))
            
        def sqrt(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.sqrt(x)
            return torch.sqrt(_as_tensor(x))
            
        def power(self, x, y): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.power(x, y)
            return torch.pow(_as_tensor(x), y)
            
        def abs(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.abs(x)
            return torch.abs(_as_tensor(x))
        
        def maximum(self, x, y, dtype=None):
            if isinstance(x, _real_numpy.ndarray) or isinstance(y, _real_numpy.ndarray):
                np_dtype = self._to_numpy_dtype(dtype) if dtype else None
                return _real_numpy.maximum(x, y, dtype=np_dtype)
            res = torch.maximum(_as_tensor(x), _as_tensor(y))
            if dtype is not None: return res.to(self._sanitize_dtype(dtype))
            return res
            
        def minimum(self, x, y, dtype=None):
            if isinstance(x, _real_numpy.ndarray) or isinstance(y, _real_numpy.ndarray):
                np_dtype = self._to_numpy_dtype(dtype) if dtype else None
                return _real_numpy.minimum(x, y, dtype=np_dtype)
            res = torch.minimum(_as_tensor(x), _as_tensor(y))
            if dtype is not None: return res.to(self._sanitize_dtype(dtype))
            return res
            
        def divide(self, x, y): return torch.div(_as_tensor(x), _as_tensor(y))
        
        def sum(self, x, axis=None, **kwargs): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.sum(x, axis=axis, **kwargs)
            return torch.sum(_as_tensor(x), dim=axis, **kwargs) if axis is not None else torch.sum(_as_tensor(x), **kwargs)
            
        def mean(self, x, axis=None, **kwargs):
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.mean(x, axis=axis, **kwargs)
            return torch.mean(_as_tensor(x), dim=axis, **kwargs) if axis is not None else torch.mean(_as_tensor(x), **kwargs)
            
        def std(self, x, axis=None, **kwargs):
             return torch.std(_as_tensor(x), dim=axis, **kwargs) if axis is not None else torch.std(_as_tensor(x), **kwargs)

        def where(self, condition, x=None, y=None):
            if x is None and y is None:
                return torch.nonzero(_as_tensor(condition), as_tuple=True)
            return torch.where(_as_tensor(condition), _as_tensor(x), _as_tensor(y))

        def isnan(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.isnan(x)
            return torch.isnan(_as_tensor(x))
        
        def isfinite(self, x):
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.isfinite(x)
            return torch.isfinite(_as_tensor(x))
            
        def isinf(self, x): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.isinf(x)
            return torch.isinf(_as_tensor(x))
            
        def diag(self, v, k=0): return torch.diag(_as_tensor(v), diagonal=k)
        
        def max(self, a, axis=None):
             a = _as_tensor(a)
             if axis is None: return torch.max(a)
             return torch.max(a, dim=axis).values
        def min(self, a, axis=None):
             a = _as_tensor(a)
             if axis is None: return torch.min(a)
             return torch.min(a, dim=axis).values

        def nonzero(self, a): return torch.nonzero(_as_tensor(a), as_tuple=True)
        def argsort(self, a, axis=-1): return torch.argsort(_as_tensor(a), dim=axis, descending=False)
        def argmax(self, a, axis=None): return torch.argmax(_as_tensor(a), dim=axis)
        def argmin(self, a, axis=None): return torch.argmin(_as_tensor(a), dim=axis)
        def flatnonzero(self, a): return torch.nonzero(_as_tensor(a).flatten(), as_tuple=True)[0]

        def sort(self, a, axis=-1, **kwargs):
            return torch.sort(_as_tensor(a), dim=axis, **kwargs).values

        def unique(self, ar, return_index=False, return_inverse=False, return_counts=False, axis=None, **kwargs):
            return torch.unique(_as_tensor(ar), sorted=True, return_inverse=return_inverse, return_counts=return_counts, dim=axis)

        def any(self, x, axis=None): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.any(x, axis=axis)
            return torch.any(_as_tensor(x), dim=axis) if axis is not None else torch.any(_as_tensor(x))
            
        def all(self, x, axis=None): 
            if isinstance(x, _real_numpy.ndarray): return _real_numpy.all(x, axis=axis)
            return torch.all(_as_tensor(x), dim=axis) if axis is not None else torch.all(_as_tensor(x))
            
        def logical_and(self, x, y): return torch.logical_and(_as_tensor(x), _as_tensor(y))
        def logical_or(self, x, y): return torch.logical_or(_as_tensor(x), _as_tensor(y))
        def isin(self, elements, test_elements): return torch.isin(_as_tensor(elements), _as_tensor(test_elements))
        
        def dot(self, a, b): return torch.matmul(_as_tensor(a), _as_tensor(b))
        def count_nonzero(self, input): return torch.count_nonzero(_as_tensor(input))
        def reshape(self, a, newshape): return _as_tensor(a).reshape(newshape)
        def transpose(self, a, axes=None): return _as_tensor(a).t() if axes is None else _as_tensor(a).permute(axes)
        def fill_diagonal(self, a, val): return _as_tensor(a).fill_diagonal_(val)
        
        def ix_(self, *args):
            args_t = [_as_tensor(a) for a in args]
            nd = len(args_t)
            out = []
            for i, a in enumerate(args_t):
                if a.ndim > 1: a = a.flatten()
                shape = [1] * nd
                shape[i] = -1
                out.append(a.view(*shape))
            return tuple(out)
            
        def savetxt(self, fname, X, **kwargs):
            if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
            _real_numpy.savetxt(fname, X, **kwargs)
        def load(self, file, **kwargs): return _real_numpy.load(file, **kwargs)
        def savez_compressed(self, file, *args, **kwds):
            clean_kwds = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in kwds.items()}
            _real_numpy.savez_compressed(file, *args, **clean_kwds)

        class errstate:
            def __init__(self, **kwargs): pass
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass

    np = NumpyShim()

    class FFTShim:
        def fft(self, x, n=None, **kwargs): return torch.fft.fft(x, n=n, **_map_args(kwargs))
        def ifft(self, x, n=None, **kwargs): return torch.fft.ifft(x, n=n, **_map_args(kwargs))
        def fft2(self, x, s=None, **kwargs): return torch.fft.fft2(x, s=s, **_map_args(kwargs))
        def ifft2(self, x, s=None, **kwargs): return torch.fft.ifft2(x, s=s, **_map_args(kwargs))
        def rfft(self, x, n=None, **kwargs): return torch.fft.rfft(x, n=n, **_map_args(kwargs))
        def irfft(self, x, n=None, **kwargs): return torch.fft.irfft(x, n=n, **_map_args(kwargs))
        def irfft2(self, x, s=None, **kwargs): return torch.fft.irfft2(x, s=s, **_map_args(kwargs))
        def rfft2(self, x, s=None, **kwargs): return torch.fft.rfft2(x, s=s, **_map_args(kwargs))
        def fftfreq(self, n, d=1.0): return torch.fft.fftfreq(n, d=d)
        def rfftfreq(self, n, d=1.0): return torch.fft.rfftfreq(n, d=d)
    fft = FFTShim()

    class LinalgShim:
        def inv(self, x): return torch.linalg.inv(x)
        def eig(self, x): return torch.linalg.eig(x)
        def eigh(self, x): return torch.linalg.eigh(x)
        def solve(self, a, b): return torch.linalg.solve(a, b)
        def norm(self, x, **kwargs): return torch.linalg.norm(x, **_map_args(kwargs))
        def det(self, x): return torch.linalg.det(x)
        def svd(self, x, **kwargs): return torch.linalg.svd(x, **kwargs)
    linalg = LinalgShim()

    import scipy.sparse as _scipy_sparse
    sparse = _scipy_sparse
    
    def to_host(data):
        if isinstance(data, torch.Tensor): return data.detach().cpu().numpy()
        return data

    print(f"🔋  Using PyTorch Wrapper on {backend_name}")

else:
    import numpy
    import scipy.fft
    import scipy.linalg
    import scipy.sparse
    np = numpy
    fft = scipy.fft
    linalg = scipy.linalg
    sparse = scipy.sparse
    def to_host(data): return data
    print("🖥️   Using NumPy/SciPy on CPU")

def to_cpu(data): return to_host(data)
