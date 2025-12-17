
############################################
# environment.py
############################################

#power‑spectrum‑correct GRF
from accelerator import np
from numpy.fft import rfft2, irfft2, rfftfreq, fftfreq    # ← **ADD**

def _sqrt_exp_spectrum(Ny, Nx, L, var):
    """
    √‑power‑spectrum for isotropic exp(−d/L) covariance on a torus.
    """
    # ky = rfftfreq(Ny)[:, None]          # (Ny,1)
    ky = fftfreq(Ny)[:, None]
    kx = rfftfreq(Nx)[None, :]          # (1,Nx/2+1)
    k2 = ky**2 + kx**2
    # Fourier transform of exp‑kernel in 2‑d:  1 / (1 + (2π L)^2 k^2)
    return np.sqrt(var) * (1.0 + (2*np.pi*L)**2 * k2)**(-1.0)

def generate_spatial_r(S, ny, nx, length_scale, base_r, var_r, *, seed=None):
    """
    Gaussian random field on a (ny×nx) periodic grid using FFT spectral synthesis.

    All species share the **same spatial pattern** (scaled by var_r and shifted
    by base_r[i]). independent landscapes, loop over S inside the
    caller – still O(S·ny·nx·log(ny·nx)).
    """
    rng = np.random.default_rng(seed)
    base_r  = np.atleast_1d(base_r).astype(float)
    var_r   = np.atleast_1d(var_r).astype(float)
    if base_r.size != S or var_r.size not in (1, S):
        raise ValueError("base_r / var_r must have length S")
    
    # new for multiple species landscapes
    out = np.empty((S, ny, nx), float)
    for i in range(S):
        spectrum = _sqrt_exp_spectrum(ny, nx, length_scale,
                                      var_r[i] if var_r.size == S else var_r[0])
        Z = rng.normal(size=(ny, nx//2 + 1)) + 1j*rng.normal(size=(ny, nx//2 + 1))
        
        # Sanity: spectrum and Z must both be (ny, nx//2+1)
        assert spectrum.shape == Z.shape, f"spectrum {spectrum.shape} vs Z {Z.shape}"
        field = irfft2(spectrum * Z, s=(ny, nx))
        field = (field - field.mean()) / field.std()
        sigma = np.sqrt(var_r[i] if var_r.size == S else var_r[0])
        out[i] = base_r[i] + sigma * field
    return out.astype(np.float64)
      

    # spectrum = _sqrt_exp_spectrum(ny, nx, length_scale,
    #                               var_r if var_r.size == 1 else var_r.mean())

    # # complex white noise with Hermitian symmetry is produced automatically
    # Z = rng.normal(size=(ny, nx//2 + 1)) + 1j*rng.normal(size=(ny, nx//2 + 1))
    # field = irfft2(spectrum * Z, s=(ny, nx))          # real (ny,nx)

    # # standardise to zero mean, unit sd
    # field = (field - field.mean()) / field.std()

    # # broadcast to species and rescale
    # r_field = base_r[:, None, None] + field * (
    #     np.sqrt(var_r)[:, None, None] if var_r.size == S else np.sqrt(var_r))

    # return r_field.astype(np.float64)

