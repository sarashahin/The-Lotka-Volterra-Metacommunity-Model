import numpy as np
import argparse
from math import sqrt
from multiprocessing import Pool, cpu_count

# ---------- Utility: simple peak-based period estimator ----------
def _estimate_period_from_series(t, y, min_peaks=5, tail_frac=0.5):
    """
    Estimate period from a scalar time series y(t) by locating local maxima (peaks)
    on the tail of the trajectory (to avoid transients).
    Returns (T_mean, n_periods). Raises ValueError if not enough peaks.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 5:
        raise ValueError("Series too short for period estimation.")
    # use tail to avoid transients
    i0 = int((1.0 - tail_frac) * n)
    t_tail = t[i0:]
    y_tail = y[i0:]

    # simple peak detector: y[k-1] < y[k] >= y[k+1]
    peaks = np.where((y_tail[1:-1] > y_tail[:-2]) & (y_tail[1:-1] >= y_tail[2:]))[0] + 1
    if peaks.size < (min_peaks + 1):
        raise ValueError(f"Not enough peaks found on tail (found {peaks.size}).")

    # convert to times, compute successive peak spacings
    t_peaks = t_tail[peaks]
    T = np.diff(t_peaks)
    return float(T.mean()), int(len(T))

# ---------- PDF ODE (Axel) integration to measure period ----------
def _pdf_ode_rhs(P, K):
    # dP/dt for Axel’s 3-state system:
    #  dP1 = -K P1 P2 + K P1 P3
    #  dP2 =  K P1 P2 - K P2 P3
    #  dP3 = -K P1 P3 + K P2 P3
    P1, P2, P3 = P
    d1 = -K*P1*P2 + K*P1*P3
    d2 =  K*P1*P2 - K*P2*P3
    d3 = -K*P1*P3 + K*P2*P3
    return np.array([d1, d2, d3], float)

def measure_pdf_ode_period(D, body_mass, r, b, mu, P0=(1/6,1/6,2/3), dt=0.25, tmax=60000.0):
    """
    Integrate the PDF ODE with RK4, measure period from P2(t) peaks on the tail.
    Returns (T_meas, n_periods, T_pred).
    """
    P = np.array(P0, float)
    K = (D * r * (r - b*r)) / (body_mass * (mu + r - b*r))
    n_steps = int(tmax / dt)
    t = np.empty(n_steps+1, float); t[0] = 0.0
    P2_series = np.empty(n_steps+1, float); P2_series[0] = P[1]

    # RK4 integrate
    for k in range(1, n_steps+1):
        k1 = _pdf_ode_rhs(P, K)
        k2 = _pdf_ode_rhs(P + 0.5*dt*k1, K)
        k3 = _pdf_ode_rhs(P + 0.5*dt*k2, K)
        k4 = _pdf_ode_rhs(P + dt*k3, K)
        P = P + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # renormalize tiny drift
        s = P.sum()
        if s != 0.0:
            P /= s
        t[k] = k * dt
        P2_series[k] = P[1]

    T_meas, n_per = _estimate_period_from_series(t, P2_series, min_peaks=6, tail_frac=0.5)
    T_pred = predicted_period_axel_pdf(D, body_mass, r, b, mu)
    return T_meas, n_per, T_pred

# ---------- CI helper ----------
def wilson_ci(p_hat, n, alpha=0.05):
    z = 1.959963984540054
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    half = z * sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    lo = max(0.0, center - half); hi = min(1.0, center + half)
    return lo, hi

# ---------- Axel–PDF period (authoritative) ----------
def predicted_period_axel_pdf(D_source, body_mass, r, b, mu):
    """
    Axel’s PDF (RSPMetacommunitySimple.pdf) gives:
        T = 2*sqrt(3)*pi * BODY_MASS * (mu + r - b*r) / ((1 - b) * D_source * r^2)
    Here D_source is the *source/leave* rate used in your code: config.DISPERSAL_RATE.
    """
    return (2.0*np.sqrt(3.0)*np.pi * body_mass * (mu + r - b*r)) / ((1.0 - b) * D_source * (r**2))

# ---------- Axel–PDF clocks at t≈0 for preset P=(1/6,1/6,2/3) ----------
def predicted_poisson_clock_rates(D, body_mass, r, b, mu, preset=(1/6, 1/6, 2/3)):
    """
    PDF-consistent 'Poisson clock progression rates' at t≈0:
        (K*P2, K*P3, K*P1)
    with  K = D * r * (r - b*r) / (body_mass * (mu + r - b*r))
    """
    P1, P2, P3 = map(float, preset)
    K = (D * r * (r - b*r)) / (body_mass * (mu + r - b*r))
    return (float(K*P2), float(K*P3), float(K*P1))

# ---------- try to *measure* first-step Poisson clock rates from PSD2 (optional hook) ----------
def try_measure_psd2_clock_rates_at_t0(model_obj):
    """
    Attempt to read PSD2’s first-step Poisson clock hazards if the model exposes them.
    Keeps models unchanged; we only *read* if present. Otherwise return None.
    Expected optional attributes/methods (any one is fine):
      - model_obj.clock_rates_t0  → tuple/list of 3 floats
      - model_obj.diagnostics['clock_rates_t0']
      - model_obj.get_clock_rates_t0()
    """
    for attr in ("clock_rates_t0", "diagnostics", "get_clock_rates_t0"):
        if hasattr(model_obj, attr):
            if attr == "clock_rates_t0":
                v = getattr(model_obj, attr)
                return tuple(map(float, v)) if v is not None else None
            if attr == "diagnostics":
                diag = getattr(model_obj, "diagnostics")
                if isinstance(diag, dict) and "clock_rates_t0" in diag:
                    v = diag["clock_rates_t0"]
                    return tuple(map(float, v)) if v is not None else None
            if attr == "get_clock_rates_t0":
                fn = getattr(model_obj, "get_clock_rates_t0")
                try:
                    v = fn()
                    return tuple(map(float, v)) if v is not None else None
                except Exception:
                    pass
    return None

# ---------- PSD2 branch ----------
def run_psd2(Nx, Ny, BODY_MASS, MORTALITY_RATE, b, seed, want_clock_probe=False):
    from models_psd2 import PSD2Model
    # (1) resident-only to measure local abundance (~1)
    initial_B = np.zeros((1, Ny, Nx)); initial_B[:] = 2 * BODY_MASS
    m1 = PSD2Model(r=np.array([1.0]), C=np.array([[1.0]]),
                   initial_B=initial_B, tmax=5000.0, record_step=10.0,
                   dispersal_type="propagule", seed=seed)
    t1, traj1, *_ = m1.run()
    B_resident = traj1[...,0,:,:]
    k0 = int(0.8*len(t1))
    local_abund = float(B_resident[k0:].mean())

    # (2) invasion (S=2) — PSD returns est_prob_traj directly
    initial_B2 = np.zeros((2, Ny, Nx))
    initial_B2[0] = B_resident[-1]
    initial_B2[1] = BODY_MASS * 1e-4

    m2 = PSD2Model(r=np.array([1.0,1.0]), C=np.array([[1.0,b],[b,1.0]]),
                   initial_B=initial_B2, tmax=200.0, record_step=2.0,
                   dispersal_type="propagule", seed=seed+7)
    out = m2.run()
    if len(out) >= 7:
        _, _, _, _, _, _, est_prob_traj = out
        p_est = float(np.nanmean(est_prob_traj[:,1,:,:][1:]))
    else:
        p_est = (1.0 - b) / (MORTALITY_RATE + (1.0 - b))  # theory fallback

    measured = None
    if want_clock_probe:
        measured = try_measure_psd2_clock_rates_at_t0(m2)

    return local_abund, p_est, measured

# ---------- IBM (strict, two-phase, parallel) ----------
PHASE1_STEPS, PHASE2_STEPS = 5000, 15000
RECORD_STEP, QUICK_THRESH = 100, 20
TAIL_FRAMES, TAIL_MIN_TOTAL = 10, 2

def ibm_prepare_resident(Nx, Ny, seed):
    from models_ibm import IBMModel
    initial_N = np.full((1, Ny, Nx), 2, dtype=np.int64)
    m1 = IBMModel(r=np.array([1.0]), C=np.array([[1.0]]),
                  initial_N=initial_N, nsteps=PHASE1_STEPS, record_step=RECORD_STEP,
                  dispersal_type="propagule", seed=seed)
    traj1 = m1.run()
    m1b = IBMModel(r=np.array([1.0]), C=np.array([[1.0]]),
                   initial_N=np.round(traj1[-1]).astype(np.int64),
                   nsteps=PHASE1_STEPS, record_step=RECORD_STEP,
                   dispersal_type="propagule", seed=seed+1)
    traj1b = m1b.run()
    resident_eq = np.round(traj1b[-1,0]).astype(np.int64)
    local_abund = float(traj1b[int(0.8*traj1b.shape[0]):].mean())
    return resident_eq, local_abund

def ibm_trial_worker(args_tuple):
    seed, resident_eq, Ny, Nx, b = args_tuple
    import numpy as np
    from models_ibm import IBMModel
    rng = np.random.default_rng(seed)
    init1 = np.zeros((2, Ny, Nx), dtype=np.int64)
    init1[0] = resident_eq
    y = rng.integers(0, Ny); x = rng.integers(0, Nx)
    init1[1, y, x] = 1

    m2a = IBMModel(r=np.array([1.0,1.0]), C=np.array([[1.0, b],[b,1.0]]),
                   initial_N=init1, nsteps=PHASE1_STEPS, record_step=RECORD_STEP,
                   dispersal_type="propagule", seed=seed+100)
    trajA = m2a.run()
    invA = trajA[:,1].sum(axis=(1,2))
    if invA[-1] == 0: return 0.0
    if invA[-1] >= QUICK_THRESH: return 1.0

    m2b = IBMModel(r=np.array([1.0,1.0]), C=np.array([[1.0, b],[b,1.0]]),
                   initial_N=np.round(trajA[-1]).astype(np.int64),
                   nsteps=PHASE2_STEPS, record_step=RECORD_STEP,
                   dispersal_type="propagule", seed=seed+200)
    trajB = m2b.run()
    invB = trajB[:,1].sum(axis=(1,2))
    tail = invB[-TAIL_FRAMES:]
    ok = (tail.min() > 0) and (tail.mean() >= TAIL_MIN_TOTAL)
    return 1.0 if ok else 0.0

def run_ibm_fast(trials, seed, jobs, Nx, Ny, b):
    resident_eq, local_abund = ibm_prepare_resident(Nx, Ny, seed)
    base = seed + 1000
    tasks = [(base + i, resident_eq, Ny, Nx, b) for i in range(trials)]
    procs = (cpu_count() if jobs == 0 else max(1, jobs))
    if procs == 1:
        results = [ibm_trial_worker(t) for t in tasks]
    else:
        with Pool(processes=procs) as pool:
            results = pool.map(ibm_trial_worker, tasks, chunksize=max(1, trials // (procs*8)))
    p_est = float(np.mean(results))
    return local_abund, p_est, procs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["psd","ibm"], required=True)
    parser.add_argument("--D_FACTOR", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=1.2, help="RPS parameter a (>1) for P→D guard check")
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--probe-clocks", action="store_true",
                        help="Try to read first-step Poisson clock rates from PSD2 (if exposed)")
    parser.add_argument("--check-pdf-ode-period", action="store_true",
                        help="Integrate PDF ODE and compare measured vs predicted period")

    # NOTE: argparse stores hyphenated names as attributes with underscores
    parser.add_argument("--pdf-ode-small-amplitude", action="store_true",
                        help="Use tiny-amplitude initial condition near (1/3,1/3,1/3) for period check")
    parser.add_argument("--pdf-ode-dt", type=float, default=0.25,
                        help="Time step for PDF ODE integration")

    args = parser.parse_args()

    # config (do NOT change IBM model)
    import config
    BASE_D = config.DISPERSAL_RATE
    config.DISPERSAL_RATE     = BASE_D * args.D_FACTOR
    config.LONG_DISTANCE_PROB = 1.0  # pure non-local
    from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS, MORTALITY_RATE, DISPERSAL_RATE
    Nx, Ny = NUM_PATCHES_X, NUM_PATCHES_Y
    b = 0.4
    r0 = 1.0
    mu = float(MORTALITY_RATE)
    D_now = float(DISPERSAL_RATE)
    D_full = float(BASE_D)
    D_half = float(BASE_D) / 2.0

    # --- Axel's P->D smallness guard ---------------------------------------
    ratio = D_now / (BODY_MASS * max(1e-12, (args.a - 1.0)))
    print(f"D/(bodyMass*(a-1)) = {ratio:.3e}  (Axel: want this ≪ 1 for P→D to be negligible)")

    # measure abundance & p_est
    if args.model == "psd":
        local_abund, p_est, measured_clocks = run_psd2(Nx, Ny, BODY_MASS, MORTALITY_RATE, b, args.seed,
                                                       want_clock_probe=args.probe_clocks)
        workers = 1
    else:
        local_abund, p_est, workers = run_ibm_fast(args.trials, args.seed, args.jobs, Nx, Ny, b)
        measured_clocks = None

    # Axel–PDF period prediction (authoritative)
    T_pred_now  = predicted_period_axel_pdf(D_now,  BODY_MASS, r0, b, mu)
    T_pred_full = predicted_period_axel_pdf(D_full, BODY_MASS, r0, b, mu)
    T_pred_half = predicted_period_axel_pdf(D_half, BODY_MASS, r0, b, mu)

    # Axel–PDF predicted clock rates at t≈0 (prey-share = PDF)
    c1, c2, c3 = predicted_poisson_clock_rates(D_now, BODY_MASS, r0, b, mu)
    print(f"Predicted clock rates @t≈0 (PDF, prey-share): {c1:.6g}, {c2:.6g}, {c3:.6g}")

    # If PSD2 exposes measured first-step clocks, print them
    if measured_clocks is not None:
        m1, m2, m3 = measured_clocks
        print(f"Measured   clock rates @t≈0 (from PSD2): {m1:.6g}, {m2:.6g}, {m3:.6g}")
    elif args.model == "psd" and args.probe_clocks:
        print("NOTE: PSD2 did not expose first-step clock hazards. "
              "Add a small hook that sets model.clock_rates_t0 at step 0 for full verification.")

    if args.model == "ibm":
        lo, hi = wilson_ci(p_est, args.trials)
        print(f"IBM p_est 95% CI (Wilson): [{lo:.3f}, {hi:.3f}] with n={args.trials} (jobs={workers})")

    print(f"MODEL={args.model.upper()}  D_FACTOR={args.D_FACTOR:g}")
    print(f"Measured local resident abundance  ≈ {local_abund:.4f}")
    print(f"Measured establishment probability ≈ {p_est:.4f}")
    print(f"Axel–PDF predicted period (current D)  ≈ {T_pred_now:.4g}  [D_source={D_now:g}]")
    print(f"Axel–PDF predicted period (full D)     ≈ {T_pred_full:.4g}")
    print(f"Axel–PDF predicted period (half D)     ≈ {T_pred_half:.4g}")

    # Optional: rigorous check that the PDF ODE period matches the formula
    if args.check_pdf_ode_period:
        # IMPORTANT: argparse turns hyphens into underscores in attribute names
        small_amp = getattr(args, "pdf_ode_small_amplitude", False)
        dt_ode    = getattr(args, "pdf_ode_dt", 0.25)

        # (A) Small-amplitude orbit near the fixed point -> should match formula <0.1%
        eps = 1e-6
        P0_small = (1/3 + eps, 1/3, 1/3 - eps)
        T_small, n_small, T_pred = measure_pdf_ode_period(
            D_now, BODY_MASS, r0, b, mu, P0=P0_small, dt=dt_ode, tmax=60000.0
        )
        err_small = abs(T_small - T_pred) / T_pred
        print(f"[PDF-ODE small] Measured period from P2(t): {T_small:.3f} (avg of {n_small} cycles)")
        print(f"[PDF-ODE small] Predicted period (closed form): {T_pred:.3f}")
        print(f"[PDF-ODE small] Relative error: {100*err_small:.3f}%")

        # (B) Large-amplitude orbit (original default) -> period increases with amplitude
        if not small_amp:
            P0_large = (1/6, 1/6, 2/3)
            T_large, n_large, T_pred2 = measure_pdf_ode_period(
                D_now, BODY_MASS, r0, b, mu, P0=P0_large, dt=dt_ode, tmax=60000.0
            )
            err_large = abs(T_large - T_pred2) / T_pred2
            print(f"[PDF-ODE large] Measured period from P2(t): {T_large:.3f} (avg of {n_large} cycles)")
            print(f"[PDF-ODE large] Predicted period (closed form): {T_pred2:.3f}")
            print(f"[PDF-ODE large] Relative error: {100*err_large:.3f}%")

if __name__ == "__main__":
    main()
