############################################
# run_all_rps.py
############################################
"""
run_all_rps.py  – benchmark RPS OR infinite-pool assembly pipeline
------------------------------------------------------------------
Examples
  3-sp benchmark           :  python run_all_rps.py
  infinite pool (120)      :  python run_all_rps.py --pool 120 --tmax 20000
Quick CI smoke-test        :  python run_all_rps.py --dry-run
Outputs → results/{data,plots,movies}/
"""
from __future__ import annotations
import cupy, os
print("CuPy OK on device:", cupy.cuda.runtime.getDevice())
print("LD_LIBRARY_PATH =", os.getenv("LD_LIBRARY_PATH","<unset>"))
import argparse, json, time, pathlib, datetime as _dt
import numpy as np, matplotlib.pyplot as plt
import scipy
from  scipy.fft import rfft, rfftfreq
import gpu_patch
import sys





# ─── project modules ─────────────────────────────────────────────────────
import dispersal
from config import BODY_MASS, NUM_PATCHES_X, NUM_PATCHES_Y, STEP_SIZE, DISPERSAL_RATE,LONG_DISTANCE_PROB
from models_psd2 import PSD2Model
from models_ibm  import IBMModel
from models_ode  import ODEModel
from assembly_stepwise_psd2 import stepwise_assembly_psd2
from assembly_stepwise_ibm  import stepwise_assembly_ibm
from run_rps_dynamics       import animate_spatial
# visualisation helpers  <‑‑ add these two lines
from utils_vis import make_mosaic
from colour_bank import random_colour_table

# ─────────────────────────────────────────────────────────────────────────

# ─── helpers ─────────────────────────────────────────────────────────────
def dominant_period(t: np.ndarray, x: np.ndarray) -> float:
    if len(t) < 4:
        return float('nan')
    freqs = rfftfreq(len(t), np.mean(np.diff(t)))[1:]
    return float(1.0 / freqs[np.argmax(np.abs(rfft(x))[1:])])

def ensure_dirs(*paths: pathlib.Path) -> None:
    for p in paths:  p.mkdir(parents=True, exist_ok=True)

def first_extinction_time(B: np.ndarray, ts: np.ndarray) -> np.ndarray:
    extinct = (B.sum(axis=(2, 3)) == 0)
    out     = np.full(B.shape[1], -1., float)
    if extinct.any():
        idx  = extinct.argmax(axis=0)
        out[extinct.any(0)] = ts[idx[extinct.any(0)]]
    return out
# ─────────────────────────────────────────────────────────────────────────

def cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--tmax',    type=float, default=1200)
    p.add_argument('--record',  type=float, default=10.)
    p.add_argument('--fps',     type=int,   default=10)
    p.add_argument('--no-movie',action='store_true')
    p.add_argument('--pool',    type=int,
                   help='trigger infinite-pool assembly (number used only for meta)')
    p.add_argument('--assemble-horizon', type=float, default=1_000,
                   help='model-time horizon for each invasion attempt during assembly')
    p.add_argument('--dry-run', action='store_true',
                   help='CI / smoke-test: tmax=100, record=20, skip movies/plots')
    p.add_argument('--skip-ode', action='store_true',
                   help='do not run the ODE reference model')
    return p.parse_args(argv)

# ─────────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = cli(argv)

    # ------- dry-run patch ------------------------------------------------
    if args.dry_run:
        args.tmax   = 100
        args.record = 20
        args.no_movie = True
        print("[dry-run] tmax set to 100, record_step to 20, movies disabled.")

    # ---------------------------------------------------------------
    # COMMON RANDOM STARTING MOSAIC  (same biomass everywhere)
    rng      = np.random.default_rng(123)
    B_seed   =  (rng.random((3, NUM_PATCHES_Y, NUM_PATCHES_X)) < 0.5).astype(float) * BODY_MASS
    N_ibm   = (B_seed / BODY_MASS).astype(int)         # integer counts for IBM
    # ---------------------------------------------------------------

    # ------- scenario selection ------------------------------------------
    if args.pool is None:                                  # 3-species RPS
        a, b = 1.7, 0.4
        r0 = np.ones(3)
        C0 = np.array([[1,a,b],[b,1,a],[a,b,1]], float)
        tag, use_assembly = 'RPS', False
    else:                                                  # infinite pool
        tag, use_assembly = f'pool{args.pool}', True       # exact size decided later

    # choose colours ONCE, so PSD & IBM panels share them
    # choose colours ONCE, but pick as many as you need
    max_richness_est = 50 if not use_assembly else 2*args.pool
    colour_table = random_colour_table(max_richness_est)
                    

    # ------- folders ------------------------------------------------------
    root      = pathlib.Path('results')
    d_data, d_plot, d_mov = root/'data', root/'plots', root/'movies'
    ensure_dirs(d_data, d_plot, d_mov)

    data, meta = {}, {}

    # =====================================================================
    #  PSD2
    # =====================================================================
    if use_assembly:
        r_psd, C_psd, extra_psd = stepwise_assembly_psd2(
            window_time=args.assemble_horizon,
            record_step=args.record,
            max_rounds=30000,
            F_sat=6)
        # data['PSD2_occ'] = extra_psd['occ_counts']          # NEW
        B_seed  = extra_psd['B_seed']
        data['PSD2_occ'] = extra_psd['occ_counts']
        dispersal.set_invasion_pressure(None)            # safety
    else:
        r_psd, C_psd = r0, C0
        # B_seed      = None           # start from default biomass
        extra_psd = {}  # Empty dict to avoid errors when referenced

    m_psd = PSD2Model(r_psd, C_psd, initial_B=B_seed,
                      tmax=args.tmax, record_step=args.record,
                      dispersal_type='propagule', seed=42)
    t_psd, B_psd, w_psd, pc_psd, g_psd, inv_psd, est_psd = m_psd.run()

    # -------- snapshots for the 4×2 patchy mosaic --------------------------
    snap_times = [0, 0.25*args.tmax, 0.5*args.tmax, 0.75*args.tmax,
                0.85*args.tmax, 0.9*args.tmax, 0.95*args.tmax, args.tmax]
    snap_idx   = [np.abs(t_psd - tt).argmin() for tt in snap_times]
    frames_psd = [B_psd[i] for i in snap_idx]    # each is (S,Ny,Nx)

    if len(r_psd) > colour_table.shape[0]:
        colour_table = random_colour_table(len(r_psd))

    make_mosaic(frames_psd, snap_times,
                colour_table[:len(r_psd)],        # truncate to richness
                save_to=d_plot / f"{tag}_PSD2_panels.png", ncols=4, dpi=300)


    meta['PSD2'] = dict(S=len(r_psd),
                        period=dominant_period(t_psd, B_psd.mean((2,3))[:,0]))
    data.update({f'PSD2_{k}': v for k,v in dict(
        t=t_psd, B=B_psd, wait=w_psd, pclock=pc_psd,
        growth=g_psd, invasion=inv_psd, est_prob=est_psd).items()})

    if not args.no_movie:
        animate_spatial(B_psd, f'PSD2 {tag}', str(d_mov/f'PSD2_{tag}.mp4'), args.fps)

    # =====================================================================
    #  IBM
    # =====================================================================
    if use_assembly:
        r_ibm, C_ibm, N_ibm, extra_ibm = stepwise_assembly_ibm(
            window_steps=int(args.assemble_horizon/STEP_SIZE),
            record_step=int(args.record/STEP_SIZE),
            seed_size=5,
            F_sat=6)
    else:
        r_ibm, C_ibm = r0, C0
        # N_ibm        = None                                 # nothing to save
        extra_ibm     = {}          # nothing to add later

    m_ibm = IBMModel(r_ibm, C_ibm, initial_N=N_ibm,
                     nsteps=int(args.tmax/STEP_SIZE),
                     record_step=int(args.record/STEP_SIZE),
                     dispersal_type='propagule', seed=1)
    B_ibm = m_ibm.run()
    t_ibm = np.arange(args.record, args.tmax + args.record, args.record)

    snap_idx   = [np.abs(t_ibm - tt).argmin() for tt in snap_times]  # reuse same times
    frames_ibm = [B_ibm[i] for i in snap_idx]

    if len(r_ibm) > colour_table.shape[0]:
        colour_table = random_colour_table(len(r_ibm))
    make_mosaic(frames_ibm, snap_times,
                colour_table[:len(r_ibm)],
                save_to=d_plot / f"{tag}_IBM_panels.png")
    # # -------- snapshots for the 4×2 patchy mosaic --------------------------

    meta['IBM'] = dict(S=len(r_ibm),
                       period=dominant_period(t_ibm, B_ibm.mean((2,3))[:,0]))
    ibm_blk = dict(t=t_ibm, B=B_ibm,
                   ext_step=first_extinction_time(B_ibm, t_ibm))
    if N_ibm is not None:            # only when we actually assembled
        ibm_blk['N_final'] = N_ibm
    if extra_ibm:
        ibm_blk['occ_counts'] = extra_ibm['occ_counts']
        
        
    data.update({f'IBM_{k}': v for k,v in ibm_blk.items()})

    if not args.no_movie:
        animate_spatial(B_ibm, f'IBM {tag}', str(d_mov/f'IBM_{tag}.mp4'), args.fps)

    # # =====================================================================
    # #  ODE (benchmark only)
    # # =====================================================================
    # if not use_assembly:
    #     m_ode = ODEModel(r0, C0,
    #                      tmax=args.tmax, record_step=args.record,
    #                      dispersal_type='propagule', seed=7)
    #     t_ode, B_ode = m_ode.run()
    #     meta['ODE'] = dict(S=3,
    #                        period=dominant_period(t_ode, B_ode.mean((2,3))[:,0]))
    #     data.update({'ODE_t': t_ode, 'ODE_B': B_ode})

    # # =====================================================================
    #  SAVE
    # =====================================================================
    meta.update(dict(
        scenario   = tag,
        timestamp  = _dt.datetime.utcnow().isoformat(timespec='seconds')+'Z',
        grid       = f"{NUM_PATCHES_Y}×{NUM_PATCHES_X}",
        dispersal  = dict(LONG_DISTANCE_PROB=float(dispersal.LONG_DISTANCE_PROB)),
        body_mass  = BODY_MASS))
    fout = d_data / f'{tag.lower()}_dataset.npz'
    np.savez_compressed(fout, meta=json.dumps(meta, indent=2), **data)
    print('[save] →', fout)

    # =====================================================================
    #  QUICK-LOOK PLOT
    # =====================================================================
    if not args.no_movie and not args.dry_run:
        plt.figure(figsize=(8,6))
        plt.plot(t_psd,  B_psd.mean((1,2,3)), 'b', label='PSD2')
        plt.plot(t_ibm,  B_ibm.mean((1,2,3)), 'g', label='IBM')
        if 'ODE_t' in data:
            plt.plot(data['ODE_t'],
                     data['ODE_B'].mean((2,3))[:,0], 'r', label='ODE')
        txt = '   '.join(f"{k}:{v['period']:.1f}"
                         for k,v in meta.items()
                         if isinstance(v, dict) and 'period' in v)
        plt.title(f'{tag} – dominant periods   {txt}')
        plt.xlabel('time'); plt.ylabel('mean biomass'); plt.legend()
        plt.tight_layout()
        plt.savefig(d_plot/f'{tag}_means.png', dpi=150); plt.close()

    print('[done] finished successfully.')

# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
# if __name__ == "__main__" and "--quick-exit" in sys.argv:
#     print("Quick-exit sanity-check OK"); sys.exit(0)
    t0 = time.time()
    main()
    print(f'⏱  {time.time()-t0:.1f}s')
