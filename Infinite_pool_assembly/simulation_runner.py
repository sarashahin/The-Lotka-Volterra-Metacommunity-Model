############################################
# simulation_runner.py
############################################
"""
Object-oriented framework for running ecological simulations.

Defines a base `SimulationRunner` class to handle command-line parsing,
directory setup, data post-processing, and saving. Engine-specific logic
for assembly and dynamics is implemented in subclasses.
"""
import abc
import argparse
import json
import time
import logging
from pathlib import Path
import datetime as dt
from accelerator import np

# Import project modules
import dispersal
from config import BODY_MASS, NUM_PATCHES_X, NUM_PATCHES_Y, STEP_SIZE, THRESHOLD, INV
from models_ibm import IBMModel
from models_psd2 import PSD2Model
from models_ode import ODEModel
from community_assembler import IBMAssembler, PSD2Assembler
from run_rps_dynamics import animate_spatial
from assembly_utils import draw_interactions, expand_RC
import simulation_utils as sim_utils

class SimulationRunner(abc.ABC):
    """Base class for managing and executing simulation workflows."""

    def __init__(self, argv=None):
        self.start_time = time.time()
        self.args = self._parse_cli(argv)
        self._setup_logging()
        self._setup_paths()
        self._configure_dispersal()

        self.world_tag = sim_utils.build_world_tag(
            base=f"{self.args.tag}_{self.args.engine}" + (f"_{self.args.world_tag_extra}" if self.args.world_tag_extra else ""),
            ls=self.args.env_length_scale, vr=self.args.env_var_r, thr=self.args.detection_threshold,
            env_seed=self.args.env_seed_field, grid_y=NUM_PATCHES_Y, grid_x=NUM_PATCHES_X,
            disp=float(dispersal.DISPERSAL_RATE), ldd=float(dispersal.LONG_DISTANCE_PROB)
        )
        self.checkpoint_path = self.paths['checkpoints'] / f"{self.world_tag}_assembly_latest.npz"
        self.last_checkpoint_time = time.time()

    def _parse_cli(self, argv):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Run ecological simulations.")
        parser.add_argument('--engine', type=str, choices=['ibm', 'psd2', 'ode'], default='ibm')
        parser.add_argument('--tmax', type=float, default=1200)
        parser.add_argument('--record', type=float, default=10.)
        parser.add_argument('--pool', type=int, help='Trigger infinite-pool assembly.')
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--no-movie', action='store_true')
        parser.add_argument('--disp-rate', type=float, default=None)
        parser.add_argument('--ldd-prob', type=float, default=None)
        parser.add_argument('--random-seed', type=int, default=0)        
        # Assembly generic
        parser.add_argument('--window-duration', type=float, default=500, help="Duration of one assembly window (steps for IBM, time for PSD2).")
        parser.add_argument('--F-sat', type=str, default='None')
        parser.add_argument('--max-rounds', type=str, default='None')
        parser.add_argument('--max-attempts', type=str, default='10000')
        parser.add_argument('--detection-threshold', type=str, default=None)
        parser.add_argument('--resume', type=str, default=None)
        parser.add_argument('--save-every-rounds', type=int, default=5)
        parser.add_argument('--save-every-seconds', type=float, default=180.0)
        # Dynamics generic
        parser.add_argument('--record-mode', choices=['full', 'mean', 'none'], default='full')
        # Environment specific
        parser.add_argument('--env-length-scale', type=float, default=None)
        parser.add_argument('--env-var-r', type=float, default=None)
        parser.add_argument('--env-seed-field', type=int, default=None)
        # Output options
        parser.add_argument('--fp16-time-series', action='store_true')
        parser.add_argument('--world-tag-extra', type=str, default='')
        parser.add_argument('--n-species', type=int, default=3)

        args = parser.parse_args(argv)
        args.use_assembly = args.pool is not None
        args.tag = f'pool{args.pool}' if args.use_assembly else 'RPS'

        for key in ['F_sat', 'max_rounds', 'max_attempts', 'detection_threshold']:
            val = getattr(args, key)
            if isinstance(val, str) and val.lower() == 'none':
                setattr(args, key, None)
            elif val is not None:
                setattr(args, key, float(val) if '.' in str(val) else int(val))

        if args.dry_run:
            args.tmax, args.record, args.no_movie = 100, 20, True
            logging.info("[dry-run] Overriding tmax=100, record=20, no-movie=True.")

        return args

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', handlers=[logging.StreamHandler()])

    def _setup_paths(self):
        root = Path('results')
        self.paths = {
            'data': root / 'data',
            'plots': root / 'plots',
            'movies': root / 'movies',
            'checkpoints': root / 'checkpoints'
        }
        sim_utils.ensure_dirs(*self.paths.values())

    def _configure_dispersal(self):
        if self.args.disp_rate is not None:
            dispersal.DISPERSAL_RATE = self.args.disp_rate
        if self.args.ldd_prob is not None:
            dispersal.LONG_DISTANCE_PROB = self.args.ldd_prob
        logging.info(f"Dispersal configured: Rate={dispersal.DISPERSAL_RATE}, LDD Prob={dispersal.LONG_DISTANCE_PROB}")

    def _checkpoint_callback(self, state):
        do_round = (state['round'] > 0 and state['round'] % self.args.save_every_rounds == 0)
        do_time  = (time.time() - self.last_checkpoint_time) >= self.args.save_every_seconds
        if (do_round or do_time):
            sim_utils.atomic_save_npz(self.checkpoint_path, **state)
            self.last_checkpoint_time = time.time()
            logging.info(f"[checkpoint] -> {self.checkpoint_path.name}")

    def execute(self):
        if self.args.use_assembly:
            r_final, C_final, final_state, extra_assembly = self.run_assembly()
        else:
            rng = np.random.default_rng(self.args.random_seed)
            if self.args.n_species == 3:
                logging.info("Running 3-species RPS scenario (no assembly).")
                r_final = np.ones(3)
                C_final = np.array([[1, 1.7, 0.4], [0.4, 1, 1.7], [1.7, 0.4, 1]], dtype=float)
            else: # set up a community with n_species species
                  # permanently invading at rate INV
                logging.info(f"Running {self.args.n_species}-species scenario (no assembly).")
                r_final = np.array([], dtype=float)
                C_final = np.array([[]], dtype=float)
                for _ in range(self.args.n_species):
                    row, col = draw_interactions(len(r_final))
                    r_final, C_final = expand_RC(r_final, C_final, 1, row, col)
            B_seed = ( rng.random((self.args.n_species, NUM_PATCHES_Y, NUM_PATCHES_X)) <
                       1.5/self.args.n_species ).astype(float) * BODY_MASS
            dispersal.set_invasion_pressure( INV * np.ones_like(B_seed) )

            if self.args.engine == 'ibm':
                final_state = (B_seed / BODY_MASS).astype(int)
            elif self.args.engine == 'psd2':
                W_seed = np.zeros_like(B_seed, dtype=bool)
                PC_seed = np.ones_like(B_seed, dtype=float)
                final_state = (B_seed, W_seed, PC_seed)
            else:
                final_state = B_seed
            extra_assembly = {}

        dyn_start_time = time.time()
        results = self.run_dynamics(r_final, C_final, final_state)
        results['runtime_s'] = time.time() - dyn_start_time
        results.update({
            'r_final': r_final, 'C_final': C_final,
            'final_state_assembly': final_state, 'extra_assembly': extra_assembly
        })

        self.post_process_and_save(results)
        total_runtime = time.time() - self.start_time
        logging.info(f"Simulation finished successfully in {total_runtime:.1f}s.")

    def post_process_and_save(self, results):
        logging.info("Post-processing and saving results...")

        r_final, C_final = results['r_final'], results['C_final']
        B_dynamics, t_dynamics = results.get('B_dynamics'), results.get('t_dynamics')

        detection_thr_count = self.args.detection_threshold if self.args.detection_threshold is not None else (THRESHOLD / BODY_MASS)
        thr_mass = detection_thr_count * BODY_MASS

        P_t = (B_dynamics >= thr_mass).astype(np.uint8) if B_dynamics is not None and B_dynamics.ndim == 4 else None
        B_last = B_dynamics[-1].astype(np.float16) if B_dynamics is not None and B_dynamics.ndim == 4 else None

        sp_events = sim_utils.species_event_times(P_t, t_dynamics) if P_t is not None else {}
        deg_in, deg_out = sim_utils.summarize_interactions(C_final)
        C_top_idx, C_top_w = sim_utils.topk_interactions(C_final, k=16)

        output_payload = {
            # FIX: Use Python int/float constructors; np.int64/float32 are dtypes in the accelerator shim.
            "gamma": int(len(r_final)), "r_base": r_final.astype(np.float32),
            "deg_in": deg_in, "deg_out": deg_out, "C_topk_idx": C_top_idx, "C_topk_w": C_top_w,
            "t_dynamics": t_dynamics,
            "B_dynamics": B_dynamics.astype(np.float16) if self.args.fp16_time_series and B_dynamics is not None else B_dynamics,
            "P_t": P_t, "B_last": B_last, "runtime_s": float(results['runtime_s']),
            "detection_threshold": float(detection_thr_count),
            **results.get('model_specific_outputs', {}), **results.get('extra_assembly', {}), **sp_events
        }

        train_path = self.paths['data'] / f"{self.world_tag.lower()}_training.npz"
        sim_utils.atomic_save_npz(train_path, **output_payload)
        logging.info(f"[save] -> {train_path}")

        meta = {"scenario": self.args.tag, "engine": self.args.engine, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "grid": f"{NUM_PATCHES_Y}x{NUM_PATCHES_X}", "dispersal": {"rate": float(dispersal.DISPERSAL_RATE), "ldd_prob": float(dispersal.LONG_DISTANCE_PROB)}, "total_runtime_s": time.time() - self.start_time, "cli_args": vars(self.args)}
        meta_path = self.paths['data'] / f"{self.world_tag.lower()}_meta.json"
        with open(meta_path, 'w') as f: json.dump(meta, f, indent=2)
        logging.info(f"[save] -> {meta_path}")

        if not self.args.no_movie and B_dynamics is not None and B_dynamics.ndim == 4:
            movie_path = self.paths['movies'] / f'{self.world_tag}.mp4'
            
            # 1. Bring to CPU
            B_host = sim_utils.to_host(B_dynamics)

            # 2. FIX: Force Float64 and Add Dithering
            #    We add tiny random noise (1e-10) to ensure Max > Min.
            #    This prevents Matplotlib from dividing by zero on uniform frames.
            B_host = B_host.astype(np.float64)
            B_host += np.random.normal(0, 1e-10, B_host.shape)

            # 3. Sanitize and Clip
            B_host = np.nan_to_num(B_host, nan=0.0, posinf=0.0, neginf=0.0)
            B_host = np.clip(B_host, 0.0, 1.0)

            # 4. Generate Movie
            animate_spatial(B_host, f'{self.args.engine.upper()} {self.args.tag}', str(movie_path))
            logging.info(f"[save] -> {movie_path}")

    @abc.abstractmethod
    def run_assembly(self): pass

    @abc.abstractmethod
    def run_dynamics(self, r, C, initial_state): pass

class IBMSimulation(SimulationRunner):
    def run_assembly(self):
        init_r, init_C, init_state, init_attempts, init_round = None, None, None, 0, -1
        if self.args.resume:
            try:
                ck = np.load(self.args.resume, allow_pickle=True)
                init_r, init_C = ck['r'], ck['C']
                init_state = ck.get('state') or ck.get('N')
                init_attempts, init_round = int(ck.get('attempts', 0)), int(ck.get('round', -1))
                logging.info(f"Resuming IBM from checkpoint: {self.args.resume}")
            except Exception as e:
                logging.error(f"Failed to load IBM checkpoint: {e}", exc_info=True)

        assembler = IBMAssembler(
            max_attempts=self.args.max_attempts, max_rounds=self.args.max_rounds,
            F_sat=self.args.F_sat, detection_threshold=self.args.detection_threshold,
            checkpoint_fn=self._checkpoint_callback, seed=self.args.random_seed,
            init_r=init_r, init_C=init_C, init_state=init_state,
            init_attempts=init_attempts, init_round=init_round,
            nsteps=int(self.args.window_duration), record_step=int(self.args.window_duration)
        )
        return assembler.run()

    def run_dynamics(self, r, C, initial_N):
        logging.info("Running dynamics with IBM engine...")
        ### CHECK: The AI version used `tmax=self.args.tmax` here. 
        ### I reverted to `nsteps=int(self.args.tmax / STEP_SIZE)` to ensure exact time-stepping behavior.
        model = IBMModel(r, C, initial_N=initial_N,
                         nsteps=int(self.args.tmax / STEP_SIZE),
                         record_step=int(self.args.record / STEP_SIZE),
                         record_mode=self.args.record_mode, seed=1,
                         length_scale=self.args.env_length_scale, var_r=self.args.env_var_r,
                         seed_field=self.args.env_seed_field)
        
        B_dynamics_raw = model.run()
        t_dynamics = np.arange(1, model.nrecords + 1) * self.args.record
        
        return {
            "B_dynamics": B_dynamics_raw, "t_dynamics": t_dynamics,
            "model_specific_outputs": {"final_assembly_N": initial_N}
        }

class PSD2Simulation(SimulationRunner):
    def run_assembly(self):
        assembler = PSD2Assembler(
            max_attempts=self.args.max_attempts, max_rounds=self.args.max_rounds,
            F_sat=self.args.F_sat, detection_threshold=self.args.detection_threshold,
            checkpoint_fn=self._checkpoint_callback, seed=self.args.random_seed,
            tmax=self.args.window_duration, record_step=self.args.window_duration
        )
        return assembler.run()

    def run_dynamics(self, r, C, initial_state):
        B0, W0, PC0 = initial_state
        logging.info("Running dynamics with PSD2 engine...")
        model = PSD2Model(r, C, initial_B=B0, initial_wait=W0, initial_clock=PC0,
                          tmax=self.args.tmax, record_step=self.args.record, seed=1,
                          length_scale=self.args.env_length_scale, var_r=self.args.env_var_r,
                          seed_field=self.args.env_seed_field)
        
        t, B, W, PC, G, INV, EST = model.run()
        
        B_final_assembly, _, _ = initial_state
        return {
            "t_dynamics": t, "B_dynamics": B,
            "model_specific_outputs": {
                'PSD2_wait': W, 'PSD2_pclock': PC, 'PSD2_growth': G,
                'PSD2_invasion': INV, 'PSD2_est_prob': EST,
                'final_assembly_B': B_final_assembly
            }
        }

class ODESimulation(SimulationRunner):
    def run_assembly(self):
        logging.error("Assembly is not supported for the ODE engine.")
        raise NotImplementedError

    def run_dynamics(self, r, C, initial_state=None):
        logging.info("Running dynamics with ODE engine...")
        model = ODEModel(r, C, tmax=self.args.tmax, record_step=self.args.record, seed=1,
                         length_scale=self.args.env_length_scale, var_r=self.args.env_var_r,
                         seed_field=self.args.env_seed_field)
        t, B = model.run()
        return {
            "t_dynamics": t, "B_dynamics": B
        }
