############################################
# simulation_runner.py
############################################
import abc
import argparse
import json
import time
import logging
from pathlib import Path
import datetime as dt
from accelerator import np
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
    def __init__(self, argv=None):
        self.start_time = time.time()
        self.args = self._parse_cli(argv)
        self._setup_logging()
        self._setup_paths()
        self._configure_dispersal()
        self.world_tag = sim_utils.build_world_tag(
            base=f"{self.args.tag}_{self.args.engine}" + (f"_{self.args.world_tag_extra}" if self.args.world_tag_extra else ""),
            ls=self.args.env_length_scale, vr=self.args.env_var_r,
            thr=THRESHOLD, env_seed=self.args.env_seed_field,
            grid_y=NUM_PATCHES_Y, grid_x=NUM_PATCHES_X,
            disp=dispersal.DISPERSAL_RATE, ldd=dispersal.LONG_DISTANCE_PROB
        )

    def _parse_cli(self, argv):
        parser = argparse.ArgumentParser(description="Ecological Simulation Runner")
        parser.add_argument('--engine', type=str, choices=['ibm', 'psd2', 'ode'], required=True)
        parser.add_argument('--pool', type=int, default=100)
        parser.add_argument('--tmax', type=int, default=20000)
        parser.add_argument('--record', type=int, default=1000)
        parser.add_argument('--tag', type=str, default="sim")
        parser.add_argument('--no-movie', action='store_true')
        parser.add_argument('--F-sat', type=float, default=10.0)
        parser.add_argument('--world-tag-extra', type=str, default="")
        parser.add_argument('--env-length-scale', type=float, default=None)
        parser.add_argument('--env-var-r', type=float, default=None)
        parser.add_argument('--env-seed-field', type=int, default=0)
        return parser.parse_args(argv)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    def _setup_paths(self):
        self.out_dir = Path("results")
        self.data_dir = self.out_dir / "data"
        self.plot_dir = self.out_dir / "plots"
        self.movie_dir = self.out_dir / "movies"
        sim_utils.ensure_dirs(self.data_dir, self.plot_dir, self.movie_dir)

    def _configure_dispersal(self):
        logging.info(f"Dispersal configured: Rate={dispersal.DISPERSAL_RATE}, LDD Prob={dispersal.LONG_DISTANCE_PROB}")

    def execute(self):
        r_final, C_final, final_state, extra_assembly = self.run_assembly()
        
        if len(r_final) == 0:
            logging.warning("Assembly resulted in 0 species. Skipping dynamics phase.")
            return

        results = self.run_dynamics(r_final, C_final, final_state)
        self.save_results(r_final, C_final, final_state, extra_assembly, results)
        if not self.args.no_movie:
            self.visualize(results)
        logging.info(f"Simulation completed in {time.time() - self.start_time:.1f}s")

    @abc.abstractmethod
    def run_assembly(self): pass

    @abc.abstractmethod
    def run_dynamics(self, r, C, initial_state): pass

    def save_results(self, r, C, final_state, extra_assembly, results):
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = self.data_dir / f"{self.world_tag}_{timestamp}.npz"
        sim_utils.atomic_save_npz(fname, r=r, C=C, **results, **extra_assembly)
        logging.info(f"Saved results to {fname}")

    def visualize(self, results):
        if 'B_dynamics' in results:
            fname = self.movie_dir / f"{self.world_tag}_dynamics.mp4"
            traj = sim_utils.to_host(results['B_dynamics'])
            animate_spatial(traj, title=f"{self.args.engine} Dynamics", filename=str(fname))

class IBMSimulation(SimulationRunner):
    def run_assembly(self):
        assembler = IBMAssembler(richness_cap=self.args.pool, F_sat=self.args.F_sat, tmax=2000, record_step=2000)
        return assembler.run()

    def run_dynamics(self, r, C, initial_state):
        logging.info("Running dynamics with IBM engine...")
        model = IBMModel(r, C, initial_N=initial_state, tmax=self.args.tmax, record_step=self.args.record, seed=1, length_scale=self.args.env_length_scale, var_r=self.args.env_var_r, seed_field=self.args.env_seed_field)
        traj = model.run()
        return {"t_dynamics": np.arange(0, self.args.tmax + self.args.record, self.args.record), "B_dynamics": traj}

class PSD2Simulation(SimulationRunner):
    def run_assembly(self):
        assembler = PSD2Assembler(richness_cap=self.args.pool, F_sat=self.args.F_sat, tmax=2000, record_step=2000)
        return assembler.run()

    def run_dynamics(self, r, C, initial_state):
        B0, W0, PC0 = initial_state
        logging.info("Running dynamics with PSD2 engine...")
        model = PSD2Model(r, C, initial_B=B0, initial_wait=W0, initial_clock=PC0, tmax=self.args.tmax, record_step=self.args.record, seed=1, length_scale=self.args.env_length_scale, var_r=self.args.env_var_r, seed_field=self.args.env_seed_field)
        t, B, W, PC, G, INV, EST = model.run()
        B_final_assembly, _, _ = initial_state
        return {"t_dynamics": t, "B_dynamics": B, "model_specific_outputs": {'PSD2_wait': W, 'PSD2_pclock': PC, 'PSD2_growth': G, 'PSD2_invasion': INV, 'PSD2_est_prob': EST, 'final_assembly_B': B_final_assembly}}

class ODESimulation(SimulationRunner):
    def run_assembly(self):
        logging.error("Assembly is not supported for the ODE engine.")
        raise NotImplementedError
    def run_dynamics(self, r, C, initial_state=None):
        logging.info("Running dynamics with ODE engine...")
        model = ODEModel(r, C, tmax=self.args.tmax, record_step=self.args.record, seed=1)
        t, y = model.run()
        return {"t_dynamics": t, "B_dynamics": y}
