############################################
# models_ode.py
############################################
"""
Pure ODE approach, solved with Assimulo.
We now incorporate multi‐patch dynamics and dispersal.
The governing equations in log-space are:
    d(logB_i)/dt = (r_i - sum_j C_ij * B_j) + (dispersal_influx_i) / B_i,
i.e. in direct form:
    dB/dt = B*(r - C@B) + dispersal_influx,
where dispersal_influx is computed from the dispersal module.
"""
import numpy as np
import logging
from config import (
    BODY_MASS,
    TMAX,
    RECORDING_STEP_SIZE,
    RTOL,
    ATOL,
    MAX_STEPS,  # if needed
    CONNECTANCE,            # ◀◀◀
    INTERACTION_STRENGTH    # ◀◀◀
)
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
from euler_simple import EulerSimple  # Import your own solver
from dispersal import compute_dispersal, LOCAL_DISPERSAL_MATRIX   # Use our dispersal module
from environment import generate_spatial_r     # ◀◀◀
# add at the very top, together with the other imports
from utils_analysis import count_invasions 


# It is recommended to remove any constant INV from config now,
# because we will compute the dispersal flux on the fly.
logger = logging.getLogger(__name__)

# For multi-patch models, we also need NUM_PATCHES_X and NUM_PATCHES_Y
# (these can be imported from config or defined elsewhere)
from config import NUM_PATCHES_X, NUM_PATCHES_Y

class ODEModel:
    """
    Pure ODE approach with multi-patch dynamics.
    We solve for y = logB, where B is of shape (S, NUM_PATCHES_Y, NUM_PATCHES_X).
    
    The ODE is:
        d(logB_ijk)/dt = [r_i - sum_j C_ij * B_j_ij] + (incoming_flux_ijk) / B_ijk,
    where the incoming_flux is computed from the dispersal module (compute_dispersal).
    """
    def __init__(self,
                 r,
                #  C,
                 C=None,               # ◀◀◀ can auto‐generate
                 r_field=None,         # ◀◀◀ spatial override
                 initial_B=None, 
                 length_scale=None,
                 var_r=None,
                 seed_field=None,
                 tmax=None,
                 record_step=None,
                 seed=123,
                 dispersal_type='propagule',
                 dispersal_away_rate=None
                ):
        
    
        np.random.seed(seed)
        # self.r = np.asarray(r, dtype=float).flatten()   # shape: (S,)
        # self.C = np.asarray(C, dtype=float)              # shape: (S,S)
        self.S = len(r)
        
        # ◀◀◀ CHANGED: competition matrix
        if C is None:
            rng = np.random.default_rng(seed_field)
            C = np.eye(self.S, dtype=float)
            for i in range(self.S):
                for j in range(self.S):
                    if i != j and rng.random() < CONNECTANCE:
                        C[i,j] = INTERACTION_STRENGTH * rng.random()
        self.C = np.asarray(C, float)

        # ◀◀◀ CHANGED: spatial r_field
        if r_field is None:
            if (length_scale is not None) and (var_r is not None):
                self.r_field = generate_spatial_r(
                    self.S, NUM_PATCHES_Y, NUM_PATCHES_X,
                    length_scale, r, var_r, seed=seed_field
                )
            else:
                self.r_field = np.broadcast_to(
                    np.asarray(r, float).reshape(self.S,1,1),
                    (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
                )
        else:
            assert r_field.shape == (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
            self.r_field = r_field

        self.r_flat = self.r_field.reshape(self.S, -1)
        
        self.tmax = tmax if tmax is not None else TMAX
        self.record_step = record_step if record_step is not None else RECORDING_STEP_SIZE
        
        # ---------- dispersal parameters ----------

        # dispersal parameters
        self.dispersal_type = dispersal_type
        if dispersal_away_rate is None:
            # sum of outgoing weights for each patch
            flat = np.asarray(LOCAL_DISPERSAL_MATRIX.sum(axis=0)).flatten()
            self.dispersal_away_rate = flat.reshape(NUM_PATCHES_Y, NUM_PATCHES_X)
        else:
            self.dispersal_away_rate = dispersal_away_rate

        # Initialize biomass in each patch.
        # Use initial biomass = BODY_MASS/10 for each species in every patch.
        # init_biomass = BODY_MASS / 10
        # # Initialize logB as a 3D array of shape (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        # self.logB = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X), np.log(init_biomass))

        if initial_B is not None:
            assert initial_B.shape == (self.S, NUM_PATCHES_Y, NUM_PATCHES_X)
            # add 1e‑100 so log() is safe even if some patches are empty
            self.logB = np.log(np.maximum(initial_B, 1e-100))
        else:
            init_biomass = BODY_MASS / 10
            self.logB = np.full((self.S, NUM_PATCHES_Y, NUM_PATCHES_X),
                                np.log(init_biomass))
        

        # Determine number of records based on tmax and record_step.
        # storage
        self.nrecords = int(self.tmax // self.record_step) + 1
        self.trajectory = np.zeros((self.nrecords,
                                    self.S,
                                    NUM_PATCHES_Y,
                                    NUM_PATCHES_X))
        self.time_points = np.zeros(self.nrecords)

    def _deriv(self, t, logB_flat, sw=None):
        """
        Compute d(logB)/dt given the current state.
        1. Reshape the 1D state vector into a 3D array of logB with shape (S, NUM_PATCHES_Y, NUM_PATCHES_X).
        2. Compute B = exp(logB).
        3. Compute local growth: for each patch, local_growth = r - C dot B (patch-wise).
        4. Compute dispersal influx using compute_dispersal(B).
        5. The derivative in log-space is: d(logB)/dt = local_growth + (dispersal_influx / B).
        6. Return a flattened array of derivatives.
        accepts the extra 'sw' argument that EulerSimple will pass in.
        We ignore it for this continuous‐time ODE.
        """
        # Reshape the flat state vector to (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        logB = logB_flat.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        np.clip(logB, -50.0, 50.0, out=logB)
        # B = np.exp(logB, dtype=float) 
        B = np.exp(logB)  # Compute biomass from logB

        # Calculate local growth for every patch:
        # For each patch, we want: local_growth_i = r_i - sum_j C_ij * B_j,
        # where B_j is the biomass of species j in that same patch.
        # We vectorize this by reshaping B to (S, num_patches)
        # local net growth -----------------------------------------------------
        # 1) local competitive growth
        B_mat = B.reshape(self.S, -1)
        # ◀◀◀ CHANGED: spatial r_flat
        growth_flat = self.r_flat - (self.C @ B_mat)# (S, patches)
        # growth_flat = self.r[:, None] - (self.C @ B_mat)  # (S, patches)
        local_growth = growth_flat.reshape(B.shape) 

        # 2) subtract adult dispersal‐away if requested
        if self.dispersal_type == 'adult':
            local_growth -= self.dispersal_away_rate

        # --- Incorporate dispersal ---
        # Compute dispersal flux from the dispersal module.
        # We are not using any constant INV but compute it on the fly.
        # The dispersal module returns a tuple: (outgoing_flux, incoming_flux),
        # both having the same shape as B.
        
        incoming_flux = compute_dispersal(B)

        # ---------- invasion pressure (cap ≃ 10 like in PSD2) ----------
        invasion_pressure = incoming_flux / (B + 1e-300)
        # invasion_pressure = np.clip(invasion_pressure, a_min=None, a_max=10.0)
        # build d(logB)/dt
        # dlogB = local_growth + incoming_flux / (B + 1e-300)
        # 
        dlogB = local_growth + invasion_pressure
        # very stiff systems can still diverge – put a final guard
        np.clip(dlogB, -20.0, 20.0, out=dlogB) 

        # Debug information (can be commented out if too verbose)
        logger.debug(f"[ODEModel _deriv] t={t:.2f}, sample logB (flattened)={logB.flatten()[:10]}")
        logger.debug(f"Local growth (sample)={local_growth.flatten()[:10]}")
        return dlogB.flatten()  # Return as 1D vector for the ODE solver

    def run(self):
        """
        Run the ODE simulation using the EulerSimple solver (a fixed step Euler scheme).
        The state is maintained as a 1D vector representing the flattened logB array for
        all species over all patches.
        """
        logger.info("Starting ODE simulation with EulerSimple (multi-patch and dispersal)...")

        # try:
        #     from assimulo.solvers import CVode
        #     HAVE_CVODE = True
        # except Exception:
        #     HAVE_CVODE = False
        
        # # Flatten the initial state (logB) to a 1D vector.
        y0 = self.logB.flatten()
        sw0 = np.zeros_like(y0)   # dummy, because _deriv ignores it
        problem = Explicit_Problem(self._deriv, y0, 0.0, sw0=sw0)
        problem.name = 'ODEModel' 
        solver = EulerSimple(problem)
        # Set solver options:
        solver.options['inith'] = 1
        solver.options['maxsteps'] = 10000000  # large enough for tmax=2000
        solver.store_event_points = False

        # ── choose solver ───────────────────────────────────────────────
        # if HAVE_CVODE:
        #     solver = CVode(problem)           # adaptive, stiff
        #     solver.discr          = 'BDF'
        #     solver.iter           = 'Newton'
        #     solver.linear_solver  = 'SPGMR'
        #     solver.rtol           = RTOL
        #     solver.atol           = ATOL
        #     solver.store_event_points = False
        #     solver.options['maxsteps'] = MAX_STEPS
        # else:                                 # fixed step fallback
        #     solver = EulerSimple(problem)
        #     solver.options['inith']    = 0.1   # *** much smaller step ***
        #     solver.options['maxsteps'] = int(self.tmax/0.1)+10
        #     solver.store_event_points = False

        # Define the record times based on tmax and record_step.
        times = np.arange(0, self.tmax + self.record_step, self.record_step, dtype=float)

        # Record the initial biomass trajectory.
        # self.trajectory[0, :, :, :] = np.exp(y0.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X)))
        # Record the initial biomass trajectory (clipped to avoid overflow). new
        init_logB = y0.reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
        np.clip(init_logB, -50.0, 50.0, out=init_logB)
        self.trajectory[0, ...] = np.exp(init_logB)
        self.time_points[0] = 0.0
        self.record_idx = 1

        # Now, run the integration using EulerSimple.
        # EulerSimple is called like a function: solver(t_final, number_of_records)
        t, y = solver(self.tmax, len(times) - 1)
        
        total_elements = self.S * NUM_PATCHES_Y * NUM_PATCHES_X
        
        # Loop over the recorded times and populate the trajectory and time_points.
        for step in range(len(times)):
            # Find the index in t closest to the record time.
            rec_idx = np.argmin(np.abs(t - times[step]))
            y_rec = y[rec_idx, :]
            logB = y_rec[:total_elements].reshape((self.S, NUM_PATCHES_Y, NUM_PATCHES_X))
            # (If your model uses additional state, this part might include pclock, etc.)
            np.clip(logB, -50.0, 50.0, out=logB)
            B = np.exp(logB)
            self.trajectory[step, :, :, :] = B
            self.time_points[step] = t[rec_idx]
            logger.info(f"ODE progress: t={t[rec_idx]:.2f}, record {step+1} of {len(times)}.")
            if t[rec_idx] >= self.tmax:
                break

        logger.info("ODE simulation completed.")
        # Return the recorded time points and trajectory (up to record_idx).
        return self.time_points[:step+1], self.trajectory[:step+1, :]


############################################
# Testing ODE Model – Updated Test Code
############################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def test_ode_model():
        """
        Updated test for the ODEModel that:
          - Computes the analytical equilibrium biomass per species.
          - Runs the ODEModel simulation with multipatch and dispersal.
          - Computes the time series (mean and variance) of biomass across patches.
          - Plots the time series for one species.
          - Prints final mean biomass and relative error from equilibrium.
        """
        np.random.seed(42)
        
        # Define test parameters: number of species, growth rates and competition matrix.
        S = 3  # number of species
        tmax = 2000      # total simulation time
        
        # r = np.array([0.8, 0.6, 0.7])
        # C = np.array([
        #     [0.2, 0.1, 0.1],
        #     [0.1, 0.2, 0.1],
        #     [0.1, 0.1, 0.2]
        # ])
        r = np.array([1.0, 1.0, 1.0])
        C = np.array([
            [1.0, 1.7, 0.4],
            [0.4, 1.0, 1.7],
            [1.7, 0.4, 1.0]
        ])
        
        # Calculate analytical equilibrium biomass (B_eq = inv(C) * r)
        print("\nAnalytical Equilibrium Analysis:")
        try:
            C_inv = np.linalg.inv(C)
            B_eq = C_inv @ r
            print(f"Analytical equilibrium biomass (per species): {B_eq}")
        except np.linalg.LinAlgError:
            print("Warning: Competition matrix is not invertible")
            B_eq = None
        # Define simulation parameters.
        ode_model = ODEModel(r, C, tmax=tmax, record_step=10, seed=42)
        
        # Run the simulation. The trajectory returned has shape:
        # (n_records, S, NUM_PATCHES_Y, NUM_PATCHES_X)
        time_points, trajectory = ode_model.run()
        nrecords = trajectory.shape[0]
        print(f"\nRecorded {nrecords} time steps.")
        
        # Compute the time series: average and variance over all patches
        mean_time_series = np.mean(trajectory, axis=(2, 3))  # shape: (nrecords, S)
        var_time_series  = np.var(trajectory, axis=(2, 3))    # shape: (nrecords, S)
        
        # Plot the time series for the first species (Species 0)
        species = 0
        plt.figure(figsize=(8, 5))
        plt.plot(time_points, mean_time_series[:, species], label=f'Mean Biomass (Species {species})')
        std_dev = np.sqrt(var_time_series[:, species])
        plt.fill_between(time_points,
                         mean_time_series[:, species] - std_dev,
                         mean_time_series[:, species] + std_dev,
                         color='blue', alpha=0.2, label='Std. Dev.')
        plt.xlabel("Time")
        plt.ylabel("Biomass")
        plt.title(f"ODE Model: Mean Biomass and Std. Dev. (Species {species})")
        plt.legend()
        plt.show()
        
        # Compute the final biomass per species by averaging over the patches
        final_biomass = trajectory[-1, :, :, :]  # shape: (S, NUM_PATCHES_Y, NUM_PATCHES_X)
        mean_final = np.mean(final_biomass, axis=(1, 2))  # one value per species
        print("\nFinal Mean Biomass for each species:", mean_final)
        
        if B_eq is not None:
            rel_error = (mean_final - B_eq) / B_eq
            print("Relative error from analytical equilibrium:", rel_error)
    
    test_ode_model()






