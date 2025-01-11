
import os
import numpy as np
import scipy.io as sio
import time
from pathlib import Path
from scipy.integrate import solve_ivp
from collections.abc import Iterable
import time
import logging
import traceback
import signal
from scipy.io import savemat  # For saving .mat files
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import copy

from assimulo.solvers import CVode
print("Assimulo and CVode imported successfully.")

from topography import Topography
from species import Species
from communitydynamics import CommunityDynamics

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Metacommunity:
    def __init__(self, spp, a_init,a_bMat, a_xMat, a_scMat, a_invMax, a_tMax, a_outputDirectory,
                 a_c1, a_c2, a_c3, a_emRate, a_dispL, a_pProducer, a_prodComp, a_symComp,
                 a_alpha, a_sigma, a_sigma_t, a_rho, a_comp_dist, a_omega, a_dispNorm,
                 a_no_nodes, a_lattice_height, a_lattice_width, a_phi, a_envVar, a_skVec,
                 a_var_e, a_randGraph, a_gabriel, a_T_int, a_envMat, a_parOut=0, a_experiment="default", a_rep=0, storeTraj=0, bMatFileName="default_bMat.mat", g_block_transitions=False, g_form_of_dynamics="default_value",simTime = 0.0):
        if a_init:
            self.invMax = a_invMax
            self.tMax = a_tMax
            self.g_form_of_dynamics = g_form_of_dynamics
            self.simTime = simTime  # Initialize simTime
            self.parOut = a_parOut
            self.storeTraj = storeTraj
            self.experiment = a_experiment
            self.rep = a_rep
            self.date = time.strftime('%Y-%m-%d', time.localtime())
            self.outputDirectory = a_outputDirectory
            self.bMatFileName = bMatFileName
            self.g_block_transitions = g_block_transitions

            if not isinstance(a_envVar, int):
                logger.warning(f"Converting `a_envVar` from {a_envVar} to integer.")
                a_envVar = int(a_envVar)

            # Initialize Topography
            topo = Topography(
                no_nodes=a_no_nodes,
                lattice_height=a_lattice_height,
                lattice_width=a_lattice_width,
                phi=a_phi,
                envVar=a_envVar,
                skVec=a_skVec,
                var_e=a_var_e,
                randGraph=a_randGraph,
                gabriel=a_gabriel,
                T_int=a_T_int,
                network_file="",
                scVec=a_skVec if a_skVec is not None else np.ones(a_no_nodes)
            )
            # Log the success of Topography creation
            logger.debug(f"Topography created with {topo.no_nodes} nodes.")

            # Pass only relevant arguments to Species constructor
            self.spp = Species(
                topo=topo,
                c1=a_c1,
                c2=a_c2,
                c3=a_c3,
                emRate=a_emRate,
                dispL=a_dispL,
                pProducer=a_pProducer,
                prodComp=a_prodComp,
                symComp=a_symComp,
                alpha=a_alpha,
                sigma=a_sigma,
                sigma_t=a_sigma_t,
                rho=a_rho,
                comp_dist=a_comp_dist,
                omega=a_omega,
                dispNorm=a_dispNorm
            )

            # Log that Species has been successfully initialized
            logger.debug("Species initialized successfully.")


            if self.spp.dMat is None or self.spp.dMat.size == 0:
                logger.error("Dispersal matrix (`dMat`) must be properly initialized and non-empty in Species.")
                raise ValueError("Dispersal matrix must be properly initialized and non-empty in Species.")
            self.dMat = self.spp.dMat
            logger.debug(f"dMat initialized with shape: {self.dMat.shape}")

        else:
            self.loadMC(a_bMat)
            self.spp.topo.scMatFileName = a_scMat
            self.outputDirectory = a_outputDirectory
            if a_invMax != 0:
                self.invMax += a_invMax
                print(f"\ninvMax updated to {self.invMax}")
            if a_tMax != self.tMax:
                self.tMax = a_tMax
                print(f"\ntMax updated to {self.tMax}")



    def meta_c_dynamics(self, T):
        """
        Numerical solver for metacommunity dynamics.
        :param T: Relaxation time
        """

        # Ensure dMat is properly initialized
        if self.spp.dMat is None or self.spp.dMat.size == 0:
            raise ValueError("The dispersal matrix (`dMat`) must be properly initialized before starting the dynamics.")

        # Identify nonzero biomass indices
        indices_S = np.nonzero(self.spp.xMat > 0)[1]  # Non-zero biomass indices
        indices_DP = np.arange(self.spp.xMat.size)  # All indices for dynamic populations

        # Initialize CommunityDynamics object
        dynamics = CommunityDynamics(
            xMat=self.spp.xMat, spp=self.spp, cMat=self.spp.cMat, rMat=self.spp.rMat,
            indices_S=indices_S, indices_DP=indices_DP, dMat=self.spp.dMat
        )

        # Validate dispersal matrix dimensions
        if dynamics.dMat_sp is None or dynamics.dMat_sp.size == 0:
            raise ValueError("The dispersal matrix (`dMat_sp`) in `CommunityDynamics` must be properly initialized.")

        # Handle scaling vectors consistency
        if self.spp.topo.scVec.shape[0] != dynamics.xMat.shape[1]:
            logger.warning("Resizing scVec to match xMat columns.")
            dynamics.scVec = np.resize(self.spp.topo.scVec, dynamics.xMat.shape[1])

        if self.spp.scVec_prime is not None:
            if self.spp.scVec_prime.ndim == 1:
                self.spp.scVec_prime = self.spp.scVec_prime.reshape(1, -1)


        logger.info("Starting dynamics with properly initialized matrices.")

        # Handling file paths for storing trajectories
        Bpath, bMatDir, newFolder = "", "", ""
        if self.storeTraj != 0:
            bMatDir = os.path.dirname(self.bMatFileName)
            newFolder = os.path.splitext(os.path.basename(self.bMatFileName))[0]
            Bpath = os.path.join(bMatDir, newFolder, "trajectory")

            if not os.path.exists(Bpath):
                os.makedirs(Bpath)
            print(f"\nSaving matrices to {Bpath}")

        # Metacommunity relaxation step
        previous_state = self.spp.xMat.copy()

        # Define the right-hand side function for ODE solver
        def rhs(t, y):
            """
            Right-hand side function for the ODE solver.
            Handles shape mismatches, broadcasting, and undefined dimensions robustly.
            :param t: Current time (required by solver)
            :param y: State vector, flattened
            :return: Time derivative of the state vector, flattened
            """
            try:
                # Reshape the input state vector 'y' into the correct shape of dynamics.xMat
                state = y.reshape(dynamics.xMat.shape)  # state.shape = (N_species, N_nodes)

                # Step 1: Compute intrinsic growth rates
                Gt = dynamics.compute_intrinsic_growth_rates(state)
                dXdt = Gt.copy()

                # Apply dispersal only to dynamic indices
                dXdt.flat[dynamics.indices_DP] *= state.flat[dynamics.indices_DP]

                # Step 2: Handle dispersal and emMat operations
                if dynamics.dMat_sp is None or dynamics.dMat_sp.size == 0:
                    raise ValueError("The dispersal matrix (dMat_sp) is not initialized or empty.")

                if dynamics.emMat is None or dynamics.emMat.size == 0:
                    # Normal dispersal without emMat
                    massEffect = state @ dynamics.dMat_sp
                else:
                    # Validate emMat dimensions
                    if dynamics.emMat.ndim == 1:
                        # Reshape 1D emMat to match state (N_species, 1)
                        emMat_N = dynamics.emMat[:, np.newaxis]
                    elif dynamics.emMat.ndim == 2 and dynamics.emMat.shape[0] == state.shape[0]:
                        # Broadcast emMat to match the shape of state (N_species, N_nodes)
                        if dynamics.emMat.shape[1] == 1:
                            emMat_N = np.tile(dynamics.emMat, (1, state.shape[1]))
                        elif dynamics.emMat.shape == state.shape:
                            emMat_N = dynamics.emMat
                        else:
                            raise ValueError(
                                f"Incompatible emMat shape {dynamics.emMat.shape} for state shape {state.shape}."
                            )
                    else:
                        raise ValueError(
                            f"Unsupported emMat dimensions {dynamics.emMat.shape}. Expected (N_species, 1) or (N_species, N_nodes)."
                        )

                    # Compute mass effect with emMat and ensure broadcasting works correctly
                    massEffect = (emMat_N * state) @ dynamics.dMat_sp

                # Step 3: Add dispersal effect to the derivatives
                dXdt += massEffect

                # Step 4: Handle edge cases: non-negativity, finite values, and extreme values
                dXdt[state < 0] = -state[state < 0]
                dXdt[~np.isfinite(dXdt)] = 0
                dXdt[np.abs(dXdt) > 1e10] = 0

                # Return flattened time derivative
                return dXdt.flatten()

            except ValueError as e:
                logger.error(f"ValueError in RHS function: {e}")
                print(f"Error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in RHS function: {e}")
                print(f"Unexpected error: {e}")
                raise


        # Initialize the solver
        initial_state = self.spp.xMat.flatten()
        problem = Explicit_Problem(rhs, initial_state, 0)
        solver = CVode(problem)
        solver.atol = 1e-5
        solver.rtol = 1e-5
        solver.maxh = 0.1

        # Simulation parameters
        previous_state = self.spp.xMat.copy()
        convergence_threshold = 1e-9
        max_iterations = int(T * 1000)
        iterations = 0

        while dynamics.current_time < T:
            solver.simulate(dynamics.current_time + 1.0)

            try:
                if solver.y.ndim == 1:
                    current_state = solver.y.reshape(self.spp.xMat.shape)
                else:
                    current_state = solver.y[-1].reshape(self.spp.xMat.shape)
            except ValueError as e:
                logger.error(f"Error reshaping solver output: {e}")
                raise

            # Compute state change
            state_change = np.linalg.norm(current_state - previous_state)
            previous_state = current_state.copy()

            # Check for equilibrium
            if state_change < convergence_threshold:
                logger.info(f"System reached equilibrium at time {solver.t:.2f} with change {state_change:.2e}.")
                return

            iterations += 1
            if iterations >= max_iterations:
                logger.warning("Maximum iterations reached without convergence.")
                return

        logger.info("Simulation completed without reaching equilibrium.")
        
     
    def remove_species(self, indices):
        """
        Helper function to remove species from all matrices.

        Parameters:
        indices (list[int]): List of species indices to remove.
        """
        logger.debug(f"Removing species at indices: {indices}")

        # Sort indices in descending order to avoid shifting issues during deletion
        indices = sorted(indices, reverse=True)

        for idx in indices:
            if idx >= self.spp.xMat.shape[0]:
                logger.warning(f"Index {idx} is out of bounds for xMat with shape {self.spp.xMat.shape}. Skipping.")
                continue

            # Update producer and consumer counts in spp
            if idx < self.spp.S_p:  # Producer index
                self.spp.S_p -= 1
            else:
                self.spp.S_c -= 1

            # Remove species from core matrices
            self.spp.xMat = np.delete(self.spp.xMat, idx, axis=0)
            if self.spp.cMat is not None and self.spp.cMat.shape[0] > 0:
                self.spp.cMat = np.delete(self.spp.cMat, idx, axis=0)
                self.spp.cMat = np.delete(self.spp.cMat, idx, axis=1)
            if self.spp.rMat is not None and self.spp.rMat.shape[0] > 0:
                self.spp.rMat = np.delete(self.spp.rMat, idx, axis=0)

            # Remove species from optional matrices
            optional_matrices = {
                'sMat': self.spp.sMat,
                'tMat': self.spp.tMat,
                'emMat': self.spp.emMat,
                'efMat': self.spp.efMat,
                'ouMat': self.spp.ouMat,
                'trajectories': self.spp.trajectories,
                'fluctuations': self.spp.fluctuations,
                'scVec': self.spp.scVec,
                'scVec_prime': self.spp.scVec_prime
            }

            for mat_name, mat in optional_matrices.items():
                if mat is not None and mat.size > 0 and idx < mat.shape[0]:
                    new_mat = np.delete(mat, idx, axis=0)
                    setattr(self.spp, mat_name, new_mat)

        logger.debug(f"Species removal complete. Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")
    
    
    def invader_sample(self, trophLev, no_invaders):
        """
        Introduce new random species and test for positive growth rates.
        """
        min_b = 1e-6
        invExcess = 3  # Excess invaders to ensure success
        suc_inv = 0

        while suc_inv < no_invaders:
            logger.debug(f"Attempting to invade. Successful so far: {suc_inv}/{no_invaders}")

            for _ in range(invExcess * no_invaders):
                self.spp.invade(trophLev)
                # Synchronize optional matrices
                # Ensure additional matrices (e.g., efMat, emMat) are updated
                for mat_name in ['efMat', 'emMat']:
                    mat = getattr(self.spp, mat_name, None)
                    if mat is not None and mat.size > 0:
                        setattr(self.spp, mat_name, np.vstack([mat, np.zeros((1, mat.shape[1]))]))
                    elif mat is not None:
                        setattr(self.spp, mat_name, np.zeros((self.spp.xMat.shape[0], self.spp.xMat.shape[1])))
                    else:
                        logger.warning(f"{mat_name} is not initialized. Skipping update.")

            # Determine which species have positive growth rates
            if trophLev == 1:
                B = np.maximum(0, self.spp.xMat[:self.spp.S_p])
                indices = np.arange(self.spp.S_p, self.spp.xMat.shape[0])
            else:
                B = np.maximum(0, self.spp.xMat)
                indices = np.arange(self.spp.S_p, self.spp.xMat.shape[0])

            try:
                if trophLev == 0:  # Producers
                    bInv = self.spp.rMat[self.spp.S_p:] - (self.spp.cMat[self.spp.S_p:, :] @ B)
                else:  # Consumers
                    bInv = self.spp.rho * (self.spp.cMat[indices, :self.spp.S_p] @ B - 1)
            except ValueError as e:
                logger.error(f"Error in growth rate calculation: {e}")
                raise

            logger.debug(f"Calculated growth rates: {bInv}")

            if bInv.size == 0 or np.all(bInv <= 0):
                logger.warning("No positive growth rates found. Adjusting parameters may be necessary.")
                break

            # bInv corresponds to species starting at self.spp.S_p for producers or 'indices' for consumers
            bInv_max = bInv.max(axis=1) if len(bInv.shape) > 1 else bInv
            # posGrowth_local are indices into bInv array
            posGrowth_local = np.where(bInv_max >= min_b)[0][: no_invaders - suc_inv]
            
            # Convert posGrowth_local into absolute species indices
            posGrowth_abs = indices[posGrowth_local]
            negGrowth = np.setdiff1d(indices, posGrowth_abs)

            logger.debug(f"Positive Growth Indices (absolute): {posGrowth_abs}, Negative Growth Indices: {negGrowth}")

            suc_inv += len(posGrowth_abs)
            for idx in reversed(negGrowth):
                self.remove_species([idx])

            logger.debug(f"State after removal - Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")

        # Update species counts based on final xMat shape
        if trophLev == 0:
            self.spp.S_p = self.spp.xMat.shape[0] - self.spp.S_c
        else:
            self.spp.S_c = self.spp.xMat.shape[0] - self.spp.S_p

        logger.info(f"Invader sampling complete. Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")
    
    


    def env_fluct(self):
        """
        Update abiotic turnover and simulate a single CVode timestep.
        """
        # Initialize CommunityDynamics object
        dynamics = CommunityDynamics(
            self.spp,
            self.spp.xMat,
            self.spp.cMat,
            self.spp.rMat,
            efMat=self.spp.efMat,
            scVec=self.spp.scVec,
            dMat=self.spp.dMat,
        )

        dynamics.xMat = self.spp.xMat
        dynamics.rMat = self.spp.rMat

        if self.spp.cMat.shape[0] != 0:
            dynamics.cMat = self.spp.cMat

        dynamics.dMat = self.spp.dMat
        dynamics.rho = self.spp.rho

        self.spp.ou_process()  # Update OU process and efMat
        dynamics.efMat = self.spp.efMat

        def rhs(t, y):
            """
            Right-hand side function for the solver.
            Ensures dimensions are consistent and operations are safe.
            """
            try:
                state = y.reshape(self.spp.xMat.shape)  # Reshape flattened state to (species, nodes)

                # Step 1: Intrinsic growth rates
                Gt = dynamics.compute_intrinsic_growth_rates(state)
                dXdt = Gt.copy()

                # Step 2: Mass effect and dispersal with emMat validation
                if dynamics.emMat is None:
                    if dynamics.dMat_sp.shape[1] != state.shape[1]:
                        raise ValueError(
                            f"Incompatible dMat_sp shape {dynamics.dMat_sp.shape} for state shape {state.shape}."
                        )
                    massEffect = state @ dynamics.dMat_sp
                else:
                    emMat = dynamics.emMat
                    if emMat.shape[0] != state.shape[0] or emMat.shape[1] not in [1, state.shape[1]]:
                        raise ValueError(
                            f"emMat shape {emMat.shape} incompatible with state shape {state.shape}."
                        )
                    if emMat.shape[1] == 1:
                        emMat = np.tile(emMat, (1, state.shape[1]))  # Broadcast emMat to match state
                    massEffect = (emMat * state) @ dynamics.dMat_sp

                dXdt += massEffect

                # Handle edge cases
                dXdt[state < 0] = -state[state < 0]  # Fix negative values
                dXdt[~np.isfinite(dXdt)] = 0  # Replace non-finite values with 0
                dXdt[np.abs(dXdt) > 1e10] = 0  # Cap extreme values

                return dXdt.flatten()

            except Exception as e:
                logger.error(f"Error in RHS function: {e}")
                raise ValueError(f"Error in RHS function: {e}")

        # Initialize solver with the problem definition
        initial_state = self.spp.xMat.flatten()
        problem = Explicit_Problem(rhs, initial_state, 0)
        solver = CVode(problem)
        solver.atol = 1e-5
        solver.rtol = 1e-5
        solver.maxh = 0.1

        # Run solver for one timestep and validate output
        solver.simulate(1)

        if solver.y.ndim == 1:
            # Handle unexpected scalar outputs or incorrect solver output size
            if solver.y.size != self.spp.xMat.size:
                raise ValueError(
                    f"Solver output size {solver.y.size} does not match xMat size {self.spp.xMat.size}."
                )
            tmp = solver.y.reshape(self.spp.xMat.shape)
        else:
            tmp = solver.y[-1].reshape(self.spp.xMat.shape)

        # Ensure trajectories match dimensions before stacking
        if self.spp.trajectories.shape[1] != tmp.shape[1]:
            logger.warning(
                f"Shape mismatch: trajectories has {self.spp.trajectories.shape[1]} columns, "
                f"but tmp has {tmp.shape[1]} columns. Resetting trajectories."
            )
            self.spp.trajectories = np.zeros((0, tmp.shape[1]))

        self.spp.trajectories = np.vstack((self.spp.trajectories, tmp))  # Store trajectory

        # Compute RE matrix (rMat + efMat) and validate dimensions
        RE = self.spp.rMat + self.spp.efMat
        RE = RE.reshape(1, -1)

        if self.spp.fluctuations.shape[1] != RE.shape[1]:
            logger.warning(
                f"Fluctuation shape mismatch: fluctuations has {self.spp.fluctuations.shape[1]} columns, "
                f"RE has {RE.shape[1]} columns. Resetting fluctuations."
            )
            self.spp.fluctuations = np.zeros((0, RE.shape[1]))

        self.spp.fluctuations = np.vstack((self.spp.fluctuations, RE))  # Store current fluctuations
           

    def warming(self, dTdt, res, time):
        """
        Updates temperature gradient and rMat, then simulates a single timestep.

        Args:
            dTdt (float): Rate of temperature increase.
            res (int): Number of steps per unit time.
            time (int): Total time for warming simulation.
        """

        # Define path to save outputs
        pos1 = self.bMatFileName.rfind("/")
        bMatDir = self.bMatFileName[:pos1]
        pos1 = self.bMatFileName.rfind(")")
        pos2 = self.bMatFileName.rfind(".")
        newFolder = self.bMatFileName[pos1 + 1:pos2]
        Bpath = os.path.join(bMatDir, newFolder, f"dTdt={dTdt}")

        if not os.path.exists(Bpath):
            os.makedirs(Bpath)

        # Initial dimensions based on rMat
        species_count, node_count = self.spp.rMat.shape

        # Ensure all matrices align with these dimensions.
        # We assume dimensions remain constant throughout the simulation.
        # If they must change, that requires a different approach.
        if self.spp.xMat.shape != (species_count, node_count):
            logger.warning(f"Aligning xMat to ({species_count},{node_count})")
            self.spp.xMat = np.zeros((species_count, node_count))
        if self.spp.efMat.shape != (species_count, node_count):
            logger.warning("Aligning efMat to rMat dimensions.")
            self.spp.efMat = np.zeros((species_count, node_count))
        if self.spp.cMat.shape != (species_count, species_count):
            logger.warning("Aligning cMat to square shape.")
            self.spp.cMat = np.zeros((species_count, species_count))
        # Ensure scVec_prime is 2D and matches xMat columns if CommunityDynamics expects that
        if self.spp.scVec_prime.shape != (species_count, node_count):
            logger.warning(f"Aligning scVec_prime to ({species_count},{node_count})")
            self.spp.scVec_prime = np.zeros((species_count, node_count))

        # Initialize CommunityDynamics with stable, consistent dimensions
        dynamics = CommunityDynamics(
            self.spp,
            self.spp.xMat,
            self.spp.cMat,
            self.spp.rMat,
            efMat=self.spp.efMat,
            scVec=self.spp.scVec,
            scVec_prime=self.spp.scVec_prime,
            dMat=self.spp.dMat,
            bodymass=self.spp.bodymass,
            mu=self.spp.mu
        )

        def rhs(t, y):
            """
            Right-hand side function for the solver. Ensures safe reshaping.
            """
            try:
                # Reshape solver state to match xMat shape
                state = y.reshape(self.spp.xMat.shape)

                # Compute intrinsic growth rates
                Gt = dynamics.compute_intrinsic_growth_rates(state)
                dXdt = Gt.copy()

                # Mass effect and dispersal
                if dynamics.emMat is not None:
                    emMat_N = np.tile(dynamics.emMat[:, np.newaxis], (1, state.shape[1]))
                    if emMat_N.shape != state.shape:
                        logger.warning(f"Reshaping emMat to match state shape: {state.shape}")
                        # Replace emMat_N with zeros to avoid broadcasting errors
                        emMat_N = np.zeros_like(state)
                    massEffect = (emMat_N * state) @ dynamics.dMat_sp
                else:
                    massEffect = state @ dynamics.dMat_sp

                dXdt += massEffect

                # Handle edge cases: negative, infinite, or excessively large values
                dXdt[state < 0] = -state[state < 0]
                dXdt[~np.isfinite(dXdt)] = 0
                dXdt[np.abs(dXdt) > 1e10] = 0

                return dXdt.flatten()
            except Exception as e:
                logger.error(f"Error in RHS function: {e}")
                raise ValueError(f"Error in RHS function: {e}")

        # Simulation loop
        for t in range(res):
            try:
                self.spp.topo.T_int += dTdt
                self.spp.update_r_vec_temp()

                # Check if dimensions changed after temperature update:
                new_species_count, new_node_count = self.spp.rMat.shape
                if (new_species_count != species_count) or (new_node_count != node_count):
                    raise ValueError(
                        f"Dimensions of rMat changed from ({species_count},{node_count}) to ({new_species_count},{new_node_count}). "
                        "Dimensions must remain constant during simulation."
                    )

                # Reassign rMat to dynamics
                dynamics.rMat = self.spp.rMat

                # No further reshaping should be needed if dimensions are stable
                # Initialize and solve
                problem = Explicit_Problem(rhs, self.spp.xMat.flatten(), 0)
                solver = CVode(problem)
                solver.simulate(1)

                # Validate solver output
                if solver.y.size != (species_count * node_count):
                    raise ValueError(
                        f"Solver output size mismatch: expected {species_count * node_count}, got {solver.y.size}."
                    )
                self.spp.xMat = solver.y.reshape((species_count, node_count))

            except Exception as e:
                logger.error(f"Error during simulation step {t}: {e}")
                raise

        # Save outputs
        Bfile = os.path.join(Bpath, f"bMat_w{time}.mat")
        Rfile = os.path.join(Bpath, f"rMat_w{time}.mat")
        np.savetxt(Bfile, self.spp.xMat, fmt='%e')
        np.savetxt(Rfile, self.spp.rMat, fmt='%e')

        if time == 0:
            filenameP = os.path.join(Bpath, "pars.mat")
            self.write_params(filenameP)




    def long_dist_disp(self, tMax, edges):
        """
        Adds long-distance dispersal links to the metacommunity.
        """
        stepwise = True if edges >= 0 else False
        row_min = np.min(np.abs(self.spp.dMat), axis=1)

        non_zero_d = np.count_nonzero(self.spp.dMat)
        zero_d = (self.spp.dMat.size - non_zero_d) // 2
        pert_record = np.zeros((zero_d, 3))

        if not stepwise:
            edges = zero_d // 1
            print(f"Edges reset to {edges}")

        pos1 = self.bMatFileName.rfind("/")
        bMatDir = self.bMatFileName[:pos1]
        pos1 = self.bMatFileName.rfind(")")
        pos2 = self.bMatFileName.rfind(".")
        newFolder = self.bMatFileName[pos1 + 1:pos2]

        if stepwise:
            Bpath = os.path.join(bMatDir, newFolder, f"dkdt={edges}")
        else:
            Bpath = os.path.join(bMatDir, newFolder, f"delta_e={edges}")

        if not os.path.exists(Bpath):
            os.makedirs(Bpath)

        t_interval = 1
        t_step = 0
        edge_count = 0

        while not np.min(row_min):
            if t_step >= pert_record.shape[0]:
                # Dynamically resize pert_record if t_step exceeds its current size
                new_size = pert_record.shape[0] + 100  # Add additional rows
                pert_record = np.vstack((pert_record, np.zeros((100, 3))))

            node = np.where(row_min == 0)[0]
            np.random.shuffle(node)
            edge = np.where(self.spp.dMat[node[0]] == 0)[0]
            np.random.shuffle(edge)

            self.spp.topo.adjMat[node[0], edge[0]] = -1.0
            self.spp.topo.adjMat[edge[0], node[0]] = -1.0
            self.spp.gen_disp_mat()

            pert_record[t_step, 0] = node[0]
            pert_record[t_step, 1] = edge[0]
            pert_record[t_step, 2] = t_interval
            t_step += 1

            row_min = np.min(np.abs(self.spp.dMat), axis=1)
            edge_count += 1

            print(f"Edge count {edge_count}")

            if edge_count == edges:
                if stepwise:
                    print(f"{edges * (t_interval + 1)} new edges allocated")
                    self.meta_c_dynamics(tMax)
                    Bfile = os.path.join(Bpath, f"bMat_d{t_interval}.mat")
                    np.savetxt(Bfile, self.spp.xMat, fmt='%e')
                    filenameRR = os.path.join(Bpath, "pert_record.mat")
                    np.savetxt(filenameRR, pert_record[:t_step], fmt='%e')
                    t_interval += 1
                    edge_count = 0

                    if t_interval > 100:
                        row_min = np.ones_like(row_min)
                else:
                    print("Single connectivity perturbation")
                    pert_record = pert_record[:t_step]
                    dynamics = CommunityDynamics(self.spp, self.spp.xMat, self.spp.cMat, self.spp.rMat, dMat=self.spp.dMat, rho=self.spp.rho)

                    def rhs(t, y):
                        state = y.reshape(self.spp.xMat.shape)
                        Gt = dynamics.compute_intrinsic_growth_rates(state)
                        dXdt = Gt.copy()
                        dXdt.flat[dynamics.indices_DP] *= state.flat[dynamics.indices_DP]

                        # Handle emigration and mass effect calculations
                        if dynamics.emMat is None:
                            massEffect = state @ dynamics.dMat_sp
                            dXdt += massEffect
                        else:
                            emMat_N = np.repeat(dynamics.emMat[:, np.newaxis], state.shape[1], axis=1)
                            massEffect = (emMat_N * state) @ dynamics.dMat_sp
                            dXdt += massEffect

                        # Handle negative biomass values
                        negativeB = np.where(state < 0)
                        if len(negativeB[0]) > 0:
                            dXdt[negativeB] = -state[negativeB]

                        # Replace non-finite values
                        nonfiniteX = np.where(~np.isfinite(dXdt))
                        if len(nonfiniteX[0]) > 0:
                            dXdt[nonfiniteX] = 0

                        # Check for blow-up values in dXdt
                        blowupX = np.where(np.abs(dXdt) > 1e10)
                        if len(blowupX[0]) > 0:
                            dXdt[blowupX] = 0

                        return dXdt.flatten()

                    problem = Explicit_Problem(rhs, self.spp.xMat.flatten(), 0)
                    solver = CVode(problem)

                    for t in range(tMax):
                        print(f"\rt = {t}", end="")
                        solver.simulate(t)
                        if t > 0 and t % 1 == 0:
                            Bfile = os.path.join(Bpath, f"bMat_d{t}.mat")
                            np.savetxt(Bfile, self.spp.xMat, fmt='%e')
                    row_min = np.ones_like(row_min)

        if edge_count:
            self.meta_c_dynamics(tMax)
            Bfile = os.path.join(Bpath, f"bMat_d{t_interval}.mat")
            np.savetxt(Bfile, self.spp.xMat, fmt='%e')
            filenameRR = os.path.join(Bpath, "pert_record.mat")
            np.savetxt(filenameRR, pert_record[:t_step], fmt='%e')



    def node_removal(self, tMax, no_removals):
        """
        Removes nodes from the network and simulates the dynamics.
        """
        self.storeTraj = 3
        if self.spp.topo.scVec.shape[0] > 0 and self.spp.topo.scVec_prime.shape[0] > 0:
            print("\nLocal interaction matrices scaled")

        pos1 = self.bMatFileName.rfind("/")
        bMatDir = self.bMatFileName[:pos1]
        pos1 = self.bMatFileName.rfind(")")
        pos2 = self.bMatFileName.rfind(".")
        newFolder = self.bMatFileName[pos1 + 1:pos2]
        Bpath = os.path.join(bMatDir, newFolder, "nodeRemoval")

        if not os.path.exists(Bpath):
            os.makedirs(Bpath)

        # Initial dimensions
        init_species_count, init_node_count = self.spp.xMat.shape

        # A helper function to realign matrices after node removal
        def align_dimensions(species_count, node_count):
            # Ensure cMat is still square
            if self.spp.cMat.shape != (species_count, species_count):
                logger.warning("Realigning cMat to square shape after node removal.")
                self.spp.cMat = np.zeros((species_count, species_count))
            # Ensure efMat matches rMat
            if self.spp.efMat.shape != (species_count, node_count):
                logger.warning("Realigning efMat shape to match rMat after node removal.")
                self.spp.efMat = np.zeros((species_count, node_count))
            # Ensure scVec_prime matches current dimensions
            if self.spp.scVec_prime.shape != (species_count, node_count):
                logger.warning(f"Realigning scVec_prime to ({species_count},{node_count}) after node removal.")
                self.spp.scVec_prime = np.zeros((species_count, node_count))

        node_id = np.linspace(no_removals, self.spp.topo.no_nodes - 1, self.spp.topo.no_nodes - no_removals, dtype=int)
        spp_copy = copy.deepcopy(self.spp)  # Use deepcopy to preserve original spp object

        for x in node_id:
            if x >= self.spp.xMat.shape[1]:
                print(f"Skipping node {x}: Index out of bounds for xMat with size {self.spp.xMat.shape[1]}")
                continue

            print(f"\nNode {x}")
            self.spp = copy.deepcopy(spp_copy)  # Restore original spp for each iteration

            # Remove node data (columns) from matrices and vectors (since nodes are columns)
            self.spp.xMat = np.delete(self.spp.xMat, x, axis=1)
            if self.spp.rMat.shape[0] > 0:
                self.spp.rMat = np.delete(self.spp.rMat, x, axis=1)

            # Update network and distance matrices (nodes are rows/columns)
            self.spp.topo.network = np.delete(self.spp.topo.network, x, axis=0)
            self.spp.topo.distMat = np.delete(self.spp.topo.distMat, x, axis=0)
            self.spp.topo.distMat = np.delete(self.spp.topo.distMat, x, axis=1)

            # Update dMat (dispersal), same dimension change as distMat
            self.spp.dMat = np.delete(self.spp.dMat, x, axis=0)
            self.spp.dMat = np.delete(self.spp.dMat, x, axis=1)

            # Check if dMat is empty
            if self.spp.dMat is None or self.spp.dMat.size == 0:
                print("Warning: `dMat` became empty after node removal. Stopping further node removals.")
                break

            # scVec and scVec_prime represent conditions per node, so remove along axis=1 as well
            if self.spp.topo.scVec.shape[0] > 0 and self.spp.topo.scVec.ndim == 2:
                if self.spp.topo.scVec.shape[1] > x:  # ensure column exists
                    self.spp.topo.scVec = np.delete(self.spp.topo.scVec, x, axis=1)
            if self.spp.topo.scVec_prime.shape[0] > 0 and self.spp.topo.scVec_prime.ndim == 2:
                if self.spp.topo.scVec_prime.shape[1] > x:
                    self.spp.topo.scVec_prime = np.delete(self.spp.topo.scVec_prime, x, axis=1)

            # Adjust envMat if it depends on nodes (columns)
            if hasattr(self.spp.topo, 'envMat') and self.spp.topo.envMat is not None and self.spp.topo.envMat.ndim > 1:
                if self.spp.topo.envMat.shape[1] > x:
                    self.spp.topo.envMat = np.delete(self.spp.topo.envMat, x, axis=1)

            # Decrement node count
            self.spp.topo.no_nodes -= 1

            # Now re-check dimensions
            species_count, node_count = self.spp.xMat.shape

            # Ensure dimensions remain consistent or handle shape changes if needed
            if species_count != init_species_count:
                raise ValueError(
                    f"Species count changed from {init_species_count} to {species_count}. "
                    f"Dimensions must remain consistent during simulation."
                )

            # Align other matrices after node removal
            align_dimensions(species_count, node_count)

            # Initialize CommunityDynamics with updated shapes
            try:
                dynamics = CommunityDynamics(
                    spp=self.spp,
                    xMat=self.spp.xMat,
                    cMat=self.spp.cMat,
                    rMat=self.spp.rMat,
                    dMat=self.spp.dMat,
                    efMat=self.spp.efMat,  # ensure efMat passed if previously used
                    scVec=self.spp.topo.scVec if self.spp.topo.scVec.shape[0] > 0 else None,
                    scVec_prime=self.spp.topo.scVec_prime if self.spp.topo.scVec_prime.shape[0] > 0 else None
                )
            except ValueError as e:
                print(f"Error initializing CommunityDynamics after node removal: {e}")
                break

            dynamics.rho = self.spp.rho
            if self.spp.topo.scVec.shape[0] != 0:
                dynamics.scVec = self.spp.topo.scVec
                dynamics.scVec_prime = self.spp.topo.scVec_prime
            dynamics.S_p = self.spp.S_p
            dynamics.S_c = self.spp.S_c

            Bfile = os.path.join(Bpath, f"bMat_nr_init{x}.mat")
            np.savetxt(Bfile, self.spp.xMat, fmt='%e')
            if self.spp.S_c > 0:
                Bfile_c = os.path.join(Bpath, f"bMat_nr_c_init{x}.mat")
                B_c = self.spp.xMat[self.spp.S_p:, :]
                np.savetxt(Bfile_c, B_c, fmt='%e')

            def rhs(t, y):
                state = y.reshape(self.spp.xMat.shape)
                Gt = dynamics.compute_intrinsic_growth_rates(state)
                dXdt = Gt.copy()
                dXdt.flat[dynamics.indices_DP] *= state.flat[dynamics.indices_DP]

                if dynamics.emMat is None:
                    massEffect = state @ dynamics.dMat_sp
                    dXdt += massEffect
                else:
                    emMat_N = np.repeat(dynamics.emMat[:, np.newaxis], state.shape[1], axis=1)
                    if emMat_N.shape != state.shape:
                        logger.warning("Realigning emMat to state shape after node removal.")
                        emMat_N = np.zeros_like(state)
                    massEffect = (emMat_N * state) @ dynamics.dMat_sp
                    dXdt += massEffect

                dXdt[state < 0] = -state[state < 0]
                dXdt[~np.isfinite(dXdt)] = 0
                dXdt[np.abs(dXdt) > 1e10] = 0

                return dXdt.flatten()

            problem = Explicit_Problem(rhs, self.spp.xMat.flatten(), 0)
            solver = CVode(problem)
            solver.simulate(tMax)

            Bfile = os.path.join(Bpath, f"bMat_nr{x}.mat")
            np.savetxt(Bfile, self.spp.xMat, fmt='%e')
            if self.spp.S_c > 0:
                Bfile_c = os.path.join(Bpath, f"bMat_nr_c{x}.mat")
                B_c = self.spp.xMat[self.spp.S_p:, :]
                np.savetxt(Bfile_c, B_c, fmt='%e')


    def gen_jacobian(self):
        """
        Generate the numerical approximation of the Jacobian matrix for computing regional competitive overlap matrix.
        """
        try:
            dynamics = CommunityDynamics(
                spp=self.spp,
                xMat=self.spp.xMat,
                cMat=self.spp.cMat,
                rMat=self.spp.rMat,
                dMat=self.spp.dMat
            )

            # Ensure dimensions are consistent
            species_count, node_count = self.spp.xMat.shape

            # emMat should be 1D: (species_count,)
            if dynamics.emMat is not None:
                if dynamics.emMat.ndim > 1:
                    logger.warning("Reshaping emMat to 1D vector for dimension consistency.")
                    # Assuming emMat is uniform across nodes or take first column
                    # Adjust this as per your model's logic
                    dynamics.emMat = dynamics.emMat[:, 0]

                if dynamics.emMat.shape[0] != species_count:
                    raise ValueError(
                        f"emMat must have length {species_count}, got {dynamics.emMat.shape[0]}."
                    )

            dynamics.indices_DP = self.spp.indices_DP
            dynamics.indices_S = self.spp.indices_S
            dynamics.rho = self.spp.rho
            dynamics.bodymass = self.spp.bodymass
            dynamics.bodymass_inv = self.spp.bodymass_inv
            dynamics.mu = self.spp.mu

            def compute_rhs(y):
                """
                Computes the right-hand side function for the dynamics.
                """
                state = y.reshape(self.spp.xMat.shape)
                Gt = dynamics.compute_intrinsic_growth_rates(state)
                dXdt = Gt.copy()
                dXdt.flat[dynamics.indices_DP] *= state.flat[dynamics.indices_DP]

                # Apply mass effect
                if dynamics.emMat is None:
                    # No emMat, simple mass effect
                    massEffect = state @ dynamics.dMat_sp
                else:
                    # emMat is now (species_count,)
                    # Extend emMat to match state dimensions: (species_count, node_count)
                    emMat_N = np.tile(dynamics.emMat[:, np.newaxis], (1, node_count))
                    if emMat_N.shape != state.shape:
                        raise ValueError(
                            f"emMat_N shape {emMat_N.shape} does not match state {state.shape}."
                        )
                    massEffect = (emMat_N * state) @ dynamics.dMat_sp

                dXdt += massEffect

                # Handle edge cases
                dXdt[state < 0] = -state[state < 0]
                dXdt[~np.isfinite(dXdt)] = 0
                dXdt[np.abs(dXdt) > 1e10] = 0

                return dXdt.flatten()

            # Finite difference Jacobian approximation
            y0 = self.spp.xMat.flatten()
            epsilon = 1e-6  # Perturbation size
            n = y0.size
            jacobian = np.zeros((n, n))

            for i in range(n):
                perturb = np.zeros(n)
                perturb[i] = epsilon

                f_plus = compute_rhs(y0 + perturb)
                f_minus = compute_rhs(y0 - perturb)

                # Finite difference approximation
                jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)

            self.jacobian = jacobian
            print(f"Jacobian matrix successfully computed with shape: {self.jacobian.shape}")

        except Exception as e:
            raise RuntimeError(f"Error during Jacobian generation: {e}")


    def gen_c_mat_reg(self, h=0.001, regularization=1e-6):
        """
        Generate a numerically approximated regional interaction matrix via a computation harvesting experiment.
        :param h: Harvesting rate
        :param regularization: Small constant added to the diagonal for numerical stability
        """
        self.gen_jacobian()
        
        if self.jacobian is None:
            raise ValueError("Jacobian matrix is not initialized.")
        
        try:
            # Invert Jacobian
            J_inv = np.linalg.inv(self.jacobian)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Failed to invert Jacobian matrix: {e}")
        
        S_tot = self.spp.xMat.shape[0]
        index = np.arange(0, self.spp.topo.no_nodes) * S_tot
        self.cMat_reg = np.zeros((S_tot, S_tot))

        for i in range(S_tot):
            H_i = np.zeros(S_tot * self.spp.topo.no_nodes)
            H_i[index + i] = h * self.spp.xMat[i, :]
            dB_jx = -1 * (J_inv @ H_i) / h
            dB_jx_mat = dB_jx.reshape(S_tot, self.spp.topo.no_nodes)
            self.cMat_reg[:, i] = np.sum(dB_jx_mat, axis=1)

        self.jacobian = None
        
        # Add small regularization to avoid singularity
        self.cMat_reg += np.eye(self.cMat_reg.shape[0]) * regularization

        try:
            self.cMat_reg = np.linalg.inv(self.cMat_reg)
        except np.linalg.LinAlgError:
            raise RuntimeError("Regularized cMat_reg is still singular. Adjust regularization or investigate input data.")
        
        # Normalize regional competitive overlap matrix
        norm = np.zeros_like(self.cMat_reg)
        sgn = np.zeros_like(self.cMat_reg)
        norm[np.diag_indices_from(self.cMat_reg)] = 1 / np.sqrt(np.abs(np.diag(self.cMat_reg)))
        sgn[np.diag_indices_from(self.cMat_reg)] = np.diag(self.cMat_reg) / np.abs(np.diag(self.cMat_reg))
        self.cMat_reg = sgn @ norm @ self.cMat_reg @ norm
        print("Regional competitive overlap matrix successfully generated and normalized.")

               

    def gen_source_sink(self, t_full_relax):
        """
        Infer which populations are dependent upon immigration for local detectability by switching off dispersal and relaxing to equilibrium
        :param t_full_relax: Long relaxation time
        """
        D_store = np.copy(self.spp.dMat)
        B_store_p = np.copy(self.spp.xMat)
        Src_p = np.zeros_like(self.spp.xMat)
        Snk_p = np.zeros_like(self.spp.xMat)

        self.spp.dMat = np.zeros_like(D_store)
        np.fill_diagonal(self.spp.dMat, np.diag(D_store))
        self.meta_c_dynamics(t_full_relax)

        Src_p[self.spp.xMat > self.spp.thresh] = 1

        self.spp.xMat = B_store_p
        self.spp.dMat = D_store

        Snk_p[self.spp.xMat > self.spp.thresh] = 1
        Snk_p = Src_p - Snk_p
        Snk_p[Snk_p == 1] = 0
        self.spp.xMat_src = Src_p + Snk_p




    def print_params(self):
          """
          Print model parameterization to console
          """
          print("\nModel parameters:")
          print(f"\ntMax {self.tMax}")
          print(f"\nparOut {self.parOut}")
          print(f"\nexperiment {self.experiment}")
          print(f"\nrep {self.rep}")
          print(f"\nform_of_dynamics {self.g_form_of_dynamics}")
          print(f"\nsimTime {self.simTime}")
          print(f"\ndate {self.date}")
          print(f"\nc1 {self.spp.c1}")
          print(f"\nc2 {self.spp.c2}")
          print(f"\nemRate {self.spp.emRate}")
          print(f"\ndispL {self.spp.dispL}")
          print(f"\npProducer {self.spp.pProducer}")
          print(f"\nalpha {self.spp.alpha}")
          print(f"\nsigma {self.spp.sigma}")
          print(f"\nrho {self.spp.rho}")
          print(f"\ncomp_dist {self.spp.comp_dist}")
          print(f"\nomega {self.spp.omega}")
          print(f"\nno_nodes {self.spp.topo.no_nodes}")
          print(f"\nphi {self.spp.topo.phi}")
          if self.spp.topo.skVec.size > 0:
              print(f"\nsk {self.spp.topo.skVec[0]}")
          print(f"\nenvVar {self.spp.topo.envVar}")
          print(f"\nT_int {self.spp.topo.T_int}")
          print(f"\nvar_e {self.spp.topo.var_e}")
          print(f"\nrandGraph {self.spp.topo.randGraph}")
          print(f"\ngabriel {self.spp.topo.gabriel}")
          print(f"\ninvMax {self.invMax}")
          print(f"\ndispNorm {self.spp.dispNorm}")


    def write_params(self, filenameP):
        # Summary:
        # Write model parameterization to a file
        with open(filenameP, 'w') as params:
            params.write(f"date {self.date}\n")
            params.write(f"experiment {self.experiment}\n")
            params.write(f"form_of_dynamics {self.g_form_of_dynamics}\n")
            params.write(f"invMax {self.invMax}\n")
            params.write(f"parOut {self.parOut}\n")
            params.write(f"rep {self.rep}\n")
            params.write(f"simTime {self.simTime}\n")
            params.write(f"no_nodes {self.spp.topo.no_nodes}\n")
            params.write(f"randGraph {self.spp.topo.randGraph}\n")
            params.write(f"phi {self.spp.topo.phi}\n")
            params.write(f"envVar {self.spp.topo.envVar}\n")
            params.write(f"c1 {self.spp.c1}\n")
            params.write(f"c2 {self.spp.c2}\n")
            if self.spp.c3 > 0:
                params.write(f"c3 {self.spp.c3}\n")
            params.write(f"emRate {self.spp.emRate}\n")
            params.write(f"dispL {self.spp.dispL}\n")
            params.write(f"invasion {self.spp.invasion}\n")
            params.write(f"tMax {self.tMax}\n")
            params.write(f"pProducer {self.spp.pProducer}\n")
            params.write(f"prodComp {self.spp.prodComp}\n")
            params.write(f"var_e {self.spp.topo.var_e}\n")
            params.write(f"alpha {self.spp.alpha}\n")
            params.write(f"sigma {self.spp.sigma}\n")
            params.write(f"rho {self.spp.rho}\n")
            params.write(f"comp_dist {self.spp.comp_dist}\n")
            if self.spp.topo.T_int > 0:
                params.write(f"T_int {self.spp.topo.T_int}\n")
            params.write(f"omega {self.spp.omega}\n")
            if len(self.spp.topo.skVec) > 0:
                params.write(f"sk {self.spp.topo.skVec[0]}\n")
            else:
                params.write(f"sk NULL\n")
            params.write(f"delta_g {self.spp.delta_g}\n")
            params.write(f"sigma_r {self.spp.sigma_r}\n")
            params.write(f"symComp {self.spp.symComp}\n")


    def save_video(self, path):
        self.gen_c_mat_reg()
        self.gen_source_sink(t_full_relax=1000)  # Pass a suitable value for t_full_relax

        if self.cMat_reg is not None:
            np.savetxt(f"{path}cMat_reg_{self.spp.invasion}.mat", self.cMat_reg)
        if self.spp.rMat is not None:
            np.savetxt(f"{path}rMat_{self.spp.invasion}.mat", self.spp.rMat)
        if self.spp.xMat is not None:
            np.savetxt(f"{path}bMat_{self.spp.invasion}.mat", self.spp.xMat)
        if self.spp.xMat_src is not None:
            np.savetxt(f"{path}bMat_src_{self.spp.invasion}.mat", self.spp.xMat_src)

        self.cMat_reg = None
        self.spp.xMat_src = None

        if self.store_params:
            self.write_params(f"{path}params.mat")
            if self.spp.topo.network is not None:
                np.savetxt(f"{path}network.mat", self.spp.topo.network)
            if self.spp.topo.envMat is not None:
                np.savetxt(f"{path}envMat.mat", self.spp.topo.envMat)
            if self.spp.dMat is not None:
                np.savetxt(f"{path}dMat.mat", self.spp.dMat)
            self.store_params = False




    def saveMC(self):
        """
        Generate file names and save model matrices to file.
        """
        print("\nOutputting data...")
        
        # Generate paths for saving
        if len(self.bMatFileName) == 0:
            self.outputDirectory = Path(f"SimulationData/N={self.spp.topo.no_nodes}/{self.experiment}_experiment/{self.date}/")
            self.outputDirectory.mkdir(parents=True, exist_ok=True)
            base_filename = f"{self.date}_{self.experiment}({self.invMax}){self.parOut}"
            self.bMatFileName = str(self.outputDirectory / f"{base_filename}bMat{self.rep}.mat")

            base_filename = f"{self.date}_{self.experiment}({self.invMax}){self.parOut}"
            filenameR = self.outputDirectory / f"{base_filename}rMat{self.rep}.mat"
            filenameSR = self.outputDirectory / f"{base_filename}sMat{self.rep}.mat"
            filenameB = self.outputDirectory / f"{base_filename}bMat{self.rep}.mat"
            filenameBC = self.outputDirectory / f"{base_filename}bMat_c{self.rep}.mat"
            filenameP = self.outputDirectory / f"{base_filename}params{self.rep}.mat"
            filenameN = self.outputDirectory / f"{base_filename}network{self.rep}.mat"
            filenameTr = self.outputDirectory / f"{base_filename}trajec{self.rep}.mat"
            filenameEf = self.outputDirectory / f"{base_filename}envFluct{self.rep}.mat"
            filenameS = self.outputDirectory / f"{base_filename}S{self.rep}.mat"
            filenameD = self.outputDirectory / f"{base_filename}dMat{self.rep}.mat"
            filenameEM = self.outputDirectory / f"{base_filename}emMat{self.rep}.mat"
            filenameEMC = self.outputDirectory / f"{base_filename}emMat_c{self.rep}.mat"
            filenameC = self.outputDirectory / f"{base_filename}cMat_reg{self.rep}.mat"
            filenameT = self.outputDirectory / f"{base_filename}tMat{self.rep}.mat"
            filenameE = self.outputDirectory / f"{base_filename}envMat{self.rep}.mat"
            filenameBs = self.outputDirectory / f"{base_filename}bMat_src{self.rep}.mat"
            filenameBCs = self.outputDirectory / f"{base_filename}bMat_c_src{self.rep}.mat"
            self.bMatFileName = str(filenameB)
            print(f"Biomass matrix file name: {self.bMatFileName}")
        else:
            # Use existing bMatFileName to generate paths for other matrices
            filenameBC = self.bMatFileName.replace("bMat", "bMat_c")
            filenameD = self.bMatFileName.replace("bMat", "dMat")
            filenameEM = self.bMatFileName.replace("bMat", "emMat")
            filenameEMC = self.bMatFileName.replace("bMat", "emMat_c")
            filenameI = self.bMatFileName.replace("bMat", "cMat")
            filenameC = self.bMatFileName.replace("bMat", "cMat_reg")
            filenameA = self.bMatFileName.replace("bMat", "aMat")
            filenameN = self.bMatFileName.replace("bMat", "network")
            filenameP = self.bMatFileName.replace("bMat", "params")
            filenameR = self.bMatFileName.replace("bMat", "rMat")
            filenameSR = self.bMatFileName.replace("bMat", "sMat")
            filenameS = self.bMatFileName.replace("bMat", "S")
            filenameT = self.bMatFileName.replace("bMat", "tMat")
            filenameE = self.bMatFileName.replace("bMat", "envMat")
            filenameTr = self.bMatFileName.replace("bMat", "trajec")
            filenameEf = self.bMatFileName.replace("bMat", "envFluct")
            filenameBs = self.bMatFileName.replace("bMat", "bMat_src")
            filenameBCs = self.bMatFileName.replace("bMat", "bMat_c_src")

        # Store matrix objects accordingly
        if self.spp.trajectories is not None and self.spp.trajectories.size != 0:
            filenameTr = self.bMatFileName.replace("bMat", "trajec")
            savemat(filenameTr, {'trajectories': self.spp.trajectories})

        if self.spp.efMat is not None and self.spp.efMat.size != 0:
            R = self.spp.rMat.flatten()
            N = np.full((1, R.shape[0]), np.nan)
            R = np.vstack((R, N))
            if self.spp.fluctuations.shape[1] != R.shape[1]:
                pad_width = abs(R.shape[1] - self.spp.fluctuations.shape[1])
                if R.shape[1] > self.spp.fluctuations.shape[1]:
                    self.spp.fluctuations = np.pad(self.spp.fluctuations, ((0, 0), (0, pad_width)), constant_values=np.nan)
                else:
                    R = np.pad(R, ((0, 0), (0, pad_width)), constant_values=np.nan)
            self.spp.fluctuations = np.vstack((R, self.spp.fluctuations))
            filenameEf = self.bMatFileName.replace("bMat", "envFluct")
            savemat(filenameEf, {'fluctuations': self.spp.fluctuations})

        B_p = self.spp.xMat[:self.spp.S_p, :]
        savemat(self.bMatFileName, {'B_p': B_p})
        print(f"Data successfully saved to {self.outputDirectory}")



    def loadMC(self, bMatFile):
        """
        Import metacommunity model.
        """
        if not bMatFile:
            raise ValueError("No valid bMat file path provided. Please provide a valid file path.")

        self.bMatFileName = bMatFile
        print(f"Importing data, file name {self.bMatFileName}")

        # Generate file paths based on bMatFileName
        try:
            filenameBC = self.bMatFileName.replace("bMat", "bMat_c")
            filenameD = self.bMatFileName.replace("bMat", "dMat")
            filenameEM = self.bMatFileName.replace("bMat", "emMat")
            filenameEMC = self.bMatFileName.replace("bMat", "emMat_c")
            filenameI = self.bMatFileName.replace("bMat", "cMat")
            filenameA = self.bMatFileName.replace("bMat", "aMat")
            filenameN = self.bMatFileName.replace("bMat", "network")
            filenameP = self.bMatFileName.replace("bMat", "params")
            filenameR = self.bMatFileName.replace("bMat", "rMat")
            filenameSR = self.bMatFileName.replace("bMat", "sMat")
            filenameS = self.bMatFileName.replace("bMat", "S")
            filenameT = self.bMatFileName.replace("bMat", "tMat")
            filenameE = self.bMatFileName.replace("bMat", "envMat")
            filenameTr = self.bMatFileName.replace("bMat", "trajec")
            filenameIPp = self.bMatFileName.replace("bMat", "invProb_p")
            filenameIPc = self.bMatFileName.replace("bMat", "invProb_c")

            # Load matrix objects
            self.spp.xMat = sio.loadmat(self.bMatFileName)['B_p']
            self.spp.S_p = self.spp.xMat.shape[0]

            self.spp.dMat = sio.loadmat(filenameD)['dMat']
            if os.path.exists(filenameI):
                self.spp.cMat = sio.loadmat(filenameI)['C']
            if os.path.exists(filenameEM):
                self.spp.emMat = sio.loadmat(filenameEM)['Em_p']
            self.spp.topo.network = sio.loadmat(filenameN)['network']
            if os.path.exists(filenameT):
                self.spp.tMat = sio.loadmat(filenameT)['tMat']
            if os.path.exists(filenameE):
                self.spp.topo.envMat = sio.loadmat(filenameE)['envMat']
            self.spp.rMat = sio.loadmat(filenameR)['rMat']
            if os.path.exists(filenameSR):
                self.spp.sMat = sio.loadmat(filenameSR)['sMat']
            self.spp.sppRichness = sio.loadmat(filenameS)['sppRichness']

            print(f"\nImported network rows 0-4 = \n{self.spp.topo.network[:min(self.spp.topo.network.shape[0], 4)]}")
            self.spp.topo.genDistMat()
            print(f"\nSpecies richness, S_p = {self.spp.rMat.shape[0]}, S_c = {self.spp.xMat.shape[0] - self.spp.rMat.shape[0]}")

        except KeyError as e:
            print(f"Error while importing data: Missing key in MAT file - {e}")
            raise
        except FileNotFoundError as e:
            print(f"Error while importing data: File not found - {e}")
            raise






# Example usage

# Example Usage Scenario
if __name__ == "__main__":
    # Step 1: Initialize Topography, Species, and Metacommunity
    # Create Topography
    topo = Topography(
        no_nodes=10,                # Number of nodes in the landscape
        lattice_height=5,           # Lattice height for spatial structure
        lattice_width=5,            # Lattice width for spatial structure
        phi=0.3,                    # Environmental heterogeneity parameter
        envVar=1,                   # Environmental variability switch
        skVec=np.array([0.1]),      # Environmental skewness vector
        var_e=0.5,                  # Variance in environmental tolerance
        randGraph=False,            # Random graph generation flag
        gabriel=True,               # Use Gabriel graph connectivity
        T_int=25.0,                 # Initial temperature
    network_file="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",  # Environment data file
        scVec=np.array([0.05]) 
    )

    # Create Species
    spp = Species(
        topo=topo,
        c1=0.5,
        c2=0.3,
        c3=0.1,
        emRate=0.2,
        dispL=0.7,
        pProducer=0.4,
        prodComp=True,
        symComp=True,
        alpha=0.02,
        sigma=1.5,
        sigma_t=0.3,
        rho=0.8,
        comp_dist=1,
        omega=0.9,
        dispNorm=1.0,
        bodymass=1e-4,  # Default value if needed
        mu=0.1  # Default value if needed
    )

    meta_community = Metacommunity(spp,
        a_init=True,
        a_bMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
        a_xMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
        a_scMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1params0.mat",
        a_invMax=100,
        a_tMax=500,
        a_outputDirectory="output/",
        a_c1=0.5,
        a_c2=0.3,
        a_c3=0.1,
        a_emRate=0.2,
        a_dispL=0.7,
        a_pProducer=0.4,
        a_prodComp=True,
        a_symComp=True,
        a_alpha=0.02,
        a_sigma=1.5,
        a_sigma_t=0.3,
        a_rho=0.8,
        a_comp_dist=1,
        a_omega=0.9,
        a_dispNorm=1.0,
        a_no_nodes=10,
        a_lattice_height=5,
        a_lattice_width=5,
        a_phi=0.3,
        a_envVar=1,
        a_skVec=np.array([0.1]),
        a_var_e=0.5,
        a_randGraph=False,
        a_gabriel=True,
        a_T_int=25.0,
        a_envMat="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1rMat0.mat", # Environment data file
        a_parOut=1,
        a_experiment="/Users/sarashahin/Desktop/model/SimulationData/autonomous_turnover_example_pars.txt",
        a_rep=1,
        storeTraj=1,  # Enable storing of trajectories
        bMatFileName="/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
        g_block_transitions=False
    )

    # Step 2: Simulate Metacommunity Dynamics
    print("\nStarting Metacommunity Dynamics Simulation...")
    meta_community.meta_c_dynamics(T=200)  # Simulate dynamics up to T=200

    # Step 3: Invader Sampling
    print("\nInvader Sampling...")
    meta_community.invader_sample(trophLev=1, no_invaders=2)  # Sample 5 new producer species

    # Step 4: Environmental Fluctuation
    print("\nSimulating Environmental Fluctuation...")
    meta_community.env_fluct()  # Apply one timestep of environmental fluctuation

    # Step 5: Warming Event Simulation
    print("\nSimulating Warming Event...")
    meta_community.warming(dTdt=0.5, res=5, time=20)  # Simulate warming with temperature increase rate of 0.5

    # Step 6: Long-Distance Dispersal
    print("\nSimulating Long-Distance Dispersal...")
    meta_community.long_dist_disp(tMax=100, edges=3)  # Randomly add 3 dispersal edges and simulate for 100 timesteps

    # Step 7: Node Removal
    print("\nRemoving Nodes from Metacommunity...")
    meta_community.node_removal(tMax=100, no_removals=2)  # Remove 2 nodes from the community

    # Step 8: Generate and Save Model Outputs
    print("\nGenerating and Saving Outputs...")
    meta_community.gen_jacobian()  # Generate Jacobian matrix
    meta_community.gen_c_mat_reg(h=0.001)  # Generate regional interaction matrix
    meta_community.gen_source_sink(t_full_relax=1000)  # Generate source-sink matrix
    meta_community.saveMC()  # Save the metacommunity model matrices to files
    meta_community.print_params()  # Print parameters to console
    print("\nSimulation Completed.")
