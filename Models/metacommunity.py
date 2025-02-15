
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
import random
from functools import partial

from assimulo.solvers import CVode
print("Assimulo and CVode imported successfully.")

from topography import Topography
from species import Species
from communitydynamics import CommunityDynamics

# logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Metacommunity:
    def __init__(self, spp, a_init, a_bMat, a_xMat, a_scMat, a_invMax, a_tMax, a_outputDirectory,
                 a_c1, a_c2, a_c3, a_emRate, a_dispL, a_pProducer, a_prodComp, a_symComp,
                 a_alpha, a_sigma, a_sigma_t, a_rho, a_comp_dist, a_omega, a_dispNorm,
                 a_no_nodes, a_lattice_height, a_lattice_width, a_phi, a_envVar, a_skVec,
                 a_var_e, a_randGraph, a_gabriel, a_T_int, a_envMat, a_parOut=0, 
                 a_experiment="default", a_rep=0, storeTraj=0, 
                 bMatFileName="default_bMat.mat", sc_file="default", 
                 g_block_transitions=False, g_form_of_dynamics="default_value",
                 simTime=0.0):
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
                
                
            #scVec and scVec_prime are (1, nodes)
            if a_skVec is not None:
                if a_skVec.ndim == 1:
                    scVec = a_skVec.reshape(1, -1)
                    logger.debug(f"Reshaped a_skVec to scVec with shape: {scVec.shape}")
                elif a_skVec.ndim == 2 and a_skVec.shape[0] == 1:
                    scVec = a_skVec
                    logger.debug(f"Assigned a_skVec to scVec with shape: {scVec.shape}")
                else:
                    logger.warning(
                        f"`a_skVec` has unexpected shape {a_skVec.shape}. Averaging across species to reshape."
                    )
                    scVec = a_skVec.mean(axis=0, keepdims=True)
                    logger.debug(f"Reshaped scVec by averaging to shape: {scVec.shape}")
            else:
                scVec = np.ones((1, a_no_nodes))
                logger.debug(f"Initialized scVec to ones with shape: {scVec.shape}")

            # initialize scVec_prime
            scVec_prime = np.ones((1, a_no_nodes))
            logger.debug(f"Initialized scVec_prime to ones with shape: {scVec_prime.shape}")              

            # Assign scVec and scVec_prime to self.spp
            self.scVec = scVec
            self.scVec_prime = scVec_prime
            logger.debug(f"Assigned scVec and scVec_prime to spp with shapes: {self.scVec.shape}, {self.scVec_prime.shape}")            

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
                sc_file=sc_file,
                scVec=scVec  # Ensure scVec is (1, nodes)
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
                dispNorm=a_dispNorm,
                scVec=None,

            )

            # Log that Species has been successfully initialized
            logger.debug("Species initialized successfully.")
            
            
            self.bodyMass = self.spp.bodymass  #  some bodymass

            if self.spp.dMat is None or self.spp.dMat.size == 0:
                logger.error("Dispersal matrix (`dMat`) must be properly initialized and non-empty in Species.")
                raise ValueError("Dispersal matrix must be properly initialized and non-empty in Species.")
            self.dMat = self.spp.dMat
            logger.debug(f"dMat initialized with shape: {self.dMat.shape}")
            
            # Initialize PSD-related attributes
            self.initialize_PSD_initialization(a_init=True)
            
            # Ensure bodymass_inv and mu are available
            self.bodymass_inv = self.spp.bodymass_inv if hasattr(self.spp, 'bodymass_inv') else 1.0
            self.mu = self.spp.mu if hasattr(self.spp, 'mu') else 1.0

            # Initialize CommunityDynamics once
            self.community_dynamics = CommunityDynamics(
                spp=self.spp,
                xMat=self.spp.xMat,
                cMat=self.spp.cMat,
                rMat=self.spp.rMat,
                efMat=self.spp.efMat,
                scVec=self.spp.scVec,
                scVec_prime=self.spp.scVec_prime,
                rho=self.spp.rho,
                g_form_of_dynamics=self.g_form_of_dynamics,  # Ensure this flag is set
                emMat=self.spp.emMat,               # Ensure emMat is provided
                dMat=self.spp.dMat,
                bodymass_inv=self.bodymass_inv,    # Ensure bodymass_inv is defined
                mu=self.mu                          # Ensure mu is defined
            )
            logger.debug("CommunityDynamics instance initialized within Metacommunity.")

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
                
            # Initialize PSD-related attributes
            self.initialize_PSD_initialization(a_init=False)
            
        # Initialize CommunityDynamics in both cases
        if a_init:
            # Already initialized above
            pass
        else:
            # Ensure bodymass_inv and mu are available
            self.bodymass_inv = self.spp.bodymass_inv if hasattr(self.spp, 'bodymass_inv') else 1.0
            self.mu = self.spp.mu if hasattr(self.spp, 'mu') else 1.0
            
            # Initialize CommunityDynamics for loaded metacommunity
            self.community_dynamics = CommunityDynamics(
                spp=self.spp,
                xMat=self.spp.xMat,
                cMat=self.spp.cMat,
                rMat=self.spp.rMat,
                efMat=self.spp.efMat,
                scVec=self.spp.scVec,
                scVec_prime=self.spp.scVec_prime,
                rho=self.spp.rho,
                g_form_of_dynamics=self.g_form_of_dynamics,  # this flag is set
                emMat=self.spp.emMat,               #  emMat is provided
                dMat=self.spp.dMat,
                bodymass_inv=self.bodymass_inv,    # bodymass_inv is defined
                mu=self.mu                          
            )
            logger.debug("CommunityDynamics instance initialized within Metacommunity (loaded).")


    def initialize_PSD_initialization(self, a_init):
        """
        Initialize PSD-related attributes based on the initialization mode.
        """
        if a_init:
            #  self.spp.xMat.shape is (S, N).
            # We want PSD arrays of length S, not S*N.
            num_species = self.spp.xMat.shape[0]

            self.bodyMass = self.spp.bodymass  # some bodymass
            self.PSD_states = np.zeros(num_species, dtype=int) 
            self.PoissonClocks = np.full(num_species, -np.log(random.uniform(0, 1))) 

            # Make BRelaxed shape (S,), not (S, N)
            self.BRelaxed = np.full(num_species, self.bodyMass / 10) 
            self.logBRelaxed = np.log(np.maximum(self.BRelaxed, 1e-10))

            if self.storeTraj:
                self.trajectories = []
                self.PSD_trajectories = []

            logger.debug("PSD state variables initialized for new metacommunity.")

        else:
            # For loading  MC, also ensure these are shape (S,).
            if not hasattr(self, 'PSD_states'):
                num_species = self.spp.xMat.shape[0]
                self.PSD_states = np.zeros(num_species, dtype=int)
                self.PoissonClocks = np.full(num_species, -np.log(random.uniform(0, 1)))
                self.BRelaxed = np.full(num_species, self.spp.bodymass / 10)
                self.logBRelaxed = np.log(np.maximum(self.BRelaxed, 1e-10))
                if self.storeTraj:
                    self.trajectories = []
                    self.PSD_trajectories = []

            logger.debug("PSD state variables initialized for loaded metacommunity.")

                              
                
    def update_PSD_states(self, dynamics=None):
        """
        Update the PSD states of each species based on aggregated growth rates and biomass.
        :param dynamics: Instance of CommunityDynamics.
        """
        
    ### If `dynamics` is not passed, try using self.community_dynamics
        if dynamics is None:
            if not hasattr(self, 'community_dynamics') or (self.community_dynamics is None):
                raise ValueError("No valid `dynamics` provided and `self.community_dynamics` is not set.")
            dynamics = self.community_dynamics        
        
        growth_rates = dynamics.compute_intrinsic_growth_rates(self.spp.xMat)
        
        # Debugging statements
        logger.debug(f"Growth rates shape: {growth_rates.shape}")
        
        # Aggregate growth rates per species
        if growth_rates.ndim > 1:
            # Assuming species are along axis=0
            aggregated_growth_rates = np.mean(growth_rates, axis=1)
            logger.debug(f"Aggregated growth rates (mean) shape: {aggregated_growth_rates.shape}")
        else:
            # Already 1D
            aggregated_growth_rates = growth_rates
            logger.debug("Growth rates are already 1D.")
        
        # aggregated_growth_rates is 1D and matches the number of species
        assert aggregated_growth_rates.ndim == 1, "Aggregated growth rates must be 1D."
        assert aggregated_growth_rates.size == len(self.PSD_states), "Size mismatch between growth rates and PSD_states."
        
        for i in range(len(self.PSD_states)):
            biomass_sum = self.spp.xMat[i, :].sum()
            logger.debug(f"Processing species {i}: PSD_state={self.PSD_states[i]}, "
                        f"Aggregated Growth Rate={aggregated_growth_rates[i]:.4f}, "
                        f"Biomass Sum={biomass_sum:.4f}, "
                        f"Body Mass={self.bodyMass:.4f}")
            
            # aggregated_growth_rates[i] is a scalar
            if self.PSD_states[i] == 0:  # Deterministic
                if aggregated_growth_rates[i] > 0 and biomass_sum < self.bodyMass:
                    self.PSD_states[i] = 1  # Switch to Stochastic
                    logger.debug(f"Species {i} switched to Stochastic.")
            elif self.PSD_states[i] == 1:  # Stochastic
                if aggregated_growth_rates[i] <= 0:
                    self.PSD_states[i] = 0  # Switch back to Deterministic
                    logger.debug(f"Species {i} switched back to Deterministic.")
            elif self.PSD_states[i] == 2:  # Probabilistic
                # Define transitions 
                pass


    def handle_invasions(self, time_step):
        """
        Handle species invasions based on Poisson clocks.
        :param time_step: Current time step.
        """
        for i in range(len(self.PoissonClocks)):
            if self.PoissonClocks[i] <= 0 and self.PSD_states[i] == 1:
                # Invasion occurs
                self.spp.xMat[i, :] += self.bodyMass  # Add biomass
                # Reset Poisson clock with a new random interval
                self.PoissonClocks[i] = -np.log(random.uniform(0, 1))
                # switch state if biomass exceeds threshold
                if self.spp.xMat[i, :].sum() >= self.bodyMass:
                    self.PSD_states[i] = 0  # Switch to Deterministic

    def update_biomass(self, dynamics, time_step):
        """
        Update biomass based on current PSD states and growth rates.
        :param dynamics: Instance of CommunityDynamics.
        :param time_step: Current time step.
        """
        growth_rates = dynamics.compute_intrinsic_growth_rates(self.spp.xMat)
        
        for i in range(len(self.PSD_states)):
            if self.PSD_states[i] == 0:  # Deterministic
                self.spp.xMat[i, :] += growth_rates[i] * time_step
            elif self.PSD_states[i] == 1:  # Stochastic
                # Apply stochastic updates random births and deaths
                #Binomial death and Poisson births
                deaths = np.random.binomial(self.spp.xMat[i, :].astype(int), 0.1)  # 10% death rate
                births = np.random.poisson(growth_rates[i] * self.spp.xMat[i, :])
                self.spp.xMat[i, :] = self.spp.xMat[i, :] - deaths + births
            elif self.PSD_states[i] == 2:  # Probabilistic
                # Implement probabilistic dynamics 
                pass

            #  biomass does not drop below zero
            self.spp.xMat[i, :] = np.maximum(self.spp.xMat[i, :], 0)

    def record_trajectories(self, current_time):
        """
        Record the current state of biomass and PSD states.
        :param current_time: Current simulation time.
        """
        if self.storeTraj:
            self.trajectories.append({
                'time': current_time,
                'biomass': self.spp.xMat.copy(),
                'PSD_states': self.PSD_states.copy()
            })
            #  record relaxed biomass
            self.PSD_trajectories.append({
                'time': current_time,
                'BRelaxed': self.BRelaxed.copy(),
                'logBRelaxed': self.logBRelaxed.copy()
            })      

    def meta_c_dynamics(self, T):
        """
        Numerical solver for metacommunity dynamics incorporating the PSD scheme.
        :param T: Total simulation time.
        """
        
        # Define the time step for discrete updates
        time_step = 1.0  #1 unit of time per iteration
        current_time = 0.0
        
        # Initialize CommunityDynamics object
        dynamics = CommunityDynamics(
            xMat=self.spp.xMat, 
            spp=self.spp, 
            cMat=self.spp.cMat, 
            rMat=self.spp.rMat,
            indices_S=np.nonzero(self.spp.xMat > 0)[1],  # Non-zero biomass indices
            indices_DP=np.arange(self.spp.xMat.size),    # All indices for dynamic populations
            dMat=self.spp.dMat
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
                
        # Validate xMat shape
        species_count, node_count = self.spp.xMat.shape
        logger.debug(f"xMat shape: {self.spp.xMat.shape}")
        
        # Validate cMat shape
        if self.spp.cMat.shape != (species_count, species_count):
            logger.error(f"cMat shape {self.spp.cMat.shape} does not match expected shape ({species_count}, {species_count}).")
            raise ValueError("Competitive matrix shape mismatch.")
            
            logger.info("Starting dynamics with PSD integration.")
            
            # Handling file paths for storing trajectories
            Bpath, bMatDir, newFolder = "", "", ""
            if self.storeTraj != 0:
                bMatDir = os.path.dirname(self.bMatFileName)
                newFolder = os.path.splitext(os.path.basename(self.bMatFileName))[0]
                Bpath = os.path.join(bMatDir, newFolder, "trajectory")
                
                if not os.path.exists(Bpath):
                    os.makedirs(Bpath)
                print(f"\nSaving trajectories to {Bpath}")
            
            # Initialize previous_state for convergence checking
            previous_state = self.spp.xMat.copy()
            
            # Initialize trajectory storage 
            if self.storeTraj:
                self.trajectories = []
                self.PSD_trajectories = []
            
            # Main simulation loop integrating PSD dynamics
            while current_time < T:
                logger.debug(f"Simulation time: {current_time}/{T}")
                
                # 1. Update PSD States based on current growth rates and biomass
                self.update_PSD_states(dynamics)
                
                # 2. Handle invasions based on Poisson clocks
                self.handle_invasions(time_step)
                
                # 3. Update biomass based on current PSD states
                self.update_biomass(dynamics, time_step)
                
                # 4. Update Poisson clocks
                self.PoissonClocks -= time_step
                
                # 5. Record trajectories if enabled
                if self.storeTraj:
                    self.record_trajectories(current_time)
                
                # 6. Increment current time
                current_time += time_step
                
                # Save biomass and state at specific intervals
                # every 100 time units
                if self.storeTraj and current_time % 100 == 0:
                    timestamp = f"time_{int(current_time)}.mat"
                    biomass_file = os.path.join(Bpath, f"biomass_{timestamp}")
                    savemat(biomass_file, {'xMat': self.spp.xMat})
                    logger.info(f"Biomass state saved at time {current_time} to {biomass_file}")
            
            logger.info("Metacommunity dynamics simulation with PSD integration completed.")


    def remove_species(self, indices):
        """
        Helper function to remove species from all matrices and PSD-related variables.
        """
        logger.debug(f"Removing species at indices: {indices}")
        indices = sorted(indices, reverse=True)

        for idx in indices:
            if idx >= self.spp.xMat.shape[0]:
                logger.warning(f"Index {idx} is out of bounds for xMat with shape {self.spp.xMat.shape}. Skipping.")
                continue

            # Update producer counts
            if idx < self.spp.S_p:
                self.spp.S_p -= 1
            else:
                self.spp.S_c -= 1

            # Remove from xMat, cMat, rMat along axis=0
            self.spp.xMat = np.delete(self.spp.xMat, idx, axis=0)
            if self.spp.cMat is not None and self.spp.cMat.shape[0] > 0:
                self.spp.cMat = np.delete(self.spp.cMat, idx, axis=0)
                self.spp.cMat = np.delete(self.spp.cMat, idx, axis=1)
            if self.spp.rMat is not None and self.spp.rMat.shape[0] > 0:
                self.spp.rMat = np.delete(self.spp.rMat, idx, axis=0)

            # Remove from optional matrices if they have rows
            optional_matrices = {
                'sMat': self.spp.sMat,
                'tMat': self.spp.tMat,
                'emMat': self.spp.emMat,
                'efMat': self.spp.efMat,
                'ouMat': self.spp.ouMat,
                'trajectories': self.spp.trajectories,
                'fluctuations': self.spp.fluctuations,
            }
            for mat_name, mat in optional_matrices.items():
                if mat is not None and mat.size > 0 and idx < mat.shape[0]:
                    new_mat = np.delete(mat, idx, axis=0)
                    setattr(self.spp, mat_name, new_mat)
                    logger.debug(f"{mat_name} shape after removal: {new_mat.shape}")

            # For the 1D PSD arrays, remove by index with no axis
            if len(self.PSD_states) > idx:
                self.PSD_states = np.delete(self.PSD_states, idx)
            if len(self.PoissonClocks) > idx:
                self.PoissonClocks = np.delete(self.PoissonClocks, idx)
            if len(self.BRelaxed) > idx:
                self.BRelaxed = np.delete(self.BRelaxed, idx)
                self.logBRelaxed = np.log(np.maximum(self.BRelaxed, 1e-10))

        # Re-initialize CommunityDynamics
        self.community_dynamics = CommunityDynamics(
            spp=self.spp,
            xMat=self.spp.xMat,
            cMat=self.spp.cMat,
            rMat=self.spp.rMat,
            efMat=self.spp.efMat,
            scVec=self.spp.scVec,
            scVec_prime=self.spp.scVec_prime,
            rho=self.spp.rho,
            g_form_of_dynamics=self.g_form_of_dynamics,
            emMat=self.spp.emMat,
            dMat=self.spp.dMat,
            bodymass_inv=self.bodymass_inv,
            mu=self.mu
        )
        logger.debug("CommunityDynamics instance re-initialized after species removal.")
        

    def invader_sample(self, trophLev, no_invaders):
        """
        Introduce new random species and test for positive growth rates based on PSD scheme.
        """
        min_b = 1e-6
        invExcess = 3  # Excess invaders to ensure success
        suc_inv = 0

        while suc_inv < no_invaders:
            logger.debug(f"Attempting to invade. Successful so far: {suc_inv}/{no_invaders}")

            # Introduce invExcess * no_invaders species
            for _ in range(invExcess * no_invaders):
                self.spp.invade(trophLev)

                # If producers, optionally bump up rMat range
                if trophLev == 0:
                    newly_added_index = self.spp.xMat.shape[0] - 1
                    self.spp.rMat[newly_added_index, :] = np.random.uniform(0.2, 0.4, size=self.spp.xMat.shape[1])
                    # set it near zero for simpler competition:
                    self.spp.cMat[newly_added_index, :] = np.random.uniform(-0.01, 0.01, self.spp.cMat.shape[1])

                    #  minimal effect on existing species:
                    self.spp.cMat[:, newly_added_index] = np.random.uniform(-0.01, 0.01, self.spp.cMat.shape[0])
                    

                # Synchronize matrices
                for mat_name in ['efMat', 'emMat']:
                    mat = getattr(self.spp, mat_name, None)
                    if mat is not None and mat.size > 0:
                        new_mat = np.vstack([mat, np.zeros((1, mat.shape[1]))])
                        setattr(self.spp, mat_name, new_mat)
                        logger.debug(f"{mat_name} shape after invasion: {new_mat.shape}")
                    elif mat is not None:
                        new_mat = np.zeros((self.spp.xMat.shape[0], self.spp.xMat.shape[1]))
                        setattr(self.spp, mat_name, new_mat)
                        logger.debug(f"{mat_name} initialized with shape: {new_mat.shape}")
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
                    bInv = (
                        self.spp.rMat[self.spp.S_p:] 
                        - (self.spp.cMat[self.spp.S_p:, :] @ B)
                    )
                else:  # Consumers
                    bInv = self.spp.rho * (
                        self.spp.cMat[indices, :self.spp.S_p] @ B - 1
                    )
            except ValueError as e:
                logger.error(f"Error in growth rate calculation: {e}")
                raise

            logger.debug(f"Calculated growth rates: {bInv}")

            if bInv.size == 0 or np.all(bInv <= 0):
                logger.warning("No positive growth rates found. Adjusting parameters may be necessary.")
                break

            if bInv.ndim > 1:
                bInv_max = bInv.max(axis=1)
            else:
                bInv_max = bInv

            # Identify positively growing
            posGrowth_local = np.where(bInv_max >= min_b)[0][: no_invaders - suc_inv]
            posGrowth_abs = indices[posGrowth_local]
            negGrowth = np.setdiff1d(indices, posGrowth_abs)

            logger.debug(f"Positive Growth Indices (absolute): {posGrowth_abs}, Negative Growth Indices: {negGrowth}")

            # Increase successful invasions
            suc_inv += len(posGrowth_abs)

            # ### NO REMOVAL FOR PRODUCERS ###
            # Only remove negative if these are consumers (trophLev=1).
            if trophLev == 1:
                for idx in reversed(negGrowth):
                    self.remove_species([idx])
            else:
                logger.debug("Skipping removal of negative producers. They remain in the community.")

            logger.debug(f"State after removal - Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")

        # Update species counts
        if trophLev == 0:
            self.spp.S_p = self.spp.xMat.shape[0] - self.spp.S_c
        else:
            self.spp.S_c = self.spp.xMat.shape[0] - self.spp.S_p

        # Initialize PSD states for new species
        new_species_count = self.spp.xMat.shape[0] - len(self.PSD_states)
        if new_species_count > 0:
            new_PSD_states = np.ones(new_species_count, dtype=int)
            self.PSD_states = np.append(self.PSD_states, new_PSD_states)

            new_PoissonClocks = -np.log(np.random.uniform(0, 1, new_species_count))
            self.PoissonClocks = np.append(self.PoissonClocks, new_PoissonClocks)

            new_BRelaxed = np.full(new_species_count, self.bodyMass / 10)
            self.BRelaxed = np.append(self.BRelaxed, new_BRelaxed)
            self.logBRelaxed = np.log(self.BRelaxed)

            logger.debug(f"BRelaxed appended with {new_species_count} new relaxed biomass values.")

        logger.info(f"Invader sampling complete. Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")

        
    def compute_rhs(self, t, y, dynamics):
        """
        Right-hand side function for ODE solver.
        :param t: Current time
        :param y: Current state vector (flattened xMat)
        :param dynamics: Instance of CommunityDynamics.
        :return: Flattened derivative vector
        """
        try:
            # 'state' is shape (S, N)
            state = y.reshape(self.spp.xMat.shape)
            logger.debug(f"Reshaped state shape: {state.shape}")

            #  Intrinsic growth
            Gt = dynamics.compute_intrinsic_growth_rates(state)  # shape (S, N)
            dXdt = Gt.copy()                                     # shape (S, N)

            #  Dispersal + mass effect
            if dynamics.emMat is None:
                massEffect = state @ dynamics.dMat_sp
            else:
                emMat = dynamics.emMat
                # Possibly broadcast emMat to (S, N)
                if emMat.shape[1] == 1:
                    emMat = np.tile(emMat, (1, state.shape[1]))
                massEffect = (emMat * state) @ dynamics.dMat_sp
            dXdt += massEffect  # still shape (S, N)

            # Incorporate PSD updates
            #    Because BRelaxed is shape (S,),  reduce 'state' to shape (S,)
            avg_state = np.mean(state, axis=1)  ###shape (S,)

            #    Now do element-wise update:
            self.BRelaxed = self.BRelaxed + (avg_state - self.BRelaxed) * 0.1  
            self.logBRelaxed = np.log(np.maximum(self.BRelaxed, 1e-10))

            #    PSD states, invasion checks
            self.update_PSD_states(dynamics)
            self.handle_invasions(time_step=1.0)

            # Clean up edge cases
            dXdt[state < 0] = -state[state < 0]
            dXdt[~np.isfinite(dXdt)] = 0
            dXdt[np.abs(dXdt) > 1e10] = 0

            return dXdt.flatten()

        except Exception as e:
            logger.error(f"Error in RHS function: {e}")
            raise ValueError(f"Error in RHS function: {e}")
            

    def env_fluct(self):
        """
        Apply one timestep of environmental fluctuation.
        """

        dynamics = self.community_dynamics
        
        # Validate matrices using the CommunityDynamics instance
        dynamics.validate_matrices()
        
        #  scVec and scVec_prime have correct shapes**
        assert dynamics.scVec.shape == (1, self.spp.xMat.shape[1]), f"scVec shape mismatch: {dynamics.scVec.shape}, expected (1, {self.spp.xMat.shape[1]})"
        assert dynamics.scVec_prime.shape == (1, self.spp.xMat.shape[1]), f"scVec_prime shape mismatch: {dynamics.scVec_prime.shape}, expected (1, {self.spp.xMat.shape[1]})"
        logger.debug(f"Verified scVec and scVec_prime shapes before simulation: {dynamics.scVec.shape}, {dynamics.scVec_prime.shape}")

        # Define the initial state vector
        y0 = self.spp.xMat.flatten()

        # Define the RHS function with partial application of dynamics
        rhs_partial = partial(self.compute_rhs, dynamics=dynamics)

        # Initialize the solver with correct parameters
        problem = Explicit_Problem(rhs_partial, y0, t0=0.0)  # Simulate for 1 timestep
        logger.debug(f"Initialized Explicit_Problem with t0=0.0 and y0 shape: {y0.shape}")

        # Initialize the solver
        solver = CVode(problem)
        solver.atol = 1e-5
        solver.rtol = 1e-5
        solver.maxh = 0.1
        logger.debug("Initialized CVode solver with atol=1e-5, rtol=1e-5, maxh=0.1")

        # Run the simulation
        try:
            solver.simulate(1.0)  # 'tf' is correctly passed here
            logger.debug("Simulation started.")

            # Extract and reshape solver output
            if solver.y.ndim == 1:
                if solver.y.size != self.spp.xMat.size:
                    raise ValueError(f"Solver output size {solver.y.size} does not match xMat size {self.spp.xMat.size}.")
                tmp = solver.y.reshape(self.spp.xMat.shape)
                logger.debug(f"Reshaped solver output to {tmp.shape}")
            else:
                tmp = solver.y[-1].reshape(self.spp.xMat.shape)
                logger.debug(f"Reshaped last solver output to {tmp.shape}")

            # Ensure trajectories match dimensions before stacking
            if self.storeTraj and self.spp.trajectories.shape[1] != tmp.shape[1]:
                logger.warning(
                    f"Shape mismatch: trajectories has {self.spp.trajectories.shape[1]} columns, "
                    f"but tmp has {tmp.shape[1]} columns. Resetting trajectories."
                )
                self.spp.trajectories = np.zeros((0, tmp.shape[1]))
                logger.debug(f"Reset trajectories to shape {self.spp.trajectories.shape}")

            # Store trajectory 
            if self.storeTraj:
                self.spp.trajectories = np.vstack((self.spp.trajectories, tmp))  # Store trajectory
                self.PSD_trajectories.append({
                    'time': 1.0,  # Updated to current time after simulation
                    'PSD_states': self.PSD_states.copy(),
                    'PoissonClocks': self.PoissonClocks.copy(),
                    'BRelaxed': self.BRelaxed.copy(),
                    'logBRelaxed': self.logBRelaxed.copy()
                })
                logger.debug("Trajectory and PSD states recorded.")

            # Compute RE matrix (rMat + efMat) and validate dimensions
            RE = self.spp.rMat + self.spp.efMat
            RE = RE.reshape(1, -1)
            logger.debug(f"Computed RE matrix shape: {RE.shape}")

            if self.spp.fluctuations.shape[1] != RE.shape[1]:
                logger.warning(
                    f"Fluctuation shape mismatch: fluctuations has {self.spp.fluctuations.shape[1]} columns, "
                    f"RE has {RE.shape[1]} columns. Resetting fluctuations."
                )
                self.spp.fluctuations = np.zeros((0, RE.shape[1]))
                logger.debug(f"Reset fluctuations to shape {self.spp.fluctuations.shape}")

            self.spp.fluctuations = np.vstack((self.spp.fluctuations, RE))  # Store current fluctuations
            logger.debug("Fluctuations updated.")

        except Exception as e:
            logger.error(f"Error during environmental fluctuation simulation: {e}")
            raise


           
    def warming(self, dTdt, res, time):
        """
        Updates temperature gradient and rMat, then simulates multiple timesteps incorporating the PSD scheme.
        
        Args:
            dTdt (float): Rate of temperature increase.
            res (int): Number of steps per unit time.
            time (int): Total time for warming simulation.
        """
        import os

        # Define path to save outputs
        pos1 = self.bMatFileName.rfind("/")
        bMatDir = self.bMatFileName[:pos1] if pos1 != -1 else "."
        pos1 = self.bMatFileName.rfind(")")
        pos2 = self.bMatFileName.rfind(".")
        newFolder = self.bMatFileName[pos1 + 1:pos2] if pos1 != -1 and pos2 != -1 else "default_folder"
        Bpath = os.path.join(bMatDir, newFolder, f"dTdt={dTdt}")

        if not os.path.exists(Bpath):
            os.makedirs(Bpath)
        logger.debug(f"Saving warming outputs to {Bpath}")

        # Initial dimensions based on rMat
        species_count, node_count = self.spp.rMat.shape

        #  Force scVec_prime to match (1, node_count), because CommunityDynamics
        # expects scVec_prime to be (1, N), not (S, N). 
        if self.spp.scVec_prime.shape != (1, node_count):
            logger.warning(f"Aligning scVec_prime to (1,{node_count})")
            self.spp.scVec_prime = np.zeros((1, node_count))

        # Force scVec to match (1, node_count). single row vector.
        if self.spp.scVec.shape != (1, node_count):
            logger.warning(f"Aligning scVec to (1,{node_count})")
            self.spp.scVec = np.zeros((1, node_count))

        #  matrices align with rMat, xMat, cMat
        if self.spp.xMat.shape != (species_count, node_count):
            logger.warning(f"Aligning xMat to ({species_count},{node_count})")
            self.spp.xMat = np.zeros((species_count, node_count))
        if self.spp.efMat.shape != (species_count, node_count):
            logger.warning("Aligning efMat to rMat dimensions.")
            self.spp.efMat = np.zeros((species_count, node_count))
        if self.spp.cMat.shape != (species_count, species_count):
            logger.warning("Aligning cMat to square shape.")
            self.spp.cMat = np.zeros((species_count, species_count))

        # Initialize CommunityDynamics with stable, consistent dimensions
        dynamics = CommunityDynamics(
            spp=self.spp,
            xMat=self.spp.xMat,
            cMat=self.spp.cMat,
            rMat=self.spp.rMat,
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
                # Reshape solver state to match xMat shape (S, N)
                state = y.reshape(self.spp.xMat.shape)

                #  Compute intrinsic growth rates using CommunityDynamics
                Gt = dynamics.compute_intrinsic_growth_rates(state)  # shape (S, N)
                dXdt = Gt.copy()

                #  Mass effect and dispersal
                if dynamics.emMat is not None:
                    #  reshape emMat to (S, N) or broadcast to match 'state'
                    
                    #    emMat_N = np.tile(dynamics.emMat[:, np.newaxis], (1, state.shape[1]))
                    #original emMat was shape (S,). So let's handle shape 
                    em = dynamics.emMat
                    if em.ndim == 1 and em.shape[0] == state.shape[0]:
                        # Broadcast across nodes
                        emMat_N = np.tile(em[:, np.newaxis], (1, state.shape[1]))
                    else:
                        # If emMat is already (S,N), keep it
                        emMat_N = em
                        if emMat_N.shape != state.shape:
                            logger.warning(f"Reshaping emMat to match state shape: {state.shape}")
                            emMat_N = np.zeros_like(state)  ### fallback to zeros

                    massEffect = (emMat_N * state) @ dynamics.dMat_sp
                else:
                    massEffect = state @ dynamics.dMat_sp

                dXdt += massEffect  # shape (S, N)

                # Incorporate PSD-related updates
                #  If BRelaxed is shape (S,), average state over nodes first.
                avg_state = np.mean(state, axis=1)  ### shape (S,)
                self.BRelaxed = self.BRelaxed + (avg_state - self.BRelaxed) * 0.1
                self.logBRelaxed = np.log(np.maximum(self.BRelaxed, 1e-10))

                # Update PSD states and handle invasions
                self.update_PSD_states(dynamics)
                self.handle_invasions(time_step=1.0)

                # Handle edge cases
                dXdt[state < 0] = -state[state < 0]  # fix negative values
                dXdt[~np.isfinite(dXdt)] = 0         # replace non-finite with 0
                dXdt[np.abs(dXdt) > 1e10] = 0        # cap extreme values

                return dXdt.flatten()

            except Exception as e:
                logger.error(f"Error in RHS function during warming: {e}")
                raise ValueError(f"Error in RHS function during warming: {e}")

        # Simulation loop for warming
        for step in range(res):
            try:
                # Update temperature
                self.spp.topo.T_int += dTdt
                self.spp.update_r_vec_temp()

                #rMat didn't change dimension
                new_species_count, new_node_count = self.spp.rMat.shape
                if (new_species_count != species_count) or (new_node_count != node_count):
                    raise ValueError(
                        f"Dimensions of rMat changed from ({species_count},{node_count}) to "
                        f"({new_species_count},{new_node_count}). Must remain constant."
                    )

                # Reassign updated rMat to dynamics
                dynamics.rMat = self.spp.rMat

                # (Re-)Initialize solver
                initial_state = self.spp.xMat.flatten()
                problem = Explicit_Problem(rhs, initial_state, step)
                solver = CVode(problem)
                solver.atol = 1e-5
                solver.rtol = 1e-5
                solver.maxh = 0.1

                # Simulate one sub-timestep
                solver.simulate(step + 1)

                # Validate output
                if solver.y.size != (species_count * node_count):
                    raise ValueError(
                        f"Solver output size mismatch: expected {species_count * node_count}, "
                        f"got {solver.y.size}."
                    )
                self.spp.xMat = solver.y.reshape((species_count, node_count))

                # storing trajectories, record them
                if self.storeTraj:
                    self.trajectories.append({
                        'time': step + 1,
                        'biomass': self.spp.xMat.copy(),
                        'PSD_states': self.PSD_states.copy()
                    })
                    self.PSD_trajectories.append({
                        'time': step + 1,
                        'PoissonClocks': self.PoissonClocks.copy(),
                        'BRelaxed': self.BRelaxed.copy(),
                        'logBRelaxed': self.logBRelaxed.copy()
                    })
                    logger.debug(f"Warming step {step + 1} recorded.")

            except Exception as e:
                logger.error(f"Error during warming step {step}: {e}")
                raise

        # Save outputs after warming
        Bfile = os.path.join(Bpath, f"bMat_w{time}.mat")
        Rfile = os.path.join(Bpath, f"rMat_w{time}.mat")
        np.savetxt(Bfile, self.spp.xMat, fmt='%e')
        np.savetxt(Rfile, self.spp.rMat, fmt='%e')
        logger.info(f"Warming simulation outputs saved to {Bfile} and {Rfile}.")

        if time == 0:
            filenameP = os.path.join(Bpath, "pars.mat")
            self.write_params(filenameP)
            logger.debug(f"Parameters saved to {filenameP}.")


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
        Removes nodes from the network and simulates the dynamics incorporating the PSD scheme.
        
        Parameters:
            tMax (float): Maximum simulation time after node removal.
            no_removals (int): Number of nodes to remove.
        """
        self.storeTraj = 3  # Ensure trajectory storage is enabled for PSD tracking
        if self.spp.topo.scVec.shape[0] > 0 and self.spp.topo.scVec_prime.shape[0] > 0:
            print("\nLocal interaction matrices scaled")

        pos1 = self.bMatFileName.rfind("/")
        bMatDir = self.bMatFileName[:pos1] if pos1 != -1 else "."
        pos1 = self.bMatFileName.rfind(")")
        pos2 = self.bMatFileName.rfind(".")
        newFolder = (
            self.bMatFileName[pos1 + 1 : pos2] 
            if pos1 != -1 and pos2 != -1 
            else "nodeRemoval"
        )
        Bpath = os.path.join(bMatDir, newFolder, "nodeRemoval")

        if not os.path.exists(Bpath):
            os.makedirs(Bpath)
        logger.debug(f"Saving node removal outputs to {Bpath}")

        # Initial dimensions
        init_species_count, init_node_count = self.spp.xMat.shape

        def align_dimensions(species_count, node_count):
            """
            Helper function to fix dimension mismatches after node removal.
            """
            # 1) Ensure cMat is still square
            if self.spp.cMat.shape != (species_count, species_count):
                logger.warning("Realigning cMat to square shape after node removal.")
                self.spp.cMat = np.zeros((species_count, species_count))

            # 2) Ensure efMat matches rMat shape (S x N)
            if self.spp.efMat.shape != (species_count, node_count):
                logger.warning("Realigning efMat shape to match rMat after node removal.")
                self.spp.efMat = np.zeros((species_count, node_count))

            #  Ensure scVec_prime is shape (species_count, node_count) 
            
            if self.spp.scVec_prime.shape != (species_count, node_count):
                logger.warning(
                    f"Realigning scVec_prime to ({species_count},{node_count}) after node removal."
                )
                self.spp.scVec_prime = np.zeros((species_count, node_count))

            # Force scVec to be (1, node_count). 

            ### handle scVec that might be empty or 1D ###
            sc_temp = self.spp.topo.scVec

            if sc_temp.size == 0:
                #just initialize to zeros or ones of shape (1, node_count)
                logger.warning("scVec is empty. Initializing to zeros.")
                sc_temp = np.zeros((1, node_count))

            elif sc_temp.ndim == 1:
                #  shape (N,) or (something,)
                if sc_temp.shape[0] != node_count:
                    logger.warning(
                        f"1D scVec length {sc_temp.shape[0]} != node_count {node_count}. Forcing zeros."
                    )
                    sc_temp = np.zeros(node_count)
                #  safely reshape to (1,N)
                sc_temp = sc_temp.reshape(1, node_count)

            elif sc_temp.ndim == 2:
                # shape might be (S,N), or (1,N)
                #row count > 1, average across axis=0 => shape (N,)
                if sc_temp.shape[0] > 1:
                    sc_temp = sc_temp.mean(axis=0)  # now shape (N,)
                # If after averaging we have shape (N,), we must ensure N == node_count
                if sc_temp.ndim == 1:
                    if sc_temp.shape[0] != node_count:
                        logger.warning(
                            f"Averaged scVec has length {sc_temp.shape[0]}, "
                            f"expected node_count {node_count}. Forcing zeros."
                        )
                        sc_temp = np.zeros(node_count)
                    # reshape to (1,N)
                    sc_temp = sc_temp.reshape(1, node_count)
                else:
                    # if still 2D, check shape
                    if sc_temp.shape != (1, node_count):
                        logger.warning(
                            f"scVec had shape {sc_temp.shape}, forcing to (1,{node_count})."
                        )
                        sc_temp = np.zeros((1, node_count))

            else:
                # sc_temp.ndim > 2? Unusual. Force (1, node_count)
                logger.warning(f"scVec has ndim={sc_temp.ndim}, forcing (1,{node_count}).")
                sc_temp = np.zeros((1, node_count))

            self.spp.topo.scVec = sc_temp  # Now guaranteed shape (1, node_count)

            # Force scVec_prime to be (1, node_count)
            ###handle scVec_prime that might be empty or 1D ###
            sc_temp = self.spp.topo.scVec_prime
            if sc_temp.size == 0:
                logger.warning("scVec_prime is empty. Initializing to zeros.")
                sc_temp = np.zeros((1, node_count))

            elif sc_temp.ndim == 1:
                if sc_temp.shape[0] != node_count:
                    logger.warning(
                        f"1D scVec_prime length {sc_temp.shape[0]} != node_count {node_count}. Forcing zeros."
                    )
                    sc_temp = np.zeros(node_count)
                sc_temp = sc_temp.reshape(1, node_count)

            elif sc_temp.ndim == 2:
                if sc_temp.shape[0] > 1:
                    sc_temp = sc_temp.mean(axis=0)  # shape (N,)
                if sc_temp.ndim == 1:
                    if sc_temp.shape[0] != node_count:
                        logger.warning(
                            f"Averaged scVec_prime length {sc_temp.shape[0]}, forcing zeros."
                        )
                        sc_temp = np.zeros(node_count)
                    sc_temp = sc_temp.reshape(1, node_count)
                else:
                    if sc_temp.shape != (1, node_count):
                        logger.warning(
                            f"scVec_prime had shape {sc_temp.shape}, forcing (1,{node_count})."
                        )
                        sc_temp = np.zeros((1, node_count))
            else:
                logger.warning(
                    f"scVec_prime has ndim={sc_temp.ndim}, forcing (1,{node_count})."
                )
                sc_temp = np.zeros((1, node_count))

            self.spp.topo.scVec_prime = sc_temp  # Now guaranteed shape (1, node_count)

            #matrices match new dimensions
            optional_matrices = ['sMat', 'tMat', 'emMat', 'ouMat']
            for mat_name in optional_matrices:
                mat = getattr(self.spp, mat_name, None)
                if mat is not None and mat.ndim == 2:
                    if mat.shape[1] != node_count:
                        logger.warning(
                            f"Realigning {mat_name} to have {node_count} columns after node removal."
                        )
                        setattr(self.spp, mat_name, np.zeros((species_count, node_count)))


    def gen_jacobian(self):
        """
        Generate the numerical approximation of the Jacobian matrix for computing regional competitive overlap matrix,
        incorporating the PSD scheme.
        """
        try:
            dynamics = CommunityDynamics(
                spp=self.spp,
                xMat=self.spp.xMat,
                cMat=self.spp.cMat,
                rMat=self.spp.rMat,
                dMat=self.spp.dMat,
                efMat=self.spp.efMat,
                scVec=self.spp.topo.scVec if self.spp.topo.scVec.shape[0] > 0 else None,
                scVec_prime=self.spp.topo.scVec_prime if self.spp.topo.scVec_prime.shape[0] > 0 else None
            )

            # Ensure dimensions are consistent
            species_count, node_count = self.spp.xMat.shape

            # emMat should be 1D: (species_count,)
            if dynamics.emMat is not None:
                if dynamics.emMat.ndim > 1:
                    logger.warning("Reshaping emMat to 1D vector for dimension consistency.")
                    # emMat is uniform across nodes 
                    # Adjust
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
                dXdt[state.sum(axis=1) < 0] = -state[state.sum(axis=1) < 0]
                dXdt[~np.isfinite(dXdt)] = 0
                dXdt[np.abs(dXdt) > 1e10] = 0

                return dXdt.flatten()

            # Finite difference Jacobian approximation
            y0 = self.spp.xMat.flatten()
            epsilon = 1e-6  # Perturbation size
            n = y0.size
            jacobian = np.zeros((n, n))

            logger.debug("Starting Jacobian computation using finite differences.")

            for i in range(n):
                if i % 1000 == 0 and i > 0:
                    logger.debug(f"Computing Jacobian column {i}/{n}")
                perturb = np.zeros(n)
                perturb[i] = epsilon

                f_plus = compute_rhs(y0 + perturb)
                f_minus = compute_rhs(y0 - perturb)

                # Finite difference approximation
                jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)

            self.jacobian = jacobian
            logger.info(f"Jacobian matrix successfully computed with shape: {self.jacobian.shape}")

        except Exception as e:
            logger.error(f"Error during Jacobian generation: {e}")
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

            # Define filenames for other matrices
            filenames = {
                'rMat': f"{base_filename}rMat{self.rep}.mat",
                'sMat': f"{base_filename}sMat{self.rep}.mat",
                'bMat': f"{base_filename}bMat{self.rep}.mat",
                'bMat_c': f"{base_filename}bMat_c{self.rep}.mat",
                'params': f"{base_filename}params{self.rep}.mat",
                'network': f"{base_filename}network{self.rep}.mat",
                'trajec': f"{base_filename}trajec{self.rep}.mat",
                'envFluct': f"{base_filename}envFluct{self.rep}.mat",
                'S': f"{base_filename}S{self.rep}.mat",
                'dMat': f"{base_filename}dMat{self.rep}.mat",
                'emMat': f"{base_filename}emMat{self.rep}.mat",
                'emMat_c': f"{base_filename}emMat_c{self.rep}.mat",
                'cMat_reg': f"{base_filename}cMat_reg{self.rep}.mat",
                'tMat': f"{base_filename}tMat{self.rep}.mat",
                'envMat': f"{base_filename}envMat{self.rep}.mat",
                'bMat_src': f"{base_filename}bMat_src{self.rep}.mat",
                'bMat_c_src': f"{base_filename}bMat_c_src{self.rep}.mat",
                'scVec': f"{base_filename}scVec{self.rep}.mat",
                'scVec_prime': f"{base_filename}scVec_prime{self.rep}.mat"
            }

            # Save scVec and scVec_prime
            savemat(filenames['scVec'], {'scVec': self.spp.scVec})
            savemat(filenames['scVec_prime'], {'scVec_prime': self.spp.scVec_prime})
            logger.debug(f"Saved scVec and scVec_prime to {filenames['scVec']} and {filenames['scVec_prime']} respectively.")
            
            print(f"Biomass matrix file name: {self.bMatFileName}")
        else:
            # Use existing bMatFileName to generate paths for other matrices
            filenames = {key: self.bMatFileName.replace("bMat", key) for key in [
                'bMat_c', 'dMat', 'emMat', 'emMat_c', 'cMat', 'cMat_reg',
                'aMat', 'network', 'params', 'rMat', 'sMat', 'S',
                'tMat', 'envMat', 'trajec', 'envFluct', 'bMat_src', 'bMat_c_src',
                'scVec', 'scVec_prime'
            ]}
        
        # Store matrix objects accordingly
        if self.spp.trajectories is not None and self.spp.trajectories.size != 0:
            savemat(filenames['trajec'], {'trajectories': self.spp.trajectories})

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
            savemat(filenames['envFluct'], {'fluctuations': self.spp.fluctuations})

        # Save biomass matrices
        B_p = self.spp.xMat[:self.spp.S_p, :]
        savemat(self.bMatFileName, {'B_p': B_p})

        # Save scVec and scVec_prime
        savemat(filenames['scVec'], {'scVec': self.spp.scVec})
        savemat(filenames['scVec_prime'], {'scVec_prime': self.spp.scVec_prime})
        logger.debug(f"Saved scVec and scVec_prime to {filenames['scVec']} and {filenames['scVec_prime']} respectively.")

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
            filenames = {
                'bMat_c': self.bMatFileName.replace("bMat", "bMat_c"),
                'dMat': self.bMatFileName.replace("bMat", "dMat"),
                'emMat': self.bMatFileName.replace("bMat", "emMat"),
                'emMat_c': self.bMatFileName.replace("bMat", "emMat_c"),
                'cMat': self.bMatFileName.replace("bMat", "cMat"),
                'aMat': self.bMatFileName.replace("bMat", "aMat"),
                'network': self.bMatFileName.replace("bMat", "network"),
                'params': self.bMatFileName.replace("bMat", "params"),
                'rMat': self.bMatFileName.replace("bMat", "rMat"),
                'sMat': self.bMatFileName.replace("bMat", "sMat"),
                'S': self.bMatFileName.replace("bMat", "S"),
                'tMat': self.bMatFileName.replace("bMat", "tMat"),
                'envMat': self.bMatFileName.replace("bMat", "envMat"),
                'trajec': self.bMatFileName.replace("bMat", "trajec"),
                'invProb_p': self.bMatFileName.replace("bMat", "invProb_p"),
                'invProb_c': self.bMatFileName.replace("bMat", "invProb_c"),
                'scVec': self.bMatFileName.replace("bMat", "scVec"),
                'scVec_prime': self.bMatFileName.replace("bMat", "scVec_prime")
            }

            # Load matrix objects
            self.spp.xMat = sio.loadmat(self.bMatFileName)['B_p']
            self.spp.S_p = self.spp.xMat.shape[0]

            self.spp.dMat = sio.loadmat(filenames['dMat'])['dMat']
            if os.path.exists(filenames['cMat']):
                self.spp.cMat = sio.loadmat(filenames['cMat'])['C']
            if os.path.exists(filenames['emMat']):
                self.spp.emMat = sio.loadmat(filenames['emMat'])['Em_p']
            self.spp.topo.network = sio.loadmat(filenames['network'])['network']
            if os.path.exists(filenames['tMat']):
                self.spp.tMat = sio.loadmat(filenames['tMat'])['tMat']
            if os.path.exists(filenames['envMat']):
                self.spp.topo.envMat = sio.loadmat(filenames['envMat'])['envMat']
            self.spp.rMat = sio.loadmat(filenames['rMat'])['rMat']
            if os.path.exists(filenames['sMat']):
                self.spp.sMat = sio.loadmat(filenames['sMat'])['sMat']
            self.spp.sppRichness = sio.loadmat(filenames['S'])['sppRichness']

            print(f"\nImported network rows 0-4 = \n{self.spp.topo.network[:min(self.spp.topo.network.shape[0], 4)]}")
            self.spp.topo.genDistMat()
            print(f"\nSpecies richness, S_p = {self.spp.rMat.shape[0]}, S_c = {self.spp.xMat.shape[0] - self.spp.rMat.shape[0]}")

            # Load scVec and scVec_prime
            if os.path.exists(filenames['scVec']):
                self.spp.scVec = sio.loadmat(filenames['scVec'])['scVec'].reshape(1, -1)
                logger.debug(f"Loaded scVec with shape: {self.spp.scVec.shape}")
            else:
                logger.warning(f"scVec file {filenames['scVec']} not found. Initializing to ones.")
                self.spp.scVec = np.ones((1, self.spp.xMat.shape[1]))
            
            if os.path.exists(filenames['scVec_prime']):
                self.spp.scVec_prime = sio.loadmat(filenames['scVec_prime'])['scVec_prime'].reshape(1, -1)
                logger.debug(f"Loaded scVec_prime with shape: {self.spp.scVec_prime.shape}")
            else:
                logger.warning(f"scVec_prime file {filenames['scVec_prime']} not found. Initializing to ones.")
                self.spp.scVec_prime = np.ones((1, self.spp.xMat.shape[1]))

            # scVec and scVec_prime remain (1, nodes)**
            if self.spp.scVec.shape != (1, self.spp.xMat.shape[1]):
                logger.warning(f"scVec has shape {self.spp.scVec.shape}, expected (1, {self.spp.xMat.shape[1]}). Reshaping.")
                self.spp.scVec = self.spp.scVec.reshape(1, -1) if self.spp.scVec.ndim == 1 else self.spp.scVec.mean(axis=0, keepdims=True)
                logger.debug(f"Reshaped scVec to {self.spp.scVec.shape}.")

            if self.spp.scVec_prime.shape != (1, self.spp.xMat.shape[1]):
                logger.warning(f"scVec_prime has shape {self.spp.scVec_prime.shape}, expected (1, {self.spp.xMat.shape[1]}). Reshaping.")
                self.spp.scVec_prime = self.spp.scVec_prime.reshape(1, -1) if self.spp.scVec_prime.ndim == 1 else self.spp.scVec_prime.mean(axis=0, keepdims=True)
                logger.debug(f"Reshaped scVec_prime to {self.spp.scVec_prime.shape}.")

        except KeyError as e:
            print(f"Error while importing data: Missing key in MAT file - {e}")
            raise
        except FileNotFoundError as e:
            print(f"Error while importing data: File not found - {e}")
            raise






# Example usage

# Example Usage Scenario
if __name__ == "__main__":
    #Initialize Topography, Species, and Metacommunity
    # Create Topography
        # Initialize scaling vector with 2 entries
    scVec = np.linspace(0.05, 0.15, 2)  # Adjust as needed for your model

    # Initialize species scaling vectors (skVec) with envVar=2
    skVec = np.array([0.1, 0.2])  # match envVar=2
    
    topo = Topography(
        no_nodes=2,
        lattice_height=2,
        lattice_width=1,
        phi=1.0,
        envVar=2,
        var_e=1.0,
        randGraph=False,
        gabriel=True,
        scVec=np.array([0.05]),
        skVec=np.array([0.1, 0.2, 0.3]),
        T_int=25.0,
        network_file="",  #  ASCII file for network
        env_file="",  # file for environment
        sc_file="",  # file for scaling
        network_file_type='ascii',
        env_file_type='ascii',
        sc_file_type='ascii'
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
        bodymass=1e-4,  
        mu=0.2,  
    )

    meta_community = Metacommunity(spp,
        a_init=True,
        a_bMat="",
        a_xMat="",
        a_scMat="",
        a_invMax=100,
        a_tMax=500,
        a_outputDirectory="/Users/model/output/",
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
        a_rho=0.3,
        a_comp_dist=1,
        a_omega=0.9,
        a_dispNorm=1.0,
        a_no_nodes=2,
        a_lattice_height=2,
        a_lattice_width=1,
        a_phi=0.3,
        a_envVar=1,
        a_skVec=np.array([0.1]),
        a_var_e=0.5,
        a_randGraph=False,
        a_gabriel=True,
        a_T_int=25.0,
        a_envMat="", # Environment data file
        a_parOut=1,
        a_experiment="",
        a_rep=1,
        storeTraj=1,  # Enable storing of trajectories
        bMatFileName="/Users/model/output/",
  # Example file for environment
        sc_file="",
        g_block_transitions=False
    )

    # Simulate Metacommunity Dynamics
    print("\nStarting Metacommunity Dynamics Simulation...")
    meta_community.meta_c_dynamics(T=200)  # Simulate dynamics up to T=200

    # Invader Sampling
    print("\nInvader Sampling...")
    meta_community.invader_sample(trophLev=1, no_invaders=3)  # Sample 3 new producer species
    # producers
    meta_community.invader_sample(trophLev=0, no_invaders=3)
    
    # Remove a species
    print("\nRemoving species at indices 0 and 2...")
    meta_community.remove_species([10, 12])

    #  Environmental Fluctuation
    print("\nSimulating Environmental Fluctuation...")
    meta_community.env_fluct()  # Apply one timestep of environmental fluctuation

    # Warming Event Simulation
    print("\nSimulating Warming Event...")
    meta_community.warming(dTdt=0.5, res=5, time=20)  # Simulate warming with temperature increase rate of 0.5

    # Long-Distance Dispersal
    print("\nSimulating Long-Distance Dispersal...")
    meta_community.long_dist_disp(tMax=100, edges=3)  # Randomly add 3 dispersal edges and simulate for 100 timesteps

    #  Node Removal
    print("\nRemoving Nodes from Metacommunity...")
    meta_community.node_removal(tMax=100, no_removals=2)  # Remove 2 nodes from the community

    #  Generate and Save Model Outputs
    print("\nGenerating and Saving Outputs...")
    meta_community.gen_jacobian()  # Generate Jacobian matrix
    meta_community.gen_c_mat_reg(h=0.001)  # Generate regional interaction matrix
    meta_community.gen_source_sink(t_full_relax=1000)  # Generate source-sink matrix
    meta_community.saveMC()  # Save the metacommunity model matrices to files
    meta_community.print_params()  # Print parameters to console
    print("\nSimulation Completed.")