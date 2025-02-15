# -*- coding: utf-8 -*-
"""Species.py

Updated and completed version with all missing methods implemented.
"""

from topography import Topography  # Importing Topography class
from lvmcm_rng import LVMCM_rng



import numpy as np
import logging
from typing import Optional, Tuple, List
from scipy.io import loadmat
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Setup logging
logging.basicConfig(
    filename='species_simulation.log',  # Log to a file
    filemode='w',  # the log file each run
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)



class Species:
    def __init__(self, topo: 'Topography', scVec=None, dMat=None, tMat=None, c1=0.2, c2=0.25, c3=0.05, efMat=None, emRate=0.04,
                 dispL=0.3, pProducer=0.5, prodComp=True, symComp=False, alpha=0.2, sigma=0.5,
                 sigma_t=0.05, rho=0.3, comp_dist=0, omega=0.4,
                 dispNorm=1, delta_g=1.5, bodymass=1e-4, mu=0.001):
        """
        Initialize the Species class with topography and model parameters.
        """
        self.topo = topo
        no_nodes = self.topo.no_nodes

        # Initialize species-related matrices with zero species
        self.xMat = np.zeros((0, no_nodes))  # Biomass matrix
        self.rMat = np.zeros((0, no_nodes))  # Growth rates
        self.cMat = np.zeros((0, 0))         # Interaction matrix
        self.dMat = dMat if dMat is not None else np.zeros((no_nodes, no_nodes))
        self.efMat = np.zeros((0, no_nodes))  # Environmental factors
        self.ouMat = np.zeros((0, no_nodes))  # Ornstein-Uhlenbeck process
        self.sMat = np.zeros((0, no_nodes))   # Competition matrix
        self.tMat = tMat if tMat is not None else np.zeros((0, 2))  # Temperature optima and scaling factors
        self.trajectories = np.zeros((0, no_nodes))
        self.fluctuations = np.zeros((0, no_nodes))
        self.emMat = np.zeros((0, no_nodes))  # Emigration matrix

        # Parameters
        self.c1 = c1  # Interspecific competition parameter 1
        self.c2 = c2  # Interspecific competition parameter 2
        self.c3 = c3  # Interspecific competition parameter 3
        self.rho = rho  # Consumer mortality
        self.sigma = sigma  # Standard deviation of attack rate distribution
        self.alpha = alpha  # Base attack rate
        self.pProducer = pProducer  # Probability of invading producer species
        self.emRate = emRate  # Emigration rate
        self.dispL = dispL  # Dispersal length
        self.thresh = 0.2  # Detection/extinction threshold
        self.sigma_t = sigma_t  # Temporal autocorrelation (OU process)
        self.omega = omega  # Temperature niche width
        self.delta_g = delta_g  # Range of environmental optima
        self.bodymass = bodymass  # Body mass
        self.bodymass_inv = 1 / self.bodymass  # Inverse body mass
        self.mu = mu  # Mortality
        self.comp_dist = comp_dist  # Competition distribution type
        self.symComp = symComp  # Symmetric competition
        self.dispNorm = dispNorm  # Dispersal normalization method

        # Initialize species richness
        self.sppRichness = np.empty((0, 2))  # Each row: [S_p, S_c]

        # Initialize scVec and scVec_prime as empty arrays
        if scVec is not None and scVec.size != 0:
            raise ValueError("scVec must be empty at initialization.")
        self.scVec = np.ones((0, no_nodes))  # Start empty, rows added per species
        self.scVec_prime = np.zeros((0, no_nodes))

        # Initialize indices_S and indices_DP as empty 1D NumPy arrays
        self.indices_S = np.array([], dtype=int)  # For species indices
        self.indices_DP = np.array([], dtype=int)  # For derived parameters or specific indices
        
        # New PSD-related attributes
        self.poisson_clock = np.zeros((0, no_nodes))        # Poisson clocks per species per node
        self.establishment_prob = np.zeros((0, no_nodes))   # Establishment probabilities per species per node
        self.i = np.zeros((0, no_nodes))                    # Invasion rates per species per node
        self.waiting = np.zeros((0, no_nodes), dtype=bool)  # Waiting states per species per node

        self.body_mass = 0.15                    # Biomass units (can be parameterized)
        # Initialize logB as empty; will be updated upon invasions
        self.logB = np.array([])  # Log biomass, initially empty
       

        # Counters
        self.S_p = 0  # Producer species richness
        self.S_c = 0  # Consumer species richness
        self.I_p = 0  # Multispecies invasion counter (producers)
        self.I_c = 0  # Multispecies invasion counter (consumers)

        # Generate dispersal matrix during initialization
        self.gen_disp_mat()

        # If no producers, add default producer
        if self.S_p == 0:
            logger.info("Initializing with default invader as no producers exist.")
            self.invade(0)  # Add one default producer to ensure rMat is not empty

        # Initialize rMat with a default growth rate if there are producers
        if self.S_p > 0:
            self.rMat = np.array([self.gen_r_vec() for _ in range(self.S_p)])
            self.emMat = np.zeros_like(self.rMat)  # Initialize emMat with same number of species
            logger.debug(f"Initialized emMat with shape: {self.emMat.shape}")
        else:
            self.rMat = np.zeros((0, no_nodes))
            # self.efMat = np.zeros((0, no_nodes))
            self.emMat = np.zeros((0, no_nodes))
            logger.debug("Initialized rMat, efMat, and emMat as empty arrays.")

        # Ensure sMat and rMat dimensions align
        if self.rMat.shape != self.sMat.shape:
            logger.warning("sMat and rMat dimensions do not align. Resizing sMat.")
            self.sMat = np.tile(self.sMat, (self.rMat.shape[0], 1))

        # Validate dimensions of xMat and rMat
        if self.rMat.shape != self.xMat.shape:
            logger.warning("rMat and xMat dimensions do not align. Resizing xMat.")
            self.xMat = np.zeros_like(self.rMat)

        if self.tMat is None:
            self.tMat = np.zeros((0, 2))  # Initialize as empty matrix

        # Validate dimensions
        total_species = self.S_p + self.S_c
        if total_species > 0:
            assert self.xMat.shape == (total_species, no_nodes)
            assert self.rMat.shape == (total_species, no_nodes)
            assert self.cMat.shape == (total_species, total_species)
            assert self.sMat.shape == (total_species, no_nodes)
            assert self.scVec.shape == (total_species, no_nodes)
            assert self.scVec_prime.shape == (total_species, no_nodes)
            assert self.tMat.shape == (total_species, 2), "tMat must have 2 columns."

    def gen_r_vec(self, z_vec_ext=None):
        """
        Generate a spatially correlated random field for growth rates.

        Parameters:
        z_vec_ext (np.ndarray): Optional vector of random variables to control randomness.

        Returns:
        np.ndarray: Spatially autocorrelated and biologically meaningful growth rate vector.
        """
        logger.debug("Generating spatially correlated random growth vector.")

        desired_mean = 1.0  # Positive mean to ensure producers can grow
        desired_std = 0.5    # Standard deviation

        no_nodes = self.topo.no_nodes

        # Generate or use the provided random vector
        if z_vec_ext is None:
            z_vec = np.random.randn(no_nodes)  # Must be length no_nodes
            logger.debug("Generated new random vector for spatial correlation.")
        else:
            z_vec = z_vec_ext
            if z_vec.shape[0] != no_nodes:
                raise ValueError(f"z_vec_ext length {z_vec.shape[0]} does not match no_nodes={no_nodes}")
            logger.debug("Using external random vector for spatial correlation.")

        if not hasattr(self.topo, 'sigEVec') or not hasattr(self.topo, 'sigEVal'):
            raise ValueError("Topography eigen decomposition not initialized. Ensure sigEVec and sigEVal exist.")

        # Ensure sigEVal is a 1D array
        sigEVal = self.topo.sigEVal
        if sigEVal.ndim != 1:
            # If sigEVal is diagonal stored in another form, extract the diagonal
            # if sigEVal is (N,N), use np.diag:
            if sigEVal.ndim == 2 and sigEVal.shape[0] == sigEVal.shape[1]:
                sigEVal = np.diag(sigEVal)
            else:
                raise ValueError("sigEVal must be a 1D array or a square diagonal matrix.")

        if sigEVal.shape[0] != no_nodes:
            raise ValueError(f"sigEVal length {sigEVal.shape[0]} does not match no_nodes={no_nodes}")

        # Perform element-wise multiplication (N,) * (N,) -> (N,)
        scaled_z = sigEVal * z_vec  # Result should be shape (no_nodes,)

        # sigEVec should be (no_nodes, no_nodes) and scaled_z is (no_nodes,)
        r_i = self.topo.sigEVec @ scaled_z  # Result: (no_nodes,)

        if r_i.ndim != 1 or r_i.shape[0] != no_nodes:
            raise ValueError(f"r_i must be a 1D vector of length {no_nodes}, got shape {r_i.shape}")

        current_mean = np.mean(r_i)
        current_std = np.std(r_i)
        if current_std == 0:
            logger.error("Standard deviation of growth rates is zero. Cannot normalize.")
            raise ValueError("Standard deviation of growth rates is zero.")

        r_i = (r_i - current_mean) / current_std
        r_i = r_i * desired_std + desired_mean
        logger.debug(f"Shifted and scaled growth vector: mean={np.mean(r_i):.4f}, std={np.std(r_i):.4f}")

        min_growth = 0.5
        max_growth = 1.5
        r_i = np.clip(r_i, min_growth, max_growth)
        logger.debug(f"Clipped growth vector to range [{min_growth}, {max_growth}]: {r_i}")

        # Now r_i is (no_nodes,) and can be reshaped to (1, no_nodes)
        return r_i


    def update_r_vec_temp(self):
        """
        Update rMat to reflect abiotic changes due to temperature shifts.
        """
        logger.debug("Updating growth rate vector for temperature changes.")
        self.topo.gen_temp_grad()

        #  matrices are initialized
        if self.sMat.size == 0 or self.topo.envMat.size == 0 or self.tMat.size == 0:
            raise ValueError("One or more matrices (sMat, envMat, tMat) are not initialized or empty.")

        no_nodes = self.topo.no_nodes
        total_species = self.sMat.shape[0]

        if self.topo.envMat.shape[0] != no_nodes:
            raise ValueError(
                f"envMat rows ({self.topo.envMat.shape[0]}) do not match number of nodes ({no_nodes})."
            )
        if self.tMat.shape[0] != total_species:
            raise ValueError(
                f"tMat rows ({self.tMat.shape[0]}) do not match number of species ({total_species})."
            )

        # Shape them properly:
        envMat_SN = self.topo.envMat[:, np.newaxis]  # (no_nodes, 1)
        tMat_SN = self.tMat[:, 0][:, np.newaxis]     # (total_species, 1)

        logger.debug(f"sMat shape: {self.sMat.shape}, envMat_SN shape: {envMat_SN.shape}, "
                    f"tMat_SN shape: {tMat_SN.shape}")

        try:
            rMat_cc = self.sMat.copy()

            # Each species i, each node j:
            for i in range(total_species):
                for j in range(no_nodes):
                    if self.omega > 0:
                        rMat_cc[i, j] -= self.omega * (envMat_SN[j, 0] - tMat_SN[i, 0]) ** 2
                    else:
                        #  use the second column of tMat:
                        rMat_cc[i, j] -= (envMat_SN[j, 0] - tMat_SN[i, 0]) ** 2 * self.tMat[i, 1]

            # Clip negative values to at least -1
            rMat_cc = np.clip(rMat_cc, -1, None)
            self.rMat = rMat_cc
            logger.debug("Updated rMat for temperature changes.")

        except Exception as e:
            logger.error(f"Error during rMat computation: {e}")
            raise


    def gen_r_vec_quad(self):
        """
        Generate growth rate vector based on quadratic environmental response.

        Returns:
        np.ndarray: Quadratic environmental response growth rate vector.
        """
        logger.debug("Generating growth rate vector based on quadratic environmental response.")
        g_ik = np.zeros(self.topo.envVar)
        for k in range(self.topo.envVar):
            g_i = np.random.rand() * self.topo.range_env[k]
            g_i += self.topo.min_env[k]
            g_i *= self.delta_g
            g_ik[k] = g_i

        r_i = np.ones(self.topo.no_nodes)
        for k in range(self.topo.envVar):
            emg = self.topo.envMat[k, :] - g_ik[k]
            r_i -= self.topo.skVec[k] * (emg ** 2)

        r_i = np.clip(r_i, -1, None)

        if self.tMat is None or self.tMat.size == 0:
            self.tMat = g_ik[np.newaxis, :]
        else:
            self.tMat = np.vstack([self.tMat, g_ik])

        logger.debug(f"Quadratic growth vector: {r_i}")
        return r_i

    def ou_process(self):
        """
        Simulate abiotic turnover using an Ornstein-Uhlenbeck process.
        """
        logger.debug("Running Ornstein-Uhlenbeck process for abiotic turnover.")

        if self.ouMat is None or self.ouMat.shape != self.xMat.shape:
            self.ouMat = np.zeros_like(self.xMat)
        if self.efMat is None or self.efMat.shape != self.xMat.shape:
            self.efMat = np.zeros_like(self.xMat)

        zMat = np.random.randn(*self.efMat.shape)
        self.ouMat = (self.ouMat + self.sigma_t * zMat) / np.sqrt(1 + self.sigma_t ** 2)

        for i in range(self.rMat.shape[0]):
            self.efMat[i] = self.gen_r_vec()  # generate growth rates

        logger.debug("Updated `ouMat` and `efMat` using Ornstein-Uhlenbeck process.")

    def gen_r_vec_erf(self):
        """
        Generate growth rates based on environmental response function (ERF).
        """
        logger.debug("Generating growth rates using Environmental Response Function (ERF).")
        tol = np.random.randn(self.topo.envVar)
        tol /= np.sqrt(np.sum(tol**2)) * np.sqrt(self.topo.envVar)

        r_i = 1 + np.dot(tol, self.topo.envMat) / np.sqrt(2 * self.topo.envVar)
        r_i[r_i < -1] = -1  # Clip values below -1

        if self.tMat is None or self.tMat.size == 0:
            self.tMat = tol[np.newaxis, :]
        else:
            self.tMat = np.vstack([self.tMat, tol])

        logger.debug(f"Generated `r_i`: {r_i}")
        return r_i
    

    def invade(self, trophLev):
        """
        Introduce an invader species into the system with consistent initialization
        and interaction matrix updates, incorporating PSD scheme dynamics.

        Parameters:
        - trophLev (int): Trophic level of the invader (0 for producer, 1 for consumer).
        """
        logger.debug(f"Invading with trophic level {trophLev}.")
        # inv_biomass = np.random.uniform(5e-3, 2e-2)
        inv_biomass = 0.2
        variation = 0.2
        no_nodes = self.topo.no_nodes

        # Randomly scale each node in [1-variation, 1+variation]
        scale_factors = 1.0 + (np.random.rand(no_nodes) - 0.5) * 2 * variation
        scale_factors = np.clip(scale_factors, 0.5, 1.5)  # Some bounded random
        new_x = inv_biomass * scale_factors.reshape(1, no_nodes)
        self.xMat = np.vstack([self.xMat, new_x])

        # Generate a new growth rate vector
        new_r = self.gen_r_vec().reshape(1, no_nodes)
        self.rMat = np.vstack([self.rMat, new_r])

        # Update cMat
        total_species_before = self.S_p + self.S_c
        if self.cMat.size == 0:
            # First species
            self.cMat = np.zeros((1, 1))
        else:
            n_existing = self.cMat.shape[0]
            interaction_row = []
            for i in range(n_existing):
                if trophLev == 0:
                    # Invading producer
                    if i < self.S_p:
                        # Producer-Producer competition (negative)
                        interaction_coeff = np.random.uniform(-0.2, -0.05)
                    else:
                        # Producer relative to existing consumers
                        interaction_coeff = np.random.uniform(-0.01, 0.05)
                else:
                    # Invading consumer
                    if i < self.S_p:
                        # Consumer exploits producer (positive)
                        interaction_coeff = np.random.uniform(0.05, 0.2)
                    else:
                        # Consumer-Consumer competition (negative)
                        interaction_coeff = np.random.uniform(-0.1, -0.01)
                interaction_row.append(interaction_coeff)
            interaction_row = np.array(interaction_row)

            # Add the new row to cMat
            self.cMat = np.vstack([self.cMat, interaction_row])

            # Add a new column for the new species
            new_col = np.zeros((self.cMat.shape[0], 1))
            new_col[:-1, 0] = interaction_row  # Symmetric if symComp is True
            self.cMat = np.hstack([self.cMat, new_col])

            # Symmetric competition if enabled
            if self.symComp:
                self.cMat[-1, :-1] = interaction_row
                self.cMat[:-1, -1] = interaction_row

        # Update S_p or S_c
        if trophLev == 0:
            self.S_p += 1
            logger.debug("Producer added to the community.")
        else:
            self.S_c += 1
            logger.debug("Consumer added to the community.")

        # Add a row to sMat for this new species
        s_add = np.random.uniform(0.1, 1.0, (1, no_nodes))
        self.sMat = np.vstack([self.sMat, s_add])

        # Update scVec and scVec_prime to match the new species count
        add_vec = np.ones((1, no_nodes))
        self.scVec = np.vstack([self.scVec, add_vec])

        add_prime = np.zeros((1, no_nodes))
        self.scVec_prime = np.vstack([self.scVec_prime, add_prime])

        # Add a row to tMat for this new species
        temp_opt = np.random.uniform(0, 1)
        scaling_factor = np.random.uniform(0.05, 0.2)
        t_add = np.array([[temp_opt, scaling_factor]])
        self.tMat = np.vstack([self.tMat, t_add])

        # Initialize PSD attributes for the new species across all nodes
        new_poisson_clock = np.log(np.random.uniform(low=1e-3, high=1.0, size=no_nodes))  #  positive values before log
        new_establishment_prob = np.zeros(no_nodes)
        new_i = np.full(no_nodes, self.mu)
        new_waiting = np.ones(no_nodes, dtype=bool)

        if self.poisson_clock.size == 0:
            # First species
            self.poisson_clock = new_poisson_clock.reshape(1, -1)
            self.establishment_prob = new_establishment_prob.reshape(1, -1)
            self.i = new_i.reshape(1, -1)
            self.waiting = new_waiting.reshape(1, -1)
        else:
            # Append new species' state
            self.poisson_clock = np.vstack([self.poisson_clock, new_poisson_clock])
            self.establishment_prob = np.vstack([self.establishment_prob, new_establishment_prob])
            self.i = np.vstack([self.i, new_i])
            self.waiting = np.vstack([self.waiting, new_waiting])
            
        # create logB 
        if self.logB.size == 0:
            self.logB = np.log(new_x + 1e-10)
        else:
            # existing logB, so just append a new row
            new_logB = np.log(new_x + 1e-10)
            self.logB = np.vstack([self.logB, new_logB])


        # Validate final dimensions
        total_species = self.S_p + self.S_c
        assert self.xMat.shape == (total_species, no_nodes), f"xMat shape mismatch: {self.xMat.shape} vs {(total_species, no_nodes)}"
        assert self.rMat.shape == (total_species, no_nodes), f"rMat shape mismatch: {self.rMat.shape} vs {(total_species, no_nodes)}"
        assert self.cMat.shape == (total_species, total_species), "cMat dimension mismatch"
        assert self.sMat.shape == (total_species, no_nodes), f"sMat shape mismatch: {self.sMat.shape} vs {(total_species, no_nodes)}"
        assert self.scVec.shape == (total_species, no_nodes), f"scVec shape mismatch: {self.scVec.shape} vs {(total_species, no_nodes)}"
        assert self.scVec_prime.shape == (total_species, no_nodes), f"scVec_prime shape mismatch: {self.scVec_prime.shape} vs {(total_species, no_nodes)}"
        assert self.tMat.shape == (total_species, 2), f"tMat shape mismatch: {self.tMat.shape} vs {(total_species, 2)}"
        assert self.poisson_clock.shape == (total_species, no_nodes), f"poisson_clock shape mismatch: {self.poisson_clock.shape} vs {(total_species, no_nodes)}"
        assert self.establishment_prob.shape == (total_species, no_nodes), f"establishment_prob shape mismatch: {self.establishment_prob.shape} vs {(total_species, no_nodes)}"
        assert self.i.shape == (total_species, no_nodes), f"i shape mismatch: {self.i.shape} vs {(total_species, no_nodes)}"

        logger.info(f"Species invaded - Trophic Level: {trophLev}, Producers: {self.S_p}, Consumers: {self.S_c}.")
        logger.debug(f"xMat shape: {self.xMat.shape}, rMat shape: {self.rMat.shape}, cMat shape: {self.cMat.shape}, tMat shape: {self.tMat.shape}")

        # Perform consistency checks
        self.consistency_checks()

        return

    
    
    def consistency_checks(self):
        """
        Perform consistency checks to ensure that all matrices align correctly.
        """
        total_species = self.S_p + self.S_c
        assert self.xMat.shape[0] == total_species, f"xMat species count mismatch: {self.xMat.shape[0]} vs {total_species}"
        assert self.rMat.shape[0] == total_species, f"rMat species count mismatch: {self.rMat.shape[0]} vs {total_species}"
        assert self.cMat.shape == (total_species, total_species), f"cMat shape mismatch: {self.cMat.shape} vs {(total_species, total_species)}"
        assert self.sMat.shape == (total_species, self.topo.no_nodes), f"sMat shape mismatch: {self.sMat.shape} vs {(total_species, self.topo.no_nodes)}"
        assert self.scVec.shape == (total_species, self.topo.no_nodes), f"scVec shape mismatch: {self.scVec.shape} vs {(total_species, self.topo.no_nodes)}"
        assert self.scVec_prime.shape == (total_species, self.topo.no_nodes), f"scVec_prime shape mismatch: {self.scVec_prime.shape} vs {(total_species, self.topo.no_nodes)}"
        assert self.tMat.shape == (total_species, 2), f"tMat shape mismatch: {self.tMat.shape} vs {(total_species, 2)}"
        assert self.poisson_clock.shape == (total_species, self.topo.no_nodes), f"poisson_clock shape mismatch: {self.poisson_clock.shape} vs {(total_species, self.topo.no_nodes)}"
        assert self.establishment_prob.shape == (total_species, self.topo.no_nodes), f"establishment_prob shape mismatch: {self.establishment_prob.shape} vs {(total_species, self.topo.no_nodes)}"
        assert self.i.shape == (total_species, self.topo.no_nodes), f"i shape mismatch: {self.i.shape} vs {(total_species, self.topo.no_nodes)}"

        logger.debug("Consistency checks passed.")


    def extinct(self, wholeDom=True, species_to_remove=None):
        """
        Remove extinct species from the community.

        Parameters:
        - wholeDom (bool): Whether to check all species for extinction based on thresholds.
        - species_to_remove (array-like, optional): Indices of species to remove.

        Returns:
        - list: List containing arrays of removed producer and consumer indices.
        """
        if species_to_remove is None:
            species_to_remove = np.array([], dtype=int)

        indReturn = [None, None]  # Store extinction indices

        if wholeDom and species_to_remove.size == 0:
            logger.info("No specific species provided for extinction.")
            return indReturn

        ind_p = []
        ind_c = []

        for idx in species_to_remove:
            if idx < self.S_p:
                ind_p.append(idx)
                logger.debug(f"Producer species {idx} marked for extinction.")
            else:
                ind_c.append(idx)
                logger.debug(f"Consumer species {idx} marked for extinction.")

        ind_p = np.array(ind_p, dtype=int)
        ind_c = np.array(ind_c, dtype=int)
        indReturn[0] = ind_p
        indReturn[1] = ind_c

        species_to_remove = np.unique(species_to_remove)

        if species_to_remove.size > 0:
            logger.debug(f"Removing species at indices: {species_to_remove}")
            logger.info(f"Removing {len(species_to_remove)} extinct species from the community.")

            # Create a boolean mask for species to keep
            mask = np.ones(self.xMat.shape[0], dtype=bool)
            mask[species_to_remove] = False

            # Apply mask to all relevant matrices
            self.xMat = self.xMat[mask, :]
            self.rMat = self.rMat[mask, :]
            self.cMat = self.cMat[np.ix_(mask, mask)]
            self.sMat = self.sMat[mask, :]
            self.scVec = self.scVec[mask, :]
            self.scVec_prime = self.scVec_prime[mask, :]
            self.tMat = self.tMat[mask, :]
            self.waiting = self.waiting[mask, :]
            self.poisson_clock = self.poisson_clock[mask, :]
            self.establishment_prob = self.establishment_prob[mask, :]
            self.i = self.i[mask, :]
            self.logB = self.logB[mask, :]  #logB is updated consistently**

            # Update species counts
            self.S_p -= len(ind_p)
            self.S_c -= len(ind_c)

            logger.info(f"Removed {len(species_to_remove)} extinct species from the community.")
            logger.info(f"Extinct Producers Removed: {ind_p}")
            logger.info(f"Extinct Consumers Removed: {ind_c}")
        else:
            logger.info("No species extinct in this iteration.")

        # Update species richness
        new_row = np.array([[self.S_p, self.S_c]])
        if self.sppRichness.size == 0:
            self.sppRichness = new_row
            logger.debug("Initialized sppRichness.")
        else:
            self.sppRichness = np.vstack([self.sppRichness, new_row])
            logger.debug(f"Updated sppRichness: {self.sppRichness}")

        # Log current richness
        logger.info(f"Current Species Richness - Producers: {self.S_p}, Consumers: {self.S_c}")

        # Perform consistency checks
        self.consistency_checks()

        return indReturn

    @staticmethod
    def _update_indices(indices, extinct_indices, S_before):
        """
        Update indices for species after extinction events.

        Parameters:
        indices (array-like): Current indices.
        extinct_indices (array-like): Indices of extinct species.
        S_before (int): Number of species before extinction.

        Returns:
        np.ndarray: Updated indices.
        """
        indices = np.array(indices)
        extinct_indices = np.array(extinct_indices)
        # Update indices for extinct species
        updated_indices = indices.copy()
        for j, extinct in enumerate(extinct_indices):
            mask = updated_indices > extinct
            updated_indices[mask] -= 1
        return updated_indices

    def gen_disp_mat(self):
        """
        Generate the dispersal matrix with improved clarity, robustness, and validation.
        """
        logger.debug("Generating dispersal matrix.")

        if self.topo.no_nodes <= 1:
            # Handle single-node case
            self.dMat = np.zeros((1, 1))
            logger.debug("Dispersal matrix generated for a single node (no dispersal possible).")
            return

        if self.topo.adjMat is None:
            self.topo.gen_adj_mat()
            logger.debug("Adjacency matrix generated as it was not initialized.")

        # Compute dispersal matrix with exponential decay
        max_dist = np.max(self.topo.distMat)
        if max_dist == 0:
            logger.warning("All distances in distMat are zero. Check network generation.")
            self.dMat = np.zeros_like(self.topo.distMat)
            return

        self.dMat = np.exp(-self.topo.distMat / self.dispL)
        logger.debug(f"Initial dispersal matrix computed with dispersal length {self.dispL}.")

        # Apply adjacency constraints
        self.dMat *= self.topo.adjMat
        logger.debug("Dispersal matrix adjusted with adjacency matrix.")

        # Normalize dispersal rates
        kMat = np.zeros_like(self.dMat)
        for i in range(self.topo.no_nodes):
            neighbors = np.where(abs(self.topo.adjMat[:, i]) == 1)[0]
            num_neighbors = len(neighbors)
            if num_neighbors > 0:
                if self.dispNorm == 0:  # Effort-weighted dispersal
                    d_sum = np.sum(self.dMat[:, i])
                    if d_sum > 0:
                        for j in neighbors:
                            kMat[i, j] = self.emRate * self.dMat[j, i] / d_sum
                elif self.dispNorm == 1:  # Degree-weighted dispersal
                    for j in neighbors:
                        kMat[i, j] = self.emRate / num_neighbors
                elif self.dispNorm == 2:  # Passive dispersal
                    for j in neighbors:
                        kMat[i, j] = self.emRate

        # Multiply with kMat to finalize dispersal matrix
        self.dMat = self.dMat * kMat
        logger.debug(f"Dispersal matrix normalized using dispersal normalization method {self.dispNorm}.")

        # Include diagonal terms for self-dispersal
        # Final validation
        if not np.all(np.isfinite(self.dMat)):
            logger.error("Dispersal matrix contains invalid values (NaN or inf). Check parameters.")
            raise ValueError("Dispersal matrix contains invalid values.")

        logger.info("Dispersal matrix generated successfully.")
        logger.debug(f"Dispersal matrix: \n{self.dMat}")
                

    def update_biomass_psd(self, stepsize=1):
        if self.logB.size == 0:
            logger.error("logB is not initialized. Please initialize logB before running PSD simulation.")
            raise AttributeError("Species object has no attribute 'logB'.")

        # Convert log biomass to biomass
        B = np.exp(self.logB) * (~self.waiting)  # Shape: (species, nodes)
        B = np.maximum(B, 1e-5)  # Prevent extremely low biomass

        # Calculate local growth rates for all nodes
        interaction = self.cMat @ B  # Shape: (species, nodes)

        # Introduce a mortality rate
        death_rate = 0.5 # Adjust as needed
        local_growth_rate = self.rMat - interaction - death_rate  # Allow for negative growth rates
        local_growth_rate = np.maximum(local_growth_rate, -2.0)  # Allow slight negative rates

        # Update establishment probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            establishment_prob = 1 / (1 + self.mu / (local_growth_rate + 0.1))  # Avoid division by zero
        establishment_prob = np.nan_to_num(establishment_prob, nan=0.0, posinf=1.0, neginf=0.0)
        establishment_prob = np.clip(establishment_prob, 0, 1)
        self.establishment_prob = establishment_prob  # (species, nodes)
        logger.debug(f"Establishment Probabilities: {self.establishment_prob}")

        # Dynamic update of invasion rates based on establishment probabilities
        scaling_factor = 0.05  # Define an appropriate scaling factor
        self.i = self.establishment_prob * scaling_factor
        self.i = np.clip(self.i, 0.0, 1.0)
        logger.debug(f"Updated invasion rates (i): {self.i}")

        # Update Poisson clocks
        poisson_increment = (self.xMat / self.body_mass) * self.establishment_prob * stepsize  # (species, nodes)
        self.poisson_clock += poisson_increment  # (species, nodes)
        self.poisson_clock = np.clip(self.poisson_clock, a_min=0, a_max=None)  # Prevent negative values
        logger.debug(f"Updated Poisson Clocks: {self.poisson_clock}")

        # Determine invasions
        invaders = self.poisson_clock > 0  # (species, nodes)
        species_idx, node_idx = np.where(invaders)
        for s, n in zip(species_idx, node_idx):
            # Invasion occurs
            self.xMat[s, n] += self.body_mass  # Increment biomass
            # self.poisson_clock[s, n] = np.log(np.random.uniform())  # Reset Poisson clock
            self.poisson_clock[s, n] = -np.random.exponential(1.0)
            self.waiting[s, n] = False  # Exit waiting state
            self.logB[s, n] = np.log(self.xMat[s, n] + 1e-10)  # Update logB after invasion
            logger.debug(f"Invasion occurred for species {s} at node {n}. New biomass: {self.xMat[s, n]:.4f}")

        # Update biomass based on local_growth_rate
        d_logB = (local_growth_rate) * stepsize  # (species, nodes)
        self.logB += d_logB  # (species, nodes)

        # maximum and minimum cap for logB to prevent overflow and underflow
        logB_max = 10  # Define a maximum logB to prevent overflow
        logB_min = -10  # Define a minimum logB to prevent underflow
        self.logB = np.clip(self.logB, logB_min, logB_max)  # (species, nodes)
        logger.debug(f"Updated logB after growth and mortality: {self.logB}")

        # Convert logB back to biomass
        self.xMat = np.exp(self.logB) * (~self.waiting)  # (species, nodes)

        # Calculate total biomass per species
        species_biomass_sum = np.sum(self.xMat, axis=1)  # Sum across nodes for each species
        logger.debug(f"Species biomass sums: {species_biomass_sum}")

        extinct_species = species_biomass_sum <= self.thresh  # Boolean array

        if np.any(extinct_species):
            extinct_indices = np.where(extinct_species)[0]
            logger.debug(f"Species marked for extinction: {extinct_indices}")
            self.extinct(wholeDom=True, species_to_remove=extinct_indices)
        else:
            logger.info("No species extinct in this iteration.")

        # Perform consistency checks
        self.consistency_checks()

        return


    def simulate_psd(self, tmax=20000, stepsize=1, recording_stepsize=100):
        """
        Simulate population dynamics using the PSD scheme.

        Parameters:
        - tmax (int): Total simulation time.
        - stepsize (float): Time increment for each simulation step.
        - recording_stepsize (int): Interval at which to record data.
        """
        nsteps = int(tmax / stepsize)
        nrecords = int(tmax / recording_stepsize)
        record_number = 0

        # Check if there are species to simulate
        if self.xMat.shape[0] == 0:
            logger.error("No species present. Cannot simulate PSD.")
            raise ValueError("No species present. Cannot simulate PSD.")

        # Lists to store variable-shaped snapshots
        PSDtrajectory_list = []  # Each entry is a 2D array shape (current_species, nodes)
        PSDwaiting_list = []     # Each entry is also shape (current_species, nodes)

        for step_i in range(1, nsteps + 1):
            # If no species remain
            if self.xMat.shape[0] == 0:
                logger.info("No species left, stopping PSD simulation.")
                break

            # Update PSD
            self.update_biomass_psd(stepsize)

            # Recording step
            if step_i % recording_stepsize == 0:
                # Append the current xMat and waiting arrays
                PSDtrajectory_list.append(self.xMat.copy())
                PSDwaiting_list.append(self.waiting.copy())
                record_number += 1

                # Record species richness
                new_row = np.array([[self.S_p, self.S_c]])
                if self.sppRichness.size == 0:
                    self.sppRichness = new_row
                else:
                    self.sppRichness = np.vstack([self.sppRichness, new_row])

                logger.info(f"Recorded Species Richness at step {step_i} -> "
                            f"Producers: {self.S_p}, Consumers: {self.S_c}")

                if record_number % 10 == 0 or record_number == 1:
                    logger.info(f"PSD Simulation Progress: Record {record_number}/{nrecords}")
                    logger.debug(f"Time Step: {step_i}, "
                                f"Max logB: {np.max(self.logB):.4f}, Min logB: {np.min(self.logB):.4f}")
                    logger.debug(f"Establishment Probabilities (sample): {self.establishment_prob[:5, :5]}")
                    logger.debug(f"Poisson Clocks (sample): {self.poisson_clock[:5, :5]}")


        # 1) Determine the max number of species across all records
        max_species_across_time = max(snapshot.shape[0] for snapshot in PSDtrajectory_list) if PSDtrajectory_list else 0
        no_nodes = self.topo.no_nodes  # Typically same across all
        n_records = len(PSDtrajectory_list)

        def pad_array_2d(arr, target_species, target_nodes, pad_value=0.0, pad_bool=False):
            """
            Pad or truncate a 2D array (species, nodes) to shape (target_species, target_nodes).
            """
            curr_species, curr_nodes = arr.shape
            # Typically curr_nodes == target_nodes
            if curr_species < target_species:
                pad_shape = (target_species - curr_species, curr_nodes)
                if pad_bool:
                    padding = np.full(pad_shape, False, dtype=bool)
                else:
                    padding = np.full(pad_shape, pad_value, dtype=arr.dtype)
                arr_padded = np.vstack([arr, padding])
                return arr_padded
            elif curr_species > target_species:
                return arr[:target_species, :]  # Truncate if needed
            else:
                return arr

        # 2) Create final arrays with consistent shape: (n_records, max_species, nodes)
        PSDtrajectory = []
        PSDwaiting = []

        for snapshot_xMat, snapshot_wait in zip(PSDtrajectory_list, PSDwaiting_list):
            # Pad them to shape (max_species_across_time, no_nodes)
            xMat_padded = pad_array_2d(snapshot_xMat, max_species_across_time, no_nodes, pad_value=0.0, pad_bool=False)
            wait_padded = pad_array_2d(snapshot_wait, max_species_across_time, no_nodes, pad_value=False, pad_bool=True)

            PSDtrajectory.append(xMat_padded)
            PSDwaiting.append(wait_padded)

        if PSDtrajectory:
            self.PSDtrajectoryFull = np.array(PSDtrajectory)  # shape: (n_records, max_species, no_nodes)
            self.PSDwaitingFull = np.array(PSDwaiting)        # same shape
        else:
            # If no snapshots were recorded, store empty arrays
            self.PSDtrajectoryFull = np.zeros((0, 0, 0))
            self.PSDwaitingFull = np.zeros((0, 0, 0), dtype=bool)

        logger.debug(f"Final PSDtrajectoryFull shape: {self.PSDtrajectoryFull.shape}")
        logger.debug(f"Final PSDwaitingFull shape: {self.PSDwaitingFull.shape}")

        # Final consistency checks
        self.consistency_checks()

    
    def log_state_psd(self, current_time):
        """
        Log the current state of the species, including PSD-related attributes.
        
        Parameters:
        - current_time (float): The current simulation time.
        """
        logger.debug(f"Time: {current_time}")
        logger.debug(f"Producers: {self.S_p}, Consumers: {self.S_c}")
        logger.debug(f"xMat shape: {self.xMat.shape}")
        logger.debug(f"rMat shape: {self.rMat.shape}")
        logger.debug(f"cMat shape: {self.cMat.shape}")
        logger.debug(f"sMat shape: {self.sMat.shape}")
        logger.debug(f"tMat shape: {self.tMat.shape}")
        logger.debug(f"Waiting States: {self.waiting}")
        logger.debug(f"Poisson Clocks: {self.poisson_clock}")
        logger.debug(f"Establishment Probabilities: {self.establishment_prob}")
        
        
    def generate_visualizations(self):
        """
        Generate comprehensive visualizations for the PSD scheme simulation.
        """
        try:
            # Ensure that trajectories and richness data exist
            if not hasattr(self, 'PSDtrajectoryFull') or not hasattr(self, 'PSDwaitingFull'):
                logger.error("Simulation trajectories not found. Run simulate_psd first.")
                return
            if not hasattr(self, 'sppRichness'):
                logger.error("Species richness data not found.")
                return
            
            if self.PSDtrajectoryFull.size == 0:
                logger.error("PSDtrajectoryFull is empty. No visualizations will be generated.")
                return

            #  Plot Species Richness Over Time
            self.plot_species_richness()

            #  Plot Biomass Trajectories for Selected Species and Community Average
            self.plot_biomass_trajectories(num_species_to_plot=20)

            #  Plot Biomass Distribution at Final Time Point
            self.plot_biomass_distribution_final()

            #  Plot Heatmap of Biomass Over Time
            self.plot_biomass_heatmap()

            #  Plot Invasion and Extinction Events
            self.plot_invasion_extinction_events()

            #  Plot Final Biomass Over All Species
            self.plot_biomass_over_species()

            logger.info("All visualizations generated successfully.")

        except Exception as e:
            logger.error(f"Error during visualization generation: {e}")
            raise

    def plot_species_richness(self, save_fig=True):
        """
        Plot the number of producers and consumers over time.

        Parameters:
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            time = np.arange(self.sppRichness.shape[0]) * 100  # Adjust based on recording_stepsize
            S_p = self.sppRichness[:, 0]
            S_c = self.sppRichness[:, 1]

            plt.figure(figsize=(10, 6))
            plt.plot(time, S_p, label='Producers', color='green', linewidth=2)
            plt.plot(time, S_c, label='Consumers', color='red', linewidth=2)
            plt.title('Species Richness Over Time')
            plt.xlabel('Time')
            plt.ylabel('Number of Species')
            plt.legend()
            plt.grid(True)
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/species_richness_over_time.png', dpi=300)
                logger.info("Species richness plot saved as 'species_richness_over_time.png'.")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting species richness: {e}")
            raise

    def plot_biomass_trajectories(self, num_species_to_plot=10, save_fig=True):
        """
        Plot biomass trajectories for a subset of species and overall community average.

        Parameters:
        - num_species_to_plot (int): Number of individual species to plot.
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            n_records, n_species, n_nodes = self.PSDtrajectoryFull.shape
            time = np.arange(n_records) * 100  # adjust recording_stepsize

            if n_species == 0:
                logger.warning("No species present to plot biomass trajectories.")
                return

            # Select species to plot (random or first N)
            species_indices = np.random.choice(n_species, size=min(num_species_to_plot, n_species), replace=False)

            plt.figure(figsize=(12, 8))
            for idx in species_indices:
                # Plot biomass of species averaged over all nodes
                avg_biomass_species = np.mean(self.PSDtrajectoryFull[:, idx, :], axis=1)
                plt.plot(time, avg_biomass_species, alpha=0.6, label=f'Species {idx+1}')

            # Plot average biomass across all species and nodes
            avg_biomass = np.mean(self.PSDtrajectoryFull, axis=(1,2))
            plt.plot(time, avg_biomass, color='black', linewidth=2, label='Average Biomass')
            plt.title('PSD Biomass Trajectories of Selected Species and Community Average')
            plt.xlabel('Time')
            plt.ylabel('Biomass')
            plt.legend(loc='upper right', fontsize='small')
            plt.grid(True)
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/biomass_trajectories.png', dpi=300)
                logger.info("Biomass trajectories plot saved as 'biomass_trajectories.png'.")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting biomass trajectories: {e}")
            raise


    def plot_biomass_distribution_final(self, save_fig=True):
        """
        Plot the distribution of biomass across species at the final time point.

        Parameters:
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            final_biomass = np.mean(self.PSDtrajectoryFull[-1, :, :], axis=1)  # Average across nodes
            plt.figure(figsize=(10, 6))
            sns.histplot(final_biomass, bins=30, kde=True, color='skyblue')
            plt.title('PSD Biomass Distribution Across Species at Final Time Point')
            plt.xlabel('Biomass')
            plt.ylabel('Number of Species')
            plt.grid(True)
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/biomass_distribution_final.png', dpi=300)
                logger.info("Final biomass distribution plot saved as 'biomass_distribution_final.png'.")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting biomass distribution at final time point: {e}")
            raise


    def plot_biomass_heatmap(self, save_fig=True):
        """
        Plot a heatmap of biomass over time for all species, averaged across nodes.
        
        Parameters:
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            # Average biomass across nodes to reduce to 2D: (n_records, n_species)
            avg_biomass = np.mean(self.PSDtrajectoryFull, axis=2)  # Shape: (32, 301)
            
            # Transpose to have species on y-axis and time on x-axis
            avg_biomass = avg_biomass.T  # Shape: (301, 32)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(avg_biomass, cmap='viridis', cbar=True, 
                        xticklabels=5, yticklabels=False)
            plt.title('PSD Heatmap of Biomass Over Time for All Species (Averaged Across Nodes)')
            plt.xlabel('Time Steps')
            plt.ylabel('Species')
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/biomass_heatmap.png', dpi=300)
                logger.info("PSD Biomass heatmap plot saved as 'biomass_heatmap.png'.")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting biomass heatmap: {e}")
            raise



    def plot_invasion_extinction_events(self, save_fig=True):
        """
        Plot the changes in species richness to infer invasion and extinction events,
        and overlay individual extinction events from extinction_log.

        Parameters:
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            # Time array (adjust recording_stepsize)
            time = np.arange(self.sppRichness.shape[0]) * 100  # Example: step size of 100

            # Species richness
            S_p = self.sppRichness[:, 0]
            S_c = self.sppRichness[:, 1]

            # Compute differences to detect invasions (+1) and extinctions (-1)
            delta_S_p = np.diff(S_p, prepend=S_p[0])
            delta_S_c = np.diff(S_c, prepend=S_c[0])

            invasions_p = np.where(delta_S_p > 0)[0]
            extinctions_p = np.where(delta_S_p < 0)[0]
            invasions_c = np.where(delta_S_c > 0)[0]
            extinctions_c = np.where(delta_S_c < 0)[0]

            plt.figure(figsize=(14, 6))

            # Producer Invasions and Extinctions
            plt.subplot(1, 2, 1)
            plt.scatter(time[invasions_p], S_p[invasions_p], color='green', marker='^', label='Producer Invasion')
            plt.scatter(time[extinctions_p], S_p[extinctions_p], color='darkgreen', marker='v', label='Producer Extinction')

            # Overlay Extinction IDs for Producers
            if hasattr(self, 'extinction_log') and isinstance(self.extinction_log, dict) and self.extinction_log:
                for species_id, extinction_iter in self.extinction_log.items():
                    # Ensure extinction_iter is scalar
                    if np.isscalar(extinction_iter):
                        extinction_time = extinction_iter * 100  # Convert iteration to time
                        # Find the nearest time index
                        extinction_idx = np.argmin(np.abs(time - extinction_time))
                        plt.text(extinction_time, S_p[extinctions_p].max()*0.95, f'ID:{species_id}', 
                                fontsize=6, color='green', rotation=45)
                    else:
                        logger.warning(f"Extinction iteration for species {species_id} is not scalar.")

            plt.title('PSD Producer Invasion and Extinction Events')
            plt.xlabel('Time')
            plt.ylabel('Number of Producers')
            plt.legend()
            plt.grid(True)

            # Consumer Invasions and Extinctions
            plt.subplot(1, 2, 2)
            plt.scatter(time[invasions_c], S_c[invasions_c], color='red', marker='^', label='Consumer Invasion')
            plt.scatter(time[extinctions_c], S_c[extinctions_c], color='darkred', marker='v', label='Consumer Extinction')

            # Overlay Extinction IDs for Consumers
            if hasattr(self, 'extinction_log') and isinstance(self.extinction_log, dict) and self.extinction_log:
                for species_id, extinction_iter in self.extinction_log.items():
                    # Ensure extinction_iter is scalar
                    if np.isscalar(extinction_iter):
                        extinction_time = extinction_iter * 100  # Convert iteration to time
                        # Find the nearest time index
                        extinction_idx = np.argmin(np.abs(time - extinction_time))
                        plt.text(extinction_time, S_c[extinctions_c].max()*0.95, f'ID:{species_id}', 
                                fontsize=6, color='red', rotation=45)
                    else:
                        logger.warning(f"Extinction iteration for species {species_id} is not scalar.")

            plt.title('PSD Consumer Invasion and Extinction Events')
            plt.xlabel('Time')
            plt.ylabel('Number of Consumers')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/invasion_extinction_events.png', dpi=300)
                logger.info("Invasion and extinction events plot saved as 'invasion_extinction_events.png'.")
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting invasion and extinction events: {e}")
            raise

    def plot_biomass_over_species(self, save_fig=True):
        """
        Plot the final biomass of all species to visualize biomass distribution.

        Parameters:
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            # Check if PSDtrajectoryFull has the expected 3D shape
            if not hasattr(self, 'PSDtrajectoryFull') or self.PSDtrajectoryFull.ndim != 3:
                logger.error("PSDtrajectoryFull is not properly initialized or is not 3D.")
                raise ValueError("PSDtrajectoryFull must be a 3D array with shape (n_records, species, nodes).")

            # Aggregate biomass across nodes at the final time point
            final_biomass = np.mean(self.PSDtrajectoryFull[-1, :, :], axis=1)  # Shape: (species,)

            species_ids = np.arange(1, len(final_biomass) + 1)

            plt.figure(figsize=(14, 6))
            plt.bar(species_ids, final_biomass, color='skyblue')
            plt.title('PSD Final Biomass of All Species')
            plt.xlabel('Species ID')
            plt.ylabel('Biomass')
            plt.xlim(0, len(final_biomass) + 1)
            plt.grid(axis='y')
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/final_biomass_all_species.png', dpi=300)
                logger.info("Final PSD biomass of all species plot saved as 'final_biomass_all_species.png'.")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting final biomass over species: {e}")
            raise


    def plot_psd_trajectory(self, save_fig=True):
        """
        Plot the PSD biomass trajectories over time as a heatmap.

        Parameters:
        - save_fig (bool): Whether to save the plot as an image.
        """
        try:
            # Check if PSDtrajectoryFull has the expected 3D shape
            if not hasattr(self, 'PSDtrajectoryFull') or self.PSDtrajectoryFull.ndim != 3:
                logger.warning("PSD trajectory data not found or not properly initialized. Run simulate_psd first.")
                return

            # Aggregate biomass across nodesfor each species at each time step
            avg_biomass = np.mean(self.PSDtrajectoryFull, axis=2)  # Shape: (n_records, species)

            # Transpose to have species on the y-axis and time on the x-axis
            avg_biomass = avg_biomass.T  # Shape: (species, n_records)

            plt.figure(figsize=(12, 8))
            sns.heatmap(avg_biomass, cmap='viridis', cbar=True, 
                        xticklabels=5, yticklabels=False)
            plt.title('PSD Biomass Trajectory Heatmap (Averaged Across Nodes)')
            plt.xlabel('Time Steps')
            plt.ylabel('Species')
            if save_fig:
                plt.savefig('/Users/model/Species_plots_PSD_vis/PSD_Biomass_Trajectory_Heatmap.png', dpi=300)
                logger.info("PSD biomass trajectory heatmap saved as 'PSD_Biomass_Trajectory_Heatmap.png'.")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting PSD biomass trajectory heatmap: {e}")
            raise


    def main_visualization(self):
        """
        Main function to execute all visualization steps.
        """
        try:
            # Run simulation
            self.simulate_psd(tmax=2000, stepsize=1, recording_stepsize=100)
            
            # Optionally, log the current state
            self.log_state_psd(current_time=2000)
            
        except Exception as e:
            logger.error(f"Error in main visualization pipeline: {e}")
            raise
                
            
# Example usage
# instantiation of Topography in species.py
if __name__ == "__main__":
    #  initialize the Topography class 
    topo = Topography(
        no_nodes=2,
        lattice_height=2,
        lattice_width=1,
        phi=1.0,
        envVar=3,        
        skVec=np.array([0.1, 0.2, 0.3]),  # skVec is still 
        var_e=1.0,
        randGraph=True,
        gabriel=True,
        T_int=25.0,
        network_file="",
        sc_file="",  # file for scaling
        scVec=np.array([0.05]),
        consArea_bin=np.array([1 if i % 2 == 0 else 0 for i in range(2)]),  #  binary conservation area
        consArea_multiplicative=True  # Conservation area perturbation mode
    )

    # Generate the network explicitly
    topo.gen_network()

    # Create an instance of Species using the Topography instance
    species = Species(topo)
    species.gen_disp_mat()  # Generate dispersal matrix
    
    # Simulation Parameters
    time_steps = 200  # Define the number of simulation steps
    stepsize = 1
    recording_stepsize = 10
    
    # Initial Invasions to Populate the Community
    for _ in range(300):
        troph_level = np.random.choice([0, 1])  # Randomly choose producer or consumer
        species.invade(troph_level)
        
    # Initialize logB after initial invasions
 
    if len(species.xMat) > 0:
        species.logB = np.log(species.xMat + 1e-10)  # Correct: Shape (301, 32)
    else:
        logger.error("No species present after initial invasions.")
        raise ValueError("No species present after initial invasions.")
    
    # Initialize species richness
    if species.sppRichness.size == 0:
        species.sppRichness = np.array([[species.S_p, species.S_c]])
        logger.debug(f"Initialized sppRichness: {species.sppRichness}")

    
    # Run PSD Simulation
    species.simulate_psd(tmax=time_steps, stepsize=stepsize, recording_stepsize=recording_stepsize)
    
    # Initialize species richness
    # species.sppRichness = np.array([[species.S_p, species.S_c]])

    # Run PSD Simulation and Generate Visualizations

    species.generate_visualizations()



    # log the current state
    species.log_state_psd(current_time=time_steps)
    
    # except Exception as e:
    # logger.critical(f"Critical error in main execution: {e}")

    