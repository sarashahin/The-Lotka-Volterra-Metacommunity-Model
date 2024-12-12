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


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Species:
    def __init__(self, topo: Topography, scVec=None, dMat=None, tMat=None, c1=0.1, c2=0.2, c3=-1.0, efMat=None, emRate=0.05, dispL=1.0, 
                 pProducer=0.5, prodComp=True, symComp=False, alpha=0.5, sigma=0.1, sigma_t=0.05, rho=0.01, comp_dist=0, omega=0.5, 
                 dispNorm=0, delta_g=1.25, bodymass=1e-4, mu=0.1):
        self.topo = topo


        # Matrices and storage objects
        self.xMat = np.zeros((0, topo.no_nodes))  # Biomass matrix
        self.rMat = np.zeros((1, topo.no_nodes))  # Growth rate matrix (ensure at least one row initially)
        self.cMat = np.zeros((1, 1))  # Competition matrix
        self.dMat = dMat  # Dispersal matrix
        self.efMat = np.zeros_like(self.rMat)  # Environmental fluctuations matrix
        self.ouMat = np.zeros_like(self.rMat)  # Ornstein-Uhlenbeck matrix
        self.sMat = np.random.uniform(low=0.1, high=1.0, size=(1, topo.no_nodes))  # Abiotic growth matrix
        self.tMat = tMat if tMat is not None else np.zeros((1, topo.no_nodes))  # Temperature tolerance matrix
        self.trajectories = np.zeros((1, topo.no_nodes))  # Initialize as an empty array
        self.fluctuations = np.zeros((1, topo.no_nodes))  # Initialize fluctuations as an empty array
        logger.debug(f"Initialized sMat with shape: {self.sMat.shape}")
        self.emMat = np.zeros((self.xMat.shape[0], self.xMat.shape[1]))  # Initialize as a zero matrix
        


        # Parameters
        self.c1 = 0.1  # interspecific competition parameter 1
        self.c2 = 0.2  # interspecific competition parameter 2
        self.c3 = -1.0  # interspecific competition parameter 3
        # Initialize invasion counter
        self.invasion = 0
        self.rho = 0.01  # consumer mortality
        self.sigma = 0.1  # standard deviation of attack rate distribution
        self.alpha = 0.5  # base attack rate
        self.pProducer = 0.5  # probability of invading producer species
        self.emRate = 0.05  # emigration rate
        self.emMat = np.array([])
        self.dispL = 1.0  # dispersal length
        self.thresh = 1e-4  # detection/extinction threshold
        self.sigma_t = 0.05  # temporal autocorrelation (OU process)
        self.omega = 0.5  # temperature niche width
        self.delta_g = 1.25  # range of environmental optima
        self.bodymass = 1e-4  # body mass
        self.bodymass_inv = 1 / self.bodymass  # inverse body mass
        self.mu = 0.1  # mortality
        self.sppRichness = np.zeros((0, 2))
        self.scVec = scVec
        self.efMat = np.zeros_like(self.xMat)  # Environmental fluctuations matrix
        self.ouMat = np.zeros_like(self.xMat)  # Ornstein-Uhlenbeck matrix
        


        # Initialize additional attributes that might be needed in metacommunity dynamics
        self.indices_DP = []  # Define indices_DP as an empty list or array initially
        self.indices_S = []  # Defin

        # Switches
        self.prodComp = True  # producer competition
        self.comp_dist = 0  # competition distribution type
        self.symComp = False  # symmetric competition
        self.dispNorm = 0  # dispersal normalization method



        # Ensure dispersal matrix is generated during initialization
        self.gen_disp_mat()


        # Counters
        self.S_p = 1 # Producer species richness
        self.S_c = 0  # Consumer species richness
        self.I_p = 1  # Multispecies invasion counter (producers)
        self.I_c = 0  # Multispecies invasion counter (consumers)



        # Validate sMat during initialization
        # if self.sMat is None or self.sMat.size == 0:
        #     self.sMat = np.random.uniform(low=0.1, high=1.0, size=(1, topo.no_nodes))
        #     logger.warning("sMat was not initialized. Default values have been set.")
        # logger.debug(f"sMat initialized with shape: {self.sMat.shape}")


        if self.rMat.shape[0] != self.sMat.shape[0]:
            logger.warning("sMat and rMat dimensions do not align. Resizing sMat.")
            self.sMat = np.tile(self.sMat, (self.rMat.shape[0], 1))

        # Validate dimensions
        if self.rMat.shape != self.xMat.shape:
            logger.warning("rMat and xMat dimensions do not align. Resizing xMat.")
            self.xMat = np.zeros_like(self.rMat)


        if self.dMat is None or self.dMat.size == 0:
            logger.error("Species: Dispersal matrix (`dMat`) is empty or uninitialized.")
        else:
            logger.debug(f"Species: Dispersal matrix (`dMat`) initialized with shape: {self.dMat.shape}")




        # Initialize rMat with a default growth rate if there are no producers yet
        if self.S_p > 0:
            self.rMat = np.array([self.gen_r_vec() for _ in range(self.S_p)])
        else:
            logger.info("Initializing with default invader as no producers exist.")
            # self.invade(0)  # Adding a default producer to ensure rMat is not empty
            self.invade(0)  # Call `invade` with the required argument

        
        if self.tMat is None:
            self.tMat = np.zeros((0, self.topo.no_nodes))  # Initialize as empty


    def gen_r_vec(self, z_vec_ext=None):
        """
        Generate a spatially correlated random field for growth rates.

        Parameters:
        z_vec_ext (np.ndarray): Optional vector of random variables to control randomness.

        Returns:
        np.ndarray: Spatially autocorrelated and biologically meaningful growth rate vector.
        """
        logger.debug("Generating spatially correlated random growth vector.")
        
        # Default mean for random vector generation
        mu = 0.0
        
        # Generate or use the provided random vector
        if z_vec_ext is None:
            z_vec = np.random.randn(self.topo.no_nodes)  # Generate standard normal random vector
            logger.debug("Generated new random vector for spatial correlation.")
        else:
            z_vec = z_vec_ext
            logger.debug("Using external random vector for spatial correlation.")
        
        # Apply eigen decomposition for spatial autocorrelation
        try:
            r_i = self.topo.sigEVec @ (self.topo.sigEVal @ z_vec)
        except AttributeError:
            raise ValueError("Topography eigen decomposition not initialized. Ensure sigEVec and sigEVal are generated.")
        
        # Normalize growth rates to mean=0 and std=1
        r_i = (r_i - np.mean(r_i)) / np.std(r_i)
        logger.debug(f"Normalized growth vector: {r_i}")

        # Clip growth rates to ensure biologically meaningful bounds
        min_growth = -2.0  # Minimum growth rate
        max_growth = 3.0   # Maximum growth rate
        r_i = np.clip(r_i, min_growth, max_growth)
        logger.debug(f"Clipped growth vector to range [{min_growth}, {max_growth}]: {r_i}")

        return r_i



    def gen_r_vec_temp(self):
        """
        Generate growth rate vector based on temperature response.

        Returns:
        np.ndarray: Temperature-dependent growth rate vector.
        """
        logger.debug("Generating temperature-dependent growth rate vector.")
        t_i = np.random.rand() * self.topo.T_int
        t_vec = np.full(self.topo.no_nodes, t_i)

        s_i = self.gen_r_vec()  # Generate spatially autocorrelated abiotic growth rates

        if self.sMat is None:
            self.sMat = s_i[np.newaxis, :]
        else:
            self.sMat = np.vstack([self.sMat, s_i])

        if self.omega > 0:
            parab_width = (2 / self.omega) ** 2
            r_i = s_i - parab_width * (self.topo.env_mat[0] - t_vec) ** 2
            r_i = np.clip(r_i, -1, None)

            if self.tMat is None:
                self.tMat = np.array([[t_i]])
            else:
                self.tMat = np.vstack([self.tMat, [t_i]])
        else:
            logger.error("Dynamic omega sampling not implemented in this method.")
            raise NotImplementedError("Dynamic omega sampling is not supported yet.")

        logger.debug(f"Temperature-dependent growth vector: {r_i}")
        return r_i



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
            emg = self.topo.env_mat[k, :] - g_ik[k]
            r_i -= self.topo.skVec[k] * (emg ** 2)

        r_i = np.clip(r_i, -1, None)

        if self.tMat is None:
            self.tMat = g_ik[np.newaxis, :]
        else:
            self.tMat = np.vstack([self.tMat, g_ik])

        logger.debug(f"Quadratic growth vector: {r_i}")
        return r_i



    def update_r_vec_temp(self):
        """
        Update `rMat` to reflect abiotic changes due to temperature shifts.
        """
        logger.debug("Updating growth rate vector for temperature changes.")
        self.topo.gen_temp_grad()

        if self.sMat is None or self.sMat.size == 0:
            raise ValueError("sMat is not initialized or empty.")
        if self.topo.env_mat is None or self.topo.env_mat.size == 0:
            raise ValueError("env_mat is not initialized or empty.")
        if self.tMat is None or self.tMat.size == 0:
            raise ValueError("tMat is not initialized or empty.")

        # Align dimensions
        envMat_SN = np.tile(self.topo.env_mat[:, np.newaxis], (1, self.rMat.shape[1]))
        tMat_SN = np.tile(self.tMat[:, 0][:, np.newaxis], (1, self.rMat.shape[1]))

        logger.debug(f"sMat shape: {self.sMat.shape}, envMat_SN shape: {envMat_SN.shape}, tMat_SN shape: {tMat_SN.shape}")

        # Compute growth rate changes
        if self.omega > 0:
            rMat_cc = self.sMat - self.omega * (envMat_SN - tMat_SN) ** 2
        else:
            rMat_t = (envMat_SN - tMat_SN) ** 2
            for i in range(rMat_t.shape[0]):
                rMat_t[i] *= self.tMat[i, 1]
            rMat_cc = self.sMat - rMat_t

        # Clip negative values to -1
        rMat_cc[rMat_cc < 0] = -1
        self.rMat = rMat_cc
        logger.debug("Updated `rMat` for temperature changes.")




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
            self.efMat[i] = self.gen_r_vec()  # Use a valid function to generate growth rates

        logger.debug("Updated `ouMat` and `efMat` using Ornstein-Uhlenbeck process.")


    def gen_r_vec_erf(self):
        """
        Generate growth rates based on environmental response function (ERF).
        """
        logger.debug("Generating growth rates using Environmental Response Function (ERF).")
        tol = np.random.randn(self.topo.env_var)
        tol /= np.sqrt(np.sum(tol**2)) * np.sqrt(self.topo.env_var)

        r_i = 1 + np.dot(tol, self.topo.env_mat) / np.sqrt(2 * self.topo.env_var)
        r_i[r_i < -1] = -1  # Clip values below -1

        if self.tMat is None:
            self.tMat = tol[np.newaxis, :]
        else:
            self.tMat = np.vstack([self.tMat, tol])

        logger.debug(f"Generated `r_i`: {r_i}")
        return r_i



    def invade(self, trophLev):
        """
        Introduce an invader species into the system with consistent initialization and interaction matrix updates.

        Parameters:
        trophLev (int): Trophic level of the invader (0 for producer, 1 for consumer).
        """
        logger.debug(f"Invading with trophic level {trophLev}.")
        inv_biomass = 1e-2  # Initial biomass for the invader species

        # Initialize matrices if they are not already
        if self.xMat is None:
            self.xMat = np.zeros((0, self.topo.no_nodes))
            logger.debug("Initialized xMat for the first invader.")

        if self.rMat is None:
            self.rMat = np.zeros((0, self.topo.no_nodes))
            logger.debug("Initialized rMat for the first invader.")

        if self.cMat is None:
            self.cMat = np.zeros((0, 0))
            logger.debug("Initialized cMat for the first invader.")

        # Adding a producer (trophic level 0)
        if trophLev == 0:
            # Add producer to xMat and rMat
            self.xMat = np.vstack([self.xMat, np.full((1, self.topo.no_nodes), inv_biomass)])
            self.rMat = np.vstack([self.rMat, self.gen_r_vec()])

            # Update cMat for the new producer
            if self.cMat.size == 0:
                self.cMat = np.zeros((1, 1))  # Initialize for the first producer
            else:
                interaction_row = np.random.uniform(-0.1, 0.1, self.cMat.shape[0])  # Weak interactions
                self.cMat = np.vstack([self.cMat, interaction_row])
                self.cMat = np.hstack([self.cMat, np.append(interaction_row, 0).reshape(-1, 1)])

            self.S_p += 1  # Increment producer count
            logger.debug("Producer added to the community.")

        # Adding a consumer (trophic level 1)
        else:
            # Add consumer to xMat and rMat
            self.xMat = np.vstack([self.xMat, np.full((1, self.topo.no_nodes), inv_biomass)])
            self.rMat = np.vstack([self.rMat, np.zeros((1, self.topo.no_nodes))])  # Consumers do not have growth rates

            # Update cMat for the new consumer
            if self.cMat.size == 0:
                self.cMat = np.zeros((1, 1))  # Initialize for the first consumer
            else:
                interaction_row = np.random.uniform(-0.1, 0.1, self.cMat.shape[0])  # Weak interactions
                self.cMat = np.vstack([self.cMat, interaction_row])
                self.cMat = np.hstack([self.cMat, np.append(interaction_row, 0).reshape(-1, 1)])

            self.S_c += 1  # Increment consumer count
            logger.debug("Consumer added to the community.")

        # Log updated matrix shapes and species counts
        logger.info(f"Species invaded - Trophic Level: {trophLev}, Producers: {self.S_p}, Consumers: {self.S_c}.")
        logger.debug(f"xMat shape: {self.xMat.shape}, rMat shape: {self.rMat.shape}, cMat shape: {self.cMat.shape}")




    def extinct(self, wholeDom=True, ind_p=None, ind_c=None):
        if ind_p is None:
            ind_p = np.array([], dtype=int)
        if ind_c is None:
            ind_c = np.array([], dtype=int)

        indReturn = [None, None]  # Store extinction indices
        S_tot = self.xMat.shape[0]  # Total species richness

        # Check for extinct producers
        if wholeDom:
            ind_p = []
            for i in range(self.S_p):
                if np.all(self.xMat[i, :] <= self.thresh):  # Check for extinction
                    ind_p.append(i)
            ind_p = np.array(ind_p, dtype=int)
        indReturn[0] = ind_p

        # Remove extinct producers
        S_before = self.S_p
        for i in reversed(ind_p):  # Iterate in reverse order
            extinct_indices = i + np.arange(0, S_before * self.topo.no_nodes, S_before)
            self.xMat = np.delete(self.xMat, i, axis=0)
            self.rMat = np.delete(self.rMat, i, axis=0)
            if hasattr(self, 'sMat') and self.sMat.shape[0] > 0:
                self.sMat = np.delete(self.sMat, i, axis=0)
            self.cMat = np.delete(self.cMat, i, axis=0)
            self.cMat = np.delete(self.cMat, i, axis=1)
            if hasattr(self, 'tMat') and self.tMat.shape[0] > 0:
                self.tMat = np.delete(self.tMat, i, axis=0)
            if self.emMat is not None and self.emMat.shape[0] > 0:  # Check if emMat is not None
                self.emMat = np.delete(self.emMat, i, axis=0)

            # Update indices_DP and indices_S
            self.indices_DP = self._update_indices(self.indices_DP, extinct_indices, S_before)
            self.indices_S = self._update_indices(self.indices_S, extinct_indices, S_before)
            S_before -= 1

        # Check for extinct consumers
        if wholeDom:
            ind_c = []
            for i in range(self.rMat.shape[0], self.xMat.shape[0]):
                if np.all(self.xMat[i, :] <= self.thresh):
                    ind_c.append(i)
            ind_c = np.array(ind_c, dtype=int)
        indReturn[1] = ind_c

        # Remove extinct consumers
        for i in reversed(ind_c):
            self.xMat = np.delete(self.xMat, i, axis=0)
            self.cMat = np.delete(self.cMat, i, axis=0)
            self.cMat = np.delete(self.cMat, i, axis=1)
            if self.emMat is not None and self.emMat.shape[0] > 0:  # Check if emMat is not None
                self.emMat = np.delete(self.emMat, i, axis=0)

        # Update species richness
        if self.sppRichness.shape[0] == 0:
            self.sppRichness = np.array([[self.rMat.shape[0], self.xMat.shape[0] - self.rMat.shape[0]]])
        else:
            new_row = np.array([[self.rMat.shape[0], self.xMat.shape[0] - self.rMat.shape[0]]])
            self.sppRichness = np.vstack([self.sppRichness, new_row])

        return indReturn

    @staticmethod
    def _update_indices(indices, extinct_indices, S_before):
        # Update indices for extinct species
        updated_indices = indices.copy()
        for j in range(len(extinct_indices) - 1):
            mask = (updated_indices > extinct_indices[j]) & (updated_indices < extinct_indices[j + 1])
            updated_indices[mask] -= (j + 1)
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
                for j in neighbors:
                    if self.dispNorm == 0:  # Effort-weighted dispersal
                        if np.sum(self.dMat[:, i]) > 0:
                            kMat[i, j] = self.emRate / np.sum(self.dMat[:, i])
                    elif self.dispNorm == 1:  # Degree-weighted dispersal
                        kMat[i, j] = self.emRate / num_neighbors
                    elif self.dispNorm == 2:  # Passive dispersal
                        kMat[i, j] = self.emRate

        # Multiply with kMat to finalize dispersal matrix
        self.dMat *= kMat
        logger.debug(f"Dispersal matrix normalized using dispersal normalization method {self.dispNorm}.")

        # Include diagonal terms for self-dispersal (optional)
        if self.emRate < 0:
            np.fill_diagonal(self.dMat, -abs(self.emRate))
            logger.debug("Negative emigration rate applied to diagonal for self-dispersal.")

        # Final validation
        if not np.all(np.isfinite(self.dMat)):
            logger.error("Dispersal matrix contains invalid values (NaN or inf). Check parameters.")
            raise ValueError("Dispersal matrix contains invalid values.")

        logger.info("Dispersal matrix generated successfully.")
        logger.debug(f"Dispersal matrix: \n{self.dMat}")




    def log_state(self):
        """Log the current state of the species."""
        logger.debug(f"Producers: {self.S_p}, Consumers: {self.S_c}")
        logger.debug(f"xMat shape: {self.xMat.shape}")
        logger.debug(f"rMat shape: {self.rMat.shape}")
        logger.debug(f"cMat shape: {self.cMat.shape}")


            
# Example usage
# Corrected instantiation of Topography in species.py
if __name__ == "__main__":
    # Properly initialize the Topography class with all required arguments
    topo = Topography(
        no_nodes=100,
        lattice_height=10,
        lattice_width=10,
        phi=1.0,
        envVar=3,
        
        skVec=np.array([0.1, 0.2, 0.3]),  # skVec is still passed as a parameter if needed elsewhere
        var_e=1.0,
        randGraph=True,
        gabriel=True,
        T_int=25.0,
        network_file="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
        scVec=np.array([0.05]),
        consArea_bin=np.array([1 if i % 2 == 0 else 0 for i in range(100)]),  # Example binary conservation area
        consArea_multiplicative=True  # Conservation area perturbation mode
    )

    # Generate the network explicitly
    topo.gen_network()

    # Create an instance of Species using the Topography instance
    species = Species(topo)
    species.gen_disp_mat()  # Generate dispersal matrix
    species.invade(0)       # Invade with a producer
    species.invade(1)       # Invade with a consumer
    species.extinct()       # Remove extinct species
    species.log_state()     # Log the state of the Species








