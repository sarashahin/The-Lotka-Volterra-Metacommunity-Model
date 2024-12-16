# import os
# import numpy as np
# import scipy.io as sio
# import time
# from pathlib import Path
# from scipy.integrate import solve_ivp
# from collections.abc import Iterable
# import time
# import logging
# import traceback
# import signal



# from topography import Topography
# from species import Species
# from communitydynamics import CommunityDynamics
# from ode import ODEState, ODEDynamicalObject, ODEVector, ODEMatrix


# # Configure logging for debugging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)



# class Metacommunity:
#     def __init__(self, spp, a_init,a_bMat, a_xMat, a_scMat, a_invMax, a_tMax, a_outputDirectory,
#                  a_c1, a_c2, a_c3, a_emRate, a_dispL, a_pProducer, a_prodComp, a_symComp,
#                  a_alpha, a_sigma, a_sigma_t, a_rho, a_comp_dist, a_omega, a_dispNorm,
#                  a_no_nodes, a_lattice_height, a_lattice_width, a_phi, a_envVar, a_skVec,
#                  a_var_e, a_randGraph, a_gabriel, a_T_int, a_envMat, a_parOut=0, a_experiment="default", a_rep=0, storeTraj=0, bMatFileName="default_bMat.mat", g_block_transitions=False, g_form_of_dynamics="default_value",simTime = 0.0):
#         if a_init:
#             self.invMax = a_invMax
#             self.tMax = a_tMax
#             self.g_form_of_dynamics = g_form_of_dynamics
#             self.simTime = simTime  # Initialize simTime
#             self.parOut = a_parOut
#             self.storeTraj = storeTraj
#             self.experiment = a_experiment
#             self.rep = a_rep
#             self.date = time.strftime('%Y-%m-%d', time.localtime())
#             self.outputDirectory = a_outputDirectory
#             self.bMatFileName = bMatFileName
#             self.g_block_transitions = g_block_transitions
               
#             if not isinstance(a_envVar, int):
#                 logger.warning(f"Converting `a_envVar` from {a_envVar} to integer.")
#                 a_envVar = int(a_envVar)


#             # Initialize Topography
#             topo = Topography(
#                 no_nodes=a_no_nodes,
#                 lattice_height=a_lattice_height,
#                 lattice_width=a_lattice_width,
#                 phi=a_phi,
#                 envVar=a_envVar,
#                 skVec=a_skVec,
#                 var_e=a_var_e,
#                 randGraph=a_randGraph,
#                 gabriel=a_gabriel,
#                 T_int=a_T_int,
#                 network_file="",
#                 scVec=a_skVec if a_skVec is not None else np.ones(a_no_nodes)
#             )
#             # Log the success of Topography creation
#             logger.debug(f"Topography created with {topo.no_nodes} nodes.")

#             # Pass only relevant arguments to Species constructor
#             self.spp = Species(
#                 topo=topo,
#                 c1=a_c1,
#                 c2=a_c2,
#                 c3=a_c3,
#                 emRate=a_emRate,
#                 dispL=a_dispL,
#                 pProducer=a_pProducer,
#                 prodComp=a_prodComp,
#                 symComp=a_symComp,
#                 alpha=a_alpha,
#                 sigma=a_sigma,
#                 sigma_t=a_sigma_t,
#                 rho=a_rho,
#                 comp_dist=a_comp_dist,
#                 omega=a_omega,
#                 dispNorm=a_dispNorm
#             )

#             # Log that Species has been successfully initialized
#             logger.debug("Species initialized successfully.")
            
#             if self.spp.dMat is None or self.spp.dMat.size == 0:
#                 logger.error("Dispersal matrix (`dMat`) must be properly initialized and non-empty in Species.")
#                 raise ValueError("Dispersal matrix must be properly initialized and non-empty in Species.")
#             self.dMat = self.spp.dMat
#             logger.debug(f"dMat initialized with shape: {self.dMat.shape}")




#         else:
#             self.loadMC(a_bMat)
#             self.spp.topo.scMatFileName = a_scMat
#             self.outputDirectory = a_outputDirectory
#             if a_invMax != 0:
#                 self.invMax += a_invMax
#                 print(f"\ninvMax updated to {self.invMax}")
#             if a_tMax != self.tMax:
#                 self.tMax = a_tMax
#                 print(f"\ntMax updated to {self.tMax}")


#     def meta_c_dynamics(self, T):
#         """
#         Numerical solver for metacommunity dynamics.
#         :param T: Relaxation time
#         """

#                 # Ensure dMat is properly initialized
#         if self.spp.dMat is None or self.spp.dMat.size == 0:
#             raise ValueError("The dispersal matrix (`dMat`) must be properly initialized before starting the dynamics.")

#         # Update in meta_c_dynamics function or relevant context
#         indices_S = np.nonzero(self.spp.xMat > 0)[1]  # Example: Find nonzero biomass indices

#         # Calculate indices_DP based on the requirements
#         # Assuming it represents all indices of xMat for now
#         indices_DP = np.arange(self.spp.xMat.size)

#         dynamics = CommunityDynamics(xMat=self.spp.xMat,spp=self.spp, cMat=self.spp.cMat, rMat=self.spp.rMat, indices_S=indices_S, indices_DP=indices_DP, dMat=self.spp.dMat,)  # ODE dynamical object
        
#         if dynamics.dMat_sp is None or dynamics.dMat_sp.size == 0:
#             raise ValueError("The dispersal matrix (`dMat_sp`) in `CommunityDynamics` must be properly initialized.")


#         # Initialize simulation parameters
#         dynamics.current_time = 0.0
#         convergence_threshold = 1e-6  # Threshold for detecting equilibrium
#         max_iterations = int(T * 1000)  # Safety cap for iterations
#         iterations = 0
        
        


#         # dynamics.current_time = 0.0  # Set the starting time for simulation
#         dynamics.xMat = self.spp.xMat
#         dynamics.indices_DP = self.spp.indices_DP
#         dynamics.indices_S = self.spp.indices_S
#         dynamics.rMat = self.spp.rMat

#         logger.info("Starting dynamics with properly initialized matrices.")

#         # Verify if all necessary attributes are properly initialized
#         if self.spp.dMat is None or self.spp.dMat.size == 0:
#             raise ValueError("The dispersal matrix (`spp.dMat`) must be initialized before starting the dynamics.")
#         logger.info("Starting dynamics with properly initialized dispersal matrix.")

#         # Verify if dMat_sp is properly initialized
#         if dynamics.dMat_sp is None or dynamics.dMat_sp.size == 0:
#             raise ValueError("The dispersal matrix (`dMat_sp`) in `CommunityDynamics` must be properly initialized.")

        

#         if self.spp.cMat.shape[0] != 0:
#             dynamics.cMat = self.spp.cMat
#         dynamics.rho = self.spp.rho
#         dynamics.bodymass = self.spp.bodymass
#         dynamics.bodymass_inv = self.spp.bodymass_inv
#         dynamics.mu = self.spp.mu

#         if self.spp.topo.scVec.shape[0] != 0:
#             dynamics.scVec = self.spp.topo.scVec
#             dynamics.scVec_prime = self.spp.topo.scVec_prime
#         dynamics.S_p = self.spp.S_p
#         dynamics.S_c = self.spp.S_c

#         # Check if emMat is not None and has the correct shape
#         if self.spp.emMat is not None and self.spp.emMat.shape[0] != 0:
#             dynamics.emMat = self.spp.emMat
#         else:
#             logger.warning("Warning: emMat is either None or empty. Dynamics may be incomplete.")

#         dynamics.dMat = self.spp.dMat

#         # Handling file paths for storing trajectories
#         Bpath, bMatDir, newFolder = "", "", ""
#         if self.storeTraj != 0:
#             # Create the directory structure for storing matrices
#             pos1 = self.bMatFileName.rfind("/")
#             bMatDir = self.bMatFileName[:pos1]
#             pos1 = self.bMatFileName.rfind(")")
#             pos2 = self.bMatFileName.rfind(".")
#             newFolder = self.bMatFileName[pos1+1:pos2]
#             if not self.g_block_transitions:
#                 Bpath = os.path.join(bMatDir, newFolder, "trajectory")
#             else:
#                 Bpath = os.path.join(bMatDir, newFolder, "trajectory_wo_transitions")

#             print(f"\nSaving matrices to {Bpath}")

#             if not os.path.exists(Bpath):
#                 os.makedirs(Bpath)

#         # Metacommunity relaxation step
#         PRINT_X_MAT = False
#         movie_mode = False
#         resPrint = 1  # Temporal resolution for printing to console
#         resSave = 100  # Temporal resolution for saving to file

#         if PRINT_X_MAT and movie_mode:
#             print("\033[2J")  # Clear screen

#             if not self.spp.topo.randGraph:
#                 # Get current terminal size
#                 try:
#                     terminal_lines, terminal_columns = os.popen('stty size', 'r').read().split()
#                     terminal_lines = int(terminal_lines)
#                     terminal_columns = int(terminal_columns)
#                 except Exception as e:
#                     raise RuntimeError("Can't get terminal size.")

#                 species_per_row = (terminal_columns + 1) // ((7 + 2 + 1) * self.spp.topo.lattice_width)
#                 rows_of_species = min((terminal_lines - 2) // (self.spp.topo.lattice_height + 1),
#                                       (self.spp.xMat.shape[0] - 1) // species_per_row + 1)


#         previous_state = self.spp.xMat.copy()
#         while dynamics.current_time < T:
#             root_indices = []
#             state = ODEState(dynamics)
#             # tNext = np.ceil(dynamics.current_time / resPrint + 1e-7) * resPrint
#             tNext = np.ceil(dynamics.current_time / 1 + 1e-7) * 1  # Time step for updates

#             # previous_state = state.get_matrix(self.spp.xMat.shape).copy()  # Use the new get_matrix method
#             while dynamics.current_time < T:
#                 # tNext = np.ceil(dynamics.current_time / resPrint + 1e-7) * resPrint
#                 state.integrate_until(tNext, root_indices)
#                 iterations += 1
#                 if root_indices:
#                   break  # Root was found

#                 # Check for convergence
#                 current_state = state.get_matrix(self.spp.xMat.shape)
#                 state_change = np.linalg.norm(current_state - previous_state)

#                 if state_change < convergence_threshold:
#                     logger.info(f"System reached equilibrium at time {dynamics.current_time:.2f} with change {state_change:.2e}.")
#                     return

#                 previous_state = current_state.copy()

#                 if iterations >= max_iterations:
#                     logger.warning("Maximum iterations reached without convergence.")
#                     return

#                 if PRINT_X_MAT:
#                     if dynamics.current_time >= tNext:
#                         print(f"Current time: {dynamics.current_time:.2f}, State Change: {state_change:.2e}")
#                         if movie_mode:
#                             os.system('sleep 0.1')  # Sleep so many microseconds
#                             print("\033[H")  # Move cursor to the top
#                             print(f"Invasions / S_p / S_c = {self.spp.invasion} / {int(self.spp.rMat.shape[0])} / {int(self.spp.xMat.shape[0]) - int(self.spp.rMat.shape[0])}")
#                         X_tmp = state.get_matrix(self.spp.xMat.shape)
#                         status_output = f"\nt = {int(tNext)} X = "
#                         print(status_output)

#                         if not (movie_mode and not self.spp.topo.randGraph):
#                             # Print one-species-per-line
#                             for i in range(X_tmp.shape[0]):
#                                 print()
#                                 for j in range(X_tmp.shape[1]):
#                                     oneDind = i + X_tmp.shape[0] * j

#                                     if oneDind in self.spp.indices_DP:
#                                         if X_tmp[i, j] > self.spp.bodymass:
#                                             print(f"\033[32m{X_tmp[i, j]:.2e}\033[0m", end=" ")
#                                         else:
#                                             print(f"\033[34m{X_tmp[i, j]:.2e}\033[0m", end=" ")
#                                     elif oneDind in self.spp.indices_S:
#                                         print(f"\033[31m{X_tmp[i, j]:.4f}\033[0m", end=" ")
#                                     else:
#                                         print("ERROR: INDEX NOT FOUND!")
#                         else:
#                             # Print each species with its own "map"
#                             for R in range(rows_of_species):
#                                 if R > 0:
#                                     print()
#                                     row_length = species_per_row * (7 + 2 + 1) * self.spp.topo.lattice_width - 1
#                                     print(''.join(['+' if (c + 1) % ((7 + 2 + 1) * self.spp.topo.lattice_width) == 0 else '-' for c in range(row_length)]))
#                                 for r in range(self.spp.topo.lattice_height):
#                                     print()
#                                     for C in range(species_per_row):
#                                         i = R * species_per_row + C
#                                         if i >= X_tmp.shape[0]:
#                                             continue
#                                         for c in range(self.spp.topo.lattice_width):
#                                             print('|' if c == 0 and C != 0 else ' ', end=' ')
#                                             j = r * self.spp.topo.lattice_width + c
#                                             oneDind = i + X_tmp.shape[0] * j
#                                             if oneDind in self.spp.indices_DP:
#                                                 if X_tmp[i, j] > self.spp.bodymass:
#                                                     print(f"\033[32m{X_tmp[i, j]:.2e}\033[0m", end=" ")
#                                                 else:
#                                                     print(f"\033[34m{X_tmp[i, j]:.2e}\033[0m", end=" ")
#                                             elif oneDind in self.spp.indices_S:
#                                                 print(f"\033[31m{X_tmp[i, j]:.4f}\033[0m", end=" ")
#                                             else:
#                                                 print("ERROR: INDEX NOT FOUND!")

#                 if dynamics.current_time >= tNext and self.storeTraj != 0:
#                     if int(tNext) % resSave == 0:
#                         # Write current state to file
#                         Btofile = state.get_matrix_as_1d()
#                         Btofile = Btofile.reshape(self.spp.xMat.shape)
#                         Bfile = os.path.join(Bpath, f"bMat{int(tNext)}.mat")
#                         np.savetxt(Bfile, Btofile, fmt='%e')

#                 if root_indices:
#                     break  # Root was found
#             dynamics.react_to_roots(root_indices)
#             root_indices.clear()


#     def remove_species(self, indices):
#         """
#         Helper function to remove species from all matrices.

#         Parameters:
#         indices (list[int]): List of species indices to remove.
#         """
#         logger.debug(f"Removing species at indices: {indices}")

#         # Sort indices in descending order to avoid shifting issues during deletion
#         indices = sorted(indices, reverse=True)

#         for idx in indices:
#             if idx >= self.spp.xMat.shape[0]:
#                 logger.warning(f"Index {idx} is out of bounds for xMat with shape {self.spp.xMat.shape}. Skipping.")
#                 continue

#             # Update producer and consumer counts
#             if idx < self.spp.S_p:  # Producer index
#                 self.spp.S_p -= 1
#             else:  # Consumer index
#                 self.spp.S_c -= 1

#             # Remove species from core matrices
#             self.spp.xMat = np.delete(self.spp.xMat, idx, axis=0)
#             if self.spp.cMat is not None and self.spp.cMat.shape[0] > 0:
#                 self.spp.cMat = np.delete(self.spp.cMat, idx, axis=0)
#                 self.spp.cMat = np.delete(self.spp.cMat, idx, axis=1)
#             if self.spp.rMat is not None and self.spp.rMat.shape[0] > 0:
#                 self.spp.rMat = np.delete(self.spp.rMat, idx, axis=0)

#             # Remove species from optional matrices
#             optional_matrices = {'sMat': self.spp.sMat, 'tMat': self.spp.tMat, 'emMat': self.spp.emMat}
#             for mat_name, mat in optional_matrices.items():
#                 if mat is not None:
#                     if mat.shape[0] > idx:
#                         setattr(self.spp, mat_name, np.delete(mat, idx, axis=0))
#                     else:
#                         logger.warning(f"Index {idx} is out of bounds for {mat_name}. Skipping.")

#         logger.debug(f"Species removal complete. Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")



#     def invader_sample(self, trophLev, no_invaders):
#         """
#         Introduce new random species and test for positive growth rates.
#         """
#         min_b = 1e-6
#         invExcess = 3  # Excess invaders to ensure success
#         suc_inv = 0

#         while suc_inv < no_invaders:
#             logger.debug(f"Attempting to invade. Successful so far: {suc_inv}/{no_invaders}")

#             for _ in range(invExcess * no_invaders):
#                 self.spp.invade(trophLev)
#                 # Synchronize optional matrices
#                 # Ensure additional matrices (e.g., efMat, emMat) are updated
#                 for mat_name in ['efMat', 'emMat']:
#                     mat = getattr(self.spp, mat_name, None)
#                     if mat is not None and mat.size > 0:
#                         setattr(self.spp, mat_name, np.vstack([mat, np.zeros((1, mat.shape[1]))]))
#                     elif mat is not None:
#                         setattr(self.spp, mat_name, np.zeros((self.spp.xMat.shape[0], self.spp.xMat.shape[1])))
#                     else:
#                         logger.warning(f"{mat_name} is not initialized. Skipping update.")

#             B = np.maximum(0, self.spp.xMat[:self.spp.S_p] if trophLev == 1 else self.spp.xMat)
#             indices = np.arange(self.spp.S_p, self.spp.xMat.shape[0])

#             try:
#                 if trophLev == 0:  # Producers
#                     bInv = self.spp.rMat[self.spp.S_p:] - (self.spp.cMat[self.spp.S_p:, :] @ B)
#                 else:  # Consumers
#                     bInv = self.spp.rho * (self.spp.cMat[indices, :self.spp.S_p] @ B - 1)
#             except ValueError as e:
#                 logger.error(f"Error in growth rate calculation: {e}")
#                 raise

#             logger.debug(f"Calculated growth rates: {bInv}")

#             if bInv.size == 0 or np.all(bInv <= 0):
#                 logger.warning("No positive growth rates found. Adjusting parameters may be necessary.")
#                 break

#             bInv_max = bInv.max(axis=1) if len(bInv.shape) > 1 else bInv
#             posGrowth = np.where(bInv_max >= min_b)[0][: no_invaders - suc_inv]
#             negGrowth = np.setdiff1d(indices, posGrowth)

#             logger.debug(f"Positive Growth Indices: {posGrowth}, Negative Growth Indices: {negGrowth}")

#             suc_inv += len(posGrowth)
#             for idx in reversed(negGrowth):
#                 self.remove_species([idx])

#             logger.debug(f"State after removal - Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")

#         if trophLev == 0:
#             self.spp.S_p = self.spp.xMat.shape[0] - self.spp.S_c
#         else:
#             self.spp.S_c = self.spp.xMat.shape[0] - self.spp.S_p

#         logger.info(f"Invader sampling complete. Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")

           

#     def env_fluct(self):
#         """
#         Update abiotic turnover and simulate a single CVode timestep.
#         """
#         dynamics = CommunityDynamics(
#             self.spp,
#             self.spp.xMat,
#             self.spp.cMat,
#             self.spp.rMat,
#             efMat=self.spp.efMat,
#             scVec=self.spp.scVec,
#             dMat=self.spp.dMat,
#         )  # ODE dynamical object

#         dynamics.xMat = self.spp.xMat
#         dynamics.rMat = self.spp.rMat

#         if self.spp.cMat.shape[0] != 0:
#             dynamics.cMat = self.spp.cMat

#         dynamics.dMat = self.spp.dMat  # If dynamics.dMat is initialized, single domain relaxation selected
#         dynamics.rho = self.spp.rho

#         self.spp.ou_process()  # Update the spatially resolved OU process to be added to the growth rate matrix
#         dynamics.efMat = self.spp.efMat  # In CommunityDynamics rMat += efMat

#         state = ODEState(dynamics)
#         root_indices = []

#         # Integrate for a single timestep
#         state.integrate_until(target_time=1, root_indices=root_indices)
#         tmp = state.get_matrix(shape=(dynamics.xMat.shape[0], dynamics.xMat.shape[1]))

#         # Handle trajectory shape mismatch
#         if self.spp.trajectories.shape[1] != tmp.shape[1]:
#             raise ValueError(
#                 f"Shape mismatch: trajectories has {self.spp.trajectories.shape[1]} columns, "
#                 f"but tmp has {tmp.shape[1]} columns."
#             )
#         self.spp.trajectories = np.vstack((self.spp.trajectories, tmp))  # Store trajectory

#         RE = self.spp.rMat + self.spp.efMat
#         RE = RE.reshape(1, -1)

#         # Handle fluctuation shape mismatch
#         if self.spp.fluctuations.shape[1] != RE.shape[1]:
#             logger.warning(
#                 f"Fluctuation shape mismatch detected. Fluctuations shape: {self.spp.fluctuations.shape}, "
#                 f"RE shape: {RE.shape}. Reshaping fluctuations to match."
#             )
#             self.spp.fluctuations = np.zeros((0, RE.shape[1]))  # Reinitialize fluctuations to match RE

#         self.spp.fluctuations = np.vstack((self.spp.fluctuations, RE))  # Store current distribution in R (= rMat + efMat)



#     def warming(self, dTdt, res, time):
#         """
#         Updates temperature gradient and rMat, then simulates a single timestep.

#         Args:
#             dTdt (float): Rate of temperature increase.
#             res (int): Number of steps per unit time.
#             time (int): Total time for warming simulation.
#         """
#         # Create directory structure
#         pos1 = self.bMatFileName.rfind("/")
#         bMatDir = self.bMatFileName[:pos1]
#         pos1 = self.bMatFileName.rfind(")")
#         pos2 = self.bMatFileName.rfind(".")
#         newFolder = self.bMatFileName[pos1 + 1:pos2]
#         Bpath = os.path.join(bMatDir, newFolder, f"dTdt={dTdt}")

#         if not os.path.exists(Bpath):
#             os.makedirs(Bpath)

#         if not hasattr(self.spp, 'scVec_prime') or self.spp.scVec_prime is None:
#             logger.error("`scVec_prime` is missing or not initialized. Using default values.")
#             self.spp.scVec_prime = np.arange(1, self.spp.topo.network.shape[0] + 1)

#         dynamics = CommunityDynamics(
#             self.spp,
#             self.spp.xMat,
#             self.spp.cMat,
#             self.spp.rMat,
#             efMat=self.spp.efMat,
#             scVec=self.spp.scVec,
#             scVec_prime=self.spp.scVec_prime,
#             dMat=self.spp.dMat,
#             bodymass=self.spp.bodymass,
#             mu=self.spp.mu
#         )

#         for t in range(res):
#             self.spp.topo.T_int += dTdt
#             self.spp.update_r_vec_temp()
#             dynamics.rMat = self.spp.rMat
#             state = ODEState(dynamics)

#             # Align state elements with rMat size
#             if state.elements.size != dynamics.rMat.size:
#                 logger.warning(f"Reshaping state elements to match rMat. State size: {state.elements.size}, rMat size: {dynamics.rMat.size}")
#                 state.elements = np.zeros(dynamics.rMat.size)

#             root_indices = []
#             try:
#                 state.integrate_until(1, root_indices)
#             except ValueError as e:
#                 logger.error(f"Error during integration: {e}")
#                 raise

#         # Save current state to file
#         Bfile = os.path.join(Bpath, f"bMat_w{time}.mat")
#         Rfile = os.path.join(Bpath, f"rMat_w{time}.mat")
#         np.savetxt(Bfile, self.spp.xMat, fmt='%e')
#         np.savetxt(Rfile, self.spp.rMat, fmt='%e')

#         if time == 0:
#             filenameP = os.path.join(Bpath, "pars.mat")
#             self.write_params(filenameP)








#     def long_dist_disp(self, tMax, edges):
#         """
#         Adds long-distance dispersal links to the metacommunity.
#         """
#         stepwise = True if edges >= 0 else False
#         row_min = np.min(np.abs(self.spp.dMat), axis=1)

#         non_zero_d = np.count_nonzero(self.spp.dMat)
#         zero_d = (self.spp.dMat.size - non_zero_d) // 2
#         pert_record = np.zeros((zero_d, 3))

#         if not stepwise:
#             edges = zero_d // 1
#             print(f"Edges reset to {edges}")

#         pos1 = self.bMatFileName.rfind("/")
#         bMatDir = self.bMatFileName[:pos1]
#         pos1 = self.bMatFileName.rfind(")")
#         pos2 = self.bMatFileName.rfind(".")
#         newFolder = self.bMatFileName[pos1 + 1:pos2]

#         if stepwise:
#             Bpath = os.path.join(bMatDir, newFolder, f"dkdt={edges}")
#         else:
#             Bpath = os.path.join(bMatDir, newFolder, f"delta_e={edges}")

#         if not os.path.exists(Bpath):
#             os.makedirs(Bpath)

#         t_interval = 1
#         t_step = 0
#         edge_count = 0

#         while not np.min(row_min):
#             node = np.where(row_min == 0)[0]
#             np.random.shuffle(node)
#             edge = np.where(self.spp.dMat[node[0]] == 0)[0]
#             np.random.shuffle(edge)

#             self.spp.topo.adjMat[node[0], edge[0]] = -1.0
#             self.spp.topo.adjMat[edge[0], node[0]] = -1.0
#             self.spp.gen_disp_mat()

#             pert_record[t_step, 0] = node[0]
#             pert_record[t_step, 1] = edge[0]
#             pert_record[t_step, 2] = t_interval
#             t_step += 1

#             row_min = np.min(np.abs(self.spp.dMat), axis=1)
#             edge_count += 1

#             print(f"Edge count {edge_count}")

#             if edge_count == edges:
#                 if stepwise:
#                     print(f"{edges * (t_interval + 1)} new edges allocated")
#                     self.meta_c_dynamics(tMax)
#                     Bfile = os.path.join(Bpath, f"bMat_d{t_interval}.mat")
#                     np.savetxt(Bfile, self.spp.xMat, fmt='%e')
#                     filenameRR = os.path.join(Bpath, "pert_record.mat")
#                     np.savetxt(filenameRR, pert_record, fmt='%e')
#                     t_interval += 1
#                     edge_count = 0

#                     if t_interval > 100:
#                         row_min = np.ones_like(row_min)
#                 else:
#                     print("Single connectivity perturbation")
#                     pert_record = pert_record[:t_step]
#                     dynamics = CommunityDynamics()
#                     dynamics.xMat = self.spp.xMat

#                     if self.spp.cMat.shape[0] != 0:
#                         dynamics.cMat = self.spp.cMat

#                     dynamics.rMat = self.spp.rMat
#                     dynamics.dMat = self.spp.dMat
#                     dynamics.rho = self.spp.rho

#                     for t in range(tMax):
#                         print(f"\rt = {t}", end="")
#                         state = ODEState(dynamics)
#                         state.integrate_until(t)
#                         if t > 0 and t % 1 == 0:
#                             Bfile = os.path.join(Bpath, f"bMat_d{t}.mat")
#                             np.savetxt(Bfile, self.spp.xMat, fmt='%e')
#                     row_min = np.ones_like(row_min)

#         if edge_count:
#             self.meta_c_dynamics(tMax)
#             Bfile = os.path.join(Bpath, f"bMat_d{t_interval}.mat")
#             np.savetxt(Bfile, self.spp.xMat, fmt='%e')
#             filenameRR = os.path.join(Bpath, "pert_record.mat")
#             np.savetxt(filenameRR, pert_record, fmt='%e')


#     def node_removal(self, tMax, no_removals):
#         """
#         Removes nodes from the network and simulates the dynamics.
#         """
#         self.storeTraj = 3
#         if self.spp.topo.scVec.shape[0] > 0 and self.spp.topo.scVec_prime.shape[0] > 0:
#             print("\nLocal interaction matrices scaled")

#         pos1 = self.bMatFileName.rfind("/")
#         bMatDir = self.bMatFileName[:pos1]
#         pos1 = self.bMatFileName.rfind(")")
#         pos2 = self.bMatFileName.rfind(".")
#         newFolder = self.bMatFileName[pos1 + 1:pos2]
#         Bpath = os.path.join(bMatDir, newFolder, "nodeRemoval")

#         if not os.path.exists(Bpath):
#             os.makedirs(Bpath)

#         node_id = np.linspace(no_removals, self.spp.topo.no_nodes - 1, self.spp.topo.no_nodes - no_removals, dtype=int)
#         spp_copy = self.spp

#         for x in node_id:
#             print(f"\nNode {x}")
#             self.spp = spp_copy
#             self.spp.xMat = np.delete(self.spp.xMat, x, axis=1)
#             if self.spp.rMat.shape[0] > 0:
#                 self.spp.rMat = np.delete(self.spp.rMat, x, axis=1)
#             self.spp.topo.network = np.delete(self.spp.topo.network, x, axis=0)
#             self.spp.topo.distMat = np.delete(self.spp.topo.distMat, x, axis=0)
#             self.spp.topo.distMat = np.delete(self.spp.topo.distMat, x, axis=1)
#             self.spp.dMat = np.delete(self.spp.dMat, x, axis=0)
#             self.spp.dMat = np.delete(self.spp.dMat, x, axis=1)
#             if self.spp.topo.scVec.shape[0] > 0:
#                 self.spp.topo.scVec = np.delete(self.spp.topo.scVec, x, axis=0)
#             if self.spp.topo.scVec_prime.shape[0] > 0:
#                 self.spp.topo.scVec_prime = np.delete(self.spp.topo.scVec_prime, x, axis=0)
#             if self.spp.topo.envMat.shape[1] > 0:
#                 self.spp.topo.envMat = np.delete(self.spp.topo.envMat, x, axis=1)
#             self.spp.topo.no_nodes -= 1

#             dynamics = CommunityDynamics()
#             dynamics.xMat = self.spp.xMat
#             dynamics.rMat = self.spp.rMat

#             if self.spp.cMat.shape[0] != 0:
#                 dynamics.cMat = self.spp.cMat

#             dynamics.rho = self.spp.rho
#             if self.spp.topo.scVec.shape[0] != 0:
#                 dynamics.scVec = self.spp.topo.scVec
#                 dynamics.scVec_prime = self.spp.topo.scVec_prime
#             dynamics.S_p = self.spp.S_p
#             dynamics.S_c = self.spp.S_c

#             Bfile = os.path.join(Bpath, f"bMat_nr_init{x}.mat")
#             np.savetxt(Bfile, self.spp.xMat, fmt='%e')
#             if self.spp.S_c > 0:
#                 Bfile_c = os.path.join(Bpath, f"bMat_nr_c_init{x}.mat")
#                 B_c = self.spp.xMat[self.spp.S_p:, :]
#                 np.savetxt(Bfile_c, B_c, fmt='%e')

#             self.meta_c_dynamics(tMax)
#             Bfile = os.path.join(Bpath, f"bMat_nr{x}.mat")
#             np.savetxt(Bfile, self.spp.xMat, fmt='%e')
#             if self.spp.S_c > 0:
#                 Bfile_c = os.path.join(Bpath, f"bMat_nr_c{x}.mat")
#                 B_c = self.spp.xMat[self.spp.S_p:, :]
#                 np.savetxt(Bfile_c, B_c, fmt='%e')


#     def gen_jacobian(self):
#             """
#             Generate the numerical approximation of the Jacobian matrix for computing regional competitive overlap matrix
#             """
#             dynamics = CommunityDynamics()
#             dynamics.xMat = self.spp.xMat
#             dynamics.indices_DP = self.spp.indices_DP
#             dynamics.indices_S = self.spp.indices_S
#             dynamics.rMat = self.spp.rMat
#             if self.spp.cMat.shape[0] != 0:
#                 dynamics.cMat = self.spp.cMat
#             dynamics.rho = self.spp.rho
#             dynamics.bodymass = self.spp.bodymass
#             dynamics.bodymass_inv = self.spp.bodymass_inv
#             dynamics.mu = self.spp.mu
#             if self.spp.topo.scVec.shape[0] != 0:
#                 dynamics.scVec = self.spp.topo.scVec
#                 dynamics.scVec_prime = self.spp.topo.scVec_prime
#             dynamics.S_p = self.spp.S_p
#             dynamics.S_c = self.spp.S_c
#             if self.spp.emMat.shape[0] != 0:
#                 dynamics.emMat = self.spp.emMat
#             dynamics.dMat = self.spp.dMat

#             state = ODEState(dynamics)
#             jacobian = np.zeros((dynamics.number_of_variables(), dynamics.number_of_variables()))
#             jacobian_pointers = [jacobian[:, i] for i in range(dynamics.number_of_variables())]

#             dynamics.numerical_jacobian(jacobian_pointers)
#             self.jacobian = jacobian

#     def gen_c_mat_reg(self, h=0.001):
#         """
#         Generate a numerically approximated regional interaction matrix via a computation harvesting experiment
#         :param h: Harvesting rate
#         """
#         self.gen_jacobian()
#         J_inv = np.linalg.inv(self.jacobian)

#         S_tot = self.spp.xMat.shape[0]
#         index = np.arange(0, self.spp.topo.no_nodes) * S_tot
#         self.cMat_reg = np.zeros((S_tot, S_tot))

#         for i in range(S_tot):
#             H_i = np.zeros(S_tot * self.spp.topo.no_nodes)
#             H_i[index + i] = h * self.spp.xMat[i, :]
#             dB_jx = -1 * (J_inv @ H_i) / h
#             dB_jx_mat = dB_jx.reshape(S_tot, self.spp.topo.no_nodes)
#             self.cMat_reg[:, i] = np.sum(dB_jx_mat, axis=1)

#         self.jacobian = None
#         self.cMat_reg = np.linalg.inv(self.cMat_reg)

#         # Normalize regional competitive overlap matrix
#         norm = np.zeros_like(self.cMat_reg)
#         sgn = np.zeros_like(self.cMat_reg)
#         norm[np.diag_indices_from(self.cMat_reg)] = 1 / np.sqrt(np.abs(np.diag(self.cMat_reg)))
#         sgn[np.diag_indices_from(self.cMat_reg)] = np.diag(self.cMat_reg) / np.abs(np.diag(self.cMat_reg))
#         self.cMat_reg = sgn @ norm @ self.cMat_reg @ norm

#     def gen_source_sink(self, t_full_relax):
#         """
#         Infer which populations are dependent upon immigration for local detectability by switching off dispersal and relaxing to equilibrium
#         :param t_full_relax: Long relaxation time
#         """
#         D_store = np.copy(self.spp.dMat)
#         B_store_p = np.copy(self.spp.xMat)
#         Src_p = np.zeros_like(self.spp.xMat)
#         Snk_p = np.zeros_like(self.spp.xMat)

#         self.spp.dMat = np.zeros_like(D_store)
#         np.fill_diagonal(self.spp.dMat, np.diag(D_store))
#         self.meta_c_dynamics(t_full_relax)

#         Src_p[self.spp.xMat > self.spp.thresh] = 1

#         self.spp.xMat = B_store_p
#         self.spp.dMat = D_store

#         Snk_p[self.spp.xMat > self.spp.thresh] = 1
#         Snk_p = Src_p - Snk_p
#         Snk_p[Snk_p == 1] = 0
#         self.spp.xMat_src = Src_p + Snk_p

#     def print_params(self):
#         """
#         Print model parameterization to console
#         """
#         print("\nModel parameters:")
#         print(f"\ntMax {self.tMax}")
#         print(f"\nparOut {self.parOut}")
#         print(f"\nexperiment {self.experiment}")
#         print(f"\nrep {self.rep}")
#         print(f"\nform_of_dynamics {self.g_form_of_dynamics}")
#         print(f"\nsimTime {self.simTime}")
#         print(f"\ndate {self.date}")
#         print(f"\nc1 {self.spp.c1}")
#         print(f"\nc2 {self.spp.c2}")
#         print(f"\nemRate {self.spp.emRate}")
#         print(f"\ndispL {self.spp.dispL}")
#         print(f"\npProducer {self.spp.pProducer}")
#         print(f"\nalpha {self.spp.alpha}")
#         print(f"\nsigma {self.spp.sigma}")
#         print(f"\nrho {self.spp.rho}")
#         print(f"\ncomp_dist {self.spp.comp_dist}")
#         print(f"\nomega {self.spp.omega}")
#         print(f"\nno_nodes {self.spp.topo.no_nodes}")
#         print(f"\nphi {self.spp.topo.phi}")
#         if self.spp.topo.skVec.size > 0:
#             print(f"\nsk {self.spp.topo.skVec[0]}")
#         print(f"\nenvVar {self.spp.topo.envVar}")
#         print(f"\nT_int {self.spp.topo.T_int}")
#         print(f"\nvar_e {self.spp.topo.var_e}")
#         print(f"\nrandGraph {self.spp.topo.randGraph}")
#         print(f"\ngabriel {self.spp.topo.gabriel}")
#         print(f"\ninvMax {self.invMax}")
#         print(f"\ndispNorm {self.spp.dispNorm}")


#     def write_params(self, filenameP):
#         # Summary:
#         # Write model parameterization to a file
#         with open(filenameP, 'w') as params:
#             params.write(f"date {self.date}\n")
#             params.write(f"experiment {self.experiment}\n")
#             params.write(f"form_of_dynamics {g_form_of_dynamics}\n")
#             params.write(f"invMax {self.invMax}\n")
#             params.write(f"parOut {self.parOut}\n")
#             params.write(f"rep {self.rep}\n")
#             params.write(f"simTime {self.simTime}\n")
#             params.write(f"no_nodes {self.spp.topo.no_nodes}\n")
#             params.write(f"randGraph {self.spp.topo.randGraph}\n")
#             params.write(f"phi {self.spp.topo.phi}\n")
#             params.write(f"envVar {self.spp.topo.envVar}\n")
#             params.write(f"c1 {self.spp.c1}\n")
#             params.write(f"c2 {self.spp.c2}\n")
#             if self.spp.c3 > 0:
#                 params.write(f"c3 {self.spp.c3}\n")
#             params.write(f"emRate {self.spp.emRate}\n")
#             params.write(f"dispL {self.spp.dispL}\n")
#             params.write(f"invasion {self.spp.invasion}\n")
#             params.write(f"tMax {self.tMax}\n")
#             params.write(f"pProducer {self.spp.pProducer}\n")
#             params.write(f"prodComp {self.spp.prodComp}\n")
#             params.write(f"var_e {self.spp.topo.var_e}\n")
#             params.write(f"alpha {self.spp.alpha}\n")
#             params.write(f"sigma {self.spp.sigma}\n")
#             params.write(f"rho {self.spp.rho}\n")
#             params.write(f"comp_dist {self.spp.comp_dist}\n")
#             if self.spp.topo.T_int > 0:
#                 params.write(f"T_int {self.spp.topo.T_int}\n")
#             params.write(f"omega {self.spp.omega}\n")
#             if len(self.spp.topo.skVec) > 0:
#                 params.write(f"sk {self.spp.topo.skVec[0]}\n")
#             else:
#                 params.write(f"sk NULL\n")
#             params.write(f"delta_g {self.spp.delta_g}\n")
#             params.write(f"sigma_r {self.spp.sigma_r}\n")
#             params.write(f"symComp {self.spp.symComp}\n")

#     def save_video(self, path):
#         self.gen_cmat_reg()
#         self.gen_source_sink()
#         np.savetxt(f"{path}cMat_reg_{self.spp.invasion}.mat", self.cMat_reg)
#         np.savetxt(f"{path}rMat_{self.spp.invasion}.mat", self.spp.rMat)
#         np.savetxt(f"{path}bMat_{self.spp.invasion}.mat", self.spp.xMat)
#         np.savetxt(f"{path}bMat_src_{self.spp.invasion}.mat", self.spp.xMat_src)
#         self.cMat_reg = None
#         self.spp.xMat_src = None
#         if self.store_params:
#             self.write_params(f"{path}params.mat")
#             np.savetxt(f"{path}network.mat", self.spp.topo.network)
#             np.savetxt(f"{path}envMat.mat", self.spp.topo.envMat)
#             np.savetxt(f"{path}dMat.mat", self.spp.dMat)
#             self.store_params = False




#     def saveMC(self):
#         """
#         Generate file names and save model matrices to file.
#         """
#         print("\nOutputting data...")
        
#         # Generate paths for saving
#         if len(self.bMatFileName) == 0:
#             self.outputDirectory = Path(f"SimulationData/N={self.spp.topo.no_nodes}/{self.experiment}_experiment/{self.date}/")
#             self.outputDirectory.mkdir(parents=True, exist_ok=True)

#             base_filename = f"{self.date}_{self.experiment}({self.invMax}){self.parOut}"
#             filenameR = self.outputDirectory / f"{base_filename}rMat{self.rep}.mat"
#             filenameSR = self.outputDirectory / f"{base_filename}sMat{self.rep}.mat"
#             filenameB = self.outputDirectory / f"{base_filename}bMat{self.rep}.mat"
#             filenameBC = self.outputDirectory / f"{base_filename}bMat_c{self.rep}.mat"
#             filenameP = self.outputDirectory / f"{base_filename}params{self.rep}.mat"
#             filenameN = self.outputDirectory / f"{base_filename}network{self.rep}.mat"
#             filenameTr = self.outputDirectory / f"{base_filename}trajec{self.rep}.mat"
#             filenameEf = self.outputDirectory / f"{base_filename}envFluct{self.rep}.mat"
#             filenameS = self.outputDirectory / f"{base_filename}S{self.rep}.mat"
#             filenameD = self.outputDirectory / f"{base_filename}dMat{self.rep}.mat"
#             filenameEM = self.outputDirectory / f"{base_filename}emMat{self.rep}.mat"
#             filenameEMC = self.outputDirectory / f"{base_filename}emMat_c{self.rep}.mat"
#             filenameC = self.outputDirectory / f"{base_filename}cMat_reg{self.rep}.mat"
#             filenameT = self.outputDirectory / f"{base_filename}tMat{self.rep}.mat"
#             filenameE = self.outputDirectory / f"{base_filename}envMat{self.rep}.mat"
#             filenameBs = self.outputDirectory / f"{base_filename}bMat_src{self.rep}.mat"
#             filenameBCs = self.outputDirectory / f"{base_filename}bMat_c_src{self.rep}.mat"
#             self.bMatFileName = str(filenameB)
#             print(f"Biomass matrix file name: {self.bMatFileName}")
#         else:
#             # Use existing bMatFileName to generate paths for other matrices
#             filenameBC = self.bMatFileName.replace("bMat", "bMat_c")
#             filenameD = self.bMatFileName.replace("bMat", "dMat")
#             filenameEM = self.bMatFileName.replace("bMat", "emMat")
#             filenameEMC = self.bMatFileName.replace("bMat", "emMat_c")
#             filenameI = self.bMatFileName.replace("bMat", "cMat")
#             filenameC = self.bMatFileName.replace("bMat", "cMat_reg")
#             filenameA = self.bMatFileName.replace("bMat", "aMat")
#             filenameN = self.bMatFileName.replace("bMat", "network")
#             filenameP = self.bMatFileName.replace("bMat", "params")
#             filenameR = self.bMatFileName.replace("bMat", "rMat")
#             filenameSR = self.bMatFileName.replace("bMat", "sMat")
#             filenameS = self.bMatFileName.replace("bMat", "S")
#             filenameT = self.bMatFileName.replace("bMat", "tMat")
#             filenameE = self.bMatFileName.replace("bMat", "envMat")
#             filenameTr = self.bMatFileName.replace("bMat", "trajec")
#             filenameEf = self.bMatFileName.replace("bMat", "envFluct")
#             filenameBs = self.bMatFileName.replace("bMat", "bMat_src")
#             filenameBCs = self.bMatFileName.replace("bMat", "bMat_c_src")

#         # Store matrix objects accordingly
#         if self.spp.trajectories.size != 0:
#             savemat(filenameTr, {'trajectories': self.spp.trajectories})
#             if self.spp.efMat.size != 0:
#                 R = self.spp.rMat.flatten()
#                 N = np.full((1, R.shape[0]), np.nan)
#                 R = np.vstack((R, N))
#                 self.spp.fluctuations = np.vstack((R, self.spp.fluctuations))
#                 savemat(filenameEf, {'fluctuations': self.spp.fluctuations})
#         if self.cMat_reg.size != 0:
#             savemat(filenameC, {'cMat_reg': self.cMat_reg})
#         if self.spp.xMat_src.size != 0:
#             B_p_src = self.spp.xMat_src[:self.spp.rMat.shape[0], :]
#             savemat(filenameBs, {'B_p_src': B_p_src})
#             if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
#                 B_c_src = self.spp.xMat_src[self.spp.rMat.shape[0]:, :]
#                 savemat(filenameBCs, {'B_c_src': B_c_src})
#         else:
#             savemat(filenameN, {'network': self.spp.topo.network})
#             savemat(filenameR, {'rMat': self.spp.rMat})
#             if self.spp.sMat.size != 0:
#                 savemat(filenameSR, {'sMat': self.spp.sMat})
#             B_p = self.spp.xMat[:self.spp.S_p, :]
#             savemat(self.bMatFileName, {'B_p': B_p})
#             if self.spp.emMat.size != 0:
#                 Em_p = self.spp.emMat[:self.spp.rMat.shape[0], :]
#                 savemat(filenameEM, {'Em_p': Em_p})
#                 if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
#                     Em_c = self.spp.emMat[self.spp.rMat.shape[0]:, :]
#                     savemat(filenameEMC, {'Em_c': Em_c})
#             if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
#                 B_c = self.spp.xMat[self.spp.rMat.shape[0]:, :]
#                 savemat(filenameBC, {'B_c': B_c})
#             if self.spp.topo.envMat.size != 0:
#                 savemat(filenameE, {'envMat': self.spp.topo.envMat})
#                 savemat(filenameT, {'tMat': self.spp.tMat})
#             savemat(filenameS, {'sppRichness': self.spp.sppRichness})
#             savemat(filenameD, {'dMat': self.spp.dMat})
#             if self.spp.S_p > 0:
#                 C = self.spp.cMat[:self.spp.S_p, :self.spp.S_p]
#                 savemat(filenameI, {'C': C})
#             if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
#                 A = self.spp.cMat[self.spp.S_p:, :self.spp.S_p]
#                 savemat(filenameA, {'A': A})
#             self.writePars(filenameP)



#     def loadMC(self, bMatFile):
#         """
#         Import metacommunity model.
#         """
#         if not bMatFile:
#             raise ValueError("No valid bMat file path provided. Please provide a valid file path.")

#         self.bMatFileName = bMatFile
#         print(f"Importing data, file name {self.bMatFileName}")

#         # Generate file paths based on bMatFileName
#         try:
#             filenameBC = self.bMatFileName.replace("bMat", "bMat_c")
#             filenameD = self.bMatFileName.replace("bMat", "dMat")
#             filenameEM = self.bMatFileName.replace("bMat", "emMat")
#             filenameEMC = self.bMatFileName.replace("bMat", "emMat_c")
#             filenameI = self.bMatFileName.replace("bMat", "cMat")
#             filenameA = self.bMatFileName.replace("bMat", "aMat")
#             filenameN = self.bMatFileName.replace("bMat", "network")
#             filenameP = self.bMatFileName.replace("bMat", "params")
#             filenameR = self.bMatFileName.replace("bMat", "rMat")
#             filenameSR = self.bMatFileName.replace("bMat", "sMat")
#             filenameS = self.bMatFileName.replace("bMat", "S")
#             filenameT = self.bMatFileName.replace("bMat", "tMat")
#             filenameE = self.bMatFileName.replace("bMat", "envMat")
#             filenameTr = self.bMatFileName.replace("bMat", "trajec")
#             filenameIPp = self.bMatFileName.replace("bMat", "invProb_p")
#             filenameIPc = self.bMatFileName.replace("bMat", "invProb_c")

#             # Load matrix objects
#             self.spp.xMat = sio.loadmat(self.bMatFileName)['B_p']
#             self.spp.S_p = self.spp.xMat.shape[0]

#             self.spp.dMat = sio.loadmat(filenameD)['dMat']
#             if os.path.exists(filenameI):
#                 self.spp.cMat = sio.loadmat(filenameI)['C']
#             if os.path.exists(filenameEM):
#                 self.spp.emMat = sio.loadmat(filenameEM)['Em_p']
#             self.spp.topo.network = sio.loadmat(filenameN)['network']
#             if os.path.exists(filenameT):
#                 self.spp.tMat = sio.loadmat(filenameT)['tMat']
#             if os.path.exists(filenameE):
#                 self.spp.topo.envMat = sio.loadmat(filenameE)['envMat']
#             self.spp.rMat = sio.loadmat(filenameR)['rMat']
#             if os.path.exists(filenameSR):
#                 self.spp.sMat = sio.loadmat(filenameSR)['sMat']
#             self.spp.sppRichness = sio.loadmat(filenameS)['sppRichness']

#             print(f"\nImported network rows 0-4 = \n{self.spp.topo.network[:min(self.spp.topo.network.shape[0], 4)]}")
#             self.spp.topo.genDistMat()
#             print(f"\nSpecies richness, S_p = {self.spp.rMat.shape[0]}, S_c = {self.spp.xMat.shape[0] - self.spp.rMat.shape[0]}")

#         except KeyError as e:
#             print(f"Error while importing data: Missing key in MAT file - {e}")
#             raise
#         except FileNotFoundError as e:
#             print(f"Error while importing data: File not found - {e}")
#             raise







# # Example Usage Scenario
# if __name__ == "__main__":
#     # Step 1: Initialize Topography, Species, and Metacommunity
#     # Create Topography
#     topo = Topography(
#         no_nodes=10,                # Number of nodes in the landscape
#         lattice_height=5,           # Lattice height for spatial structure
#         lattice_width=5,            # Lattice width for spatial structure
#         phi=0.3,                    # Environmental heterogeneity parameter
#         envVar=1,                   # Environmental variability switch
#         skVec=np.array([0.1]),      # Environmental skewness vector
#         var_e=0.5,                  # Variance in environmental tolerance
#         randGraph=False,            # Random graph generation flag
#         gabriel=True,               # Use Gabriel graph connectivity
#         T_int=25.0,                 # Initial temperature
#         network_file="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",  # Environment data file
#         scVec=np.array([0.05]) 
#     )

#     # Create Species
#     spp = Species(
#         topo=topo,
#         c1=0.5,
#         c2=0.3,
#         c3=0.1,
#         emRate=0.2,
#         dispL=0.7,
#         pProducer=0.4,
#         prodComp=True,
#         symComp=True,
#         alpha=0.02,
#         sigma=1.5,
#         sigma_t=0.3,
#         rho=0.8,
#         comp_dist=1,
#         omega=0.9,
#         dispNorm=1.0,
#         bodymass=1e-4,  # Default value if needed
#         mu=0.1  # Default value if needed
#     )


#     # Initialize Metacommunity
#     meta_community = Metacommunity(spp,
#         a_init=True,
#         a_bMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
#         a_xMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1network0.mat",
#         a_scMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1params0.mat",
#         a_invMax=100,
#         a_tMax=500,
#         a_outputDirectory="output/",
#         a_c1=0.5,
#         a_c2=0.3,
#         a_c3=0.1,
#         a_emRate=0.2,
#         a_dispL=0.7,
#         a_pProducer=0.4,
#         a_prodComp=True,
#         a_symComp=True,
#         a_alpha=0.02,
#         a_sigma=1.5,
#         a_sigma_t=0.3,
#         a_rho=0.8,
#         a_comp_dist=1,
#         a_omega=0.9,
#         a_dispNorm=1.0,
#         a_no_nodes=10,
#         a_lattice_height=5,
#         a_lattice_width=5,
#         a_phi=0.3,
#         a_envVar=1,
#         a_skVec=np.array([0.1]),
#         a_var_e=0.5,
#         a_randGraph=False,
#         a_gabriel=True,
#         a_T_int=25.0,
#         a_envMat="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1rMat0.mat",  # Adjust as necessary to match what the constructor expects
#         a_parOut=1,
#         a_experiment="/content/drive/MyDrive/Simulation_data/autonomous_turnover_example_pars.txt",
#         a_rep=1,
#         storeTraj=1,  # Enable storing of trajectories
#         bMatFileName="/content/drive/MyDrive/Simulation_data/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat",
#         g_block_transitions=False
#     )

#     # Step 2: Simulate Metacommunity Dynamics
#     print("\nStarting Metacommunity Dynamics Simulation...")
#     meta_community.meta_c_dynamics(T=200)  # Simulate dynamics up to T=200

#     # Step 3: Invader Sampling
#     print("\nInvader Sampling...")
#     meta_community.invader_sample(trophLev=1, no_invaders=2)  # Sample 5 new producer species

#     # Step 4: Environmental Fluctuation
#     print("\nSimulating Environmental Fluctuation...")
#     meta_community.env_fluct()  # Apply one timestep of environmental fluctuation

#     # Step 5: Warming Event Simulation
#     print("\nSimulating Warming Event...")
#     meta_community.warming(dTdt=0.5, res=5, time=20)  # Simulate warming with temperature increase rate of 0.5

#     # Step 6: Long-Distance Dispersal
#     print("\nSimulating Long-Distance Dispersal...")
#     meta_community.long_dist_disp(tMax=100, edges=3)  # Randomly add 3 dispersal edges and simulate for 100 timesteps

#     # Step 7: Node Removal
#     print("\nRemoving Nodes from Metacommunity...")
#     meta_community.node_removal(tMax=100, no_removals=2)  # Remove 2 nodes from the community

#     # Step 8: Generate and Save Model Outputs
#     print("\nGenerating and Saving Outputs...")
#     meta_community.gen_jacobian()  # Generate Jacobian matrix
#     meta_community.gen_c_mat_reg(h=0.001)  # Generate regional interaction matrix
#     meta_community.gen_source_sink(t_full_relax=1000)  # Generate source-sink matrix
#     meta_community.saveMC()  # Save the metacommunity model matrices to files
#     meta_community.print_params()  # Print parameters to console
#     print("\nSimulation Completed.")


























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
from scipy.io import savemat  # Ensure this import is present
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

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

        # Update in meta_c_dynamics function or relevant context
        indices_S = np.nonzero(self.spp.xMat > 0)[1]  # Example: Find nonzero biomass indices

        # Calculate indices_DP based on the requirements
        # Assuming it represents all indices of xMat for now
        indices_DP = np.arange(self.spp.xMat.size)

        dynamics = CommunityDynamics(xMat=self.spp.xMat, spp=self.spp, cMat=self.spp.cMat, rMat=self.spp.rMat, indices_S=indices_S, indices_DP=indices_DP, dMat=self.spp.dMat)  # ODE dynamical object

        if dynamics.dMat_sp is None or dynamics.dMat_sp.size == 0:
            raise ValueError("The dispersal matrix (`dMat_sp`) in `CommunityDynamics` must be properly initialized.")

        # Initialize simulation parameters
        dynamics.current_time = 0.0
        convergence_threshold = 1e-6  # Threshold for detecting equilibrium
        max_iterations = int(T * 1000)  # Safety cap for iterations
        iterations = 0

        dynamics.xMat = self.spp.xMat
        dynamics.indices_DP = self.spp.indices_DP
        dynamics.indices_S = self.spp.indices_S
        dynamics.rMat = self.spp.rMat

        logger.info("Starting dynamics with properly initialized matrices.")

        # Verify if all necessary attributes are properly initialized
        if self.spp.dMat is None or self.spp.dMat.size == 0:
            raise ValueError("The dispersal matrix (`spp.dMat`) must be initialized before starting the dynamics.")
        logger.info("Starting dynamics with properly initialized dispersal matrix.")

        # Verify if dMat_sp is properly initialized
        if dynamics.dMat_sp is None or dynamics.dMat_sp.size == 0:
            raise ValueError("The dispersal matrix (`dMat_sp`) in `CommunityDynamics` must be properly initialized.")

        if self.spp.cMat.shape[0] != 0:
            dynamics.cMat = self.spp.cMat
        dynamics.rho = self.spp.rho
        dynamics.bodymass = self.spp.bodymass
        dynamics.bodymass_inv = self.spp.bodymass_inv
        dynamics.mu = self.spp.mu

        if self.spp.topo.scVec.shape[0] != 0:
            dynamics.scVec = self.spp.topo.scVec
            dynamics.scVec_prime = self.spp.topo.scVec_prime
        dynamics.S_p = self.spp.S_p
        dynamics.S_c = self.spp.S_c

        # Check if emMat is not None and has the correct shape
        if self.spp.emMat is not None and self.spp.emMat.shape[0] != 0:
            dynamics.emMat = self.spp.emMat
        else:
            logger.warning("Warning: emMat is either None or empty. Dynamics may be incomplete.")

        dynamics.dMat = self.spp.dMat

        # Handling file paths for storing trajectories
        Bpath, bMatDir, newFolder = "", "", ""
        if self.storeTraj != 0:
            # Create the directory structure for storing matrices
            pos1 = self.bMatFileName.rfind("/")
            bMatDir = self.bMatFileName[:pos1]
            pos1 = self.bMatFileName.rfind(")")
            pos2 = self.bMatFileName.rfind(".")
            newFolder = self.bMatFileName[pos1+1:pos2]
            if not self.g_block_transitions:
                Bpath = os.path.join(bMatDir, newFolder, "trajectory")
            else:
                Bpath = os.path.join(bMatDir, newFolder, "trajectory_wo_transitions")

            print(f"\nSaving matrices to {Bpath}")

            if not os.path.exists(Bpath):
                os.makedirs(Bpath)

        # Metacommunity relaxation step
        PRINT_X_MAT = False
        movie_mode = False
        resPrint = 1  # Temporal resolution for printing to console
        resSave = 100  # Temporal resolution for saving to file

        if PRINT_X_MAT and movie_mode:
            print("\033[2J")  # Clear screen

            if not self.spp.topo.randGraph:
                # Get current terminal size
                try:
                    terminal_lines, terminal_columns = os.popen('stty size', 'r').read().split()
                    terminal_lines = int(terminal_lines)
                    terminal_columns = int(terminal_columns)
                except Exception as e:
                    raise RuntimeError("Can't get terminal size.")

                species_per_row = (terminal_columns + 1) // ((7 + 2 + 1) * self.spp.topo.lattice_width)
                rows_of_species = min((terminal_lines - 2) // (self.spp.topo.lattice_height + 1),
                                      (self.spp.xMat.shape[0] - 1) // species_per_row + 1)

        previous_state = self.spp.xMat.copy()

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

        while dynamics.current_time < T:
            solver.simulate(T)
            current_state = solver.y[-1].reshape(self.spp.xMat.shape)
            state_change = np.linalg.norm(current_state - previous_state)

            if state_change < convergence_threshold:
                logger.info(f"System reached equilibrium at time {solver.t[-1]:.2f} with change {state_change:.2e}.")
                return

            previous_state = current_state.copy()

            if iterations >= max_iterations:
                logger.warning("Maximum iterations reached without convergence.")
                return



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

            # Update producer and consumer counts
            if idx < self.spp.S_p:  # Producer index
                self.spp.S_p -= 1
            else:  # Consumer index
                self.spp.S_c -= 1

            # Remove species from core matrices
            self.spp.xMat = np.delete(self.spp.xMat, idx, axis=0)
            if self.spp.cMat is not None and self.spp.cMat.shape[0] > 0:
                self.spp.cMat = np.delete(self.spp.cMat, idx, axis=0)
                self.spp.cMat = np.delete(self.spp.cMat, idx, axis=1)
            if self.spp.rMat is not None and self.spp.rMat.shape[0] > 0:
                self.spp.rMat = np.delete(self.spp.rMat, idx, axis=0)

            # Remove species from optional matrices
            optional_matrices = {'sMat': self.spp.sMat, 'tMat': self.spp.tMat, 'emMat': self.spp.emMat}
            for mat_name, mat in optional_matrices.items():
                if mat is not None:
                    if mat.shape[0] > idx:
                        setattr(self.spp, mat_name, np.delete(mat, idx, axis=0))
                    else:
                        logger.warning(f"Index {idx} is out of bounds for {mat_name}. Skipping.")

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

            B = np.maximum(0, self.spp.xMat[:self.spp.S_p] if trophLev == 1 else self.spp.xMat)
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

            bInv_max = bInv.max(axis=1) if len(bInv.shape) > 1 else bInv
            posGrowth = np.where(bInv_max >= min_b)[0][: no_invaders - suc_inv]
            negGrowth = np.setdiff1d(indices, posGrowth)

            logger.debug(f"Positive Growth Indices: {posGrowth}, Negative Growth Indices: {negGrowth}")

            suc_inv += len(posGrowth)
            for idx in reversed(negGrowth):
                self.remove_species([idx])

            logger.debug(f"State after removal - Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")

        if trophLev == 0:
            self.spp.S_p = self.spp.xMat.shape[0] - self.spp.S_c
        else:
            self.spp.S_c = self.spp.xMat.shape[0] - self.spp.S_p

        logger.info(f"Invader sampling complete. Producers: {self.spp.S_p}, Consumers: {self.spp.S_c}")



    def env_fluct(self):
        """
        Update abiotic turnover and simulate a single CVode timestep.
        """
        dynamics = CommunityDynamics(
            self.spp,
            self.spp.xMat,
            self.spp.cMat,
            self.spp.rMat,
            efMat=self.spp.efMat,
            scVec=self.spp.scVec,
            dMat=self.spp.dMat,
        )  # ODE dynamical object

        dynamics.xMat = self.spp.xMat
        dynamics.rMat = self.spp.rMat

        if self.spp.cMat.shape[0] != 0:
            dynamics.cMat = self.spp.cMat

        dynamics.dMat = self.spp.dMat  # If dynamics.dMat is initialized, single domain relaxation selected
        dynamics.rho = self.spp.rho

        self.spp.ou_process()  # Update the spatially resolved OU process to be added to the growth rate matrix
        dynamics.efMat = self.spp.efMat  # In CommunityDynamics rMat += efMat

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
        solver.simulate(1)

        tmp = solver.y[-1].reshape(dynamics.xMat.shape)

        # Handle trajectory shape mismatch
        if self.spp.trajectories.shape[1] != tmp.shape[1]:
            raise ValueError(
                f"Shape mismatch: trajectories has {self.spp.trajectories.shape[1]} columns, "
                f"but tmp has {tmp.shape[1]} columns."
            )
        self.spp.trajectories = np.vstack((self.spp.trajectories, tmp))  # Store trajectory

        RE = self.spp.rMat + self.spp.efMat
        RE = RE.reshape(1, -1)

        # Handle fluctuation shape mismatch
        if self.spp.fluctuations.shape[1] != RE.shape[1]:
            logger.warning(
                f"Fluctuation shape mismatch detected. Fluctuations shape: {self.spp.fluctuations.shape}, "
                f"RE shape: {RE.shape}. Reshaping fluctuations to match."
            )
            self.spp.fluctuations = np.zeros((0, RE.shape[1]))  # Reinitialize fluctuations to match RE

        self.spp.fluctuations = np.vstack((self.spp.fluctuations, RE))  # Store current distribution in R (= rMat + efMat)

    def warming(self, dTdt, res, time):
        """
        Updates temperature gradient and rMat, then simulates a single timestep.

        Args:
            dTdt (float): Rate of temperature increase.
            res (int): Number of steps per unit time.
            time (int): Total time for warming simulation.
        """
        # Create directory structure
        pos1 = self.bMatFileName.rfind("/")
        bMatDir = self.bMatFileName[:pos1]
        pos1 = self.bMatFileName.rfind(")")
        pos2 = self.bMatFileName.rfind(".")
        newFolder = self.bMatFileName[pos1 + 1:pos2]
        Bpath = os.path.join(bMatDir, newFolder, f"dTdt={dTdt}")

        if not os.path.exists(Bpath):
            os.makedirs(Bpath)

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

        for t in range(res):
            self.spp.topo.T_int += dTdt
            self.spp.update_r_vec_temp()
            dynamics.rMat = self.spp.rMat

            problem = Explicit_Problem(rhs, self.spp.xMat.flatten(), 0)
            solver = CVode(problem)
            solver.simulate(1)

        # Save current state to file
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
                    np.savetxt(filenameRR, pert_record, fmt='%e')
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
            np.savetxt(filenameRR, pert_record, fmt='%e')


      

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

        node_id = np.linspace(no_removals, self.spp.topo.no_nodes - 1, self.spp.topo.no_nodes - no_removals, dtype=int)
        spp_copy = self.spp

        for x in node_id:
            print(f"\nNode {x}")
            self.spp = spp_copy
            self.spp.xMat = np.delete(self.spp.xMat, x, axis=1)
            if self.spp.rMat.shape[0] > 0:
                self.spp.rMat = np.delete(self.spp.rMat, x, axis=1)
            self.spp.topo.network = np.delete(self.spp.topo.network, x, axis=0)
            self.spp.topo.distMat = np.delete(self.spp.topo.distMat, x, axis=0)
            self.spp.topo.distMat = np.delete(self.spp.topo.distMat, x, axis=1)
            self.spp.dMat = np.delete(self.spp.dMat, x, axis=0)
            self.spp.dMat = np.delete(self.spp.dMat, x, axis=1)
            if self.spp.topo.scVec.shape[0] > 0:
                self.spp.topo.scVec = np.delete(self.spp.topo.scVec, x, axis=0)
            if self.spp.topo.scVec_prime.shape[0] > 0:
                self.spp.topo.scVec_prime = np.delete(self.spp.topo.scVec_prime, x, axis=0)
            if self.spp.topo.envMat.shape[1] > 0:
                self.spp.topo.envMat = np.delete(self.spp.topo.envMat, x, axis=1)
            self.spp.topo.no_nodes -= 1

            dynamics = CommunityDynamics()
            dynamics.xMat = self.spp.xMat
            dynamics.rMat = self.spp.rMat

            if self.spp.cMat.shape[0] != 0:
                dynamics.cMat = self.spp.cMat

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
            solver.simulate(tMax)

            Bfile = os.path.join(Bpath, f"bMat_nr{x}.mat")
            np.savetxt(Bfile, self.spp.xMat, fmt='%e')
            if self.spp.S_c > 0:
                Bfile_c = os.path.join(Bpath, f"bMat_nr_c{x}.mat")
                B_c = self.spp.xMat[self.spp.S_p:, :]
                np.savetxt(Bfile_c, B_c, fmt='%e')



    def gen_jacobian(self):
        """
        Generate the numerical approximation of the Jacobian matrix for computing regional competitive overlap matrix
        """
        dynamics = CommunityDynamics()
        dynamics.xMat = self.spp.xMat
        dynamics.indices_DP = self.spp.indices_DP
        dynamics.indices_S = self.spp.indices_S
        dynamics.rMat = self.spp.rMat
        if self.spp.cMat.shape[0] != 0:
            dynamics.cMat = self.spp.cMat
        dynamics.rho = self.spp.rho
        dynamics.bodymass = self.spp.bodymass
        dynamics.bodymass_inv = self.spp.bodymass_inv
        dynamics.mu = self.spp.mu
        if self.spp.topo.scVec.shape[0] != 0:
            dynamics.scVec = self.spp.topo.scVec
            dynamics.scVec_prime = self.spp.topo.scVec_prime
        dynamics.S_p = self.spp.S_p
        dynamics.S_c = self.spp.S_c
        if self.spp.emMat.shape[0] != 0:
            dynamics.emMat = self.spp.emMat
        dynamics.dMat = self.spp.dMat

        def compute_jacobian(t, y):
            state = y.reshape(self.spp.xMat.shape)
            Gt = dynamics.compute_intrinsic_growth_rates(state)
            dXdt = Gt.copy()
            dXdt.flat[dynamics.indices_DP] *= state.flat[dynamics.indices_DP]

            if dynamics.emMat is None:
                massEffect = state @ dynamics.dMat_sp
                dXdt += massEffect
            else:
                emMat_N = np.repeat(dynamics.emMat[:, np.newaxis], state.shape[1], axis=1)
                massEffect = (emMat_N * state) @ dynamics.dMat_sp
                dXdt += massEffect

            negativeB = np.where(state < 0)
            if len(negativeB[0]) > 0:
                dXdt[negativeB] = -state[negativeB]

            nonfiniteX = np.where(~np.isfinite(dXdt))
            if len(nonfiniteX[0]) > 0:
                dXdt[nonfiniteX] = 0

            blowupX = np.where(np.abs(dXdt) > 1e10)
            if len(blowupX[0]) > 0:
                dXdt[blowupX] = 0

            return dXdt.flatten()

        y0 = self.spp.xMat.flatten()
        problem = Explicit_Problem(compute_jacobian, y0, 0)
        solver = CVode(problem)
        solver.simulate(1)  # Simulate to generate the Jacobian

        self.jacobian = solver.problem.jac

    def gen_c_mat_reg(self, h=0.001):
        """
        Generate a numerically approximated regional interaction matrix via a computation harvesting experiment
        :param h: Harvesting rate
        """
        self.gen_jacobian()
        
        if self.jacobian is None:
            raise ValueError("Jacobian matrix is not initialized.")
        
        J_inv = np.linalg.inv(self.jacobian)

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
        self.cMat_reg = np.linalg.inv(self.cMat_reg)

        # Normalize regional competitive overlap matrix
        norm = np.zeros_like(self.cMat_reg)
        sgn = np.zeros_like(self.cMat_reg)
        norm[np.diag_indices_from(self.cMat_reg)] = 1 / np.sqrt(np.abs(np.diag(self.cMat_reg)))
        sgn[np.diag_indices_from(self.cMat_reg)] = np.diag(self.cMat_reg) / np.abs(np.diag(self.cMat_reg))
        self.cMat_reg = sgn @ norm @ self.cMat_reg @ norm

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
        # Store matrix objects accordingly
        if self.spp.trajectories is not None and self.spp.trajectories.size != 0:
            savemat(filenameTr, {'trajectories': self.spp.trajectories})
            if self.spp.efMat is not None and self.spp.efMat.size != 0:
                R = self.spp.rMat.flatten()
                N = np.full((1, R.shape[0]), np.nan)
                R = np.vstack((R, N))
                self.spp.fluctuations = np.vstack((R, self.spp.fluctuations))
                savemat(filenameEf, {'fluctuations': self.spp.fluctuations})

        if self.cMat_reg is not None and self.cMat_reg.size != 0:
            savemat(filenameC, {'cMat_reg': self.cMat_reg})

        if self.spp.xMat_src is not None and self.spp.xMat_src.size != 0:
            B_p_src = self.spp.xMat_src[:self.spp.rMat.shape[0], :]
            savemat(filenameBs, {'B_p_src': B_p_src})
            if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
                B_c_src = self.spp.xMat_src[self.spp.rMat.shape[0]:, :]
                savemat(filenameBCs, {'B_c_src': B_c_src})
        else:
            savemat(filenameN, {'network': self.spp.topo.network})
            savemat(filenameR, {'rMat': self.spp.rMat})
            if self.spp.sMat is not None and self.spp.sMat.size != 0:
                savemat(filenameSR, {'sMat': self.spp.sMat})
            B_p = self.spp.xMat[:self.spp.S_p, :]
            savemat(self.bMatFileName, {'B_p': B_p})
            if self.spp.emMat is not None and self.spp.emMat.size != 0:
                Em_p = self.spp.emMat[:self.spp.rMat.shape[0], :]
                savemat(filenameEM, {'Em_p': Em_p})
                if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
                    Em_c = self.spp.emMat[self.spp.rMat.shape[0]:, :]
                    savemat(filenameEMC, {'Em_c': Em_c})
            if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
                B_c = self.spp.xMat[self.spp.rMat.shape[0]:, :]
                savemat(filenameBC, {'B_c': B_c})
            if self.spp.topo.envMat is not None and self.spp.topo.envMat.size != 0:
                savemat(filenameE, {'envMat': self.spp.topo.envMat})
                savemat(filenameT, {'tMat': self.spp.tMat})
            savemat(filenameS, {'sppRichness': self.spp.sppRichness})
            savemat(filenameD, {'dMat': self.spp.dMat})
            if self.spp.S_p > 0:
                C = self.spp.cMat[:self.spp.S_p, :self.spp.S_p]
                savemat(filenameI, {'C': C})
            if self.spp.xMat.shape[0] > self.spp.rMat.shape[0]:
                A = self.spp.cMat[self.spp.S_p:, :self.spp.S_p]
                savemat(filenameA, {'A': A})
            self.write_params(filenameP)  # Use the correct method name



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
        a_experiment="/content/drive/MyDrive/Simulation_data/autonomous_turnover_example_pars.txt",
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