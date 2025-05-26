
############################################
# test_output.py
############################################

import numpy as np
import os


# Determine the base directory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the relative path to the data file
data_file = os.path.join(BASE_DIR, "results_seed101/model_outputs.npz")

# Load the data using NumPy
data = np.load(data_file, allow_pickle=True)

# Set print options to display the entire array
# np.set_printoptions(threshold=np.inf, linewidth=10,edgeitems=3 )

print(data.files)  # Should output: ['iteration', 'producers', 'consumers', 'xMat']
print('IBM')
print(data['IBM'])
print('PSD')
print(data['PSD'])
print('PSD2')
print(data['PSD2'])
print("PSD2_waiting")
print(data['PSD2_waiting'])
print('PSD2_poisson_clock')
print(data['PSD2_poisson_clock'])
print('PSD2_growth_rate')
print(data['PSD2_growth_rate'])
print('PSD2_invasion_rate')
print(data['PSD2_invasion_rate'])
print('PSD2_est_prob')
print(data['PSD2_est_prob'])
# print("i")
# print(data['i'])
print("ODE")
print(data['ODE'])

# print(data['xMat'].shape)
# extinction_log = data['extinction_log'].item()  # retrieve the dictionary
# print("extinction_log", extinction_log)


from numpy import load	
rows = load("results_seed101/model_outputs.npz")['PSD2'].shape[0]	
print('PSD2')
print(rows) # should be 2001 (2 000 000 / 1 000 + 1)
