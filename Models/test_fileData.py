import numpy as np
import os

# Determine the base directory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the data file 
data_file = os.path.join(BASE_DIR, "output", "trajectory_data_PSD.npz")

# Load the data using NumPy 
data = np.load(data_file, allow_pickle=True)

# Print the keys (files) stored in the NPZ file
print("Files in the NPZ data:")

print(data.files)  # Should output: ['iteration', 'producers', 'consumers', 'xMat']
print(data['iteration'])
print('producers')
print(data['producers'])
print('consumers')
print(data['consumers'])
print("xMat/")
print(data['xMat'])
print("rMat")
print(data['rMat'])
print('PSD_states')
print(data['PSD_states'])
print('PoissonClocks')
print(data['PoissonClocks'])
print('logB')
print(data['logB'])
print('establishment_prob')
print(data['establishment_prob'])
print("i")
print(data['i'])
print(data['xMat'].shape)
# extinction_log = data['extinction_log'].item()  # retrieve the dictionary
# print("extinction_log", extinction_log)
