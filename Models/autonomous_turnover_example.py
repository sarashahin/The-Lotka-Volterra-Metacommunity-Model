############################################
# autonomous_turnover_example.py
############################################
"""
This script analyzes metacommunity simulation data.
It computes four main metrics for each invasion:
    - Local source richness (alpha_mn)
    - Regional richness (gamma)
    - Temporal beta diversity (beta_t)
    - Spatial beta diversity (beta_s)

Metric definitions:
    -  Local Source Richness (alpha_mn):
    -  Calculates the average number of active sources per location.
    -   Equation: αₘₙ = (1/N) * Σᵢ Σⱼ I(b_src[j, i] == 1)
       where N is the number of sites and I() is the indicator function.
    
    - Regional Richness (gamma):
       The total number of source sites (S).
       Equation: γ = S
    
    - Temporal Beta Diversity (beta_t):
       Computed using the Bray-Curtis dissimilarity over time for each site.
       Equation (per site x): βₜ₍ₓ₎ = (2/(T*(T-1))) * Σᵢ<ⱼ d(i, j)
       and averaged over all sites: βₜ = (1/N) * Σₓ βₜ₍ₓ₎
       where T is the number of time steps.
    
    - Spatial Beta Diversity (beta_s):
       Computed from the first trajectorys spatial data using Bray-Curtis dissimilarity.
       Equation: βₛ = (2/(S*(S-1))) * Σᵢ<ⱼ d(i, j)
       where S is the number of sites.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import pairwise_distances
from IPython.display import Image, display  # For inline visualization

# Set to True to perform analysis; False to load existing results
ANALYSE_DATA = False

# Relative directory paths for the simulation data
dir_list = [
    "./SimulationData/N=32/discrTraj_experiment/2021-3-1/2021-3-1_discrTraj(10000)1bMat0.mat",
    "./SimulationData/N=32/betaDiscrTraj04_experiment/2021-3-1/2021-3-1_betaDiscrTraj04(10000)1bMat0.mat",
    "./SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat"
]
# Corresponding labels for each data directory
distr = ["discr", "betaDiscr0.4", "normDiscr0.4"]

# Number of cores for parallel processing
cores = 3

def analyse_directory(dir_path, distr_name):
    """
    Analyze the data in a given directory to compute community metrics.

    Parameters:
        dir_path (str): The relative path to the data directory.
        distr_name (str): Label for the type of distribution (experiment).

    Returns:
        pd.DataFrame: DataFrame containing computed metrics.
    """
    # Initialize lists to store metrics
    alpha_mn, gamma, beta_t, beta_s = [], [], [], []
    inv_numbers = []  # List to store invasion identifiers

    # Walk through the directory structure
    for root, _, files in os.walk(dir_path):
        # Sort files numerically based on digits in the filename
        f_list = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
        inv_numbers = [int(re.findall(r'\d+', file)[0]) for file in f_list]

        for inv in inv_numbers:
            sub_path = os.path.join(root, str(inv))
            # List all files in the subdirectory and sort them numerically
            b_list = sorted(os.listdir(sub_path), key=lambda x: int(re.findall(r'\d+', x)[0]))

            # retrieve the source matrix file (identified by 'src' in the filename)
            try:
                src_file = [f for f in b_list if 'src' in f][0]
                b_src = pd.read_csv(os.path.join(sub_path, src_file), header=None).values
            except IndexError:
                print(f"No source file found in {sub_path}, skipping.")
                continue

            # remove source file from the list to process trajectory files only
            b_list = [f for f in b_list if 'src' not in f]
            if len(b_list) == 0:
                print(f"No trajectory files found in {sub_path}, skipping.")
                continue

            # read trajectory data for each time step and flatten the matrix into a 1D array per time
            b = []
            for t in b_list:
                b_mat = pd.read_csv(os.path.join(sub_path, t), header=None).values
                b.append(b_mat.flatten())
            b = np.array(b)
            b[b <= 0] = 0  # Replace non-positive values with 0

            #metric Calculations:
            N = b_src.shape[1]  # Number of sites
            S = b_src.shape[0]  # Number of source sites (regional richness)
            
            # 1. Local Source Richness (alpha_mn)
            # αₘₙ = (1/N) * Σᵢ (number of sources at site i)
            alpha_mn.append(np.mean(np.sum(b_src == 1, axis=0)))
            
            # 2. Regional Richness (gamma)
            # γ = S
            gamma.append(S)
            
            # 3. Temporal Beta Diversity (beta_t)
            # for each site x, compute the Bray-Curtis dissimilarity over time and average:
            # βₜ₍ₓ₎ = (2/(T*(T-1))) * Σᵢ<ⱼ d(i, j), then βₜ = (1/N) * Σₓ βₜ₍ₓ₎
            mean_bc = []
            for x in range(N):
                Bxt = b[:, S * x:S * (x + 1)]
                Bxt[Bxt <= 0] = 0
                bc = pairwise_distances(Bxt, metric='braycurtis')
                # Use only the upper triangular part (excluding the diagonal) for pairwise comparisons
                mean_bc.append(np.mean(bc[np.triu_indices(len(bc), k=1)]))
            beta_t.append(np.mean(mean_bc))
            
            # 4. spatial Beta Diversity (beta_s)
            # compute Bray-Curtis dissimilarity between sites at the first time step:
            # βₛ = (2/(S*(S-1))) * Σᵢ<ⱼ d(i, j)
            b_t = pd.read_csv(os.path.join(sub_path, b_list[0]), header=None).values
            bc_spatial = pairwise_distances(b_t.T, metric='braycurtis')
            beta_s.append(np.mean(bc_spatial))
    
    # if no valid data is found, return an empty DataFrame
    if not alpha_mn:
        print(f"No valid data found in directory {dir_path}, skipping.")
        return pd.DataFrame()
    
    # create a DataFrame with the computed metrics.
    # Note: The invasion numbers are sliced to match the length of the computed metrics.
    data = pd.DataFrame({
        'inv': inv_numbers[:len(alpha_mn)],
        'it': list(range(len(gamma))),
        'alpha.mn': alpha_mn,
        'gamma': gamma,
        'beta.t': beta_t,
        'beta.s': beta_s,
        'distr': [distr_name] * len(alpha_mn)
    })
    
    return data

# run analysis if required, or load precomputed data
if ANALYSE_DATA:
    with ProcessPoolExecutor(max_workers=cores) as executor:
        results = list(executor.map(analyse_directory, dir_list, distr))
    dat = pd.concat([r for r in results if not r.empty], ignore_index=True)
    # Save the resulting DataFrame to a CSV file 
    dat.to_csv("./SimulationData/autonomous_turnover_example.csv", index=False)
else:
    dat = pd.read_csv("./SimulationData/autonomous_turnover_example.csv")

#make sure that the 'alpha.mn' column exists
if 'alpha.mn' not in dat.columns:
    raise ValueError("Column 'alpha.mn' is missing from the data. Please check the analysis process.")

# print DataFrame columns for debugging purposes
print("Columns in `dat` DataFrame:", dat.columns)

#prepare DataFrames for visualization
dat_s = pd.DataFrame({
    'inv': pd.concat([dat['inv'], dat['inv']]),
    'it': pd.concat([dat['it'], dat['it']]),
    'distr': pd.concat([dat['distr'], dat['distr']]),
    'S': pd.concat([dat['gamma'], dat['alpha.mn']]),
    'level': ['gamma'] * len(dat) + ['alpha'] * len(dat)
})

dat_b = pd.DataFrame({
    'inv': pd.concat([dat['inv'], dat['inv']]),
    'it': pd.concat([dat['it'], dat['it']]),
    'distr': pd.concat([dat['distr'], dat['distr']]),
    'beta': pd.concat([dat['beta.t'], dat['beta.s']]),
    'level': ['temp'] * len(dat) + ['spat'] * len(dat)
})

#plotting the results
# attempt to extract a threshold index for the 'discr' distribution where temporal beta exceeds 1e-2.
try:
    threshold = dat_b[(dat_b['level'] == 'temp') & (dat_b['distr'] == 'discr') & (dat_b['beta'] > 1e-2)].index.min()
except ValueError:
    print("No suitable threshold found for temp level in discr distribution. Please verify the data.")
    threshold = None

threshold_value = None  # Initialize threshold_value

plt.figure(figsize=(12, 6))

# Subplot 1: Species Richness Over Time
plt.subplot(1, 2, 1)
sns.lineplot(data=dat_s[dat_s['distr'] == 'discr'], x='it', y='S', hue='level', style='level', markers=True)
if threshold is not None and not pd.isna(threshold):
    try:
        threshold_value = dat['it'].iloc[threshold] if isinstance(threshold, int) else float(threshold)
        plt.axvline(x=threshold_value, linestyle='--', color='grey')
    except (IndexError, ValueError, KeyError) as e:
        print(f"Invalid threshold value: {threshold}. Could not plot vertical line. Error: {e}")

plt.yscale('log')
plt.xlabel('Time (iterations of assembly model)')
plt.ylabel('Species richness')
plt.title('Species Richness Over Time')
plt.legend(title='Level')

# Subplot 2: Mean Bray-Curtis Dissimilarity Over Time
plt.subplot(1, 2, 2)
sns.lineplot(data=dat_b[dat_b['distr'] == 'discr'], x='it', y='beta', hue='level', style='level', markers=True)
if threshold_value is not None:
    try:
        plt.axvline(x=threshold_value, linestyle='--', color='grey')
    except (IndexError, ValueError, KeyError) as e:
        print(f"Invalid threshold value: {threshold_value}. Could not plot vertical line. Error: {e}")

plt.ylim(0, 1)
plt.xlabel('Time (iterations of assembly model)')
plt.ylabel('Mean BC dissimilarity')
plt.title('Mean BC Dissimilarity Over Time')
plt.legend(title='Level')

plt.tight_layout()

# Save and display the figure 
plt.savefig('./metacommunity_visualization_debug.png', dpi=300)
plt.show()
display(Image(filename='./metacommunity_visualization_debug.png'))
