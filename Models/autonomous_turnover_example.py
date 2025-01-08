# -*- coding: utf-8 -*-


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import pairwise_distances
from IPython.display import Image, display  # Import Image and display for inline visualization



ANALYSE_DATA = False  # Set to True if you want to perform the analysis

# Directory paths and corresponding labels
dir_list = [    
    "/Users/sarashahin/Desktop/model/SimulationData/N=32/discrTraj_experiment/2021-3-1/2021-3-1_discrTraj(10000)1bMat0.mat",
    "/Users/sarashahin/Desktop/model/SimulationData/N=32/betaDiscrTraj04_experiment/2021-3-1/2021-3-1_betaDiscrTraj04(10000)1bMat0.mat",
    "/Users/sarashahin/Desktop/model/SimulationData/N=32/normDiscrTraj04_experiment/2021-3-1/2021-3-1_normDiscrTraj04(10000)1bMat0.mat"
]
distr = ["discr", "betaDiscr0.4", "normDiscr0.4"]
cores = 3

# Analyze data function
def analyse_directory(dir_path, distr_name):
    alpha_mn, gamma, beta_t, beta_s = [], [], [], []
    inv_numbers = []  # Initialize inv_numbers to ensure it is always defined

    # Iterate over each invasion directory in the given path
    for root, _, files in os.walk(dir_path):
        f_list = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
        inv_numbers = [int(re.findall(r'\d+', file)[0]) for file in f_list]

        for inv_dir in inv_numbers:
            sub_path = os.path.join(root, str(inv_dir))
            b_list = sorted(os.listdir(sub_path), key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Read source matrix and remove "src" related files
            try:
                src_file = [f for f in b_list if 'src' in f][0]
                b_src = pd.read_csv(os.path.join(sub_path, src_file), header=None).values
            except IndexError:
                print(f"No source file found in {sub_path}, skipping.")
                continue

            b_list = [f for f in b_list if 'src' not in f]

            if len(b_list) == 0:
                print(f"No trajectory files found in {sub_path}, skipping.")
                continue

            b = []
            # Loop through time series and create (S x N x T) trajectory object
            for t in b_list:
                b_mat = pd.read_csv(os.path.join(sub_path, t), header=None).values
                b.append(b_mat.flatten())

            b = np.array(b)
            b[b <= 0] = 0

            # Record statistics: mean alpha, gamma, beta temporal, beta spatial
            N = b_src.shape[1]
            S = b_src.shape[0]
            alpha_mn.append(np.mean(np.sum(b_src == 1, axis=0)))  # Local source richness
            gamma.append(S)  # Regional richness

            mean_bc = []
            for x in range(N):
                Bxt = b[:, S * x:S * (x + 1)]
                Bxt[Bxt <= 0] = 0
                bc = pairwise_distances(Bxt, metric='braycurtis')
                mean_bc.append(np.mean(bc[np.triu_indices(len(bc), k=1)]))
            beta_t.append(np.mean(mean_bc))

            b_t = pd.read_csv(os.path.join(sub_path, b_list[0]), header=None).values
            bc_spatial = pairwise_distances(b_t.T, metric='braycurtis')
            beta_s.append(np.mean(bc_spatial))

    if not alpha_mn:
        print(f"No valid data found in directory {dir_path}, skipping.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid data is found

    data = pd.DataFrame({
        'inv': inv_numbers[:len(alpha_mn)],  # Ensure the length of 'inv' matches other columns
        'it': list(range(len(gamma))),
        'alpha.mn': alpha_mn,
        'gamma': gamma,
        'beta.t': beta_t,
        'beta.s': beta_s,
        'distr': [distr_name] * len(alpha_mn)
    })

    return data


# Ensure the analysis is run if needed or load existing data
if ANALYSE_DATA:
    with ProcessPoolExecutor(max_workers=cores) as executor:
        results = list(executor.map(analyse_directory, dir_list, distr))
    dat = pd.concat([r for r in results if not r.empty], ignore_index=True)
    # Save the resulting DataFrame to a CSV
    dat.to_csv("/Users/sarashahin/Desktop/model/SimulationData/autonomous_turnover_example.csv", index=False)
else:
    dat = pd.read_csv("/Users/sarashahin/Desktop/model/SimulationData/autonomous_turnover_example.csv")

# Check if 'alpha.mn' exists in the DataFrame
if 'alpha.mn' not in dat.columns:
    raise ValueError("Column 'alpha.mn' is missing from the data. Please check the analysis process.")

# Print the columns to debug the issue if the KeyError persists
print("Columns in `dat` DataFrame:", dat.columns)

# Create DataFrames for visualization
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

# Proceed with the plotting part of the script

# Generate Figure 5
try:
    # Extracting the index of the threshold where the condition is satisfied
    threshold = dat_b[(dat_b['level'] == 'temp') & (dat_b['distr'] == 'discr') & (dat_b['beta'] > 1e-2)].index.min()
except ValueError:
    print("No suitable threshold found for temp level in discr distribution. Please verify the data.")
    threshold = None

threshold_value = None  # Initialize threshold_value to ensure it is always defined

plt.figure(figsize=(12, 6))

# Plot Species Richness Over Time
plt.subplot(1, 2, 1)
sns.lineplot(data=dat_s[dat_s['distr'] == 'discr'], x='it', y='S', hue='level', style='level', markers=True)

# If a valid threshold is found, plot the vertical line
if threshold is not None and not pd.isna(threshold):
    try:
        # Ensure that threshold is an integer or a valid index
        threshold_value = dat['it'].iloc[threshold] if isinstance(threshold, int) else float(threshold)
        plt.axvline(x=threshold_value, linestyle='--', color='grey')
    except (IndexError, ValueError, KeyError) as e:
        print(f"Invalid threshold value: {threshold}. Could not plot vertical line. Error: {e}")

plt.yscale('log')
plt.xlabel('Time (iterations of assembly model)')
plt.ylabel('Species richness')
plt.title('Species Richness Over Time')
plt.legend(title='Level')

# Plot Mean BC Dissimilarity Over Time
plt.subplot(1, 2, 2)
sns.lineplot(data=dat_b[dat_b['distr'] == 'discr'], x='it', y='beta', hue='level', style='level', markers=True)

# Use the threshold value for the second plot, if defined
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

# Save the figure
plt.savefig('metacommunity_visualization_debug.png', dpi=300)

# Show the plot
plt.show()

# Display the saved visualization
display(Image(filename='metacommunity_visualization_debug.png'))
