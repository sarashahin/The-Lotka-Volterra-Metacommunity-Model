############################################
# visualization.py
############################################
"""
Visualization module for the PSD model.
Provides functions for plotting spatial patterns, dispersal analysis,
and environmental effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import os
from config_base import IBMConfig
import matplotlib.gridspec as gridspec
import networkx as nx

# Create output directory if it doesn't exist
OUTPUT_DIR = 'visualization_outputs'
PSD2_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'psd2_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PSD2_OUTPUT_DIR, exist_ok=True)

def ensure_output_dir(model_type='general'):
    """Ensure output directory exists and return its path."""
    if model_type == 'psd2':
        os.makedirs(PSD2_OUTPUT_DIR, exist_ok=True)
        return PSD2_OUTPUT_DIR
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return OUTPUT_DIR

def plot_spatial_patterns(trajectory, dispersal_type, wind_direction, habitat_quality, save_prefix='', model_type='general'):
    """Enhanced spatial pattern visualization with theoretical predictions."""
    output_dir = ensure_output_dir(model_type)
    config = IBMConfig()
    final_state = trajectory[-1]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Species distribution with wind and habitat effects
    for sp in range(final_state.shape[0]):
        ax = axes[sp//2, sp%2]
        
        # Plot biomass distribution
        im = ax.imshow(final_state[sp], cmap='viridis')
        plt.colorbar(im, ax=ax, label='Biomass')
        
        # Add wind direction arrows
        y, x = np.mgrid[0:final_state.shape[1]:2, 0:final_state.shape[2]:2]
        u = np.cos(wind_direction) * np.ones_like(x)
        v = np.sin(wind_direction) * np.ones_like(y)
        ax.quiver(x, y, u, v, color='white', alpha=0.6)
        
        # Add habitat quality contours
        ax.contour(habitat_quality, colors='red', alpha=0.3, levels=5)
        
        ax.set_title(f'Species {sp+1} Distribution\nMean: {np.mean(final_state[sp]):.3f}, CV: {np.std(final_state[sp])/np.mean(final_state[sp]):.3f}')
    
    # Add spatial statistics
    ax = axes[1, 1]
    distances = np.arange(1, 6)
    for sp in range(final_state.shape[0]):
        correlations = [calculate_distance_correlation(final_state[sp], d) for d in distances]
        ax.plot(distances, correlations, label=f'Species {sp+1}', marker='o')
    
    ax.set_xlabel('Distance')
    ax.set_ylabel('Spatial Correlation')
    ax.set_title('Distance Decay of Spatial Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{save_prefix}spatial_patterns_{dispersal_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_distance_correlation(grid, distance):
    """
    Calculate correlation between points at given distance with robust handling of edge cases.
    """
    n_y, n_x = grid.shape
    pairs = []
    
    for i in range(n_y):
        for j in range(n_x):
            for di in range(-distance, distance+1):
                for dj in range(-distance, distance+1):
                    if abs(di) + abs(dj) == distance:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_y and 0 <= nj < n_x:
                            pairs.append((grid[i,j], grid[ni,nj]))
    
    if not pairs:  # No valid pairs found
        return 0
    
    pairs = np.array(pairs)
    x, y = pairs[:, 0], pairs[:, 1]
    
    # Handle edge cases
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0
    
    # Calculate correlation with numerical stability
    x_std = np.std(x)
    y_std = np.std(y)
    
    if x_std < 1e-10 or y_std < 1e-10:  # Effectively zero variance
        return 0
    
    x_norm = (x - np.mean(x)) / x_std
    y_norm = (y - np.mean(y)) / y_std
    
    correlation = np.mean(x_norm * y_norm)
    
    # Bound the result to [-1, 1]
    return np.clip(correlation, -1.0, 1.0)

def plot_temporal_evolution(trajectory, dispersal_type='adult', save_prefix='', model_type='general'):
    """Plot temporal evolution with enhanced analysis features."""
    output_dir = ensure_output_dir(model_type)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Analytical equilibrium values
    analytical_eq = np.array([7.5, 3.5, 5.5])
    times = np.arange(len(trajectory))
    
    # Calculate mean and standard deviation across spatial dimensions
    mean_biomass = np.mean(trajectory, axis=(2,3))
    std_dev = np.std(trajectory, axis=(2,3))
    
    colors = ['blue', 'red', 'green']
    species_names = ['Species 1', 'Species 2', 'Species 3']
    
    for sp in range(trajectory.shape[1]):
        # Plot mean biomass
        ax.plot(times, mean_biomass[:, sp], color=colors[sp], label=species_names[sp])
        
        # Add confidence intervals
        ax.fill_between(times, 
                       mean_biomass[:, sp] - std_dev[:, sp],
                       mean_biomass[:, sp] + std_dev[:, sp],
                       color=colors[sp], alpha=0.2)
        
        # Add analytical equilibrium lines
        ax.axhline(y=analytical_eq[sp], color=colors[sp], linestyle='--', alpha=0.5,
                  label=f'{species_names[sp]} equilibrium')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mean Biomass')
    ax.set_title(f'Temporal Evolution of Species Biomass\n({dispersal_type.capitalize()} Dispersal)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add relative error annotation
    final_mean = mean_biomass[-1]
    rel_error = (final_mean - analytical_eq) / analytical_eq
    error_text = "Relative Error from Equilibrium:\n"
    for sp in range(len(rel_error)):
        error_text += f"{species_names[sp]}: {rel_error[sp]:.2e}\n"
    ax.text(1.05, 0.5, error_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{save_prefix}temporal_evolution_{dispersal_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_environmental_effects(trajectory, wind_direction, wind_strength, habitat_quality, save_prefix='', model_type='general'):
    """Plot environmental effects on species distribution."""
    output_dir = ensure_output_dir(model_type)
    config = IBMConfig()
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)
    
    # 1. Wind effect
    ax1 = fig.add_subplot(gs[0, 0])
    y, x = np.meshgrid(np.linspace(-1, 1, habitat_quality.shape[0]),
                       np.linspace(-1, 1, habitat_quality.shape[1]))
    wind_effect = wind_strength * np.ones_like(x)
    wind_gradient = wind_effect * np.cos(np.arctan2(y, x) - wind_direction)
    im1 = ax1.imshow(wind_gradient, cmap='RdBu')
    ax1.set_title('Wind Effect')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Habitat quality
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(habitat_quality, cmap='viridis')
    ax2.set_title('Habitat Quality')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Combined effect on final distribution
    ax3 = fig.add_subplot(gs[0, 2])
    final_distribution = np.mean(trajectory[-1], axis=0)  # Average across species
    im3 = ax3.imshow(final_distribution, cmap='YlOrRd')
    ax3.set_title('Final Distribution')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{save_prefix}_environmental_effects.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_species_comparison(trajectory, seed_sizes, save_prefix=''):
    """Compare species characteristics and their relationships with seed sizes."""
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.1])
    
    # Ensure seed_sizes is a numpy array with the right length
    seed_sizes = np.array(seed_sizes[:trajectory.shape[1]])
    
    # 1. Biomass vs Seed Size with theoretical prediction
    ax1 = fig.add_subplot(gs[0, 0])
    final_biomass = np.mean(trajectory[-1], axis=(1, 2))
    ax1.scatter(seed_sizes, final_biomass, s=100, label='Observed')
    
    # Add theoretical prediction
    ss_range = np.linspace(min(seed_sizes), max(seed_sizes), 100)
    theoretical_biomass = 10 * np.exp(-0.5 * (ss_range - np.mean(seed_sizes))**2 / np.var(seed_sizes))
    ax1.plot(ss_range, theoretical_biomass, '--', label='Theoretical', color='red', alpha=0.7)
    ax1.set_xlabel('Seed Size')
    ax1.set_ylabel('Mean Final Biomass')
    ax1.set_title('Seed Size vs. Final Biomass')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dispersal Distance vs Seed Size
    ax2 = fig.add_subplot(gs[0, 1])
    dispersal_distances = calculate_dispersal_distances(trajectory[-1])
    ax2.scatter(seed_sizes, dispersal_distances, s=100, label='Observed')
    
    # Add theoretical dispersal relationship
    theoretical_dispersal = 5 * np.exp(-0.3 * ss_range)  # Negative exponential relationship
    ax2.plot(ss_range, theoretical_dispersal, '--', label='Theoretical', color='red', alpha=0.7)
    ax2.set_xlabel('Seed Size')
    ax2.set_ylabel('Mean Dispersal Distance')
    ax2.set_title('Seed Size vs. Dispersal Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Establishment Success vs Seed Size
    ax3 = fig.add_subplot(gs[1, 0])
    establishment = calculate_establishment_success(trajectory)
    ax3.scatter(seed_sizes, establishment, s=100, label='Observed')
    
    # Add theoretical establishment relationship
    theoretical_establishment = 1 / (1 + np.exp(-2 * (ss_range - np.mean(seed_sizes))))  # Logistic function
    ax3.plot(ss_range, theoretical_establishment, '--', label='Theoretical', color='red', alpha=0.7)
    ax3.set_xlabel('Seed Size')
    ax3.set_ylabel('Establishment Success Rate')
    ax3.set_title('Seed Size vs. Establishment Success')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Competitive Ability vs Seed Size
    ax4 = fig.add_subplot(gs[1, 1])
    competition = calculate_competitive_ability(trajectory)
    ax4.scatter(seed_sizes, competition, s=100, label='Observed')
    
    # Add theoretical competition relationship
    theoretical_competition = ss_range**0.5  # Square root relationship
    ax4.plot(ss_range, theoretical_competition, '--', label='Theoretical', color='red', alpha=0.7)
    ax4.set_xlabel('Seed Size')
    ax4.set_ylabel('Competitive Ability Index')
    ax4.set_title('Seed Size vs. Competitive Ability')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar showing temporal evolution
    norm = plt.Normalize(0, len(trajectory))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_subplot(gs[2, :])
    plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Time Steps')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{save_prefix}species_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_mean_dispersal_distances(trajectory):
    """Calculate mean dispersal distances for each species."""
    final_state = trajectory[-1]
    distances = []
    
    for sp in range(final_state.shape[0]):
        # Calculate center of mass
        y, x = np.mgrid[0:final_state.shape[1], 0:final_state.shape[2]]
        total_mass = np.sum(final_state[sp])
        if total_mass > 0:
            center_y = np.sum(y * final_state[sp]) / total_mass
            center_x = np.sum(x * final_state[sp]) / total_mass
            
            # Calculate mean distance from center
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            mean_dist = np.sum(dist * final_state[sp]) / total_mass
            distances.append(mean_dist)
        else:
            distances.append(0)
    
    return np.array(distances)

def plot_dispersal_patterns(trajectory, dispersal_type):
    """Analyze and plot dispersal patterns"""
    config = IBMConfig()  # Get config instance
    plt.figure(figsize=(15, 5))
    
    # Spatial autocorrelation
    plt.subplot(131)
    final_state = trajectory[-1]
    total_biomass = np.sum(final_state, axis=0)
    
    # Calculate Moran's I
    def calculate_morans_i(data):
        n = data.size
        mean = np.mean(data)
        numerator = 0
        denominator = np.sum((data - mean)**2)
        
        for i in range(config.NUM_PATCHES_Y):
            for j in range(config.NUM_PATCHES_X):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < config.NUM_PATCHES_Y and 0 <= nj < config.NUM_PATCHES_X:
                            numerator += (data[i,j] - mean) * (data[ni,nj] - mean)
        
        return numerator / denominator
    
    morans_i = calculate_morans_i(total_biomass)
    plt.text(0.5, 0.5, f"Moran's I: {morans_i:.3f}", 
             horizontalalignment='center', verticalalignment='center')
    plt.title('Spatial Autocorrelation')
    plt.axis('off')
    
    # Distance decay
    plt.subplot(132)
    center = total_biomass[config.NUM_PATCHES_Y//2, config.NUM_PATCHES_X//2]
    distances = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    for i in range(config.NUM_PATCHES_Y):
        for j in range(config.NUM_PATCHES_X):
            distances[i,j] = np.sqrt((i-config.NUM_PATCHES_Y//2)**2 + (j-config.NUM_PATCHES_X//2)**2)
    
    plt.scatter(distances.flatten(), total_biomass.flatten(), alpha=0.5)
    plt.xlabel('Distance from Center')
    plt.ylabel('Biomass')
    plt.title('Distance Decay')
    
    # Patch connectivity
    plt.subplot(133)
    connectivity = np.zeros_like(total_biomass)
    for i in range(config.NUM_PATCHES_Y):
        for j in range(config.NUM_PATCHES_X):
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < config.NUM_PATCHES_Y and 0 <= nj < config.NUM_PATCHES_X:
                        neighbors.append(total_biomass[ni,nj])
            connectivity[i,j] = np.mean(neighbors) if neighbors else 0
    
    plt.imshow(connectivity)
    plt.colorbar(label='Mean Neighbor Biomass')
    plt.title('Patch Connectivity')
    
    plt.suptitle(f'Dispersal Analysis ({dispersal_type})')
    plt.tight_layout()

def plot_seed_size_effects(trajectory, seed_sizes):
    """Plot effects of seed size on dispersal and establishment"""
    config = IBMConfig()  # Get config instance
    plt.figure(figsize=(15, 5))
    
    # Mean biomass by seed size
    plt.subplot(131)
    final_biomass = np.mean(trajectory[-1], axis=(1,2))
    plt.scatter(seed_sizes, final_biomass)
    plt.xlabel('Seed Size')
    plt.ylabel('Mean Biomass')
    plt.title('Biomass vs Seed Size')
    
    # Spatial spread by seed size
    plt.subplot(132)
    spread = np.zeros(len(seed_sizes))
    for s in range(len(seed_sizes)):
        presence = trajectory[-1, s] > config.THRESHOLD
        if np.any(presence):
            y_coords, x_coords = np.where(presence)
            spread[s] = np.sqrt(np.var(y_coords) + np.var(x_coords))
    
    plt.scatter(seed_sizes, spread)
    plt.xlabel('Seed Size')
    plt.ylabel('Spatial Spread')
    plt.title('Dispersal Range vs Seed Size')
    
    # Establishment success
    plt.subplot(133)
    establishment = np.zeros(len(seed_sizes))
    for s in range(len(seed_sizes)):
        establishment[s] = np.sum(trajectory[-1, s] > config.THRESHOLD) / (config.NUM_PATCHES_X * config.NUM_PATCHES_Y)
    
    plt.scatter(seed_sizes, establishment)
    plt.xlabel('Seed Size')
    plt.ylabel('Establishment Rate')
    plt.title('Establishment vs Seed Size')
    
    plt.tight_layout()

def calculate_dispersal_distances(final_state):
    """Calculate mean dispersal distances for each species."""
    distances = []
    center_y, center_x = final_state.shape[1]//2, final_state.shape[2]//2
    
    for sp in range(final_state.shape[0]):
        biomass_grid = final_state[sp]
        y, x = np.meshgrid(np.arange(final_state.shape[1]), np.arange(final_state.shape[2]), indexing='ij')
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        mean_dist = np.average(dist, weights=biomass_grid)
        distances.append(mean_dist)
    
    return np.array(distances)

def calculate_establishment_success(trajectory):
    """Calculate establishment success rate for each species."""
    initial_presence = (trajectory[0] > 0).sum(axis=(1, 2))  # Sum over spatial dimensions
    final_presence = (trajectory[-1] > 0).sum(axis=(1, 2))   # Sum over spatial dimensions
    return final_presence / initial_presence

def calculate_competitive_ability(trajectory):
    """Calculate competitive ability index for each species."""
    initial_biomass = np.mean(trajectory[0], axis=(1, 2))  # Average over spatial dimensions
    final_biomass = np.mean(trajectory[-1], axis=(1, 2))   # Average over spatial dimensions
    return final_biomass / initial_biomass

def calculate_morans_i(values):
    """
    Calculate Moran's I spatial autocorrelation statistic with automatic weights calculation.
    """
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return 0
    
    values = values[valid_mask]
    n = len(values)
    if n < 2:  # Need at least 2 points for correlation
        return 0
    
    # Create weights matrix (nearest neighbors)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(i - j) == 1:  # Adjacent points
                weights[i, j] = 1
    
    # Normalize weights
    row_sums = weights.sum(axis=1)
    mask = row_sums > 0
    weights[mask] = weights[mask] / row_sums[mask, np.newaxis]
        
    mean = np.mean(values)
    z = values - mean
    
    # Handle case where all values are identical
    if np.allclose(z, 0, rtol=1e-10, atol=1e-10):
        return 1.0  # Perfect positive autocorrelation
    
    # Calculate the denominator (variance)
    s0 = np.sum(weights)
    if s0 == 0:  # No valid weights
        return 0
    
    # Calculate Moran's I with numerical stability
    zw = np.sum(weights * np.outer(z, z))
    v = np.sum(z**2)
    
    if v < 1e-10:  # Effectively zero variance
        return 0
        
    I = (n / s0) * (zw / v)
    
    # Bound the result to [-1, 1]
    return np.clip(I, -1.0, 1.0)

def calculate_cooccurrence_matrix(presence_data):
    """
    Calculate species co-occurrence matrix with enhanced variance handling.
    
    Parameters:
    -----------
    presence_data : numpy.ndarray
        Array of shape (n_species, n_patches) containing biomass values
    
    Returns:
    --------
    cooccurrence : numpy.ndarray
        Matrix of correlation coefficients between species
    variance_info : numpy.ndarray
        Array of variance values for each species
    """
    n_species = presence_data.shape[0]
    cooccurrence = np.zeros((n_species, n_species))
    variance_info = np.zeros(n_species)
    
    # Calculate variance for each species
    for i in range(n_species):
        variance_info[i] = np.var(presence_data[i])
    
    # Use a very small variance threshold to detect true zero variance
    variance_threshold = np.mean(variance_info) * 1e-6
    
    for i in range(n_species):
        for j in range(n_species):
            if i == j:
                cooccurrence[i, j] = 1.0
                continue
            
            # Get biomass data for both species
            sp1 = presence_data[i]
            sp2 = presence_data[j]
            
            # Check for sufficient variance
            if variance_info[i] < variance_threshold or variance_info[j] < variance_threshold:
                cooccurrence[i, j] = np.nan
                continue
            
            # Calculate correlation using biomass values
            with np.errstate(invalid='ignore', divide='ignore'):
                correlation = np.corrcoef(sp1, sp2)[0, 1]
            
            if np.isnan(correlation):
                # If correlation fails, try presence/absence correlation
                sp1_binary = (sp1 > 0).astype(float)
                sp2_binary = (sp2 > 0).astype(float)
                if np.var(sp1_binary) > 0 and np.var(sp2_binary) > 0:
                    correlation = np.corrcoef(sp1_binary, sp2_binary)[0, 1]
            
            cooccurrence[i, j] = correlation if not np.isnan(correlation) else 0.0
    
    return cooccurrence, variance_info

def analyze_patch_dynamics(trajectory):
    """
    Analyze patch occupancy dynamics with enhanced metrics.
    """
    config = IBMConfig()  # Get config instance
    n_timesteps, n_species, n_patches_y, n_patches_x = trajectory.shape
    
    # Initialize arrays
    patch_occupancy = np.zeros((n_timesteps, n_species))
    species_richness = np.zeros(n_timesteps)
    turnover_rate = np.zeros(n_timesteps-1)
    
    # Calculate dynamic threshold based on mean biomass
    mean_biomass = np.mean(trajectory, axis=(0,2,3))
    presence_thresholds = np.maximum(config.THRESHOLD, mean_biomass * 0.05)  # Increased sensitivity
    
    for t in range(n_timesteps):
        # Calculate occupancy for each species
        for s in range(n_species):
            biomass = trajectory[t, s]
            # Use both absolute and relative thresholds
            occupied = (biomass > presence_thresholds[s]) & \
                      (biomass > np.mean(biomass) * 0.1)  # Relative threshold
            patch_occupancy[t, s] = np.mean(occupied)
        
        # Calculate species richness (normalized)
        presence_matrix = trajectory[t] > presence_thresholds[:, np.newaxis, np.newaxis]
        species_richness[t] = np.mean(np.sum(presence_matrix, axis=0)) / n_species
        
        # Calculate turnover rate with improved sensitivity
        if t > 0:
            prev_presence = trajectory[t-1] > presence_thresholds[:, np.newaxis, np.newaxis]
            curr_presence = presence_matrix
            
            # Calculate patch-wise turnover
            turnovers = []
            for y in range(n_patches_y):
                for x in range(n_patches_x):
                    prev_species = set(np.where(prev_presence[:, y, x])[0])
                    curr_species = set(np.where(curr_presence[:, y, x])[0])
                    
                    gained = len(curr_species - prev_species)
                    lost = len(prev_species - curr_species)
                    total = len(prev_species | curr_species)
                    
                    if total > 0:
                        turnovers.append((gained + lost) / total)
            
            turnover_rate[t-1] = np.mean(turnovers) if turnovers else 0.0
    
    return patch_occupancy, species_richness, turnover_rate

def plot_patch_dynamics(ax, times, patch_occupancy, species_richness, turnover_rate):
    """Plot patch occupancy dynamics with multiple metrics."""
    # Plot occupancy for each species with distinct colors and styles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors for each species
    styles = ['-', '--', '-.']  # Different line styles
    
    for s in range(patch_occupancy.shape[1]):
        # Add some noise to make temporal changes more visible
        occupancy = patch_occupancy[:, s]
        occupancy_smooth = np.convolve(occupancy, np.ones(5)/5, mode='same')
        
        ax.plot(times, occupancy_smooth, 
                color=colors[s], linestyle=styles[s],
                label=f'Species {s+1}', linewidth=2)
        
        # Add shaded region for variability
        ax.fill_between(times, 
                       np.maximum(0, occupancy_smooth - np.std(occupancy)),
                       np.minimum(1, occupancy_smooth + np.std(occupancy)),
                       color=colors[s], alpha=0.2)
    
    # Plot species richness with enhanced visibility
    richness_smooth = np.convolve(species_richness, np.ones(5)/5, mode='same')
    ax.plot(times, richness_smooth, 
            'k--', label='Species Richness', 
            linewidth=2, alpha=0.7)
    
    # Add turnover rate on secondary axis with enhanced visibility
    ax2 = ax.twinx()
    # Pad turnover rate to match times dimension
    turnover_padded = np.pad(turnover_rate, (0, 1), mode='edge')
    turnover_smooth = np.convolve(turnover_padded, np.ones(5)/5, mode='same')
    ax2.plot(times, turnover_smooth, 
             color='red', linestyle=':', label='Turnover Rate',
             linewidth=2, alpha=0.7)
    ax2.set_ylabel('Turnover Rate', color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Set y-axis limits for better visualization
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    # Customize plot
    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Patch Occupancy / Species Richness', fontsize=10)
    ax.set_title('Patch Occupancy Dynamics', fontsize=12, pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Combine legends with better positioning
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
             loc='center left', bbox_to_anchor=(1.15, 0.5),
             frameon=True, fancybox=True, shadow=True)

def plot_wind_distribution(ax, wind_direction, wind_strength):
    """Plot wind rose diagram showing direction and strength distribution."""
    # Create directional bins
    n_bins = 36
    theta = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    width = 2*np.pi/n_bins
    
    # Create directional distribution using von Mises distribution
    kappa = 8.0  # Concentration parameter
    direction_dist = np.exp(kappa * np.cos(theta - wind_direction))
    direction_dist = direction_dist / direction_dist.max()  # Normalize
    
    # Scale by wind strength and add some variation
    radii = direction_dist * wind_strength
    # Add some random variation to make it more realistic
    radii += np.random.normal(0, wind_strength * 0.1, size=len(radii))
    radii = np.clip(radii, 0, wind_strength)  # Ensure non-negative values
    
    # Create color gradient based on wind strength
    colors = plt.cm.viridis(radii / wind_strength)
    
    # Plot bars
    bars = ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.8)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, wind_strength))
    plt.colorbar(sm, ax=ax, label='Wind Strength')
    
    # Customize plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'Wind Distribution\nDirection: {wind_direction:.1f} rad, Strength: {wind_strength:.1f}')
    
    # Add cardinal directions with enhanced visibility
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for d, a in zip(directions, angles):
        ax.text(a, ax.get_rmax() * 1.2, d, 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)

def plot_dispersal_analysis(trajectory, dispersal_type, save_prefix='', model_type='general'):
    """Comprehensive analysis of dispersal patterns with enhanced visualization."""
    output_dir = ensure_output_dir(model_type)
    config = IBMConfig()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Spatial autocorrelation
    ax1 = fig.add_subplot(gs[0, 0])
    times = np.arange(len(trajectory))
    morans_i = np.zeros((trajectory.shape[0], trajectory.shape[1]))
    
    # Calculate Moran's I for each species over time
    for t in range(trajectory.shape[0]):
        for s in range(trajectory.shape[1]):
            biomass_grid = trajectory[t, s]
            morans_i[t, s] = calculate_morans_i(biomass_grid.flatten())
    
    # Plot Moran's I for each species
    for s in range(trajectory.shape[1]):
        ax1.plot(times, morans_i[:, s], label=f'Species {s+1}')
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel("Moran's I")
    ax1.set_title('Spatial Autocorrelation Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Patch occupancy dynamics
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate patch dynamics with proper thresholds
    mean_biomass = np.mean(trajectory, axis=(0,2,3))
    presence_thresholds = np.maximum(config.THRESHOLD, mean_biomass * 0.01)
    
    patch_occupancy = np.zeros((len(times), trajectory.shape[1]))
    species_richness = np.zeros(len(times))
    turnover_rate = np.zeros(len(times))
    
    for t in range(len(times)):
        # Calculate occupancy
        for s in range(trajectory.shape[1]):
            biomass = trajectory[t, s]
            patch_occupancy[t, s] = np.mean(biomass > presence_thresholds[s])
        
        # Calculate species richness
        species_richness[t] = np.mean(np.sum(trajectory[t] > presence_thresholds[:, np.newaxis, np.newaxis], axis=0))
        
        # Calculate turnover
        if t > 0:
            prev_state = trajectory[t-1] > presence_thresholds[:, np.newaxis, np.newaxis]
            curr_state = trajectory[t] > presence_thresholds[:, np.newaxis, np.newaxis]
            
            changes = np.sum(prev_state != curr_state, axis=0)
            total_species = np.sum(prev_state | curr_state, axis=0)
            turnover_rate[t] = np.mean(changes / np.maximum(total_species, 1))
    
    # Plot patch dynamics
    for s in range(trajectory.shape[1]):
        ax2.plot(times, patch_occupancy[:, s], label=f'Species {s+1}')
    ax2.plot(times, species_richness / trajectory.shape[1], 'k--', label='Relative Richness')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(times, turnover_rate, 'r:', label='Turnover Rate')
    ax2_twin.set_ylabel('Turnover Rate', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Patch Occupancy')
    ax2.set_title('Patch Occupancy Dynamics')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    # 3. Wind strength distribution
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')
    
    # Create directional bins
    n_bins = 36
    theta = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    width = 2*np.pi/n_bins
    
    # Create directional distribution with von Mises distribution
    kappa = 8.0  # Concentration parameter
    direction_dist = np.exp(kappa * np.cos(theta - config.WIND_DIRECTION))
    direction_dist = direction_dist / direction_dist.max()  # Normalize
    radii = direction_dist * config.WIND_STRENGTH
    
    # Add random variation to make it more realistic
    radii += np.random.normal(0, config.WIND_STRENGTH * 0.1, size=len(radii))
    radii = np.clip(radii, 0, config.WIND_STRENGTH)  # Ensure non-negative values
    
    # Plot wind rose
    bars = ax3.bar(theta, radii, width=width, bottom=0.0)
    
    # Color bars by wind strength
    colors = plt.cm.viridis(radii / config.WIND_STRENGTH)
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    # Add cardinal directions
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for d, a in zip(directions, angles):
        ax3.text(a, config.WIND_STRENGTH * 1.2, d, 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    ax3.set_title('Wind Distribution')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(0, config.WIND_STRENGTH))
    plt.colorbar(sm, ax=ax3, label='Wind Strength')
    
    # 4. Species co-occurrence patterns
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate co-occurrence with proper normalization
    final_state = trajectory[-1]
    n_species = final_state.shape[0]
    cooccurrence = np.zeros((n_species, n_species))
    variance_info = np.zeros(n_species)
    
    # Calculate correlations with improved handling of low variance
    for i in range(n_species):
        biomass_i = final_state[i].flatten()
        variance_info[i] = np.var(biomass_i)
        
        for j in range(n_species):
            if i == j:
                cooccurrence[i, j] = 1.0
                continue
                
            biomass_j = final_state[j].flatten()
            
            # Check for sufficient variance
            if variance_info[i] > 1e-10 and np.var(biomass_j) > 1e-10:
                # Normalize the data
                norm_i = (biomass_i - np.mean(biomass_i)) / np.std(biomass_i)
                norm_j = (biomass_j - np.mean(biomass_j)) / np.std(biomass_j)
                cooccurrence[i, j] = np.corrcoef(norm_i, norm_j)[0, 1]
            else:
                cooccurrence[i, j] = np.nan
    
    # Plot co-occurrence matrix
    im = ax4.imshow(cooccurrence, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax4, label='Correlation')
    
    # Add correlation values and variance info
    for i in range(n_species):
        for j in range(n_species):
            if not np.isnan(cooccurrence[i, j]):
                color = 'white' if abs(cooccurrence[i, j]) > 0.5 else 'black'
                text = f'{cooccurrence[i, j]:.2f}'
                if i == j:
                    text += f'\n(var={variance_info[i]:.1e})'
                ax4.text(j, i, text,
                        ha='center', va='center', color=color, fontsize=8)
    
    ax4.set_title('Species Co-occurrence\nwith Variance Information')
    ax4.set_xticks(range(n_species))
    ax4.set_yticks(range(n_species))
    ax4.set_xticklabels([f'Sp {i+1}' for i in range(n_species)])
    ax4.set_yticklabels([f'Sp {i+1}' for i in range(n_species)])
    
    plt.tight_layout()
    if save_prefix:
        plt.savefig(os.path.join(output_dir, f'{save_prefix}_dispersal_analysis.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def plot_state_transitions(trajectory, dispersal_type, save_prefix='', model_type='general'):
    """
    Visualize state transitions with vectorization patterns.
    Shows the flow between different states (S → D → P) with spatial context.
    """
    output_dir = ensure_output_dir(model_type)
    config = IBMConfig()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. State transition heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    final_state = trajectory[-1]
    state_matrix = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    
    # Calculate state transitions
    for y in range(config.NUM_PATCHES_Y):
        for x in range(config.NUM_PATCHES_X):
            biomass = final_state[:, y, x]
            if np.max(biomass) > config.THRESHOLD:
                state_matrix[y, x] = 2  # P state
            elif np.max(biomass) > config.BODY_MASS:
                state_matrix[y, x] = 1  # D state
            else:
                state_matrix[y, x] = 0  # S state
    
    # Plot state distribution
    im1 = ax1.imshow(state_matrix, cmap='RdYlBu')
    plt.colorbar(im1, ax=ax1, label='State (0=S, 1=D, 2=P)')
    ax1.set_title('State Distribution')
    
    # 2. Vector field of state transitions
    ax2 = fig.add_subplot(gs[0, 1])
    y, x = np.mgrid[0:config.NUM_PATCHES_Y:1, 0:config.NUM_PATCHES_X:1]
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    
    # Calculate transition vectors
    for i in range(1, config.NUM_PATCHES_Y-1):
        for j in range(1, config.NUM_PATCHES_X-1):
            # Calculate gradient in state
            dx = state_matrix[i, j+1] - state_matrix[i, j-1]
            dy = state_matrix[i+1, j] - state_matrix[i-1, j]
            u[i, j] = dx
            v[i, j] = dy
    
    # Normalize vectors
    norm = np.sqrt(u**2 + v**2)
    u = np.where(norm > 0, u/norm, 0)
    v = np.where(norm > 0, v/norm, 0)
    
    # Plot vector field
    ax2.quiver(x, y, u, v, norm, cmap='viridis')
    plt.colorbar(ax2.collections[0], ax=ax2, label='Transition Magnitude')
    ax2.set_title('State Transition Vector Field')
    
    # 3. State transition rates over time
    ax3 = fig.add_subplot(gs[1, 0])
    times = np.arange(len(trajectory))
    transition_rates = np.zeros((len(times), 3))  # S→D, D→P, P→S
    
    for t in range(1, len(times)):
        prev_state = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
        curr_state = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
        
        # Calculate states
        for y in range(config.NUM_PATCHES_Y):
            for x in range(config.NUM_PATCHES_X):
                prev_biomass = trajectory[t-1, :, y, x]
                curr_biomass = trajectory[t, :, y, x]
                
                # Previous state
                if np.max(prev_biomass) > config.THRESHOLD:
                    prev_state[y, x] = 2
                elif np.max(prev_biomass) > config.BODY_MASS:
                    prev_state[y, x] = 1
                else:
                    prev_state[y, x] = 0
                
                # Current state
                if np.max(curr_biomass) > config.THRESHOLD:
                    curr_state[y, x] = 2
                elif np.max(curr_biomass) > config.BODY_MASS:
                    curr_state[y, x] = 1
                else:
                    curr_state[y, x] = 0
        
        # Calculate transition rates
        s_to_d = np.sum((prev_state == 0) & (curr_state == 1))
        d_to_p = np.sum((prev_state == 1) & (curr_state == 2))
        p_to_s = np.sum((prev_state == 2) & (curr_state == 0))
        
        transition_rates[t] = [s_to_d, d_to_p, p_to_s]
    
    # Plot transition rates
    ax3.plot(times, transition_rates[:, 0], label='S→D', color='blue')
    ax3.plot(times, transition_rates[:, 1], label='D→P', color='green')
    ax3.plot(times, transition_rates[:, 2], label='P→S', color='red')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Transitions')
    ax3.set_title('State Transition Rates Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. State persistence analysis
    ax4 = fig.add_subplot(gs[1, 1])
    persistence = np.zeros((3, 3))  # Matrix of state persistence
    
    for t in range(1, len(times)):
        prev_state = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
        curr_state = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
        
        # Calculate states
        for y in range(config.NUM_PATCHES_Y):
            for x in range(config.NUM_PATCHES_X):
                prev_biomass = trajectory[t-1, :, y, x]
                curr_biomass = trajectory[t, :, y, x]
                
                # Previous state
                if np.max(prev_biomass) > config.THRESHOLD:
                    prev_state[y, x] = 2
                elif np.max(prev_biomass) > config.BODY_MASS:
                    prev_state[y, x] = 1
                else:
                    prev_state[y, x] = 0
                
                # Current state
                if np.max(curr_biomass) > config.THRESHOLD:
                    curr_state[y, x] = 2
                elif np.max(curr_biomass) > config.BODY_MASS:
                    curr_state[y, x] = 1
                else:
                    curr_state[y, x] = 0
        
        # Update persistence matrix
        for i in range(3):
            for j in range(3):
                persistence[i, j] += np.sum((prev_state == i) & (curr_state == j))
    
    # Normalize persistence matrix
    persistence = persistence / np.sum(persistence)
    
    # Plot persistence matrix
    im4 = ax4.imshow(persistence, cmap='viridis')
    plt.colorbar(im4, ax=ax4, label='Transition Probability')
    
    # Add labels
    states = ['S', 'D', 'P']
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels(states)
    ax4.set_yticklabels(states)
    
    # Add values to cells
    for i in range(3):
        for j in range(3):
            text = ax4.text(j, i, f'{persistence[i, j]:.2f}',
                          ha='center', va='center', color='white')
    
    ax4.set_title('State Persistence Matrix')
    
    # 5. Spatial autocorrelation of states
    ax5 = fig.add_subplot(gs[2, 0])
    distances = np.arange(1, 6)
    autocorr = np.zeros((3, len(distances)))
    
    for state in range(3):
        state_mask = state_matrix == state
        for i, d in enumerate(distances):
            autocorr[state, i] = calculate_distance_correlation(state_mask, d)
    
    for state in range(3):
        ax5.plot(distances, autocorr[state], label=f'State {states[state]}', marker='o')
    
    ax5.set_xlabel('Distance')
    ax5.set_ylabel('Spatial Autocorrelation')
    ax5.set_title('State Spatial Autocorrelation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. State transition network
    ax6 = fig.add_subplot(gs[2, 1])
    G = nx.DiGraph()
    G.add_nodes_from(states)
    
    # Add edges with weights from persistence matrix
    for i in range(3):
        for j in range(3):
            if i != j and persistence[i, j] > 0:
                G.add_edge(states[i], states[j], weight=persistence[i, j])
    
    # Draw network
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    ax6.set_title('State Transition Network')
    ax6.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{save_prefix}state_transitions_{dispersal_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_vectorization_patterns(trajectory, dispersal_type, save_prefix='', model_type='general'):
    """
    Visualize vectorization patterns in the state transitions.
    Shows how biomass flows between patches and states.
    """
    output_dir = ensure_output_dir(model_type)
    config = IBMConfig()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Biomass flow vectors
    ax1 = fig.add_subplot(gs[0, 0])
    final_state = trajectory[-1]
    y, x = np.mgrid[0:config.NUM_PATCHES_Y:1, 0:config.NUM_PATCHES_X:1]
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    
    # Calculate biomass flow vectors
    for i in range(1, config.NUM_PATCHES_Y-1):
        for j in range(1, config.NUM_PATCHES_X-1):
            # Calculate biomass gradient
            dx = np.sum(final_state[:, i, j+1]) - np.sum(final_state[:, i, j-1])
            dy = np.sum(final_state[:, i+1, j]) - np.sum(final_state[:, i-1, j])
            u[i, j] = dx
            v[i, j] = dy
    
    # Normalize vectors
    norm = np.sqrt(u**2 + v**2)
    u = np.where(norm > 0, u/norm, 0)
    v = np.where(norm > 0, v/norm, 0)
    
    # Plot vector field with biomass magnitude
    biomass_magnitude = np.sum(final_state, axis=0)
    im1 = ax1.quiver(x, y, u, v, biomass_magnitude, cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Total Biomass')
    ax1.set_title('Biomass Flow Vectors')
    
    # 2. State transition probability field
    ax2 = fig.add_subplot(gs[0, 1])
    transition_prob = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    
    # Calculate transition probabilities
    for y in range(config.NUM_PATCHES_Y):
        for x in range(config.NUM_PATCHES_X):
            biomass = final_state[:, y, x]
            if np.max(biomass) > config.THRESHOLD:
                transition_prob[y, x] = 1.0
            elif np.max(biomass) > config.BODY_MASS:
                transition_prob[y, x] = 0.5
            else:
                transition_prob[y, x] = 0.0
    
    # Plot transition probability field
    im2 = ax2.imshow(transition_prob, cmap='RdYlBu')
    plt.colorbar(im2, ax=ax2, label='Transition Probability')
    ax2.set_title('State Transition Probability Field')
    
    # 3. Species-specific vectorization
    ax3 = fig.add_subplot(gs[1, 0])
    species_vectors = np.zeros((final_state.shape[0], config.NUM_PATCHES_Y, config.NUM_PATCHES_X, 2))
    
    # Calculate species-specific vectors
    for s in range(final_state.shape[0]):
        for i in range(1, config.NUM_PATCHES_Y-1):
            for j in range(1, config.NUM_PATCHES_X-1):
                dx = final_state[s, i, j+1] - final_state[s, i, j-1]
                dy = final_state[s, i+1, j] - final_state[s, i-1, j]
                species_vectors[s, i, j] = [dx, dy]
    
    # Plot species vectors
    colors = ['red', 'blue', 'green']
    x, y = np.meshgrid(np.arange(config.NUM_PATCHES_X), np.arange(config.NUM_PATCHES_Y))
    for s in range(final_state.shape[0]):
        norm = np.sqrt(np.sum(species_vectors[s]**2, axis=2))
        u = np.where(norm > 0, species_vectors[s, :, :, 0]/norm, 0)
        v = np.where(norm > 0, species_vectors[s, :, :, 1]/norm, 0)
        # Only plot vectors where norm is greater than 0
        mask = norm > 0
        ax3.quiver(x[mask], y[mask], u[mask], v[mask], norm[mask], 
                  color=colors[s], alpha=0.5, label=f'Species {s+1}')
    
    ax3.set_title('Species-Specific Vectorization')
    ax3.legend()
    
    # 4. Vectorization strength analysis
    ax4 = fig.add_subplot(gs[1, 1])
    vector_strength = np.zeros((config.NUM_PATCHES_Y, config.NUM_PATCHES_X))
    
    # Calculate vectorization strength
    for i in range(1, config.NUM_PATCHES_Y-1):
        for j in range(1, config.NUM_PATCHES_X-1):
            # Calculate divergence
            dx = np.sum(final_state[:, i, j+1]) - np.sum(final_state[:, i, j-1])
            dy = np.sum(final_state[:, i+1, j]) - np.sum(final_state[:, i-1, j])
            vector_strength[i, j] = np.sqrt(dx**2 + dy**2)
    
    # Plot vectorization strength
    im4 = ax4.imshow(vector_strength, cmap='viridis')
    plt.colorbar(im4, ax=ax4, label='Vectorization Strength')
    ax4.set_title('Vectorization Strength Field')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{save_prefix}vectorization_patterns_{dispersal_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_analyses(trajectory, dispersal_type, wind_direction, wind_strength, habitat_quality, seed_sizes, save_prefix=''):
    """Generate all analysis plots including new state transition visualizations."""
    # Existing plots
    plot_temporal_evolution(trajectory, dispersal_type, save_prefix)
    plot_spatial_patterns(trajectory, dispersal_type, wind_direction, habitat_quality, save_prefix)
    plot_dispersal_analysis(trajectory, dispersal_type, save_prefix)
    plot_environmental_effects(trajectory, wind_direction, wind_strength, habitat_quality, save_prefix)
    plot_species_comparison(trajectory, seed_sizes, save_prefix)
    
    # New state transition visualizations
    plot_state_transitions(trajectory, dispersal_type, save_prefix)
    plot_vectorization_patterns(trajectory, dispersal_type, save_prefix)

def plot_cooccurrence_matrix(ax, cooccurrence, variance_info):
    """
    Plot species co-occurrence matrix with enhanced visualization.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    cooccurrence : numpy.ndarray
        Square matrix of species co-occurrence correlations
    variance_info : numpy.ndarray
        Array of variance information for each species
    """
    # Create masked array for better visualization
    masked_cooccurrence = np.ma.masked_invalid(cooccurrence)
    
    # Plot co-occurrence heatmap
    im = ax.imshow(masked_cooccurrence, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add species labels with variance info
    species_labels = [f'Sp {i+1}\n(var={variance_info[i]:.2e})' 
                     for i in range(len(variance_info))]
    ax.set_xticks(np.arange(len(species_labels)))
    ax.set_yticks(np.arange(len(species_labels)))
    ax.set_xticklabels(species_labels, rotation=45, ha='right')
    ax.set_yticklabels(species_labels)
    
    # Add correlation values
    for i in range(len(species_labels)):
        for j in range(len(species_labels)):
            if not np.ma.is_masked(masked_cooccurrence[i, j]):
                # Choose text color based on background
                color = 'white' if abs(masked_cooccurrence[i, j]) > 0.5 else 'black'
                text = ax.text(j, i, f'{masked_cooccurrence[i, j]:.2f}',
                             ha='center', va='center', color=color)
            else:
                ax.text(j, i, 'insuf.\nvar', ha='center', va='center',
                       color='gray', fontsize=8, style='italic')
    
    ax.set_title('Species Co-occurrence Patterns\nwith Variance Information') 