############################################
# analysis_extended.py
############################################
"""
Advanced/Interactive Visualizations for Verifying the Real PSD Turnover Dynamics Model

This module loads the saved output data (model_outputs.npz) produced by main.py
and creates a suite of interactive visualizations using Plotly. These include:
  1. An interactive multi-model total biomass time series.
  2. An animated 2D scatter plot of species biomass vs. time.
  3. An interactive 3D scatter plot of species biomass vs. time.
  4. An interactive histogram (with a marginal box plot) of the final biomass distribution.
  5. Interactive line plots for alpha diversity and turnover rate over time.
  6. An interactive heatmap of the covariance matrix of species biomass.
  7. An interactive box plot comparing final biomass distributions across models.
  8. Additional interactive visualizations for PSD2 diagnostics:
      - Poisson Clock trajectory (exponentiated from log scale).
      - Growth Rate trajectory.
      - Invasion Rate trajectory.
      - Establishment Probability trajectory.
  9. An interactive heatmap for state transitions.
  
All output files are saved in the "Advanced_vis" folder.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging

from config import THRESHOLD, BODY_MASS, RECORDING_STEP_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create output directory for visualizations
OUTPUT_FOLDER = "Advanced_vis"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#############################
# Data Loading and Computation
#############################

def load_model_data(filename="model_outputs.npz"):
    """
    Load the model trajectories for IBM, PSD, PSD2, and ODE from a .npz file.
    Returns a dictionary with keys: 'IBM', 'PSD', 'PSD2', 'PSD2_waiting',
    'psd2_poisson_clock', 'psd2_growth_rate', 'psd2_invasion_rate', 'psd2_est_prob', 'ODE'.
    """
    if not os.path.exists(filename):
        logger.error(f"File {filename} not found. Run main.py first to generate outputs.")
        raise FileNotFoundError(f"{filename} not found.")
    data = np.load(filename)
    logger.info(f"Loaded model outputs from {filename}.")
    return {
        "IBM": data["IBM"],
        "PSD": data["PSD"],
        "PSD2": data["PSD2"],
        "PSD2_waiting": data["PSD2_waiting"],
        "PSD2_poisson_clock": data["PSD2_poisson_clock"],
        "PSD2_growth_rate": data["PSD2_growth_rate"],
        "PSD2_invasion_rate": data["PSD2_invasion_rate"],
        "PSD2_est_prob": data["PSD2_est_prob"],
        "ODE": data["ODE"]
    }

def compute_alpha_diversity(trajectory, threshold=THRESHOLD):
    return np.sum(trajectory > threshold, axis=1)

def compute_turnover_rate(trajectory, recording_step):
    presence = (trajectory > THRESHOLD).astype(int)
    diffs = np.abs(np.diff(presence, axis=0))
    changes = np.sum(diffs, axis=1)
    return changes / float(recording_step)

#############################
# Interactive Visualizations (Plotly)
#############################

def interactive_total_biomass(model_data):
    fig = go.Figure()
    for model_name, traj in model_data.items():
        # Skip diagnostic arrays that are not direct trajectories
        if model_name in ["PSD2_waiting", "PSD2_poisson_clock", "PSD2_growth_rate", "PSD2_invasion_rate", "PSD2_est_prob"]:
            continue
        total_biomass = np.sum(traj, axis=1)
        fig.add_trace(go.Scatter(
            x=np.arange(len(total_biomass)),
            y=total_biomass,
            mode='lines',
            name=model_name
        ))
    fig.update_layout(
        title="Interactive: Total Biomass Over Time (All Models)",
        xaxis_title="Time Index",
        yaxis_title="Total Biomass"
    )
    output_file = os.path.join(OUTPUT_FOLDER, "Interactive_Total_Biomass_All_Models.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive total biomass plot as {output_file}.")

def animated_species_scatter(model_data, model_name="PSD2"):
    traj = model_data[model_name]
    n_times, n_species = traj.shape
    records = []
    for t in range(n_times):
        for s in range(n_species):
            records.append({"time": t, "species_id": s, "biomass": traj[t, s]})
    df = pd.DataFrame(records)
    fig = px.scatter(
        df, x="species_id", y="biomass",
        animation_frame="time",
        range_y=[0, df["biomass"].max()*1.1],
        title=f"Animated Scatter of Species Biomass ({model_name})",
        labels={"species_id": "Species ID", "biomass": "Biomass"}
    )
    output_file = os.path.join(OUTPUT_FOLDER, f"Animated_Species_Scatter_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved animated scatter plot for {model_name} as {output_file}.")

def interactive_3D_species_plot(model_data, model_name="PSD2"):
    traj = model_data[model_name]
    n_times, n_species = traj.shape
    records = []
    for t in range(n_times):
        for s in range(n_species):
            records.append({"time": t, "species_id": s, "biomass": traj[t, s]})
    df = pd.DataFrame(records)
    fig = px.scatter_3d(
        df, x="time", y="species_id", z="biomass",
        title=f"3D Scatter Plot of Species Biomass ({model_name})",
        labels={"time": "Time", "species_id": "Species ID", "biomass": "Biomass"}
    )
    output_file = os.path.join(OUTPUT_FOLDER, f"3D_Species_Scatter_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive 3D species scatter plot for {model_name} as {output_file}.")

def interactive_histogram(model_data, model_name="PSD2"):
    traj = model_data[model_name]
    final_biomass = traj[-1, :]
    final_biomass = final_biomass[final_biomass > 0]
    if len(final_biomass) == 0:
        logger.warning(f"No positive biomass values for {model_name} to histogram.")
        return
    df = pd.DataFrame({"log10_biomass": np.log10(final_biomass)})
    fig = px.histogram(
        df, x="log10_biomass", nbins=50,
        title=f"Interactive Histogram of Final Biomass ({model_name})",
        labels={"log10_biomass": "log10(Biomass)"},
        marginal="box"
    )
    fig.update_layout(bargap=0.1)
    output_file = os.path.join(OUTPUT_FOLDER, f"Interactive_Histogram_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive histogram for {model_name} as {output_file}.")

def interactive_alpha_diversity(model_data, model_name="PSD2"):
    traj = model_data[model_name]
    alpha = compute_alpha_diversity(traj)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(alpha)),
        y=alpha,
        mode='lines+markers',
        name="Alpha Diversity"
    ))
    fig.update_layout(
        title=f"Interactive Alpha Diversity Over Time ({model_name})",
        xaxis_title="Time Index",
        yaxis_title="Alpha Diversity (Species Count)"
    )
    output_file = os.path.join(OUTPUT_FOLDER, f"Interactive_Alpha_Diversity_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive alpha diversity plot for {model_name} as {output_file}.")

def interactive_turnover_rate(model_data, record_step, model_name="PSD2"):
    traj = model_data[model_name]
    turnover = compute_turnover_rate(traj, record_step)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(turnover)),
        y=turnover,
        mode='lines+markers',
        name="Turnover Rate"
    ))
    fig.update_layout(
        title=f"Interactive Turnover Rate Over Time ({model_name})",
        xaxis_title="Time Index",
        yaxis_title="Turnover Rate (Changes per Time Unit)"
    )
    output_file = os.path.join(OUTPUT_FOLDER, f"Interactive_Turnover_Rate_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive turnover rate plot for {model_name} as {output_file}.")

def interactive_covariance_heatmap(model_data, model_name="PSD2"):
    traj = model_data[model_name]
    half = traj.shape[0] // 2
    portion = traj[half:, :]
    covmat = np.cov(portion.T)
    fig = px.imshow(
        covmat,
        labels={"x": "Species", "y": "Species", "color": "Covariance"},
        x=np.arange(covmat.shape[1]),
        y=np.arange(covmat.shape[0]),
        title=f"Interactive Covariance Matrix ({model_name})",
        color_continuous_scale='viridis'
    )
    output_file = os.path.join(OUTPUT_FOLDER, f"Interactive_Covariance_Heatmap_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive covariance heatmap for {model_name} as {output_file}.")

def interactive_final_boxplot(model_data):
    records = []
    for model_name, traj in model_data.items():
        if model_name in ["PSD2_waiting", "PSD2_poisson_clock", "PSD2_growth_rate", "PSD2_invasion_rate", "PSD2_est_prob"]:
            continue
        final_biomass = traj[-1, :]
        final_biomass = final_biomass[final_biomass > 0]
        if len(final_biomass) > 0:
            records.append(pd.DataFrame({
                "log10_biomass": np.log10(final_biomass),
                "Model": model_name
            }))
    if records:
        df = pd.concat(records, axis=0)
        fig = px.box(
            df, x="Model", y="log10_biomass",
            title="Final Biomass Distribution by Model (log10 scale)",
            labels={"log10_biomass": "log10(Biomass)"}
        )
        output_file = os.path.join(OUTPUT_FOLDER, "Interactive_Final_Biomass_Boxplot.html")
        fig.write_html(output_file)
        fig.show()
        logger.info(f"Saved interactive final biomass box plot as {output_file}.")
    else:
        logger.warning("No valid final biomass data available for box plot.")

#############################
# Additional Visualizations for PSD2 Diagnostics
#############################

def interactive_poisson_clock(model_data, model_name="PSD2_poisson_clock"):
    # PSD2_poisson_clock is stored in log scale. Exponentiate to obtain the raw values.
    traj = model_data[model_name]
    raw_clock = np.exp(traj)
    fig = go.Figure()
    for species in range(raw_clock.shape[1]):
        fig.add_trace(go.Scatter(
            x=np.arange(raw_clock.shape[0]),
            y=raw_clock[:, species],
            mode='lines',
            name=f"Species {species}"
        ))
    fig.update_layout(
        title="Interactive Poisson Clock Trajectory (Exponentiated)",
        xaxis_title="Time Record Index",
        yaxis_title="Poisson Clock Value"
    )
    output_file = os.path.join(OUTPUT_FOLDER, "Interactive_Poisson_Clock.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive poisson clock plot as {output_file}.")

def interactive_growth_rate(model_data, model_name="PSD2_growth_rate"):
    traj = model_data[model_name]
    fig = go.Figure()
    for species in range(traj.shape[1]):
        fig.add_trace(go.Scatter(
            x=np.arange(traj.shape[0]),
            y=traj[:, species],
            mode='lines',
            name=f"Species {species}"
        ))
    fig.update_layout(
        title="Interactive Growth Rate Trajectory",
        xaxis_title="Time Record Index",
        yaxis_title="Growth Rate"
    )
    output_file = os.path.join(OUTPUT_FOLDER, "Interactive_Growth_Rate.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive growth rate plot as {output_file}.")

def interactive_invasion_rate(model_data, model_name="PSD2_invasion_rate"):
    traj = model_data[model_name]
    fig = go.Figure()
    for species in range(traj.shape[1]):
        fig.add_trace(go.Scatter(
            x=np.arange(traj.shape[0]),
            y=traj[:, species],
            mode='lines',
            name=f"Species {species}"
        ))
    fig.update_layout(
        title="Interactive Invasion Rate Trajectory",
        xaxis_title="Time Record Index",
        yaxis_title="Invasion Rate"
    )
    output_file = os.path.join(OUTPUT_FOLDER, "Interactive_Invasion_Rate.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive invasion rate plot as {output_file}.")

def interactive_est_prob(model_data, model_name="PSD2_est_prob"):
    traj = model_data[model_name]
    fig = go.Figure()
    for species in range(traj.shape[1]):
        fig.add_trace(go.Scatter(
            x=np.arange(traj.shape[0]),
            y=traj[:, species],
            mode='lines',
            name=f"Species {species}"
        ))
    fig.update_layout(
        title="Interactive Establishment Probability Trajectory",
        xaxis_title="Time Record Index",
        yaxis_title="Establishment Probability"
    )
    output_file = os.path.join(OUTPUT_FOLDER, "Interactive_Est_Prob.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive establishment probability plot as {output_file}.")

#############################
# Visualization of State Transitions
#############################

def interactive_state_transitions(model_data, model_name="PSD2"):
    if "PSD2_waiting" not in model_data:
        logger.error("PSD2 waiting state data not found in model outputs.")
        return
    waiting_data = model_data["PSD2_waiting"].astype(int)  # convert boolean to 0/1
    fig = px.imshow(
        waiting_data.T,
        labels={"x": "Time Record Index", "y": "Species ID", "color": "State (0=established, 1=waiting)"},
        title=f"Interactive State Transitions ({model_name})",
        color_continuous_scale=["green", "red"]
    )
    output_file = os.path.join(OUTPUT_FOLDER, f"Interactive_State_Transitions_{model_name}.html")
    fig.write_html(output_file)
    fig.show()
    logger.info(f"Saved interactive state transitions plot for {model_name} as {output_file}.")

#############################
# Helper to Save Model Output
#############################

def save_model_output(ibm_trajectory, psd_trajectory, psd2_trajectory, psd2_waiting,
                      psd2_poisson_clock, psd2_growth_rate, psd2_invasion_rate, psd2_est_prob,
                      ode_trajectory, filename="model_outputs.npz"):
    np.savez(filename,
             IBM=ibm_trajectory,
             PSD=psd_trajectory,
             PSD2=psd2_trajectory,
             PSD2_waiting=psd2_waiting,
             PSD2_poisson_clock=psd2_poisson_clock,
             PSD2_growth_rate=psd2_growth_rate,
             PSD2_invasion_rate=psd2_invasion_rate,
             PSD2_est_prob=psd2_est_prob,
             ODE=ode_trajectory)
    logger.info(f"Saved model outputs to {filename}.")

#############################
# Main Function
#############################

def main():
    model_data = load_model_data("model_outputs.npz")
    interactive_total_biomass(model_data)
    animated_species_scatter(model_data, model_name="PSD2")
    interactive_3D_species_plot(model_data, model_name="PSD2")
    interactive_histogram(model_data, model_name="PSD2")
    interactive_alpha_diversity(model_data, model_name="PSD2")
    interactive_turnover_rate(model_data, RECORDING_STEP_SIZE, model_name="PSD2")
    interactive_covariance_heatmap(model_data, model_name="PSD2")
    interactive_final_boxplot(model_data)
    interactive_state_transitions(model_data, model_name="PSD2")
    # Additional PSD2 diagnostics visualizations
    interactive_poisson_clock(model_data, model_name="PSD2_poisson_clock")
    interactive_growth_rate(model_data, model_name="PSD2_growth_rate")
    interactive_invasion_rate(model_data, model_name="PSD2_invasion_rate")
    interactive_est_prob(model_data, model_name="PSD2_est_prob")

if __name__ == "__main__":
    main()
