#!/usr/bin/env python3

"""
psd_visualization.py

A script that loads the PSD-based trajectory data from `trajectory_data_PSD.npz`
and generates comprehensive, detailed plots verifying alignment with the R-based PSD scheme.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd

def load_psd_data(file_path="trajectory_data_PSD.npz"):
    """
    Load PSD trajectory data from the NPZ file.
    Returns a dictionary of arrays:
      'iteration': shape (num_records,)
      'producers': shape (num_records,)
      'consumers': shape (num_records,)
      'xMat': shape (num_records, max_species, nodes)
      'rMat': shape (num_records, max_species, nodes)
      'PSD_states': shape (num_records, max_species)
      'PoissonClocks': shape (num_records, max_species)
      'logB': shape (num_records, max_species)
      'establishment_prob': shape (num_records, max_species)
      'i': shape (num_records, max_species)
    """
    data = np.load(file_path, allow_pickle=True)
    return data

def plot_time_series(data, outdir="."):
    """
    Creates a multi-panel figure showing Producer & Consumer counts, 
    total species, cumulative invasions, etc.
    Saves the figure as `psd_time_series.html`.
    """

    # Extract arrays
    iterations = data["iteration"]  # shape (num_records,)
    producers = data["producers"]   # shape (num_records,)
    consumers = data["consumers"]   # shape (num_records,)

    # x and y have matching lengths
    if len(iterations) != len(producers) or len(iterations) != len(consumers):
        print(f"Error: iteration length {len(iterations)} does not match producers or consumers length.")
        return

    #define total species as producers + consumers
    total_species = producers + consumers

    # For 'cumulative invasions' (example usage):
    producer_invasion = np.diff(producers, prepend=producers[0])
    consumer_invasion = np.diff(consumers, prepend=consumers[0])
    cum_producer_inv = np.cumsum(np.clip(producer_invasion, 0, None))
    cum_consumer_inv = np.cumsum(np.clip(consumer_invasion, 0, None))

    # Build a subplot figure with 4 panels
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            "PSD Number of Producers and Consumers Over Iterations",
            "PSD Total Species Count Over Iterations",
            "PSD Cumulative Producer and Consumer Invasions",
            "PSD Producers vs. Consumers Over Iterations"
        )
    )

    # producers vs. consumers
    fig.add_trace(
        go.Scatter(x=iterations, y=producers, mode='lines+markers', 
                   name='Producers', line_color='green'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=consumers, mode='lines+markers',
                   name='Consumers', line_color='red'),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Iterations", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # total species
    fig.add_trace(
        go.Scatter(x=iterations, y=total_species, mode='lines+markers',
                   name='Total Species', line_color='blue'),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Iterations", row=1, col=2)
    fig.update_yaxes(title_text="Total Species", row=1, col=2)

    #cumulative invasions
    fig.add_trace(
        go.Scatter(x=iterations, y=cum_producer_inv, mode='lines',
                   name='Cumulative Producers Invaded', line_color='brown'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=cum_consumer_inv, mode='lines',
                   name='Cumulative Consumers Invaded', line_color='darkred'),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Iterations", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Invasions", row=2, col=1)

    # producers vs consumers (scatter)
    fig.add_trace(
        go.Scatter(x=producers, y=consumers, mode='markers+lines',
                   name='Producers vs. Consumers', marker_color='purple'),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Producers", row=2, col=2)
    fig.update_yaxes(title_text="Consumers", row=2, col=2)

    fig.update_layout(
        title="PSD Metacommunity Simulation Time Series",
        legend_title="Legend",
        width=1300,
        height=900,
        template="plotly_dark"
    )

    outpath = os.path.join(outdir, "psd_time_series.html")
    fig.write_html(outpath)
    print(f"Saved time-series figure to {outpath}")


def plot_psd_metrics(data, outdir="."):
    """
    Creates a multi-panel figure focusing on the PSD scheme metrics:
      - proportion of species in waiting states
      - average Poisson clocks
      - average logB
      - average establishment probability
      - average invasion rate i (multi-line per species or single average)
    Saves to `psd_metrics.html`.
    """

    iterations     = data["iteration"]         # shape (num_records,)
    PSD_states     = data["PSD_states"]        # shape (num_records, max_species, [nodes]?)
    PoissonClocks  = data["PoissonClocks"]     # shape (num_records, max_species, [nodes]?)
    logB           = data["logB"]              # shape (num_records, max_species, [nodes]?)
    est_prob       = data["establishment_prob"]# shape (num_records, max_species, [nodes]?)
    inv_rate       = data["i"]                 # shape (num_records, max_species, [nodes]?)

    num_records = len(iterations)

    def check_shape(name, arr):
        if arr.ndim < 2:
            print(f"Warning: {name} has ndim={arr.ndim}, expected >=2D.")
        if arr.shape[0] != num_records:
            print(f"Error: Mismatch in record counts for {name}. Skipping PSD metrics.")
            return False
        return True

    # Check each array
    if not all([
        check_shape("PSD_states", PSD_states),
        check_shape("PoissonClocks", PoissonClocks),
        check_shape("logB", logB),
        check_shape("establishment_prob", est_prob),
        check_shape("inv_rate", inv_rate),
    ]):
        return  

    def ensure_2d(arr, arr_name):
        if arr.ndim == 3:
            print(f"{arr_name} is 3D (shape={arr.shape}). Averaging across axis=2 to get (num_records, max_species).")
            arr = arr.mean(axis=2)  # shape => (num_records, max_species)
        elif arr.ndim == 2:
            pass  #  2D
        else:
            print(f"Warning: {arr_name} has unexpected ndim={arr.ndim}. Attempting to flatten extra dims.")
            # .mean(axis=tuple(range(2, arr.ndim))) if >3D

            arr = arr.reshape(arr.shape[0], -1)  # force 2D
        return arr

    PSD_states    = ensure_2d(PSD_states, "PSD_states")
    PoissonClocks = ensure_2d(PoissonClocks, "PoissonClocks")
    logB          = ensure_2d(logB, "logB")
    est_prob      = ensure_2d(est_prob, "establishment_prob")
    inv_rate      = ensure_2d(inv_rate, "inv_rate")


    # proportion of species in state=1
    prop_waiting = np.mean(PSD_states == 1, axis=1)     # shape (num_records,)
    avg_poisson  = np.mean(PoissonClocks, axis=1)
    avg_logB     = np.mean(logB, axis=1)
    avg_est_prob = np.mean(est_prob, axis=1)
    avg_inv_rate = np.mean(inv_rate, axis=1)

    # subplot figure for the first 4 metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Proportion of Species in Waiting States",
            "Average Poisson Clocks Over Iterations",
            "Average logB Over Iterations",
            "Average Establishment Probability Over Iterations"
        )
    )

    #proportion waiting
    fig.add_trace(
        go.Scatter(x=iterations, y=prop_waiting, mode='lines+markers',
                   name='Proportion Waiting', marker_color='purple'),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Iterations", row=1, col=1)
    fig.update_yaxes(title_text="Proportion Waiting", row=1, col=1)

    #average Poisson clock
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_poisson, mode='lines+markers',
                   name='Average Poisson Clock', marker_color='orange'),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Iterations", row=1, col=2)
    fig.update_yaxes(title_text="Average Poisson Clock", row=1, col=2)

    #average logB
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_logB, mode='lines+markers',
                   name='Average logB', marker_color='teal'),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Iterations", row=2, col=1)
    fig.update_yaxes(title_text="Average logB", row=2, col=1)

    #average establishment prob
    fig.add_trace(
        go.Scatter(x=iterations, y=avg_est_prob, mode='lines+markers',
                   name='Average Est. Prob.', marker_color='red'),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Iterations", row=2, col=2)
    fig.update_yaxes(title_text="Average Establishment Probability", row=2, col=2)

    fig.update_layout(
        title="PSD Metrics Visualization",
        width=1300,
        height=900,
        template="plotly_dark"
    )
    outpath = os.path.join(outdir, "psd_metrics.html")
    fig.write_html(outpath)
    print(f"Saved PSD metrics figure to {outpath}")

    df_inv = pd.DataFrame(inv_rate, columns=[f"species_{i}" for i in range(inv_rate.shape[1])])
    df_inv["iteration"] = iterations  # shape (num_records,)

    fig2 = px.line(
        df_inv,
        x="iteration",
        y=df_inv.columns[:-1],  # all species_{i} columns, skipping 'iteration'
        labels={"value": "Invasion Rate", "variable": "Species"},
        title="PSD Invasion Rate Per Species Over Iterations"
    )
    fig2.update_layout(template="plotly_dark", width=1000, height=600)
    outpath2 = os.path.join(outdir, "invasion_rate_per_species.html")
    fig2.write_html(outpath2)
    print(f"Saved multi-line invasion rate figure to {outpath2}")



def plot_psd_state_distribution(data, outdir="."):
    """
    Shows how many species are in PSD state=0,1,2 at each iteration,
    e.g. stacked lines or separate lines.
    """
    iterations = data["iteration"]
    PSD_states = data["PSD_states"]  # shape (num_records, max_species)

    if PSD_states.shape[0] != len(iterations):
        print("Error: PSD_states mismatch with iteration length.")
        return

    # For each iteration, count how many are in state 0,1,2
    state0 = np.sum(PSD_states==0, axis=1)
    state1 = np.sum(PSD_states==1, axis=1)
    state2 = np.sum(PSD_states==2, axis=1)
    total = state0 + state1 + state2

    # stacked area plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations, y=state0, 
            mode='lines', stackgroup='one',
            name='Deterministic (State=0)',
            line_color='green'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations, y=state1,
            mode='lines', stackgroup='one',
            name='Stochastic (State=1)',
            line_color='orange'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iterations, y=state2,
            mode='lines', stackgroup='one',
            name='Probabilistic (State=2)',
            line_color='red'
        )
    )
    fig.update_layout(
        title="PSD State Distribution Over Iterations (Stacked)",
        xaxis_title="Iterations",
        yaxis_title="Number of Species",
        template="plotly_dark",
        width=1100,
        height=700
    )
    outpath = os.path.join(outdir, "psd_state_distribution.html")
    fig.write_html(outpath)
    print(f"Saved PSD state distribution figure to {outpath}")


def plot_biomass_distribution(data, outdir=".", snapshot_iterations=None):
    """
    Visualize the distribution of species biomass at certain iteration snapshots
    using hist/violin/box plots of logB or xMat sums.
    """
    xMat = data['xMat']       # shape (num_records, max_species, nodes)
    logB = data['logB']       # shape (num_records, max_species)
    iterations = data['iteration']
    num_records = iterations.shape[0]

    if snapshot_iterations is None:
        #  pick first, middle, last
        snapshot_iterations = [0, num_records//2, num_records-1]
    # clamp them
    snapshot_iterations = [i for i in snapshot_iterations if 0 <= i < num_records]

    # figure with subplots
    fig = make_subplots(
        rows=1, cols=len(snapshot_iterations),
        subplot_titles=[f"Iteration {iterations[i]}" for i in snapshot_iterations]
    )

    for idx, snapshot_idx in enumerate(snapshot_iterations):
        snapshot_x = xMat[snapshot_idx,:,:]  # shape (max_species, nodes)
        species_biomass = np.sum(snapshot_x, axis=1)  # shape (max_species,)
        species_logB = logB[snapshot_idx,:]           # shape (max_species,)

        # histogram of logB
        valid_logB = species_logB[np.isfinite(species_logB)]
        valid_logB = valid_logB[valid_logB > -1e10]

        histogram = go.Histogram(
            x=valid_logB,
            name=f"Iter {iterations[snapshot_idx]}",
            marker_color='blue',
            opacity=0.7,
            showlegend=(True if idx==0 else False)
        )
        fig.add_trace(histogram, row=1, col=idx+1)
        fig.update_xaxes(title_text="log(Biomass)", row=1, col=idx+1)
        fig.update_yaxes(title_text="Count", row=1, col=idx+1)

    fig.update_layout(
        title="PSD Distribution of log-Biomass Across Species (Snapshots)",
        template="plotly_dark",
        width=400*len(snapshot_iterations),
        height=600
    )
    outpath = os.path.join(outdir, "psd_biomass_distribution.html")
    fig.write_html(outpath)
    print(f"Saved biomass distribution figure to {outpath}")


def plot_spatial_node_biomass(data, outdir=".", iteration_index=-1):
    """
    Optional: if you want to see total biomass across nodes for one iteration,
    we do a simple bar chart. A heatmap would require node geometry.
    """
    xMat = data["xMat"]  # shape (num_records, max_species, nodes)
    iterations = data["iteration"]
    num_records = len(iterations)
    if iteration_index < 0:
        iteration_index = num_records+iteration_index

    if iteration_index < 0 or iteration_index >= num_records:
        print(f"Invalid iteration_index {iteration_index}. Cannot plot node biomass.")
        return

    snapshot_x = xMat[iteration_index]
    node_biomass = np.sum(snapshot_x, axis=0)
    iter_label = iterations[iteration_index]

    fig = px.bar(
        x=np.arange(len(node_biomass)),
        y=node_biomass,
        labels={"x":"Node Index", "y":"Total Biomass"},
        title=f"PSD Total Biomass per Node at Iteration {iter_label}",
        template="plotly_dark"
    )
    fig.update_layout(width=1000, height=600)
    outpath = os.path.join(outdir, "psd_node_biomass.html")
    fig.write_html(outpath)
    print(f"Saved node biomass figure to {outpath}")
    
    
def animate_producer_consumer(data, outdir=".", step=5):
    """
    Creates an animated scatter of (Producers, Consumers) over iteration frames.
    """
    iterations = data["iteration"]  # shape (num_records,)
    producers = data["producers"]   # shape (num_records,)
    consumers = data["consumers"]   # shape (num_records,)

    # Basic checks
    num_iters = len(iterations)
    if len(producers) != num_iters or len(consumers) != num_iters:
        print("Mismatch in length of producers/consumers vs. iterations.")
        return

    frames_list = []
    for idx in range(0, num_iters, step):
        frame_data = {
            "data": [
                go.Scatter(
                    x=[producers[idx]],
                    y=[consumers[idx]],
                    mode="markers+text",
                    text=[f"iter={int(iterations[idx])}"],
                    textposition="top center",
                    marker=dict(color="red", size=10)
                )
            ],
            "name": f"frame_{idx}"
        }
        frames_list.append(frame_data)
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[producers[0]],
                y=[consumers[0]],
                mode="markers+text",
                text=[f"iter={iterations[0]}"],
                textposition="top center",
                marker=dict(color="red", size=10)
            )
        ],
        layout=go.Layout(
            xaxis=dict(title="Producers", range=[0, max(producers)*1.1]),
            yaxis=dict(title="Consumers", range=[0, max(consumers)*1.1]),
            title="PSD Animated Producer vs. Consumer Count Over Iterations"
        ),
        frames=frames_list
    )

    fig.update_layout(
        template="plotly_dark",
        width=800,
        height=600,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, 
                                 {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 300,
                                                 "easing": "quadratic-in-out"}}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None],
                                 {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}]
                    }
                ]
            }
        ]
    )

    outpath = os.path.join(outdir, "animate_producer_consumer.html")
    fig.write_html(outpath)
    print(f"Saved animated producer vs. consumer figure to {outpath}")

def animate_node_biomass(data, outdir=".", step=5):
    """
    Creates an animated bar chart of total biomass per node across iterations.
    """
    xMat = data["xMat"]  # shape (num_records, max_species, nodes)
    iterations = data["iteration"]
    num_iters = len(iterations)

    if xMat.ndim != 3:
        print("xMat must be shape (time, species, nodes). Cannot animate node biomass.")
        return

    # sum across species => shape (num_records, nodes)
    sum_node_biomass = np.sum(xMat, axis=1)  
    frames_list = []
    node_indices = np.arange(sum_node_biomass.shape[1])

    for idx in range(0, num_iters, step):
        frame_data = {
            "data": [
                go.Bar(
                    x=node_indices,
                    y=sum_node_biomass[idx,:],
                    marker_color="blue"
                )
            ],
            "name": f"frame_{idx}"
        }
        frames_list.append(frame_data)

    fig = go.Figure(
        data=[
            go.Bar(
                x=node_indices,
                y=sum_node_biomass[0,:],
                marker_color="blue"
            )
        ],
        layout=go.Layout(
            xaxis=dict(title="Node Index"),
            yaxis=dict(title="Biomass"),
            title="PSD Animated Node-Level Biomass Over Iterations"
        ),
        frames=frames_list
    )

    fig.update_layout(
        template="plotly_dark",
        width=900,
        height=600,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None,
                                 {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 300}}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None],
                                 {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}]
                    }
                ]
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": f"{iterations[i]}",
                        "args": [
                            [f"frame_{i}"],
                            {"mode": "immediate", "frame": {"duration":300, "redraw":True}, 
                             "transition": {"duration": 0}}
                        ]
                    }
                    for i in range(0,num_iters,step)
                ],
                "currentvalue": {"prefix": "Iteration: "}
            }
        ]
    )

    outpath = os.path.join(outdir, "animate_node_biomass.html")
    fig.write_html(outpath)
    print(f"Saved animated node biomass figure to {outpath}")
    
    
def plot_producer_consumer_bar(data, outdir=".", barmode="group"):
    """
    Creates a bar chart for Producer and Consumer counts across iterations.
    
    Parameters
    ----------
    data : dict
        Dictionary returned by load_psd_data, containing:
            - data["iteration"] : array-like
            - data["producers"] : array-like
            - data["consumers"] : array-like
    outdir : str
        Directory to save the resulting HTML file.
    barmode : str
        'group' or 'stack'. 
         - 'group'  => side-by-side bars for producers and consumers at each iteration
         - 'stack'  => stacked bars showing total species in a single bar per iteration

    Creates
    -------
    An interactive Plotly bar chart (`psd_producer_consumer_bar.html`)
    in the specified outdir.
    """
    import pandas as pd
    import plotly.express as px
    import os

    iterations = data["iteration"]
    producers = data["producers"]
    consumers = data["consumers"]

    if not (len(iterations) == len(producers) == len(consumers)):
        print("Error: iteration, producers, consumers must have the same length.")
        return

    #DataFrame for easy plotting with Plotly Express
    df = pd.DataFrame({
        "iteration": iterations,
        "Producers": producers,
        "Consumers": consumers
    })
    # Convert wide => long
    df_melt = df.melt(
        id_vars="iteration", 
        value_vars=["Producers", "Consumers"],
        var_name="SpeciesType", 
        value_name="Count"
    )

    fig = px.bar(
        df_melt,
        x="iteration",
        y="Count",
        color="SpeciesType",
        barmode=barmode,
        title="PSD Producer-Consumer Bar Plot",
        labels={"iteration":"Iteration", "Count":"Population Count"},
        template="plotly_dark"
    )
    fig.update_layout(width=1100, height=700)

    outpath = os.path.join(outdir, "psd_producer_consumer_bar.html")
    fig.write_html(outpath)
    print(f"Saved producer-consumer bar plot to: {outpath}")



def interpret_boolean_pair(sto_prob_pair):
    """
    Given a pair of booleans [sto, prob], return integer-coded PSD state:
      0 => Deterministic
      1 => Stochastic
      2 => Probabilistic
    For the case [True, True], we choose state=2 (Probabilistic) 
    (adjust if your model logic differs).
    """
    sto, prob = sto_prob_pair
    if prob:
        return 2  # If 'prob' is True, label as 2
    elif sto:
        return 1  # If only 'sto' is True, label as 1
    else:
        return 0  # Both False => Deterministic


def plot_psd_states_bar(data, outdir="."):

    iterations = data["iteration"]
    PSD_states = data["PSD_states"]

    # Check shape
    if PSD_states.ndim == 3:
        # shape (num_iterations, num_species, 2)
        if PSD_states.shape[2] == 2 and PSD_states.dtype == bool:
            print("Interpreting last dim of PSD_states as [Stochastic, Probabilistic] booleans.")
            
            # Map from boolean pair => integer-coded state
            num_iterations, num_species, _ = PSD_states.shape
            int_states = np.zeros((num_iterations, num_species), dtype=int)

            for i in range(num_iterations):
                for j in range(num_species):
                    sto = PSD_states[i, j, 0]
                    prob = PSD_states[i, j, 1]
                    # interpret as 0,1,2
                    if prob:
                        int_states[i, j] = 2
                    elif sto:
                        int_states[i, j] = 1
                    else:
                        int_states[i, j] = 0

            PSD_states = int_states  # now shape (num_iterations, num_species)
        else:
            # shape or type, fallback logic or error
            PSD_states = PSD_states.max(axis=2)  
            print("Warning: PSD_states was 3D but not recognized as [sto, prob]; used .max(axis=2).")

    # PSD_states should be 2D with integer-coded states 0,1,2
    if PSD_states.shape[0] != len(iterations):
        print("Error: PSD_states length does not match iteration length. Cannot make bar chart.")
        return

    #how many species in each state at each iteration
    state0 = np.sum(PSD_states == 0, axis=1)  # Deterministic
    state1 = np.sum(PSD_states == 1, axis=1)  # Stochastic
    state2 = np.sum(PSD_states == 2, axis=1)  # Probabilistic

    df = pd.DataFrame({
        "iteration": iterations,
        "Deterministic (0)": state0,
        "Stochastic (1)": state1,
        "Probabilistic (2)": state2
    })

    # Melt for stacked bar
    df_melt = df.melt(
        id_vars="iteration", 
        value_vars=["Deterministic (0)", "Stochastic (1)", "Probabilistic (2)"],
        var_name="PSD_State", 
        value_name="Count"
    )

    #specify custom colors
    color_map = {
        "Deterministic (0)": "blue",
        "Stochastic (1)": "orange",
        "Probabilistic (2)": "green"
    }

    fig = px.bar(
        df_melt,
        x="iteration",
        y="Count",
        color="PSD_State",
        color_discrete_map=color_map,
        title="PSD State Distribution (Stacked Bar)",
        labels={"iteration": "Iteration", "Count": "Number of Species", "PSD_State": "PSD State"},
        template="plotly_dark",
        barmode="stack"
    )
    fig.update_layout(width=1200, height=700)

    outpath = os.path.join(outdir, "psd_states_bar.html")
    fig.write_html(outpath)
    print(f"Saved PSD state distribution bar chart to: {outpath}")


def plot_histogram_of_metric(
    data, 
    metric: str = "logB", 
    iteration_index=None,
    outdir=".", 
    bins=30,
    node_aggregation="flatten"
):
    """
    Plots a histogram for a chosen PSD metric (e.g., 'logB', 'PoissonClocks', 
    'establishment_prob', 'rMat', 'xMat', etc.) at either:
      - a single iteration (if iteration_index is an int),
      - or all iterations combined (if iteration_index is None).


    """

    # Check if the metric exists
    if metric not in data:
        print(f"Error: The metric '{metric}' is not found in `data` keys.")
        print(f"Available keys: {list(data.keys())}")
        return
    
    arr = data[metric]  # 2D or 3D
    iterations = data["iteration"]
    num_iterations = len(iterations)
    
    #  arr is 3D, decide how to reduce the node dimension:
    #    shape => (num_iterations, max_species, nodes)
    if arr.ndim == 3:
        if node_aggregation == "mean":
            print(f"Averaging node dimension for metric '{metric}'.")
            arr = arr.mean(axis=2)  # => shape (num_iterations, max_species)
        elif node_aggregation == "sum":
            print(f"Summing node dimension for metric '{metric}'.")
            arr = arr.sum(axis=2)
        elif node_aggregation == "flatten":
            print(f"Flattening node dimension for metric '{metric}'.")
            # shape => (num_iterations, max_species * nodes)
            arr = arr.reshape(num_iterations, -1)
        elif node_aggregation == "none":
            print(f"No node aggregation: fully flattening if iteration_index=None.")
            # do nothing if a single iteration is requested;
            # but if iteration_index=None, we'll flatten across all dims anyway
            pass
        else:
            print(f"Unrecognized node_aggregation='{node_aggregation}'. Using 'flatten' by default.")
            arr = arr.reshape(num_iterations, -1)

    # 3) Decide which iteration(s) to gather
    if iteration_index is None:
        # Combine everything from all iterations
        if arr.ndim == 2:
            # shape => (num_iterations, X)
            values = arr.flatten()
        elif arr.ndim == 3:
            # shape => (num_iterations, max_species, nodes) + "none" approach
            values = arr.flatten()
        else:
            # fallback
            values = arr.flatten()
        title_str = f"Histogram of {metric} (All Iterations, node_agg={node_aggregation})"
    else:
        # Single iteration:
        # handle negative index (like -1 => last iteration)
        if iteration_index < 0:
            iteration_index = num_iterations + iteration_index
        if iteration_index < 0 or iteration_index >= num_iterations:
            print(f"Error: iteration_index={iteration_index} out of range (0..{num_iterations-1}).")
            return

        # flatten that single iteration
        if arr.ndim == 3:
            iteration_data = arr[iteration_index, :, :]
            values = iteration_data.flatten()
        elif arr.ndim == 2:
            values = arr[iteration_index, :]
        else:
            # fallback if 1D or other shape
            values = arr
        title_str = f"Histogram of {metric} at Iter={iterations[iteration_index]}, node_agg={node_aggregation}"

    #Filter out invalid or infinite values
    values = values[np.isfinite(values)]

    # Create the histogram
    fig = px.histogram(
        x=values,
        nbins=bins,
        title=title_str,
        labels={"x": metric, "y": "Count"},
        template="plotly_dark"
    )
    fig.update_layout(width=900, height=600)

    # Save to HTML
    fname = f"hist_{metric}"
    if iteration_index is None:
        fname += "_all"
    else:
        fname += f"_iter{iteration_index}"
    fname += f"_agg-{node_aggregation}_bins{bins}.html"

    outpath = os.path.join(outdir, fname)
    fig.write_html(outpath)
    print(f"Saved histogram of '{metric}' to: {outpath}")

    

def main_viz():
    """
    Main function to load PSD results and generate visualizations.
    """
    try:
        # Determine the base directory 
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Build relative paths
        data_file = os.path.join(BASE_DIR, "output", "trajectory_data_PSD.npz")
        outdir = os.path.join(BASE_DIR, "PSD_plots")

        # Ensure that the output directory exists
        os.makedirs(outdir, exist_ok=True)

        # Load the data
        data = load_psd_data(data_file)
        print("Data loaded successfully from:", data_file)

        # Time-series style plots
        plot_time_series(data, outdir=outdir)

        # PSD metrics plots
        plot_psd_metrics(data, outdir=outdir)

        #PSD state distribution
        plot_psd_state_distribution(data, outdir=outdir)
        
        #PSD state distribution (stacked bars)
        plot_psd_states_bar(data, outdir=outdir)

        #Biomass distribution snapshots
        plot_biomass_distribution(data, outdir=outdir, snapshot_iterations=None)
        
        #    with a grouped or stacked bar setting
        plot_producer_consumer_bar(data, outdir=outdir, barmode="group")  # or barmode="stack"

        # show node-level biomass for the last iteration
        plot_spatial_node_biomass(data, outdir=outdir, iteration_index=-1)
        
        # Generate animations
        # skip frames with step=5 or step=10
        step = 10
        animate_producer_consumer(data, outdir=outdir, step=step)
        animate_node_biomass(data, outdir=outdir, step=step)
        
        
        #LogB across all iterations combined
        plot_histogram_of_metric(data, metric="logB", iteration_index=None, outdir=outdir, bins=30)
        #PoissonClocks at last iteration
        plot_histogram_of_metric(data, metric="PoissonClocks", iteration_index=None, outdir=outdir, bins=20)
        
        plot_histogram_of_metric(
            data,
            metric="xMat",
            iteration_index=None,
            outdir="PSD_plots",
            bins=25,
            node_aggregation="sum"
        )
        

        print("All PSD plots generated successfully.")

    except Exception as e:
        print("Error in main_viz:", e)

if __name__ == "__main__":
    main_viz()
