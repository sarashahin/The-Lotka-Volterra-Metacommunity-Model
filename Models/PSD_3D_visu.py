import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

###############################################################################
# 1) ANIMATE A 3D SURFACE (for continuous data)
###############################################################################
def animate_3d_surface(
    data_array,
    output_dir,
    array_name="xMat",
    cmap='viridis',
    filename_gif="xMat_3D_animation.gif",
    filename_mp4="xMat_3D_animation.mp4"
):
    """
    Creates a 3D surface animation over the 'iteration' dimension with a SINGLE colorbar
    that persists across all frames.

    data_array.shape = (num_iters, num_steps, num_nodes).

    For each iteration i:
      - X-axis = step index (0..num_steps-1)
      - Y-axis = node index (0..num_nodes-1)
      - Z-axis = data_array[i, :, :]
      - We keep a single colorbar that matches the global min..max of the data.

    The colorbar is created once with a ScalarMappable.

    The user sees a stable color scale (vmin..vmax) for the entire animation.
    """
    import os

    num_iters, num_steps, num_nodes = data_array.shape

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    step_idx = np.arange(num_steps)
    node_idx = np.arange(num_nodes)
    X_grid, Y_grid = np.meshgrid(step_idx, node_idx, indexing='ij')

    #define global min/max for the entire data
    zmin, zmax = np.min(data_array), np.max(data_array)

    # Prepare a single Norm + ScalarMappable for the colorbar
    from matplotlib import cm
    norm = plt.Normalize(vmin=zmin, vmax=zmax)
    #  set any data on the mappable;for the colorbar:
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])  # or set_array(None)

    # store the surface so can update it
    surf_plot = [None]

    def init_func():
        ax.clear()

        ax.set_xlim(0, num_steps - 1)
        ax.set_ylim(0, num_nodes - 1)
        ax.set_zlim(zmin, zmax)

        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel(f"{array_name} Value")

        ax.set_title(
            f"{array_name} Over Iteration (3D Surface)\n"
            f"Colormap: '{cmap}', global data range: [{zmin:.2f}, {zmax:.2f}]",
            pad=15
        )

        # Create a dummy initial surface (frame=0) so we have something to color
        Z_vals = data_array[0, :, :]
        surf = ax.plot_surface(
            X_grid, Y_grid, Z_vals,
            cmap=cmap, edgecolor='none', alpha=0.85, norm=norm
        )
        surf_plot[0] = surf

        # Create the colorbar ONCE
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.08)
        cbar.set_label(f"{array_name} (darker=low, brighter=high)")

        return [surf]

    def update(frame):
        ax.clear()

        # Redraw axis limits / labels
        ax.set_xlim(0, num_steps - 1)
        ax.set_ylim(0, num_nodes - 1)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel(f"{array_name} Value")

        ax.set_title(
            f"{array_name} Iteration {frame+1}/{num_iters}\n"
            f"Colormap: {cmap}, data range = [{zmin:.2f}, {zmax:.2f}]",
            pad=15
        )

        # Plot new surface
        Z_vals = data_array[frame, :, :]
        surf = ax.plot_surface(
            X_grid, Y_grid, Z_vals,
            cmap=cmap, edgecolor='none', alpha=0.85, norm=norm
        )
        surf_plot[0] = surf
        return [surf]

    # Create the FuncAnimation
    ani = animation.FuncAnimation(
        fig, update, frames=num_iters, init_func=init_func,
        interval=500, blit=False, repeat=True
    )

    # Attempt to save as GIF
    gif_path = os.path.join(output_dir, filename_gif)
    try:
        ani.save(gif_path, writer='pillow', fps=2)
        print(f"[INFO] Saved 3D surface animation (GIF) for {array_name}: {gif_path}")
    except Exception as e:
        print(f"[WARN] Could not save GIF for {array_name}: {e}")

    # Attempt to save as MP4
    mp4_path = os.path.join(output_dir, filename_mp4)
    try:
        ani.save(mp4_path, writer='ffmpeg', fps=2)
        print(f"[INFO] Saved 3D surface animation (MP4) for {array_name}: {mp4_path}")
    except Exception as e:
        print(f"[WARN] Could not save MP4 for {array_name}: {e}")

    plt.close(fig)

###############################################################################
# 2) ANIMATE A SINGLE BOOLEAN ARRAY (True/False) IN 3D
###############################################################################
def animate_boolean_3d(
    bool_array,
    output_dir,
    title="PSD_states Over Time (3D Boolean)",
    active_color='red',
    inactive_color='blue',
    active_label='True => Active',
    inactive_label='False => Inactive',
    filename_gif="PSDstates_3D_animation.gif",
    filename_mp4="PSDstates_3D_animation.mp4"
):
    """
    3D animation for a single boolean array (num_iters, num_steps, num_nodes).
    
    * iteration => frames
    * (step, node) => the 2D plane
    * Z => 1 if True (active_color), 0 if False (inactive_color)
    
    The legend clarifies the meaning of the colors:
      red => True => "Active"
      blue => False => "Inactive"
    
    The user can easily read from the title/legend exactly what each color means in the 3D plot.
    """
    num_iters, num_steps, num_nodes = bool_array.shape

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def init_func():
        ax.clear()
        ax.set_xlim(0, num_steps)
        ax.set_ylim(-0.5, num_nodes - 0.5)
        ax.set_zlim(-0.5, 1.5)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel("State (1=Active, 0=Inactive)")
        ax.set_title(title)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(0, num_steps)
        ax.set_ylim(-0.5, num_nodes - 0.5)
        ax.set_zlim(-0.5, 1.5)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel("State (1=Active, 0=Inactive)")
        ax.set_title(f"{title}\nIteration {frame+1}/{num_iters}", pad=15)

        Xs, Ys, Zs = [], [], []
        Cs, Ss = [], []
        for j in range(num_steps):
            for k in range(num_nodes):
                val = bool_array[frame, j, k]
                Xs.append(j)
                Ys.append(k)
                if val:
                    # True => Z=1 => active_color
                    Zs.append(1)
                    Cs.append(active_color)
                    Ss.append(80)
                else:
                    # False => Z=0 => inactive_color
                    Zs.append(0)
                    Cs.append(inactive_color)
                    Ss.append(30)

        Xs = np.array(Xs)
        Ys = np.array(Ys)
        Zs = np.array(Zs)
        sc = ax.scatter(Xs, Ys, Zs, c=Cs, s=Ss, alpha=0.8)

        import matplotlib.lines as mlines
        active_pt = mlines.Line2D([], [], color=active_color, marker='o', linestyle='None',
                                  markersize=8, label=active_label)
        inactive_pt = mlines.Line2D([], [], color=inactive_color, marker='o', linestyle='None',
                                    markersize=8, label=inactive_label)
        ax.legend(handles=[active_pt, inactive_pt], loc='upper right')

        return [sc]

    ani = animation.FuncAnimation(
        fig, update, frames=num_iters, init_func=init_func,
        interval=500, blit=False, repeat=True
    )

    gif_path = os.path.join(output_dir, filename_gif)
    try:
        ani.save(gif_path, writer='pillow', fps=2)
        print(f"[INFO] Saved boolean 3D animation (GIF): {gif_path}")
    except Exception as e:
        print(f"[WARN] Could not save GIF: {e}")

    mp4_path = os.path.join(output_dir, filename_mp4)
    try:
        ani.save(mp4_path, writer='ffmpeg', fps=2)
        print(f"[INFO] Saved boolean 3D animation (MP4): {mp4_path}")
    except Exception as e:
        print(f"[WARN] Could not save MP4: {e}")

    plt.close(fig)


###############################################################################
# 3) ANIMATE THE BOOLEAN 3-STATE (Prob=Blue, Stoch=Orange, Det=Red)
###############################################################################
def animate_boolean_3states(
    bool_arrays,
    labels,
    output_dir,
    title="Three PSD States Over Time",
    filename_gif="PSD_3states.gif",
    filename_mp4="PSD_3states.mp4"
):
    """
    3 boolean arrays (num_iters, num_steps, num_nodes):
      * arrayA => "Probabilistic" => color = Blue => Z=0
      * arrayB => "Stochastic" => color = Orange => Z=1
      * arrayC => "Deterministic" => color = Red => Z=2
    
    The user-provided 'labels' should be e.g. ["Probabilistic", "Stochastic", "Deterministic"].
    
    We produce an animation over iteration, with:
      - X-axis = step
      - Y-axis = node
      - Z-axis = 0 or 1 or 2 depending on which state is True at that (iteration, step, node).
      - If none are True, we skip it or color gray.
    
    The legend clarifies the color => state mapping. This ensures the final
    3D animation is unambiguously interpretable (blue=Prob, orange=Stoch, red=Det).
    """
    arrayA, arrayB, arrayC = bool_arrays
    labelA, labelB, labelC = labels
    num_iters, num_steps, num_nodes = arrayA.shape

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-60)

    def init_func():
        ax.clear()
        ax.set_xlim(0, num_steps)
        ax.set_ylim(-0.5, num_nodes - 0.5)
        ax.set_zlim(-0.5, 2.5)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel("State Index (0/1/2)")
        ax.set_title(title)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(0, num_steps)
        ax.set_ylim(-0.5, num_nodes - 0.5)
        ax.set_zlim(-0.5, 2.5)
        ax.view_init(elev=25, azim=-60)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel("State Index (0=Blue,1=Orange,2=Red)")
        ax.set_title(f"{title}\nIteration = {frame+1}/{num_iters}", pad=15)

        Xs, Ys, Zs, Cs = [], [], [], []
        for j in range(num_steps):
            for k in range(num_nodes):
                if arrayA[frame, j, k]:
                    # Prob => Z=0 => color=Blue
                    Xs.append(j); Ys.append(k); Zs.append(0)
                    Cs.append('blue')
                elif arrayB[frame, j, k]:
                    # Stoch => Z=1 => color=Orange
                    Xs.append(j); Ys.append(k); Zs.append(1)
                    Cs.append('orange')
                elif arrayC[frame, j, k]:
                    # Det => Z=2 => color=Red
                    Xs.append(j); Ys.append(k); Zs.append(2)
                    Cs.append('red')
                else:
                    # No state => skip or color gray
                    pass

        if len(Xs) > 0:
            Xs = np.array(Xs)
            Ys = np.array(Ys)
            Zs = np.array(Zs)
            ax.scatter(Xs, Ys, Zs, c=Cs, s=60, alpha=0.8)

        import matplotlib.lines as mlines
        a_pt = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                             markersize=8, label=f"{labelA} => (Blue, z=0)")
        b_pt = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                             markersize=8, label=f"{labelB} => (Orange, z=1)")
        c_pt = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                             markersize=8, label=f"{labelC} => (Red, z=2)")
        ax.legend(handles=[a_pt, b_pt, c_pt], loc='upper right')

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=num_iters, init_func=init_func,
        interval=500, blit=False, repeat=True
    )

    gif_path = os.path.join(output_dir, filename_gif)
    try:
        ani.save(gif_path, writer='pillow', fps=2)
        print(f"[INFO] Saved 3-state animation (GIF) to: {gif_path}")
    except Exception as e:
        print(f"[WARN] Could not save GIF (3-state): {e}")

    mp4_path = os.path.join(output_dir, filename_mp4)
    try:
        ani.save(mp4_path, writer='ffmpeg', fps=2)
        print(f"[INFO] Saved 3-state animation (MP4) to: {mp4_path}")
    except Exception as e:
        print(f"[WARN] Could not save MP4 (3-state): {e}")

    plt.close(fig)


###############################################################################
# 3 Distinct States in One 3D Plot
###############################################################################
def plot_three_state_3d(
    iteration, arrayA, arrayB, arrayC,
    output_dir,
    labelA="rMat > 1.0",
    labelB="xMat > 100.0",
    labelC="PSD=True",
    title="3 Different States in One 3D Plot"
):
    """
    Combines three separate boolean arrays in one static 3D scatter:
      - X-axis = iteration
      - Y-axis = step
      - Z-axis = node
    Each set of True coords is a different color/marker.
    """
    num_iters, num_steps, num_nodes = arrayA.shape

    def coords_where_true(arr):
        coords = []
        for i in range(num_iters):
            for j in range(num_steps):
                for k in range(num_nodes):
                    if arr[i, j, k]:
                        coords.append((i, j, k))
        return coords

    coordsA = coords_where_true(arrayA)
    coordsB = coords_where_true(arrayB)
    coordsC = coords_where_true(arrayC)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    if coordsA:
        xA = [p[0] for p in coordsA]
        yA = [p[1] for p in coordsA]
        zA = [p[2] for p in coordsA]
        ax.scatter(xA, yA, zA, c='green', marker='o', s=40, alpha=0.7, label=labelA)

    if coordsB:
        xB = [p[0] for p in coordsB]
        yB = [p[1] for p in coordsB]
        zB = [p[2] for p in coordsB]
        ax.scatter(xB, yB, zB, c='orange', marker='^', s=40, alpha=0.7, label=labelB)

    if coordsC:
        xC = [p[0] for p in coordsC]
        yC = [p[1] for p in coordsC]
        zC = [p[2] for p in coordsC]
        ax.scatter(xC, yC, zC, c='purple', marker='s', s=40, alpha=0.7, label=labelC)

    ax.set_title(title)
    ax.set_xlabel("Iteration Index")
    ax.set_ylabel("Step Index")
    ax.set_zlabel("Node Index")

    ax.view_init(elev=25, azim=-60)
    ax.legend(loc='best')

    outpath = os.path.join(output_dir, "three_state_3d.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved 3D scatter of 3 distinct states to: {outpath}")
    
    
###############################################################################
#STATIC 3D SCATTER FOR PSD_states (ALL TIME)
#  
###############################################################################
def plot_boolean_3d_scatter_all_time(
    bool_array, output_dir,
    title="PSD_states (All Iterations) in 3D Scatter",
    active_color='red',
    inactive_color='gray',
    active_label='PSD: True (Active)',
    inactive_label='PSD: False (Inactive)',
    filename="PSD_states_3D_scatter_all_time.png"
):
    """
    Static 3D scatter for a boolean array across all iteration, step, and node.
    - X-axis = iteration index
    - Y-axis = step index
    - Z-axis = node index
    - color-coded for True vs. False
    """
    num_iters, num_steps, num_nodes = bool_array.shape

    X_vals, Y_vals, Z_vals = [], [], []
    colors, sizes = [], []

    for i in range(num_iters):
        for j in range(num_steps):
            for k in range(num_nodes):
                X_vals.append(i)
                Y_vals.append(j)
                Z_vals.append(k)
                if bool_array[i,j,k]:
                    colors.append(active_color)
                    sizes.append(50)  # Larger marker for True
                else:
                    colors.append(inactive_color)
                    sizes.append(10)  # Smaller marker for False

    X_vals = np.array(X_vals)
    Y_vals = np.array(Y_vals)
    Z_vals = np.array(Z_vals)
    colors = np.array(colors)
    sizes = np.array(sizes)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_vals, Y_vals, Z_vals, c=colors, s=sizes, alpha=0.7)

    ax.set_title(title, pad=15)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step Index")
    ax.set_zlabel("Node Index")

    ax.view_init(elev=25, azim=-60)

    # Create a manual legend
    import matplotlib.lines as mlines
    true_dot = mlines.Line2D([], [], color=active_color, marker='o', linestyle='None',
                             markersize=8, label=active_label)
    false_dot = mlines.Line2D([], [], color=inactive_color, marker='o', linestyle='None',
                              markersize=8, label=inactive_label)
    ax.legend(handles=[true_dot, false_dot], loc='best')

    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved static 3D scatter (all time) of PSD_states: {outpath}")
    
    

###############################################################################
#CLASSIFY EACH (ITER, STEP, NODE) INTO {0,1,2} FOR THE 3 PSD STATES
###############################################################################
def classify_psd_state(xMat, rMat, PSD_states):
    """
    create a single 3-state integer-coded array, psd_3cat,
    with shape (num_iters, num_steps, num_nodes).
    
    Returns:
      psd_3cat[i,j,k] in {0,1,2}, meaning:
         0 => Probabilistic
         1 => Stochastic
         2 => Deterministic
      or -1 if no rule matched.
    
    of "Probabilistic <-> Stochastic -> Deterministic".
    
    For demonstration:
      -pick "Probabilistic" if PSD_states == True
      - Else if rMat > 1 => "Stochastic"
      - Else if xMat > 100 => "Deterministic"
      - Else => -1
    """
    num_iters, num_steps, num_nodes = xMat.shape
    psd_3cat = np.full((num_iters, num_steps, num_nodes), fill_value=-1, dtype=int)

    for i in range(num_iters):
        for j in range(num_steps):
            for k in range(num_nodes):
                if PSD_states[i,j,k]:
                    psd_3cat[i,j,k] = 0  # Probabilistic
                elif rMat[i,j,k] > 1.0:
                    psd_3cat[i,j,k] = 1  # Stochastic
                elif xMat[i,j,k] > 100.0:
                    psd_3cat[i,j,k] = 2  # Deterministic
                else:
                    # None matched => -1
                    psd_3cat[i,j,k] = -1
    
    return psd_3cat


###############################################################################
# PLOT A STATIC 3D SCATTER OF THE 3-STATE ARRAY (ALL TIME)
###############################################################################
def plot_psd_3state_static_3d(
    psd_3cat,
    output_dir,
    title="PSD Turnover (Prob,Stoch,Det) - All Iterations",
    filename="psd_3cat_all_time.png"
):
    """
    Creates one 3D plot that shows all iteration/time at once, color-coded for each of the 3 states:
      0 => Probabilistic => blue
      1 => Stochastic    => orange
      2 => Deterministic => red
    Any -1 values are ignored.
    
    Axes:
      X = iteration
      Y = step
      Z = node
    """
    num_iters, num_steps, num_nodes = psd_3cat.shape

    # store coords for each state
    coords0 = []  # Probabilistic
    coords1 = []  # Stochastic
    coords2 = []  # Deterministic
    coords_neg1 = []  # for any -1 leftover if logic didn't match

    for i in range(num_iters):
        for j in range(num_steps):
            for k in range(num_nodes):
                val = psd_3cat[i,j,k]
                if val == 0:
                    coords0.append((i,j,k))
                elif val == 1:
                    coords1.append((i,j,k))
                elif val == 2:
                    coords2.append((i,j,k))
                else:
                    coords_neg1.append((i,j,k))
    
    # Convert to arrays
    coords0 = np.array(coords0)
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    coords_neg1 = np.array(coords_neg1)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each state
    if len(coords0) > 0:
        ax.scatter(coords0[:,0], coords0[:,1], coords0[:,2],
                   c='blue', marker='o', s=40, alpha=0.7, label='Probabilistic (0)')
    if len(coords1) > 0:
        ax.scatter(coords1[:,0], coords1[:,1], coords1[:,2],
                   c='orange', marker='o', s=40, alpha=0.7, label='Stochastic (1)')
    if len(coords2) > 0:
        ax.scatter(coords2[:,0], coords2[:,1], coords2[:,2],
                   c='red', marker='o', s=40, alpha=0.7, label='Deterministic (2)')
    if len(coords_neg1) > 0:
        ax.scatter(coords_neg1[:,0], coords_neg1[:,1], coords_neg1[:,2],
                   c='gray', marker='x', s=20, alpha=0.7, label='Unassigned (-1)')

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step")
    ax.set_zlabel("Node")
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc='best')

    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved 3-state static 3D scatter to: {outpath}")


###############################################################################
# 3) ANIMATE THE 3-STATE ARRAY OVER ITERATION
###############################################################################
def animate_psd_3state(
    psd_3cat,
    output_dir,
    title="PSD Turnover (Prob,Stoch,Det) - Animation",
    filename_gif="psd_3cat_animation.gif",
    filename_mp4="psd_3cat_animation.mp4"
):
    """
    3D animation over iteration dimension.
    
    This time, we show each iteration as a separate 'frame'.:
      X-axis = step
      Y-axis = node
      Z-axis = the state value (0,1,2), so see states stacked at different Z levels.
      color-coding = same as above (blue=0, orange=1, red=2, gray for -1).
    """
    num_iters, num_steps, num_nodes = psd_3cat.shape

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-60)

    # Set axis bounds so they don't change each frame
    ax.set_xlim(0, num_steps)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 2.5)

    #define an init function (FuncAnimation)
    def init_func():
        ax.clear()
        ax.set_xlim(0, num_steps)
        ax.set_ylim(-0.5, 1.5)
        ax.set_zlim(-0.5, 2.5)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel("State (0=Prob,1=Stoch,2=Det)")
        ax.set_title(title)
        return []

    # The update function for each frame
    def update(frame):
        ax.clear()
        ax.set_xlim(0, num_steps)
        ax.set_ylim(-0.5, 1.5)
        ax.set_zlim(-0.5, 2.5)
        ax.view_init(elev=25, azim=-60)
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Node Index")
        ax.set_zlabel("State (0=Prob,1=Stoch,2=Det)")
        ax.set_title(f"{title}\nIteration = {frame+1}/{num_iters}", pad=15)

        Xs = []
        Ys = []
        Zs = []
        Cs = []
        Ss = []

        for j in range(num_steps):
            for k in range(num_nodes):
                state_val = psd_3cat[frame, j, k]
                Xs.append(j)   # step
                Ys.append(k)   # node
                Zs.append(state_val if state_val >= 0 else -0.2) 
                  # put -1 at z=-0.2 or something so it's visible below 0 
                
                if state_val == 0:
                    Cs.append('blue')
                    Ss.append(80)
                elif state_val == 1:
                    Cs.append('orange')
                    Ss.append(80)
                elif state_val == 2:
                    Cs.append('red')
                    Ss.append(80)
                else:
                    Cs.append('gray')  # unassigned
                    Ss.append(30)

        Xs = np.array(Xs)
        Ys = np.array(Ys)
        Zs = np.array(Zs)
        sc = ax.scatter(Xs, Ys, Zs, c=Cs, s=Ss, alpha=0.8)

        # Legend
        import matplotlib.lines as mlines
        prob_pt  = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                 markersize=8, label='Prob (0)')
        stoch_pt = mlines.Line2D([], [], color='orange', marker='o', linestyle='None',
                                 markersize=8, label='Stoch (1)')
        det_pt   = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                 markersize=8, label='Det (2)')
        un_pt    = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                 markersize=8, label='Unassigned (-1)')
        ax.legend(handles=[prob_pt, stoch_pt, det_pt, un_pt], loc='upper right')

        return [sc]

    ani = animation.FuncAnimation(
        fig, update, frames=num_iters, init_func=init_func,
        interval=800, blit=False, repeat=True
    )

    # Save as GIF
    gif_path = os.path.join(output_dir, filename_gif)
    try:
        ani.save(gif_path, writer='pillow', fps=1)
        print(f"[INFO] Saved 3-state animation (GIF): {gif_path}")
    except Exception as e:
        print(f"[WARN] Could not save GIF: {e}")

    # Save as MP4
    mp4_path = os.path.join(output_dir, filename_mp4)
    try:
        ani.save(mp4_path, writer='ffmpeg', fps=1)
        print(f"[INFO] Saved 3-state animation (MP4): {mp4_path}")
    except Exception as e:
        print(f"[WARN] Could not save MP4: {e}")

    plt.close(fig)


###############################################################################
# 4) MAIN SCRIPT
###############################################################################
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Build relative paths
    npz_path = os.path.join(BASE_DIR, "output", "trajectory_data_PSD.npz")
    output_dir = os.path.join(BASE_DIR, "PSD_3D_Plots")
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    iteration           = data['iteration']          # shape (100,)
    xMat                = data['xMat']               # shape (100,88,2)
    rMat                = data['rMat']               # shape (100,88,2)
    PSD_states          = data['PSD_states']         # shape (100,88,2) boolean
    PoissonClocks       = data['PoissonClocks']      # shape (100,88,2)
    logB                = data['logB']               # shape (100,88,2)
    establishment_prob  = data['establishment_prob'] # shape (100,88,2)
    i_array             = data['i']                  # shape (100,88,2)

    print("[INFO] Data loaded from:", npz_path)
    print(" iteration shape:", iteration.shape)
    print(" xMat shape:", xMat.shape)
    print(" rMat shape:", rMat.shape)
    print(" PSD_states shape:", PSD_states.shape)
  

    #----------------------------------------------------------------------
    #  Animate 3D surfaces for continuous arrays
    #    with a colorbar that clarifies numeric range
    #----------------------------------------------------------------------
    animate_3d_surface(
        xMat, output_dir,
        array_name="xMat",
        cmap='viridis',
        filename_gif="xMat_3D_animation.gif",
        filename_mp4="xMat_3D_animation.mp4"
    )

    animate_3d_surface(
        rMat, output_dir,
        array_name="rMat",
        cmap='plasma',
        filename_gif="rMat_3D_animation.gif",
        filename_mp4="rMat_3D_animation.mp4"
    )

    animate_3d_surface(
        logB, output_dir,
        array_name="logB",
        cmap='coolwarm',
        filename_gif="logB_3D_animation.gif",
        filename_mp4="logB_3D_animation.mp4"
    )

    animate_3d_surface(
        i_array, output_dir,
        array_name="Invasion Rate (i)",
        cmap='magma',
        filename_gif="i_3D_animation.gif",
        filename_mp4="i_3D_animation.mp4"
    )

    animate_3d_surface(
        establishment_prob, output_dir,
        array_name="Establishment Probability",
        cmap='cividis',
        filename_gif="establishProb_3D_animation.gif",
        filename_mp4="establishProb_3D_animation.mp4"
    )

    animate_3d_surface(
        PoissonClocks, output_dir,
        array_name="PoissonClocks",
        cmap='Spectral',
        filename_gif="PoissonClocks_3D_animation.gif",
        filename_mp4="PoissonClocks_3D_animation.mp4"
    )

    #----------------------------------------------------------------------
    # Animate a single boolean array (PSD_states)
    #    with a legend clarifying color => meaning
    #----------------------------------------------------------------------
    from functools import partial

    animate_boolean_3d(
        bool_array=PSD_states,
        output_dir=output_dir,
        title="PSD States Over Time (3D Boolean)",
        active_color='red',   # means PSD=True => red => 'Active'
        inactive_color='blue',# means PSD=False => blue => 'Inactive'
        active_label='PSD=True => Active',
        inactive_label='PSD=False => Inactive',
        filename_gif="PSD_3D_bool.gif",
        filename_mp4="PSD_3D_bool.mp4"
    )

    #----------------------------------------------------------------------
    #  Animate 3 separate boolean arrays: Prob=Blue, Stoch=Orange, Det=Red
    #----------------------------------------------------------------------
    prob_bool  = PSD_states       # interpret as Probabilistic
    stoch_bool = (rMat > 1.0)     # Stochastic
    det_bool   = (xMat > 100.0)   # Deterministic

    animate_boolean_3states(
        bool_arrays=[prob_bool, stoch_bool, det_bool],
        labels=["Probabilistic", "Stochastic", "Deterministic"],
        output_dir=output_dir,
        title="Prob(Blue)/Stoch(Orange)/Det(Red) - 3-State Over Time",
        filename_gif="ProbStochDet_3State.gif",
        filename_mp4="ProbStochDet_3State.mp4"
    )

    print("[INFO] All dynamic 3D visualizations created. Check output directory!")


    #-------------------------------------------------------------------------
    # 3 Distinct States in One 3D Plot
    #   StateA=rMat>1.0, StateB=xMat>100, StateC=PSD_states==True
    #-------------------------------------------------------------------------
    stateA = (rMat > 1.0)
    stateB = (xMat > 100.0)
    stateC = PSD_states

    plot_three_state_3d(
        iteration=iteration,
        arrayA=stateA,
        arrayB=stateB,
        arrayC=stateC,
        output_dir=output_dir,
        labelA="rMat > 1.0",
        labelB="xMat > 100.0",
        labelC="PSD=True",
        title=" 3 Different States in One 3D Plot"
    )

    print("[INFO] All plotting is complete. Check the output directory.")
    
    
    

    #-------------------------------------------------------------------------
    # Static 3D Scatter for PSD_states across entire simulation
    #-------------------------------------------------------------------------
    plot_boolean_3d_scatter_all_time(
        bool_array=PSD_states,
        output_dir=output_dir,
        title="PSD_states (All Iterations) in 3D Scatter",
        active_color='red',
        inactive_color='gray',
        active_label='PSD: True (Active)',
        inactive_label='PSD: False (Inactive)',
        filename="PSD_states_3D_scatter_all_time.png"
    )
    
    
    # function classify_psd_state.
    psd_3cat = classify_psd_state(xMat, rMat, PSD_states)

    #-------------------------------------------------------------------------
    # Plot the 3D Scatter of All Time
    #-------------------------------------------------------------------------
    plot_psd_3state_static_3d(
        psd_3cat=psd_3cat,
        output_dir=output_dir,
        title="My PSD Turnover - 3 States (All Iterations)",
        filename="psd_3cat_all_time.png"
    )

    #-------------------------------------------------------------------------
    #Animate the 3-State Over Iteration
    #-------------------------------------------------------------------------
    animate_psd_3state(
        psd_3cat=psd_3cat,
        output_dir=output_dir,
        title="My PSD Turnover - 3 States Over Time (Animation)",
        filename_gif="psd_3cat_animation.gif",
        filename_mp4="psd_3cat_animation.mp4"
    )

    print("[INFO] Done! Check output directory for 3D plots & animations.")



if __name__ == "__main__":
    main()



