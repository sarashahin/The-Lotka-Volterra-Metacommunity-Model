

############################################
# run_rps_dynamics.py
############################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

from models_psd2 import PSD2Model
from models_ibm import IBMModel

# Helper to extract dominant oscillation period via FFT
def dominant_period(t, x):
    from scipy.fft import rfftfreq, rfft
    dt = np.mean(np.diff(t))
    freqs = rfftfreq(len(t), dt)[1:]
    spec = np.abs(rfft(x))[1:]
    peak = np.argmax(spec)
    return 1.0 / freqs[peak]

# Rock–paper–scissors competition matrix
# Rock-Paper-Scissors parameters
a, b = 1.7, 0.4     # cyclic competition: species 0 beats 1, 1 beats 2, 2 beats 0
r = np.ones(3)
C = np.array([[1, a, b],  # 0 suffers less from 1 (a), more from 2 (b)
              [b, 1, a],   # 1 suffers less from 2, more from 0
              [a, b, 1]])

# Simulation settings
tmax = 1200
record_step = 10

def test_psd2_model():
    np.random.seed(0)
    model = PSD2Model(r, C, tmax=tmax, record_step=record_step,
                      seed=42, dispersal_type='propagule')
    
    # small random perturbation around the equilibrium
    model.logB += 0.01 * (np.random.rand(*model.logB.shape) - 0.5)
    
    t, traj, *_ = model.run()
    mean_ts = traj.mean(axis=(2,3))
    plt.figure()
    for i in range(3):
        plt.plot(t, mean_ts[:,i], label=f'sp {i}')
    plt.legend(); plt.title('PSD2 mean biomasses'); plt.show()
    print("PSD2 dominant period:", dominant_period(t, mean_ts[:,0]))
    return t, traj

def test_ibm_model():
    np.random.seed(1)
    model = IBMModel(r, C, nsteps=tmax, record_step=record_step,
                     seed=1, dispersal_type='propagule')
    
    traj = model.run()
    times = np.arange(record_step, tmax+1, record_step)
    mean_ts = traj.mean(axis=(2,3))
    plt.figure()
    for i in range(3):
        plt.plot(times, mean_ts[:,i], label=f'sp {i}')
    plt.legend(); plt.title('IBM mean biomasses'); plt.show()
    print("IBM dominant period:", dominant_period(times, mean_ts[:,0]))
    return times, traj

def animate_spatial(traj, title='', filename=None, fps=30):
    """
    Create an animation showing spatial patterns over time
    """
    n_t, S, ny, nx = traj.shape
    
    # Limit figure width to avoid FFmpeg issues
    max_width = 1920  # Standard HD width
    
    # Determine how many species per row based on max width
    species_per_row = min(8, S)  # Maximum 8 species per row
    rows = int(np.ceil(S / species_per_row))
    cols = min(S, species_per_row)
    
    # Create appropriate sized figure (limit size)
    fig_width = min(max_width / 100, cols * 4)  # 4 inches per subplot width max
    fig_height = rows * 3  # 3 inches per subplot height
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if S == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Hide unused subplots
    for i in range(S, len(axes)):
        axes[i].set_visible(False)
    
    # Set up initial frame/ – guarantee vmax > 0 to avoid divide-by-zero
    vmax = [max(np.max(traj[:, i]), 1e-12) for i in range(S)]
    for i in range(S):
        im = axes[i].imshow(traj[0,i], vmin=0, vmax=vmax[i] or 1)
        axes[i].set_title(f'Species {i}')
        axes[i].axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.suptitle(title, y=1.02)

    def update(frame):
        artists = []
        for i in range(S):
            im = axes[i].images[0]
            im.set_data(traj[frame,i])
            artists.append(im)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_t,
                                  blit=True, interval=100, repeat_delay=500)
    plt.suptitle(title)

    if filename:
        if filename.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            ani.save(filename, writer=Writer(fps=fps))
        elif filename.endswith('.gif'):
            ani.save(filename, writer='imagemagick', fps=fps)
        else:
            with open(filename, 'w') as f:
                f.write(ani.to_jshtml())
        print("Saved animation to", filename)
    else:
        plt.show()

    plt.close(fig)
    return ani

if __name__ == '__main__':
    t_psd, traj_psd = test_psd2_model()
    animate_spatial(traj_psd, 'PSD2 Spatial RPS', filename='psd2_rps.mp4')

    t_ibm, traj_ibm = test_ibm_model()
    animate_spatial(traj_ibm, 'IBM Spatial RPS', filename='ibm_rps.mp4')
