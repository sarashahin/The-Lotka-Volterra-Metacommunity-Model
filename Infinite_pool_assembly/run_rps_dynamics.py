############################################
# run_rps_dynamics.py  —  fixed version
############################################

from accelerator import np, to_cpu # use to_cpu for matlibplot access!
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

# from models_psd2 import PSD2Model
from models_ibm import IBMModel

# --- global “well‑mixed’’ dispersal ---------------------------------
import config
from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS, DISPERSAL_RATE        # link all 25 patches
# -------------------------------------------------------------------

# ----------------‑‑ utility ----------------------------------------
def dominant_period(t, x):
    """Return the period of the dominant FFT peak of 1‑D signal x(t)."""
    from scipy.fft import rfftfreq, rfft
    dt = np.mean(np.diff(t))
    freqs = rfftfreq(len(t), dt)[1:]          # skip the zero mode
    spec  = np.abs(rfft(x))[1:]
    return 1.0 / freqs[np.argmax(spec)]

# ----------------‑‑ LV parameters ----------------------------------
r = np.ones(3)
C = np.eye(3)
C[C == 0] = 0.4          # off‑diagonal interaction strength

# ----------------‑‑ simulation control -----------------------------
tmax        = 20_000
record_step = 100
# -------------------------------------------------------------------

def run_psd2():
    np.random.seed(0)
    model = PSD2Model(r, C, tmax=tmax, record_step=record_step,
                      seed=42, dispersal_type='propagule')
    model.logB += 0.01 * (np.random.rand(*model.logB.shape) - 0.5)
    t, traj, *_ = model.run()

    mean_ts = traj.sum(axis=1).mean(axis=(1, 2))     # community mean biomass
    plt.figure()
    plt.plot(t, mean_ts, label='PSD2')
    plt.xlabel('time'); plt.ylabel('mean biomass'); plt.legend(); plt.title('PSD2')
    plt.show()
    print("PSD2 dominant period:", dominant_period(t, mean_ts))
    return t, traj

def run_ibm():
    np.random.seed(1)
    model = IBMModel(r, C, nsteps=tmax, record_step=record_step,
                     seed=1, dispersal_type='propagule')
    traj = model.run()
    times = np.arange(record_step, tmax + 1, record_step)

    mean_ts = traj.sum(axis=1).mean(axis=(1, 2))
    plt.figure()
    plt.plot(times, mean_ts, label='IBM', color='g')
    plt.xlabel('time'); plt.ylabel('mean biomass'); plt.legend(); plt.title('IBM')
    plt.show()
    print("IBM dominant period:", dominant_period(times, mean_ts))
    return times, traj

# ----------------‑‑ animation helper --------------------------------
def animate_spatial(traj, title='', filename=None, fps=50):
    """Animate spatial biomass maps stored as traj[t, sp, y, x]."""
    n_t, S, ny, nx = traj.shape

    if S > 72:
        S = 7*8 # larger videos break the height limit!
    
    # ---- layout: ≤8 panels per row ---------------------------------
    cols = min(8, S)
    rows = int(np.ceil(S / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 3 * rows),
                             squeeze=False)
    axes = axes.flatten()

    # hide unused panes
    for i in range(S, len(axes)):
        axes[i].set_visible(False)
        

    # 1. Sanitize the data to prevent 'NaN' from breaking the plot
    traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)

    vmax = [max(traj[:, i].max(), 1e-12) for i in range(S)]
    vmax = max(vmax)
    ims  = []
    for i in range(S):
        #im = axes[i].imshow(traj[0, i], vmin=0, vmax=vmax[i], animated=True)
        im = axes[i].imshow(traj[0, i], vmin=0, vmax=1.0, animated=True)
        axes[i].set_title(f'Species {i}')
        axes[i].axis('off')
        ims.append(im)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    def update(frame):
        for i, im in enumerate(ims):
            im.set_array(traj[frame, i])
        return ims

    ani = animation.FuncAnimation(fig, update, frames=n_t,
                                  blit=True, interval=200)

    if filename:
        try:
            if filename.endswith('.mp4'):
                Writer = animation.writers['ffmpeg']
                ani.save(filename, writer=Writer(fps=fps))
            elif filename.endswith('.gif'):
                ani.save(filename, writer='pillow', fps=fps)
            else:                       # HTML for notebooks
                with open(filename, 'w') as f:
                    f.write(ani.to_jshtml())
            print(f"Saved animation to {filename}")
        except (KeyError, RuntimeError):   # FFmpeg unavailable
            print("⚠️  FFmpeg not found – falling back to GIF via Pillow")
            ani.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=fps)
    else:
        plt.show()

    plt.close(fig)
    return ani

# ----------------‑‑ main --------------------------------------------
if __name__ == '__main__':
    t_psd, traj_psd = run_psd2()
    animate_spatial(traj_psd, 'PSD2 spatial dynamics', filename='psd2_rps.mp4')

    t_ibm, traj_ibm = run_ibm()
    animate_spatial(traj_ibm, 'IBM spatial dynamics', filename='ibm_rps.mp4')
    # animate_spatial(traj_ibm, 'IBM spatial dynamics', filename='ibm_rps.gif')
