############################################
# run_rps_dynamics.py
############################################
"""
Visualization module for ecological simulations.
Optimized for high-performance direct-to-ffmpeg video encoding.
Features:
- Direct RGB piping (Viridis colormap)
- Hardware acceleration (VideoToolbox/NVENC)
- FFmpeg-side upscaling
- Centered White Padding
- Smart TV Layout (Adapts rows to field geometry Nx/Ny)
"""

import sys
import shutil
import subprocess
import platform
import logging
import numpy as std_np  # Standard numpy for CPU image ops
from accelerator import to_cpu

import matplotlib.cm

try:
    from config import NUM_PATCHES_X, NUM_PATCHES_Y, BODY_MASS
except ImportError:
    NUM_PATCHES_X, NUM_PATCHES_Y = 50, 50
    BODY_MASS = 1.0

logger = logging.getLogger(__name__)

# --- LAYOUT CONFIGURATION ---
MAX_COLS = 10
TARGET_ASPECT = 16 / 9  # Standard TV Aspect Ratio

def get_ffmpeg_encoder_args():
    """Detects the best available hardware encoder."""
    system = platform.system()
    
    if system == 'Darwin':
        try:
            res = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
            if 'h264_videotoolbox' in res.stdout:
                return ['-c:v', 'h264_videotoolbox', '-b:v', '40M', '-allow_sw', '1']
        except Exception:
            pass
            
    # return ['-c:v', 'h264_nvenc', '-preset', 'fast', '-b:v', '20M']
    return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p']

def tile_images(data_t, cols=8, padding=2, pad_value=std_np.nan):
    """
    Tiles (S, H, W) -> Single Grid.
    Centers the fields by splitting 'padding' evenly.
    """
    S, H, W = data_t.shape
    rows = int(std_np.ceil(S / cols))
    
    # Pad species dimension
    pad_count = (rows * cols) - S
    if pad_count > 0:
        extra = std_np.full((pad_count, H, W), pad_value, dtype=data_t.dtype)
        data_t = std_np.concatenate([data_t, extra], axis=0)
        
    # Spatial Padding (Centered)
    if padding > 0:
        p_half = padding // 2
        p_rest = padding - p_half
        data_t = std_np.pad(data_t, 
                            ((0,0), (p_half, p_rest), (p_half, p_rest)), 
                            mode='constant', constant_values=pad_value)
        H += padding
        W += padding

    # Reshape to grid
    grid = data_t.reshape(rows, cols, H, W)
    grid = grid.transpose(0, 2, 1, 3) 
    grid = grid.reshape(rows * H, cols * W)
    
    return grid

def animate_spatial(traj, title='', filename=None, fps=30, padding=2):
    """
    Generates a 4K colored movie.
    Dynamically calculates layout rows to fit 16:9 TV aspect ratio
    based on the shape (Nx, Ny) of the species fields.
    """
    if filename is None: return
    if shutil.which("ffmpeg") is None:
        logger.error("FFmpeg not found.")
        return

    traj = to_cpu(traj)
    if hasattr(traj, "numpy"): traj = traj.numpy()
        
    T, S, H, W = traj.shape
    
    # --- DYNAMIC TV LAYOUT ---
    # We want (Cols * Width) / (Rows * Height) ~= 1.77
    # Solve for Rows: Rows = (Cols * Width) / (Height * 1.77)
    
    eff_w = W + padding
    eff_h = H + padding
    
    calc_rows = int(std_np.round((MAX_COLS * eff_w) / (eff_h * TARGET_ASPECT)))
    calc_rows = max(1, calc_rows) # Ensure at least 1 row
    
    max_display = MAX_COLS * calc_rows
    
    if S > max_display:
        logger.warning(f"Too many species ({S}). Truncating to {max_display} to fit TV layout (based on {W}x{H} fields).")
        traj = traj[:, :max_display]
        S = max_display
        
    cols = min(MAX_COLS, S)
    rows = int(std_np.ceil(S / cols))
    
    # Input dimensions
    in_w = cols * eff_w
    in_h = rows * eff_h
    
    # Sanitize & Normalize
    traj = std_np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
    v_max = 1 # std_np.percentile(traj, 99.9) if traj.max() > 0 else 1.0
    if v_max == 0: v_max = 1.0
    
    cmap = matplotlib.colormaps['viridis']
    
    logger.info(f"Rendering {filename}")
    logger.info(f"Geometry: {W}x{H} fields. Layout: {rows}x{cols}. Upscaling to 4K.")

    encoder_args = get_ffmpeg_encoder_args()
    
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{in_w}x{in_h}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vf', 'scale=3840:-2:flags=neighbor,format=yuv420p',
        *encoder_args,
        filename
    ]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        for t in range(T):
            grid = tile_images(traj[t], cols=cols, padding=padding, pad_value=std_np.nan)
            
            is_pad = std_np.isnan(grid)
            
            grid_safe = std_np.nan_to_num(grid, nan=0.0)
            grid_norm = std_np.clip(grid_safe / v_max, 0.0, 1.0)
            
            rgba = cmap(grid_norm) 
            frame_rgb = (rgba[..., :3] * 255).astype(std_np.uint8)
            frame_rgb[is_pad] = [255, 255, 255]
            
            process.stdin.write(frame_rgb.tobytes())
            
            if t % 100 == 0:
                logger.debug(f"Frame {t}/{T}")

        process.stdin.close()
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg Error: {process.stderr.read().decode('utf-8')}")
        else:
            logger.info(f"Saved {filename}")
            
    except Exception as e:
        logger.error(f"Movie gen failed: {e}")
        try: process.kill()
        except: pass

if __name__ == '__main__':
    # Test 1: Wide strips (should allow many rows)
    print("Test 1: Wide fields (100x10)")
    data_wide = std_np.random.rand(30, 80, 10, 100).astype(std_np.float32)
    animate_spatial(data_wide, filename="test_wide.mp4", fps=30)
    
    # Test 2: Tall strips (should limit rows)
    print("Test 2: Tall fields (10x100)")
    data_tall = std_np.random.rand(30, 20, 100, 10).astype(std_np.float32)
    animate_spatial(data_tall, filename="test_tall.mp4", fps=30)
