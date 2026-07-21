#!/usr/bin/env python3
"""
3-Body Simulation Visualizer

Reads CSV from stdin (format: t,x1,y1,x2,y2,x3,y3) and produces an MP4 video
using GPU-accelerated H.264 encoding (NVENC) via ffmpeg.

The trail shows the LAST N positions of each body (no connecting lines between
non-adjacent points) for a clean look.

Usage:
    ./3body figure8.txt | python3 visualize.py
    ./3body figure8.txt | python3 visualize.py --output orbit.mp4 --fps 60
"""

import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="3-Body simulation visualizer")
parser.add_argument("--output", "-o", default="3body.mp4",
                    help="Output video filename (default: 3body.mp4)")
parser.add_argument("--fps", type=int, default=60,
                    help="Frames per second (default: 60)")
parser.add_argument("--every", type=int, default=5,
                    help="Use every Nth frame from simulation (default: 5)")
parser.add_argument("--max-frames", type=int, default=1500,
                    help="Max frames to render (0 = all, default: 1500)")
parser.add_argument("--trail", type=int, default=40,
                    help="Trail length (default: 40)")
parser.add_argument("--dpi", type=int, default=150,
                    help="Figure DPI (default: 150)")
parser.add_argument("--size", type=float, default=7.0,
                    help="Figure size in inches (default: 7.0)")
parser.add_argument("--bitrate", type=str, default="8M",
                    help="Video bitrate (default: 8M)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Read CSV from stdin
# ---------------------------------------------------------------------------
lines = [line.strip() for line in sys.stdin if line.strip()]

data_lines = []
for line in lines:
    if line.startswith("#") or line.startswith("t,"):
        continue
    data_lines.append(line)

if not data_lines:
    print("ERROR: no data rows found in input", file=sys.stderr)
    sys.exit(1)

data = np.loadtxt(data_lines, delimiter=",")
if data.ndim == 1:
    data = data.reshape(1, -1)

t = data[:, 0]
x1, y1 = data[:, 1], data[:, 2]
x2, y2 = data[:, 3], data[:, 4]
x3, y3 = data[:, 5], data[:, 6]

n_frames = len(t)

# ---------------------------------------------------------------------------
# Sub-sample
# ---------------------------------------------------------------------------
indices = np.arange(0, n_frames, args.every)
if args.max_frames > 0 and len(indices) > args.max_frames:
    indices = indices[:args.max_frames]

if len(indices) < n_frames:
    t = t[indices]
    x1 = x1[indices]; y1 = y1[indices]
    x2 = x2[indices]; y2 = y2[indices]
    x3 = x3[indices]; y3 = y3[indices]
    n_frames = len(t)
    print(f"Using {n_frames} frames (every {args.every}, sub-sampled from {len(data)})",
          file=sys.stderr)

# ---------------------------------------------------------------------------
# Plot bounds
# ---------------------------------------------------------------------------
all_x = np.concatenate([x1, x2, x3])
all_y = np.concatenate([y1, y2, y3])
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()

range_ = max(x_max - x_min, y_max - y_min) * 0.5 + 0.5
x_center = (x_min + x_max) / 2.0
y_center = (y_min + y_max) / 2.0

# ---------------------------------------------------------------------------
# Set up figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(args.size, args.size), dpi=args.dpi)
ax.set_aspect("equal")
ax.set_xlim(x_center - range_, x_center + range_)
ax.set_ylim(y_center - range_, y_center + range_)
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_title("3-Body Problem")
ax.grid(True, alpha=0.3)
fig.tight_layout()

colors = ["#e74c3c", "#3498db", "#2ecc71"]

# For trails: plot dots instead of a continuous line to avoid "weird lines"
# when the orbit crosses itself.
trail_scatters = [
    ax.scatter([], [], s=6, color=colors[i], alpha=0.35, zorder=3)
    for i in range(3)
]

# Current position — large dot with a white edge for visibility
current_dots = [
    ax.plot([], [], "o", color=colors[i], markersize=10,
            markeredgecolor="white", markeredgewidth=1.5, zorder=6)[0]
    for i in range(3)
]

# Time text
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                    fontsize=11, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def init():
    for s in trail_scatters:
        s.set_offsets(np.empty((0, 2)))
    for d in current_dots:
        d.set_data([], [])
    time_text.set_text("")
    return trail_scatters + current_dots + [time_text]


def update(frame):
    start = max(0, frame - args.trail)
    end = frame + 1

    xs = [x1[start:end], x2[start:end], x3[start:end]]
    ys = [y1[start:end], y2[start:end], y3[start:end]]

    for i in range(3):
        pts = np.column_stack([xs[i], ys[i]])
        trail_scatters[i].set_offsets(pts)
        current_dots[i].set_data([xs[i][-1]], [ys[i][-1]])

    time_text.set_text(f"t = {t[frame]:.1f}   step {frame}")
    return trail_scatters + current_dots + [time_text]


print(f"Rendering {n_frames} frames to '{args.output}' ...", file=sys.stderr)

anim = FuncAnimation(fig, update, frames=n_frames,
                     init_func=init, blit=True, repeat=False)

# ---------------------------------------------------------------------------
# Save via ffmpeg with NVENC hardware encoding
# ---------------------------------------------------------------------------
# Write frames to a temporary directory as PNGs, then encode with ffmpeg.
# This is faster than anim.save for large frame counts, and lets us use
# the GPU encoder.
with tempfile.TemporaryDirectory() as tmpdir:
    pattern = os.path.join(tmpdir, "frame_%05d.png")
    print(f"  Writing frames to {tmpdir} ...", file=sys.stderr)

    # Save each frame as PNG
    for i in range(n_frames):
        update(i)
        fig.savefig(pattern % (i + 1), dpi=args.dpi,
                    facecolor="white", edgecolor="none")
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_frames} frames", file=sys.stderr)

    print(f"  Encoding video with NVENC ...", file=sys.stderr)

    # Use ffmpeg with h264_nvenc for GPU-accelerated encoding
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", pattern,
        "-c:v", "h264_nvenc",
        "-preset", "p7",          # highest quality NVENC preset
        "-rc", "vbr",
        "-b:v", args.bitrate,
        "-maxrate", args.bitrate,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        args.output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback to software encoding
        print(f"  NVENC failed ({result.stderr.strip().split(chr(10))[-1]}), "
              f"falling back to libx264 ...", file=sys.stderr)
        cmd[5] = "libx264"
        cmd[6] = "-preset"
        cmd[7] = "medium"
        subprocess.run(cmd, check=True)

print(f"Done! Saved to '{args.output}'", file=sys.stderr)