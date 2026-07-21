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
parser.add_argument("--dpi", type=int, default=100,
                    help="Figure DPI (default: 100)")
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
        # Only update if we have points
        if end > start:
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
# Use matplotlib's built-in ffmpeg writer with pipe-based output.
# This is MUCH faster than writing individual PNG files to disk.
print(f"  Encoding video with NVENC ...", file=sys.stderr)

# Try to use ffmpeg with NVENC via matplotlib's writer
import matplotlib.animation as animation

video_written = False

# Convert bitrate string to int (matplotlib expects int, not string like "8M")
bitrate_int = int(args.bitrate.replace('M', '000').replace('k', '000').replace('K', '000'))

# Try NVENC first (GPU-accelerated)
try:
    Writer = animation.writers['ffmpeg']
    writer = Writer(
        fps=args.fps,
        codec='h264_nvenc',
        bitrate=bitrate_int,
        extra_args=[
            '-preset', 'p4',
            '-rc', 'vbr',
            '-b:v', args.bitrate,
            '-maxrate', args.bitrate,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]
    )
    
    anim.save(args.output, writer=writer, dpi=args.dpi)
    video_written = True
    print(f"  ✓ Used NVENC hardware encoding", file=sys.stderr)
except Exception as e:
    print(f"  NVENC failed: {str(e)[:100]}, trying libx264...", file=sys.stderr)

# Fallback to software encoding
if not video_written:
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(
            fps=args.fps,
            codec='libx264',
            bitrate=bitrate_int,
            extra_args=[
                '-preset', 'ultrafast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
        )
        
        anim.save(args.output, writer=writer, dpi=args.dpi)
        video_written = True
        print(f"  ✓ Used libx264 software encoding", file=sys.stderr)
    except Exception as e:
        print(f"  libx264 failed: {str(e)[:100]}", file=sys.stderr)

if not video_written:
    print("ERROR: All codecs failed", file=sys.stderr)
    sys.exit(1)

print(f"Done! Saved to '{args.output}'", file=sys.stderr)