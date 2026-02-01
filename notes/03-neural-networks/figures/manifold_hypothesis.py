"""
Manifold hypothesis illustration for lecture notes.
Shows a 2D data manifold embedded in 3D ambient space.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

# Column widths
DOUBLE_COL = 6.75

# Colors
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
    'gray': '#BBBBBB',
}

# Create figure
fig = plt.figure(figsize=(DOUBLE_COL, 4.0))
ax = fig.add_subplot(111, projection='3d')

# Create a curved manifold surface (saddle/wave shape)
u = np.linspace(-2, 2, 50)
v = np.linspace(-2, 2, 50)
U, V = np.meshgrid(u, v)

# Saddle-like surface with some curvature
X = U
Y = V
Z = 0.3 * (U**2 - V**2) + 0.2 * np.sin(U * 1.5)

# Plot the manifold surface
surf = ax.plot_surface(X, Y, Z, alpha=0.3, color=COLORS['blue'],
                        edgecolor='none', antialiased=True)

# Add wireframe for visibility
ax.plot_wireframe(X, Y, Z, alpha=0.15, color=COLORS['blue'],
                  linewidth=0.3, rstride=5, cstride=5)

# Generate points ON the manifold
np.random.seed(42)
n_points = 25
u_pts = np.random.uniform(-1.5, 1.5, n_points)
v_pts = np.random.uniform(-1.5, 1.5, n_points)
x_pts = u_pts
y_pts = v_pts
z_pts = 0.3 * (u_pts**2 - v_pts**2) + 0.2 * np.sin(u_pts * 1.5)

# Plot points on manifold
ax.scatter(x_pts, y_pts, z_pts, c=COLORS['teal'], s=40,
           edgecolors='white', linewidths=0.5, zorder=10,
           label='Observed data')

# Generate points OFF the manifold (physically implausible)
n_off = 8
x_off = np.random.uniform(-1.5, 1.5, n_off)
y_off = np.random.uniform(-1.5, 1.5, n_off)
z_manifold = 0.3 * (x_off**2 - y_off**2) + 0.2 * np.sin(x_off * 1.5)
z_off = z_manifold + np.random.uniform(0.8, 1.5, n_off) * np.random.choice([-1, 1], n_off)

# Plot off-manifold points
ax.scatter(x_off, y_off, z_off, c=COLORS['gray'], s=25,
           alpha=0.5, marker='x', linewidths=1.5,
           label='Implausible')

# Set axis properties
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-2, 2])

# Remove axis ticks but keep a minimal frame
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set viewing angle
ax.view_init(elev=20, azim=-60)

# Add annotations with larger fonts
# "Data manifold" label
ax.text(1.8, -2.2, -0.5, "Data manifold\n(low-dimensional)",
        fontsize=10, color=COLORS['blue'], ha='center',
        fontweight='medium')

# "Ambient space" label
ax.text(-2.5, 2.5, 2.0, "Ambient space",
        fontsize=11, color='black', ha='left',
        fontweight='medium')

# Arrow-like annotation for manifold points
ax.text(0, -2.8, 0.8, "Physically plausible\nconfigurations",
        fontsize=9, color=COLORS['teal'], ha='center',
        fontweight='medium')

# Label for off-manifold points
ax.text(1.5, 1.5, 1.8, "Implausible",
        fontsize=8, color=COLORS['gray'], ha='center',
        style='italic')

# Make panes more transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')

# Grid
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# Adjust layout
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Save
save_path = Path("/Users/smsharma/Projects/ds595-ai4science/notes/chapters/03-neural-networks/figures/manifold_hypothesis.pdf")
fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
fig.savefig(save_path.with_suffix('.png'), dpi=200, bbox_inches='tight', pad_inches=0.1)
print(f"Saved to {save_path}")

plt.show()
