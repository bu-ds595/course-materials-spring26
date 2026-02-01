"""
Two-panel figure showing:
(a) Data manifold in ambient space
(b) Projection to summary statistics - multiple points mapping to same summary
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
    'magenta': '#EE3377',
    'gray': '#BBBBBB',
    'purple': '#AA3377',
}

# Create figure with two panels
fig = plt.figure(figsize=(DOUBLE_COL, 3.2))

# Panel (a): Manifold in ambient space
ax1 = fig.add_subplot(121, projection='3d')

# Create manifold surface
u = np.linspace(-2, 2, 40)
v = np.linspace(-2, 2, 40)
U, V = np.meshgrid(u, v)
X = U
Y = V
Z = 0.3 * (U**2 - V**2) + 0.2 * np.sin(U * 1.5)

# Plot surface
ax1.plot_surface(X, Y, Z, alpha=0.25, color=COLORS['blue'],
                 edgecolor='none', antialiased=True)
ax1.plot_wireframe(X, Y, Z, alpha=0.1, color=COLORS['blue'],
                   linewidth=0.3, rstride=5, cstride=5)

# Generate points in GROUPS (same color = same summary)
np.random.seed(123)

# Group 1: teal - these will map to same summary
g1_u = np.array([-1.2, -0.8, -1.0, -0.6])
g1_v = np.array([0.5, 0.8, 0.2, 0.6])
g1_x, g1_y = g1_u, g1_v
g1_z = 0.3 * (g1_u**2 - g1_v**2) + 0.2 * np.sin(g1_u * 1.5)

# Group 2: orange - these will map to same summary
g2_u = np.array([0.8, 1.2, 1.0, 0.6])
g2_v = np.array([-0.5, -0.2, -0.8, -0.4])
g2_x, g2_y = g2_u, g2_v
g2_z = 0.3 * (g2_u**2 - g2_v**2) + 0.2 * np.sin(g2_u * 1.5)

# Group 3: purple - these will map to same summary
g3_u = np.array([0.0, 0.3, -0.2, 0.1])
g3_v = np.array([1.2, 1.0, 1.4, 0.9])
g3_x, g3_y = g3_u, g3_v
g3_z = 0.3 * (g3_u**2 - g3_v**2) + 0.2 * np.sin(g3_u * 1.5)

# Plot groups with different colors
ax1.scatter(g1_x, g1_y, g1_z, c=COLORS['teal'], s=50, edgecolors='white', linewidths=0.5, zorder=10)
ax1.scatter(g2_x, g2_y, g2_z, c=COLORS['orange'], s=50, edgecolors='white', linewidths=0.5, zorder=10)
ax1.scatter(g3_x, g3_y, g3_z, c=COLORS['purple'], s=50, edgecolors='white', linewidths=0.5, zorder=10)

# Settings for panel a
ax1.set_xlim([-2.5, 2.5])
ax1.set_ylim([-2.5, 2.5])
ax1.set_zlim([-1.5, 1.5])
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.view_init(elev=25, azim=-55)

# Labels
ax1.text2D(0.5, 0.95, "(a) Data manifold", transform=ax1.transAxes,
           fontsize=10, ha='center', fontweight='medium')
ax1.text(0, -3.2, 1.5, "Ambient space", fontsize=9, ha='center', color='gray')

# Make panes transparent
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('lightgray')
ax1.yaxis.pane.set_edgecolor('lightgray')
ax1.zaxis.pane.set_edgecolor('lightgray')
ax1.grid(True, alpha=0.15)

# Panel (b): Projection to summary space
ax2 = fig.add_subplot(122)

# Draw a schematic manifold (curved band) at top
theta = np.linspace(0, np.pi, 100)
manifold_x = np.linspace(0.1, 0.9, 100)
manifold_y_top = 0.85 + 0.08 * np.sin(3 * theta)
manifold_y_bot = 0.75 + 0.08 * np.sin(3 * theta)

ax2.fill_between(manifold_x, manifold_y_bot, manifold_y_top,
                  color=COLORS['blue'], alpha=0.3, edgecolor=COLORS['blue'], linewidth=1)

# Points on manifold (same groups as panel a)
# Group 1 positions on schematic manifold
g1_mx = np.array([0.15, 0.22, 0.28, 0.35])
g1_my = 0.80 + 0.08 * np.sin(3 * np.pi * g1_mx)

# Group 2 positions
g2_mx = np.array([0.55, 0.62, 0.68, 0.75])
g2_my = 0.80 + 0.08 * np.sin(3 * np.pi * g2_mx)

# Group 3 positions
g3_mx = np.array([0.40, 0.45, 0.48, 0.52])
g3_my = 0.80 + 0.08 * np.sin(3 * np.pi * g3_mx)

# Plot points on manifold
ax2.scatter(g1_mx, g1_my, c=COLORS['teal'], s=60, edgecolors='white', linewidths=0.5, zorder=10)
ax2.scatter(g2_mx, g2_my, c=COLORS['orange'], s=60, edgecolors='white', linewidths=0.5, zorder=10)
ax2.scatter(g3_mx, g3_my, c=COLORS['purple'], s=60, edgecolors='white', linewidths=0.5, zorder=10)

# Summary statistic line at bottom
summary_y = 0.2
ax2.plot([0.1, 0.9], [summary_y, summary_y], 'k-', linewidth=2)
ax2.text(0.5, 0.08, "Summary statistic space", fontsize=9, ha='center', color='gray')

# Draw projection arrows and points
# All points in a group map to ONE summary point

# Group 1 -> single summary point
s1_x = 0.25
for mx, my in zip(g1_mx, g1_my):
    ax2.annotate('', xy=(s1_x, summary_y + 0.03), xytext=(mx, my - 0.02),
                 arrowprops=dict(arrowstyle='->', color=COLORS['teal'], alpha=0.5, lw=1))
ax2.scatter([s1_x], [summary_y], c=COLORS['teal'], s=80, edgecolors='white', linewidths=1, zorder=10)

# Group 2 -> single summary point
s2_x = 0.65
for mx, my in zip(g2_mx, g2_my):
    ax2.annotate('', xy=(s2_x, summary_y + 0.03), xytext=(mx, my - 0.02),
                 arrowprops=dict(arrowstyle='->', color=COLORS['orange'], alpha=0.5, lw=1))
ax2.scatter([s2_x], [summary_y], c=COLORS['orange'], s=80, edgecolors='white', linewidths=1, zorder=10)

# Group 3 -> single summary point
s3_x = 0.46
for mx, my in zip(g3_mx, g3_my):
    ax2.annotate('', xy=(s3_x, summary_y + 0.03), xytext=(mx, my - 0.02),
                 arrowprops=dict(arrowstyle='->', color=COLORS['purple'], alpha=0.5, lw=1))
ax2.scatter([s3_x], [summary_y], c=COLORS['purple'], s=80, edgecolors='white', linewidths=1, zorder=10)

# Labels
ax2.text(0.5, 0.98, "(b) Projection to summaries", fontsize=10, ha='center',
         fontweight='medium', transform=ax2.transAxes)

ax2.text(0.92, 0.80, "Data\nmanifold", fontsize=8, ha='left', va='center', color=COLORS['blue'])

# Key insight annotation
ax2.annotate("Same summary,\ndifferent data",
             xy=(s1_x, summary_y), xytext=(0.12, 0.45),
             fontsize=8, ha='center', color=COLORS['teal'],
             arrowprops=dict(arrowstyle='->', color=COLORS['teal'], alpha=0.7, lw=0.8))

# Clean up panel b
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.axis('off')

plt.tight_layout()

# Save
save_path = Path("/Users/smsharma/Projects/ds595-ai4science/notes/chapters/03-neural-networks/figures/manifold_and_summaries.pdf")
fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
fig.savefig(save_path.with_suffix('.png'), dpi=200, bbox_inches='tight', pad_inches=0.05)
print(f"Saved to {save_path}")

plt.show()
