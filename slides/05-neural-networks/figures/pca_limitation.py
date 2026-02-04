"""
Figure showing PCA limitation: finds linear subspace but misses curved manifold.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to load publication style if available
try:
    plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())
except:
    pass

SINGLE_COL = 3.5
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'gray': '#BBBBBB',
    'red': '#CC3311',
}

np.random.seed(42)

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 1.6, SINGLE_COL * 0.8))

# Generate data on a curved manifold (swiss roll-like in 2D projection)
t = np.linspace(0, 1.5 * np.pi, 200)
noise = 0.08

# Curved manifold data
x_curve = t * np.cos(t) + np.random.randn(len(t)) * noise
y_curve = t * np.sin(t) + np.random.randn(len(t)) * noise

# Panel 1: Data with curved manifold
ax = axes[0]
ax.scatter(x_curve, y_curve, c=t, cmap='viridis', s=15, alpha=0.8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Data on curved manifold', fontsize=10)
ax.set_aspect('equal')
ax.set_xlim(-5.5, 3)
ax.set_ylim(-3, 5)

# Panel 2: PCA projection
ax = axes[1]
ax.scatter(x_curve, y_curve, c=t, cmap='viridis', s=15, alpha=0.4)

# Compute PCA
data = np.column_stack([x_curve, y_curve])
mean = data.mean(axis=0)
centered = data - mean
cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sort by eigenvalue descending
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# Draw first principal component line
pc1 = eigenvectors[:, 0]
line_extent = 5
ax.plot([mean[0] - pc1[0]*line_extent, mean[0] + pc1[0]*line_extent],
        [mean[1] - pc1[1]*line_extent, mean[1] + pc1[1]*line_extent],
        'k-', linewidth=2, label='PC1')

# Project points onto PC1
projections = centered @ pc1.reshape(-1, 1) * pc1.reshape(1, -1) + mean
ax.scatter(projections[:, 0], projections[:, 1], c=COLORS['red'], s=10, alpha=0.6, zorder=5)

# Draw projection lines for a subset
for i in range(0, len(x_curve), 15):
    ax.plot([x_curve[i], projections[i, 0]],
            [y_curve[i], projections[i, 1]],
            color=COLORS['gray'], linewidth=0.5, alpha=0.5)

ax.set_xlabel('$x_1$')
ax.set_title('PCA projection (linear)', fontsize=10)
ax.set_aspect('equal')
ax.set_xlim(-5.5, 3)
ax.set_ylim(-3, 5)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "pca_limitation.png"
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved {output_path}")
