"""
Figure showing radial basis function expansion of distances.
Two panels: (a) the basis functions, (b) how a distance becomes a vector.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
    'purple': '#AA3377',
}

# Parameters
centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Gaussian centers (Angstroms)
gamma = 1.5  # Width parameter
d_example = 2.7  # Example distance to highlight

# Distance range
d = np.linspace(0, 6, 200)

# Compute basis functions
def rbf(d, mu, gamma):
    return np.exp(-gamma * (d - mu)**2)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0), width_ratios=[1.5, 1])

# === Panel (a): Basis functions ===
ax = axes[0]
colors_list = [COLORS['blue'], COLORS['orange'], COLORS['teal'], COLORS['red'], COLORS['purple']]

for i, (mu, color) in enumerate(zip(centers, colors_list)):
    y = rbf(d, mu, gamma)
    ax.plot(d, y, color=color, linewidth=1.5)
    # Label at peak
    ax.text(mu, 1.05, f'$e_{i+1}$', ha='center', fontsize=8, color=color)

# Vertical line at example distance
ax.axvline(d_example, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
ax.text(d_example, -0.15, f'$d = {d_example}$', ha='center', fontsize=8)

# Mark intersections
for mu, color in zip(centers, colors_list):
    activation = rbf(d_example, mu, gamma)
    ax.plot(d_example, activation, 'o', color=color, markersize=5, zorder=5)

ax.set_xlabel('Distance $d$ (Å)')
ax.set_ylabel('Basis function value')
ax.set_xlim(0, 6)
ax.set_ylim(-0.05, 1.15)
ax.set_title('(a) Radial basis functions', fontsize=9)

# === Panel (b): The resulting encoding ===
ax = axes[1]

activations = [rbf(d_example, mu, gamma) for mu in centers]
bars = ax.bar(range(1, len(centers)+1), activations, color=colors_list, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Basis function index $k$')
ax.set_ylabel('$e_k(d)$')
ax.set_xticks(range(1, len(centers)+1))
ax.set_ylim(0, 1.1)
ax.set_title(f'(b) Encoding of $d = {d_example}$ Å', fontsize=9)

# Add arrow annotation
ax.annotate('', xy=(0.5, 0.5), xytext=(-0.3, 0.5),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

plt.tight_layout()
fig.savefig('radial_basis_functions.pdf', bbox_inches='tight')
plt.close()

print("Saved radial_basis_functions.pdf")
