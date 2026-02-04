"""
Figure showing ReLU activation function for neural networks lecture.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

# Figure settings
SINGLE_COL = 3.25

# Color palette
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'gray': '#BBBBBB',
}

# Create figure
fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.8, SINGLE_COL * 0.6))

# ReLU function
x = np.linspace(-3, 3, 200)
y = np.maximum(0, x)

# Plot
ax.plot(x, y, color=COLORS['blue'], linewidth=2)

# Add axis lines through origin
ax.axhline(y=0, color=COLORS['gray'], linewidth=0.5, zorder=0)
ax.axvline(x=0, color=COLORS['gray'], linewidth=0.5, zorder=0)

# Labels
ax.set_xlabel('$x$')
ax.set_ylabel(r'$\mathrm{ReLU}(x) = \max(0, x)$')

# Clean up - keep all spines visible
ax.set_xlim(-3, 3)
ax.set_ylim(-0.5, 3)

# Save
output_path = Path(__file__).parent / "relu_activation.png"
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved {output_path}")
