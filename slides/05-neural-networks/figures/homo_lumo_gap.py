"""
Create a simple illustration of the HOMO-LUMO gap concept.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(6, 5))

# Energy levels
levels = {
    'LUMO+1': 4.0,
    'LUMO': 2.5,
    'HOMO': -1.5,
    'HOMO-1': -3.0,
}

# Colors
occupied_color = '#4A90D9'  # Blue for occupied
unoccupied_color = '#E0E0E0'  # Gray for unoccupied
gap_color = '#E04040'  # Red for gap

# Draw energy levels
level_width = 0.6
x_center = 0

for name, energy in levels.items():
    is_occupied = 'HOMO' in name and 'LUMO' not in name
    color = occupied_color if is_occupied else unoccupied_color
    lw = 3 if name in ['HOMO', 'LUMO'] else 2

    ax.hlines(energy, x_center - level_width/2, x_center + level_width/2,
              colors=color, linewidths=lw)

    # Label on the right
    ax.text(x_center + level_width/2 + 0.1, energy, name,
            va='center', ha='left', fontsize=12,
            fontweight='bold' if name in ['HOMO', 'LUMO'] else 'normal')

# Draw electrons (arrows) in occupied orbitals
arrow_props = dict(head_width=0.08, head_length=0.15, fc=occupied_color, ec=occupied_color)
for name in ['HOMO', 'HOMO-1']:
    energy = levels[name]
    # Spin up
    ax.arrow(x_center - 0.12, energy - 0.3, 0, 0.5, **arrow_props)
    # Spin down
    ax.arrow(x_center + 0.12, energy + 0.2, 0, -0.5, **arrow_props)

# Draw the gap
gap_x = x_center - level_width/2 - 0.3
ax.annotate('', xy=(gap_x, levels['LUMO']), xytext=(gap_x, levels['HOMO']),
            arrowprops=dict(arrowstyle='<->', color=gap_color, lw=2))
ax.text(gap_x - 0.15, (levels['HOMO'] + levels['LUMO'])/2,
        'Gap\n$\\Delta\\epsilon$', ha='right', va='center',
        fontsize=12, fontweight='bold', color=gap_color)

# Styling
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-4.5, 5.5)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add labels for occupied/unoccupied regions
ax.text(x_center, -4.2, 'Occupied', ha='center', fontsize=10, color=occupied_color)
ax.text(x_center, 5.0, 'Unoccupied', ha='center', fontsize=10, color='#808080')

ax.axhline(0.5, color='#CCCCCC', linestyle='--', linewidth=1, zorder=0)

plt.tight_layout()
plt.savefig('/Users/smsharma/Projects/ds595-ai4science/slides/figures/05-neural-networks/homo_lumo_gap.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved homo_lumo_gap.png")
