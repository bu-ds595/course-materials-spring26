"""
Flat vs sharp minima and generalization.
Shows why flat minima generalize better: the validation loss landscape
is slightly shifted, and flat minima are more robust to this shift.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75

# Create figure
fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, 2.2))

# Parameter space
theta = np.linspace(-1.5, 3.5, 500)

# Training loss: sharp minimum on left, flat minimum on right
def train_loss(x):
    # Sharp minimum at x=0 (less deep so it's visible)
    sharp = 1.5 * np.exp(-10 * x**2)
    # Flat minimum at x=2
    flat = 0.6 * np.exp(-0.6 * (x - 2)**2)
    # Background
    background = 0.8 + 0.08 * (x - 1)**2
    return background + 1.5 - sharp - flat

# Validation loss: same shape but shifted slightly to the right
shift = 0.18
def val_loss(x):
    sharp = 1.5 * np.exp(-10 * (x - shift)**2)
    flat = 0.6 * np.exp(-0.6 * (x - shift - 2)**2)
    background = 0.8 + 0.08 * (x - shift - 1)**2
    return background + 1.5 - sharp - flat

# Plot losses
ax.plot(theta, train_loss(theta), 'k-', linewidth=1.5, label='Train loss')
ax.plot(theta, val_loss(theta), 'k--', linewidth=1.5, label='Val loss')

# Mark the minima positions
sharp_min_train = 0.0
sharp_min_val = shift
flat_min_train = 2.0
flat_min_val = 2.0 + shift

# Get loss values at minima
sharp_train_val = train_loss(sharp_min_train)
sharp_val_val = val_loss(sharp_min_train)  # val loss at train minimum
flat_train_val = train_loss(flat_min_train)
flat_val_val = val_loss(flat_min_train)  # val loss at train minimum

# Draw vertical lines showing generalization gap
# Sharp minimum - large gap (offset the arrow slightly to avoid overlap)
arrow_offset = 0.12
ax.annotate('', xy=(sharp_min_train - arrow_offset, sharp_val_val),
            xytext=(sharp_min_train - arrow_offset, sharp_train_val),
            arrowprops=dict(arrowstyle='<->', color='#CC3311', lw=1.5))
ax.text(sharp_min_train - arrow_offset - 0.12, (sharp_train_val + sharp_val_val) / 2 + 0.35, 'large\ngap',
        fontsize=7, color='#CC3311', ha='right', va='center')

# Flat minimum - small gap
ax.annotate('', xy=(flat_min_train + arrow_offset, flat_val_val),
            xytext=(flat_min_train + arrow_offset, flat_train_val),
            arrowprops=dict(arrowstyle='<->', color='#009988', lw=1.5))
ax.text(flat_min_train + 0.45, flat_train_val - 0.15, 'small gap',
        fontsize=7, color='#009988', ha='left', va='top')

# Mark minima with dots
ax.plot(sharp_min_train, sharp_train_val, 'o', color='#CC3311', markersize=5, zorder=5)
ax.plot(flat_min_train, flat_train_val, 'o', color='#009988', markersize=5, zorder=5)

# Labels - position away from curves
ax.text(sharp_min_train, sharp_train_val - 0.35, 'sharp', fontsize=8, ha='center', va='top', color='#CC3311')
ax.text(flat_min_train, flat_train_val - 0.25, 'flat', fontsize=8, ha='center', va='top', color='#009988')

# Axis labels
ax.set_xlabel(r'Parameters $\theta$')
ax.set_ylabel('Loss')
ax.legend(loc='lower right', frameon=False)

# Clean up
ax.set_xlim(-1.5, 3.5)
ax.set_ylim(0.3, 2.5)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save
save_path = Path("/Users/smsharma/Projects/ds595-ai4science/notes/chapters/03-neural-networks/figures/flat_minima.pdf")
fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
fig.savefig(save_path.with_suffix('.png'), dpi=200, bbox_inches='tight', pad_inches=0.05)
print(f"Saved to {save_path}")
