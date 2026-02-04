"""
Figure showing MNIST digits centered vs shifted.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to load publication style if available
try:
    plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())
except:
    pass

SINGLE_COL = 3.25

np.random.seed(42)

# Create synthetic MNIST-like digit (a "7")
def create_digit_7(size=28):
    img = np.zeros((size, size))
    # Horizontal stroke at top
    img[6:8, 8:20] = 1.0
    # Diagonal stroke
    for i in range(12):
        img[8+i, 18-i:20-i] = 1.0
    # Add some noise/blur effect
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=0.8)
    return img

# Create the digit
digit = create_digit_7()

# Create shifted version
shift_amount = 5
digit_shifted = np.zeros_like(digit)
digit_shifted[:, shift_amount:] = digit[:, :-shift_amount]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.6))

# Original centered
axes[0].imshow(digit, cmap='gray_r', vmin=0, vmax=1)
axes[0].set_title('Centered', fontsize=9)
axes[0].axis('off')

# Shifted
axes[1].imshow(digit_shifted, cmap='gray_r', vmin=0, vmax=1)
axes[1].set_title('Shifted 5px', fontsize=9)
axes[1].axis('off')

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "mnist_shift.png"
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved {output_path}")
