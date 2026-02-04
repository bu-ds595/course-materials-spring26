"""
Figure showing random noise vs a real image to illustrate the manifold hypothesis.
Most of image space is noise; real images are a vanishingly small subset.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

# Try to load publication style if available
try:
    plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())
except:
    pass

# Figure settings
SINGLE_COL = 3.5
DOUBLE_COL = 7.0

# Load and process the cat image
cat_path = Path("~/Downloads/PXL_20260201_020422988.jpg").expanduser()
cat_img = Image.open(cat_path)

# Crop to square (center crop)
w, h = cat_img.size
size = min(w, h)
left = (w - size) // 2
top = (h - size) // 2
cat_img = cat_img.crop((left, top, left + size, top + size))

# Resize to 64x64 to match the slide's example
cat_img = cat_img.resize((64, 64), Image.Resampling.LANCZOS)
cat_arr = np.array(cat_img)

# Generate random noise with same shape
np.random.seed(42)
noise_arr = np.random.randint(0, 256, size=cat_arr.shape, dtype=np.uint8)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.75))

# Random noise
ax = axes[0]
ax.imshow(noise_arr)
ax.set_title("Random pixels", fontsize=10, fontweight='medium')
ax.set_xlabel("Almost all of image space", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])

# Real image
ax = axes[1]
ax.imshow(cat_arr)
ax.set_title("Real image", fontsize=10, fontweight='medium')
ax.set_xlabel("A vanishingly rare configuration", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "noise_vs_real.png"
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved {output_path}")
