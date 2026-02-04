"""
Create a visualization of a linear layer showing matrix shapes.
Shows: x (input) on left, then h = Wx + b
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 3.5))

# Colors
input_color = '#4A90D9'
weight_color = '#E8E8E8'
output_color = '#50C878'
bias_color = '#FFB347'

# Dimensions (project UP: 4 -> 6)
d_in = 4   # input dimension
d_out = 6  # output dimension

# Scale for rectangles
scale = 0.25
y_center = 1.5

# Positions
pos = 0.3

# Draw input vector x alone on the left
x_width = scale
x_height = d_in * scale
rect_x = patches.FancyBboxPatch((pos, y_center - x_height/2), x_width, x_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor=input_color, edgecolor='black', linewidth=1.5)
ax.add_patch(rect_x)
ax.text(pos + x_width/2, y_center + x_height/2 + 0.2, r'$\mathbf{x}$', ha='center', fontsize=14, fontweight='bold')
ax.text(pos + x_width/2, y_center - x_height/2 - 0.25, f'{d_in}', ha='center', fontsize=11, color='#666')
ax.text(pos + x_width/2, y_center - x_height/2 - 0.55, 'input', ha='center', fontsize=10, color='#888', style='italic')
pos += x_width + 0.5

# Arrow
ax.annotate('', xy=(pos + 0.3, y_center), xytext=(pos, y_center),
            arrowprops=dict(arrowstyle='->', color='#666', lw=2))
pos += 0.6

# Draw weight matrix W (d_out x d_in)
w_width = d_in * scale
w_height = d_out * scale
rect_w = patches.FancyBboxPatch((pos, y_center - w_height/2), w_width, w_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor=weight_color, edgecolor='black', linewidth=1.5)
ax.add_patch(rect_w)
ax.text(pos + w_width/2, y_center + w_height/2 + 0.2, r'$\mathbf{W}$', ha='center', fontsize=14, fontweight='bold')
ax.text(pos + w_width/2, y_center - w_height/2 - 0.25, f'{d_out}×{d_in}', ha='center', fontsize=11, color='#666')
pos += w_width + 0.15

# Draw input vector x again (in the multiplication)
rect_x2 = patches.FancyBboxPatch((pos, y_center - x_height/2), x_width, x_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor=input_color, edgecolor='black', linewidth=1.5)
ax.add_patch(rect_x2)
ax.text(pos + x_width/2, y_center + x_height/2 + 0.2, r'$\mathbf{x}$', ha='center', fontsize=14, fontweight='bold')
pos += x_width + 0.3

# Plus symbol
ax.text(pos, y_center, r'$+$', ha='center', va='center', fontsize=18)
pos += 0.3

# Draw bias vector b (d_out x 1)
b_width = scale
b_height = d_out * scale
rect_b = patches.FancyBboxPatch((pos, y_center - b_height/2), b_width, b_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor=bias_color, edgecolor='black', linewidth=1.5)
ax.add_patch(rect_b)
ax.text(pos + b_width/2, y_center + b_height/2 + 0.2, r'$\mathbf{b}$', ha='center', fontsize=14, fontweight='bold')
ax.text(pos + b_width/2, y_center - b_height/2 - 0.25, f'{d_out}', ha='center', fontsize=11, color='#666')
pos += b_width + 0.3

# Equals symbol
ax.text(pos, y_center, r'$=$', ha='center', va='center', fontsize=18)
pos += 0.3

# Draw output vector h (d_out x 1)
h_width = scale
h_height = d_out * scale
rect_h = patches.FancyBboxPatch((pos, y_center - h_height/2), h_width, h_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor=output_color, edgecolor='black', linewidth=1.5)
ax.add_patch(rect_h)
ax.text(pos + h_width/2, y_center + h_height/2 + 0.2, r'$\mathbf{h}$', ha='center', fontsize=14, fontweight='bold')
ax.text(pos + h_width/2, y_center - h_height/2 - 0.25, f'{d_out}', ha='center', fontsize=11, color='#666')
ax.text(pos + h_width/2, y_center - h_height/2 - 0.55, 'output', ha='center', fontsize=10, color='#888', style='italic')

# Styling
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 3)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/smsharma/Projects/ds595-ai4science/slides/figures/05-neural-networks/linear_layer.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved linear_layer.png")
