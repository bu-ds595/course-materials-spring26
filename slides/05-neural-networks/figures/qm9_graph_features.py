"""
Create a clear QM9 graph visualization with node features table.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from torch_geometric.datasets import QM9

# Load dataset and get propanol (CCCO)
dataset = QM9(root='../../notes/chapters/03-neural-networks/data/qm9')
mol = dataset[21]  # Propanol: C3H8O

# Get heavy atoms only (C, C, C, O) for cleaner visualization
heavy_mask = mol.z > 1
heavy_idx = np.where(heavy_mask.numpy())[0]
z_heavy = mol.z[heavy_mask].numpy()
pos_heavy = mol.pos[heavy_mask].numpy()
x_heavy = mol.x[heavy_mask].numpy()

# Element info
elem_names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
elem_colors = {6: '#404040', 8: '#E04040'}  # C=dark gray, O=red

fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1.5]})

# Left: Graph structure
ax = axes[0]
ax.set_aspect('equal')

# Use 2D projection of 3D positions
pos_2d = pos_heavy[:, :2]
pos_2d = pos_2d - pos_2d.mean(axis=0)  # Center

# Draw edges (bonds between heavy atoms)
edge_index = mol.edge_index.numpy()
drawn_edges = set()
for i in range(edge_index.shape[1]):
    src, dst = edge_index[0, i], edge_index[1, i]
    if src in heavy_idx and dst in heavy_idx:
        edge_key = tuple(sorted([src, dst]))
        if edge_key not in drawn_edges:
            drawn_edges.add(edge_key)
            src_pos = np.where(heavy_idx == src)[0][0]
            dst_pos = np.where(heavy_idx == dst)[0][0]
            ax.plot([pos_2d[src_pos, 0], pos_2d[dst_pos, 0]],
                   [pos_2d[src_pos, 1], pos_2d[dst_pos, 1]],
                   'k-', linewidth=3, zorder=1)

# Draw nodes
for i, (z, pos) in enumerate(zip(z_heavy, pos_2d)):
    color = elem_colors.get(z, '#808080')
    circle = plt.Circle(pos, 0.25, color=color, zorder=2)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1], elem_names[z], ha='center', va='center',
            fontsize=14, fontweight='bold', color='white', zorder=3)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axis('off')
ax.set_title('Graph structure\n(heavy atoms only)', fontsize=12, pad=10)

# Right: Node features table
ax = axes[1]
ax.axis('off')

# Feature names (from PyG QM9 source)
feature_names = ['H', 'C', 'N', 'O', 'F', 'Z', 'arom', 'sp', 'sp²', 'sp³', 'nH']

# Build table data (heavy atoms only)
col_labels = feature_names
row_labels = [f'{i}: {elem_names[z]}' for i, z in enumerate(z_heavy)]

# Format features as integers where appropriate
cell_data = []
for row in x_heavy:
    cell_data.append([f'{int(v)}' if v == int(v) else f'{v:.1f}' for v in row])

table = ax.table(
    cellText=cell_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Style header row
for j, label in enumerate(col_labels):
    table[(0, j)].set_facecolor('#e0e0e0')
    table[(0, j)].set_text_props(fontweight='bold')

# Style row labels
for i in range(len(row_labels)):
    table[(i+1, -1)].set_facecolor('#f0f0f0')
    table[(i+1, -1)].set_text_props(fontweight='bold')

ax.set_title('Node features $\\mathbf{x}_i \\in \\mathbb{R}^{11}$', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig('/Users/smsharma/Projects/ds595-ai4science/slides/figures/05-neural-networks/qm9_graph_features.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/smsharma/Projects/ds595-ai4science/slides/figures/05-neural-networks/qm9_graph_features.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved qm9_graph_features.png")
