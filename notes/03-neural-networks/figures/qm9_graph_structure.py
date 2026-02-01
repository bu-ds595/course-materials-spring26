"""
QM9 graph structure visualization for lecture notes.
Shows a molecule alongside its explicit graph representation with features.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from torch_geometric.datasets import QM9
import networkx as nx

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75

# Colors for atoms
ATOM_COLORS = {
    'H': '#FFFFFF',  # white
    'C': '#404040',  # dark gray
    'N': '#3050F8',  # blue
    'O': '#FF0D0D',  # red
    'F': '#90E050',  # green
}
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F']

# Load QM9
dataset = QM9(root='./data/qm9')
raw_path = Path('./data/qm9/raw/gdb9.sdf')
supplier = Chem.SDMolSupplier(str(raw_path), removeHs=False)

# Propanol (CCCO) - simple chain molecule
mol_idx = 39

data = dataset[mol_idx]
mol = supplier[mol_idx]
smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

# Create figure
fig = plt.figure(figsize=(DOUBLE_COL, 2.8))

# Three panels: molecule image | graph structure | feature table
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1.5], wspace=0.3)

# Panel 1: RDKit molecule image
ax1 = fig.add_subplot(gs[0])
mol_no_h = Chem.RemoveHs(mol)
AllChem.Compute2DCoords(mol_no_h)
img = Draw.MolToImage(mol_no_h, size=(200, 200))
ax1.imshow(img)
ax1.set_title('(a) Molecule', fontsize=9)
ax1.axis('off')

# Panel 2: Graph representation
ax2 = fig.add_subplot(gs[1])

# Build networkx graph from PyG data - excluding hydrogens for cleaner viz
G = nx.Graph()
num_nodes = data.num_nodes

# Map from original indices to new indices (excluding H)
old_to_new = {}
new_idx = 0

# Add nodes with atom labels (skip hydrogens)
for i in range(num_nodes):
    atom_idx = data.x[i][:5].argmax().item()
    atom = ATOM_TYPES[atom_idx]
    if atom != 'H':  # Skip hydrogens for cleaner visualization
        G.add_node(new_idx, atom=atom, orig_idx=i)
        old_to_new[i] = new_idx
        new_idx += 1

# Add edges (only between non-hydrogen atoms)
edge_index = data.edge_index
for i in range(edge_index.shape[1]):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    if src in old_to_new and dst in old_to_new and src < dst:
        G.add_edge(old_to_new[src], old_to_new[dst])

# Layout
pos = nx.spring_layout(G, seed=42, k=2)

# Draw edges
nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='#888888', width=1.5)

# Draw nodes with atom colors
node_colors = [ATOM_COLORS[G.nodes[i]['atom']] for i in G.nodes()]
node_edge_colors = ['#333333' if G.nodes[i]['atom'] != 'H' else '#888888' for i in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors,
                       edgecolors=node_edge_colors, node_size=400, linewidths=1.5)

# Labels - simple text, no LaTeX subscripts
labels = {i: f"{G.nodes[i]['atom']}" for i in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=7)

ax2.set_title('(b) Graph structure', fontsize=9)
ax2.axis('off')
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

# Panel 3: Feature table
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
ax3.set_title(r'(c) Node features $\mathbf{h}_i \in \mathbf{R}^{11}$', fontsize=9)

# Create a compact table showing features for heavy atoms only
table_data = []
headers = ['Node', 'Atom', 'Z', 'nH']

# Show features for non-hydrogen atoms
for new_i, node in enumerate(G.nodes()):
    orig_i = G.nodes[node]['orig_idx']
    feat = data.x[orig_i]
    atom_idx = feat[:5].argmax().item()
    atom = ATOM_TYPES[atom_idx]
    atomic_num = int(feat[5].item())
    num_h = int(feat[10].item())
    table_data.append([str(new_i), atom, str(atomic_num), str(num_h)])
    if new_i >= 4:  # Show max 5 rows
        break

# Add table
table = ax3.table(cellText=table_data, colLabels=headers,
                  loc='upper center', cellLoc='center',
                  colColours=['#f0f0f0']*4,
                  bbox=[0.05, 0.35, 0.9, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(8)

# No legend - keep it clean

plt.suptitle(f'QM9 molecule as graph: {smiles}', fontsize=10, fontweight='bold', y=1.08)

# Save
save_path = Path("/Users/smsharma/Projects/ds595-ai4science/notes/chapters/03-neural-networks/figures/qm9_graph_structure.pdf")
fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
fig.savefig(save_path.with_suffix('.png'), dpi=200, bbox_inches='tight', pad_inches=0.1)
print(f"Saved to {save_path}")
