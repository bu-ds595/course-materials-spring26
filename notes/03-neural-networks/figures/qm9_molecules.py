"""
QM9 molecule visualization for lecture notes.
Shows several molecules with their properties.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from torch_geometric.datasets import QM9

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75

# Load QM9
dataset = QM9(root='./data/qm9')

# Property names and units (subset of the 19 targets)
# Index 7 is HOMO-LUMO gap (good for demo)
property_names = {
    0: ('Dipole moment', 'D'),
    1: ('Isotropic polarizability', 'a₀³'),
    2: ('HOMO', 'eV'),
    3: ('LUMO', 'eV'),
    4: ('HOMO-LUMO gap', 'eV'),
    5: ('Electronic spatial extent', 'a₀²'),
    6: ('Zero-point energy', 'eV'),
    7: ('Internal energy (0K)', 'eV'),
}

# Get SMILES from the raw data
raw_path = Path('./data/qm9/raw/gdb9.sdf')
supplier = Chem.SDMolSupplier(str(raw_path), removeHs=False)

# Select diverse interesting molecules
# Indices chosen to show variety
selected_indices = [5, 42, 100, 500, 1000, 2000, 5000, 10000]

fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 3.2))
axes = axes.flatten()

for ax_idx, mol_idx in enumerate(selected_indices):
    ax = axes[ax_idx]

    # Get molecule from RDKit supplier
    mol = supplier[mol_idx]
    if mol is None:
        continue

    # Remove hydrogens for cleaner visualization
    mol_no_h = Chem.RemoveHs(mol)

    # Generate 2D coordinates for drawing
    AllChem.Compute2DCoords(mol_no_h)

    # Draw molecule to image
    img = Draw.MolToImage(mol_no_h, size=(200, 200))

    # Get properties from PyG dataset
    data = dataset[mol_idx]
    gap = data.y[0, 4].item()  # HOMO-LUMO gap

    # Get SMILES
    smiles = Chem.MolToSmiles(mol_no_h)
    if len(smiles) > 15:
        smiles = smiles[:12] + '...'

    # Plot
    ax.imshow(img)
    ax.set_title(f'Gap: {gap:.2f} eV', fontsize=8)
    ax.set_xlabel(smiles, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.suptitle('QM9: Small organic molecules with quantum properties', fontsize=10, y=1.02)
plt.tight_layout()

# Save
save_path = Path("/Users/smsharma/Projects/ds595-ai4science/notes/chapters/03-neural-networks/figures/qm9_molecules.pdf")
fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
fig.savefig(save_path.with_suffix('.png'), dpi=200, bbox_inches='tight', pad_inches=0.1)
print(f"Saved to {save_path}")

plt.show()
