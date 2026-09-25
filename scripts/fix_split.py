#!/usr/bin/env python3
"""
Fix scaffold split and prepare training data for Phase 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings('ignore')

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

# Load master
master = pd.read_csv('data/processed/training_dataset.csv')
print(f"Loaded: {len(master)} compounds")

# Recompute scaffold split properly
master['scaffold'] = master['smiles'].apply(get_scaffold)
master = master.dropna(subset=['scaffold'])

from collections import defaultdict
scaffold_to_indices = defaultdict(list)
for idx, scaffold in enumerate(master['scaffold']):
    scaffold_to_indices[scaffold].append(idx)

scaffolds = list(scaffold_to_indices.keys())
np.random.seed(42)
np.random.shuffle(scaffolds)

# Distribute scaffolds ensuring minimum compounds per split
train_compounds = []
val_compounds = []
test_compounds = []

for scaffold in scaffolds:
    idxs = scaffold_to_indices[scaffold]
    # Assign scaffolds in a round-robin fashion
    n = len(train_compounds) + len(val_compounds) + len(test_compounds)
    if n % 10 < 8:
        train_compounds.extend(idxs)
    elif n % 10 < 9:
        val_compounds.extend(idxs)
    else:
        test_compounds.extend(idxs)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_compounds)}")
print(f"  Val: {len(val_compounds)}")
print(f"  Test: {len(test_compounds)}")

# Apply split
master['split'] = 'train'
master.loc[val_compounds, 'split'] = 'val'
master.loc[test_compounds, 'split'] = 'test'

# Save with proper split
output_path = 'data/processed/training_dataset_v2.csv'
master.to_csv(output_path, index=False)

# Print stats per split
print("\nSplit statistics:")
for split in ['train', 'val', 'test']:
    subset = master[master['split'] == split]
    ct_tox = subset['CT_TOX'].notna().sum()
    total = len(subset)
    toxic = (subset['CT_TOX'] == 1).sum()
    safe = (subset['CT_TOX'] == 0).sum()
    print(f"\n{split.upper()} ({total} compounds):")
    print(f"  CT_TOX labels: {ct_tox} ({ct_tox/total*100:.1f}%)")
    print(f"  Toxic: {toxic}, Safe: {safe}")
    
    # Endpoint coverage
    task_cols = [c for c in master.columns if c.startswith('NR-') or c.startswith('SR-')]
    avail = sum(subset[task_cols].notna().sum().sum() > 0)
    total_task_vals = len(subset) * len(task_cols)
    avail_vals = subset[task_cols].notna().sum().sum()
    print(f"  Endpoint values: {avail_vals}/{total_task_vals} ({avail_vals/total_task_vals*100:.1f}%)")

print("\n✓ Saved to data/processed/training_dataset_v2.csv")