#!/usr/bin/env python3
"""
Phase 1 Starter: Download and curate toxicity datasets using DeepChem
Alternative to CompTox direct download - uses DeepChem's built-in datasets
"""

import os
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup output directories
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Phase 1: Data Curation & ETL - Alternative Approach")
print("=" * 70)
print("\nUsing DeepChem built-in datasets + curated compound lists")
print("instead of direct CompTox download (API issues)")

# Download DeepChem datasets
print("\n1. Downloading DeepChem datasets...")
print("-" * 70)

import deepchem as dc
from deepchem.data import NumpyDataset

# Datasets to download
datasets_to_get = [
    ('tox21', 'Tox21 - 12 nuclear receptor and stress response endpoints'),
    ('clintox', 'ClinTox - Clinical toxicity'),
    ('sider', 'SIDER - Side effect database'),
    ('muv', 'MUV - Maximum Unbiased Validation'),
    ('toxcast', 'ToxCast - EPA high-throughput screening'),
]

loaded_datasets = {}

for name, desc in datasets_to_get:
    try:
        print(f"\nLoading {name}: {desc}")
        if name == 'tox21':
            dataset = dc.load_dataset('tox21', file_path=str(DATA_DIR))
        elif name == 'clintox':
            dataset = dc.load_dataset('clintox', file_path=str(DATA_DIR))
        elif name == 'sider':
            dataset = dc.load_dataset('sider', file_path=str(DATA_DIR))
        elif name == 'muv':
            dataset = dc.load_dataset('muv', file_path=str(DATA_DIR))
        elif name == 'toxcast':
            dataset = dc.load_dataset('toxcast', file_path=str(DATA_DIR))
        
        loaded_datasets[name] = dataset
        print(f"  ✓ Loaded: {len(dataset)} compounds")
        
    except Exception as e:
        print(f"  ✗ Failed to load {name}: {e}")

# Create merged dataset
print("\n2. Merging datasets...")
print("-" * 70)

master_data = {
    'smiles': [],
    'source': [],
    'task': [],
    'label': []
}

# Process each dataset
for name, dataset in loaded_datasets.items():
    X = dataset.X
    y = dataset.y
    ids = dataset.ids
    # Get SMILES from dataset
    # DeepChem datasets store SMILES in X or ids
    print(f"\nProcessing {name}: {len(dataset)} compounds")
    
    # This is simplified - actual implementation would need to extract SMILES
    # from the dataset's features
    
print("\n3. Loading curated compound lists...")
print("-" * 70)

# Curated list of known toxic/safe compounds with SMILES
CURATED_COMPOUNDS = {
    'toxic': [
        ('C#N', 'Cyanide'),
        ('c1ccccc1O', 'Phenol'),
        ('C1=CC=C(C=C1)O', 'Phenol'),
        ('Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1', 'DDT'),
        ('C(C(=O)O)NCP(=O)(O)O', 'Glyphosate'),
        ('CCO', 'Ethanol'),
        ('CC(=O)OC1=CC=CC=C1C(=O)O', 'Aspirin'),
        ('CC(=O)Nc1ccc(O)cc1', 'Paracetamol'),
    ],
    'safe': [
        ('O', 'Water'),
        ('[Na+].[Cl-]', 'Sodium chloride'),
        ('[Na+].[F-]', 'Sodium fluoride'),
        ('OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O', 'Glucose'),
        ('CC(=O)O', 'Acetic acid'),
        ('C(C(=O)O)N', 'Glycine'),
    ]
}

print(f"Curated toxic compounds: {len(CURATED_COMPOUNDS['toxic'])}")
print(f"Curated safe compounds: {len(CURATED_COMPOUNDS['safe'])}")

# Save curated list
curated_df = pd.DataFrame([
    {'smiles': s, 'compound_name': n, 'label': 1, 'source': 'curated'}
    for s, n in CURATED_COMPOUNDS['toxic']
] + [
    {'smiles': s, 'compound_name': n, 'label': 0, 'source': 'curated'}
    for s, n in CURATED_COMPOUNDS['safe']
])

curated_path = PROCESSED_DIR / "curated_compounds.csv"
curated_df.to_csv(curated_path, index=False)
print(f"\n✓ Saved curated compounds to: {curated_path}")

print("\n" + "=" * 70)
print("Phase 1 Starter Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Run: python scripts/etl_merge_datasets.py")
print("2. Run: python scripts/scaffold_split.py")
print("3. Run: python scripts/prepare_training_data.py")
print("\nFiles created:")
for f in PROCESSED_DIR.glob("*"):
    print(f"  - {f.name}")