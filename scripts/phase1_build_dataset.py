#!/usr/bin/env python3
"""
Phase 1: Hybrid Toxicity Dataset Builder
Builds master dataset from Tox21, ClinTox, and curated compounds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def standardize_smiles(smiles):
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

print("=" * 70)
print("Phase 1: Data Curation - Hybrid Toxicity Dataset")
print("=" * 70)

# Load Tox21
print("\n[1] Loading Tox21...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df_tox21 = pd.read_csv(url, compression='gzip')
tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
df_tox21 = df_tox21[['smiles'] + tasks].copy()
df_tox21['smiles_std'] = df_tox21['smiles'].apply(standardize_smiles)
df_tox21 = df_tox21.dropna(subset=['smiles_std'])
print(f"   Tox21: {len(df_tox21)} compounds")

# Load ClinTox
print("\n[2] Loading ClinTox...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
df_clintox = pd.read_csv(url, compression='gzip')
df_clintox = df_clintox[['smiles', 'CT_TOX', 'FDA_APPROVED']].copy()
df_clintox['smiles_std'] = df_clintox['smiles'].apply(standardize_smiles)
df_clintox = df_clintox.dropna(subset=['smiles_std'])
print(f"   ClinTox: {len(df_clintox)} compounds")

# Curated compounds
print("\n[3] Adding curated compounds...")
curated_data = [
    ('C#N', 'Cyanide', 1), ('c1ccccc1O', 'Phenol', 1), ('Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1', 'DDT', 1),
    ('C(C(=O)O)NCP(=O)(O)O', 'Glyphosate', 1), ('C1=CC=C(C=C1)O', 'Phenol_2', 1),
    ('O', 'Water', 0), ('[Na+].[Cl-]', 'Sodium chloride', 0), ('OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O', 'Glucose', 0),
    ('CC(=O)O', 'Acetic acid', 0), ('C(C(=O)O)N', 'Glycine', 0), ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine', 0),
    ('CC(=O)Nc1ccc(O)cc1', 'Paracetamol', 0), ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin', 0), ('CCO', 'Ethanol', 0),
]
df_curated = pd.DataFrame(curated_data, columns=['smiles', 'name', 'CT_TOX'])
df_curated['smiles_std'] = df_curated['smiles'].apply(standardize_smiles)
df_curated = df_curated.dropna(subset=['smiles_std'])
print(f"   Curated: {len(df_curated)} compounds")

# Merge datasets
print("\n[4] Merging datasets...")
master = df_tox21.merge(df_clintox[['smiles_std', 'CT_TOX', 'FDA_APPROVED']], 
                        on='smiles_std', how='left', suffixes=('', '_clintox'))
master['smiles'] = master['smiles_std']
master = master.drop(columns=['smiles_std'])

# Add curated compounds (update CT_TOX if exists)
for _, row in df_curated.iterrows():
    mask = master['smiles'] == row['smiles_std']
    if mask.any():
        master.loc[mask, 'CT_TOX'] = row['CT_TOX']
        master.loc[mask, 'curated_name'] = row['name']
    else:
        # Add new compound
        new_row = {'smiles': row['smiles_std'], 'CT_TOX': row['CT_TOX'], 'curated_name': row['name']}
        for task in tasks:
            new_row[task] = np.nan
        master = pd.concat([master, pd.DataFrame([new_row])], ignore_index=True)

# Remove duplicates
master = master.drop_duplicates(subset=['smiles'])
master['source'] = master['CT_TOX'].notna().astype(int)

# Save
output_path = PROCESSED_DIR / "master_toxicity_dataset.csv"
master.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print(f"✓ Master dataset created: {output_path}")
print(f"  Total compounds: {len(master)}")
print(f"  Columns: {len(master.columns)}")
print(f"\n  CT_TOX label coverage: {master['CT_TOX'].notna().sum()}/{len(master)} ({master['CT_TOX'].notna().mean()*100:.1f}%)")
print("=" * 70)

# Print sample
print("\nSample compounds:")
print(master[['smiles', 'CT_TOX']].head(10).to_string())

# Save metadata
metadata = {
    'total_compounds': int(len(master)),
    'tox21_compounds': int(len(df_tox21)),
    'clintox_compounds': int(len(df_clintox)),
    'curated_compounds': int(len(df_curated)),
    'ct_tox_coverage': int(master['CT_TOX'].notna().sum()),
    'tasks': tasks + ['CT_TOX', 'FDA_APPROVED']
}

import json
with open(PROCESSED_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✓ Phase 1 Complete! Ready for Phase 2 (Pretraining)")
