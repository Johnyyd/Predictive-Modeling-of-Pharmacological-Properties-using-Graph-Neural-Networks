#!/usr/bin/env python3
"""
Phase 1 Data Curation: Hybrid Toxicity Dataset Builder
Downloads multiple toxicity datasets and merges them for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import MolStandardize
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def standardize_smiles(smiles):
    """Standardize SMILES using RDKit."""
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        standardizer = MolStandardize.Standardizer()
        mol = standardizer.standardize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def load_tox21():
    """Load Tox21 dataset with SMILES metadata."""
    # DeepChem Tox21 dataset
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    # Keep only relevant columns
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    
    df = df[['smiles'] + tasks].copy()
    df['source'] = 'tox21'
    df['smiles_std'] = df['smiles'].apply(standardize_smiles)
    df = df.dropna(subset=['smiles_std'])
    
    print(f"Tox21: {len(df)} compounds")
    return df

def load_clintox():
    """Load ClinTox dataset."""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    df = df[['smiles', 'CT_TOX', 'FDA_APPROVED']].copy()
    df['source'] = 'clintox'
    df['smiles_std'] = df['smiles'].apply(standardize_smiles)
    df = df.dropna(subset=['smiles_std'])
    
    print(f"ClinTox: {len(df)} compounds")
    return df

def load_curated_compounds():
    """Load manually curated compounds with known toxicity."""
    curated_data = [
        # Known toxic compounds (CT_TOX = 1)
        ('C#N', 'Cyanide', 1),
        ('c1ccccc1O', 'Phenol', 1),
        ('Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1', 'DDT', 1),
        ('C(C(=O)O)NCP(=O)(O)O', 'Glyphosate', 1),
        ('C1=CC=C(C=C1)O', 'Phenol', 1),
        ('CC(=O)Oc1ccc(cc1)Cl', 'Paracetamol-chloro', 1),
        
        # Known safe compounds (CT_TOX = 0)
        ('O', 'Water', 0),
        ('[Na+].[Cl-]', 'Sodium chloride', 0),
        ('[Na+].[F-]', 'Sodium fluoride', 0),
        ('[K+].[Cl-]', 'Potassium chloride', 0),
        ('OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O', 'Glucose', 0),
        ('CC(=O)O', 'Acetic acid', 0),
        ('C(C(=O)O)N', 'Glycine', 0),
        ('CC(C(=O)O)N', 'Alanine', 0),
        ('CC(C)C(C(=O)O)N', 'Valine', 0),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine', 0),
        ('CC(=O)Nc1ccc(O)cc1', 'Paracetamol', 0),
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin', 0),
        ('CCO', 'Ethanol', 0),
        ('CO', 'Methanol', 0),
    ]
    
    df = pd.DataFrame(curated_data, columns=['smiles', 'name', 'CT_TOX'])
    df['source'] = 'curated'
    df['smiles_std'] = df['smiles'].apply(standardize_smiles)
    df = df.dropna(subset=['smiles_std'])
    
    print(f"Curated: {len(df)} compounds")
    return df

def merge_datasets():
    """Merge all datasets into master table."""
    print("=" * 70)
    print("Phase 1: Data Curation - Hybrid Toxicity Dataset")
    print("=" * 70)
    
    # Load datasets
    print("\nLoading datasets...")
    tox21 = load_tox21()
    clintox = load_clintox()
    curated = load_curated_compounds()
    
    # Merge Tox21 and ClinTox on SMILES
    print("\nMerging Tox21 and ClinTox...")
    master = pd.merge(
        tox21.drop(columns=['source']),
        clintox.drop(columns=['source']),
        left_on='smiles_std',
        right_on='smiles_std',
        how='outer',
        suffixes=('', '_clintox')
    )
    
    # Add curated compounds
    curated_subset = curated[['smiles_std', 'CT_TOX', 'name']].copy()
    curated_subset.columns = ['smiles', 'CT_TOX_curated', 'name']
    master = pd.merge(master, curated_subset, left_on='smiles_std', right_on='smiles', how='left')
    
    # Track source
    master['source'] = master['source'].fillna('curated')
    
    # Standardize column names
    master = master.rename(columns={'smiles_std': 'smiles'})
    
    # Remove duplicates
    before = len(master)
    master = master.drop_duplicates(subset=['smiles'])
    print(f"Removed {before - len(master)} duplicates")
    
    # Save
    output_path = PROCESSED_DIR / "master_toxicity_dataset.csv"
    master.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"Master dataset saved: {output_path}")
    print(f"Total compounds: {len(master)}")
    print(f"Columns: {len(master.columns)}")
    
    # Print stats
    print("\nSource distribution:")
    print(master['source'].value_counts())
    
    # Print endpoint completeness
    task_cols = [c for c in master.columns if c.startswith('NR-') or c.startswith('SR-') or c == 'CT_TOX']
    print(f"\nEndpoint completeness:")
    for col in task_cols:
        non_null = master[col].notna().sum()
        print(f"  {col}: {non_null}/{len(master)} ({non_null/len(master)*100:.1f}%)")
    
    return master

if __name__ == "__main__":
    merge_datasets()