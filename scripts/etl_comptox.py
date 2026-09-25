#!/usr/bin/env python3
"""
ETL script for CompTox v3.0 curated dataset
Processes raw CompTox files into ML-ready format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import MolStandardize
import argparse
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "comptox_v3"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target endpoints (same as download script)
TARGET_ENDPOINTS = {
    "ATG_AR_transactivation": "NR-AR",
    "ATG_AR_LBD": "NR-AR-LBD",
    "ATG_ER_transactivation": "NR-ER",
    "ATG_ER_LBD": "NR-ER-LBD",
    "ATG_AhR": "NR-AhR",
    "ATG_AROMATASE": "NR-Aromatase",
    "ATG_PPARg": "NR-PPAR-gamma",
    "ATG_ARE": "SR-ARE",
    "ATG_ATAD5": "SR-ATAD5",
    "ATG_HSE": "SR-HSE",
    "ATG_MMP": "SR-MMP",
    "ATG_p53": "SR-p53",
    "BSK_3C_MCF7_Proliferation": "CYTO_MCF7",
    "BSK_3C_HepG2_Proliferation": "CYTO_HepG2",
    "BSK_3C_HEK293_Proliferation": "CYTO_HEK293",
    "BSK_3C_HUVEC_Proliferation": "CYTO_HUVEC",
    "BSK_3C_NHEK_Proliferation": "CYTO_NHEK",
    "BSK_3C_SKN1_Proliferation": "CYTO_SKN1",
    "TOX21_Mitochondrial_Membrane_Potential": "MITO_MP",
    "TOX21_Mitochondrial_Depolarization": "MITO_Depol",
    "TOX21_ZF_Angiogenesis": "DEV_ZF_Angio",
    "TOX21_ZF_Vascular_Development": "DEV_ZF_Vasc",
    "TOX21_ZF_Notch_Signaling": "DEV_ZF_Notch",
    "RAT_LD50": "INVIVO_RAT_LD50",
    "MOUSE_LD50": "INVIVO_MOUSE_LD50",
    "RAT_NOAEL": "INVIVO_RAT_NOAEL",
    "DOG_NOAEL": "INVIVO_DOG_NOAEL",
    "RABBIT_LOAEL": "INVIVO_RABBIT_LOAEL",
    "RAT_CARCINOGENICITY": "CARCINO_RAT",
    "MOUSE_CARCINOGENICITY": "CARCINO_MOUSE",
    "RAT_REPRO_TOXICITY": "REPRO_RAT",
    "MOUSE_REPRO_TOXICITY": "REPRO_MOUSE",
    "RAT_NEUROTOXICITY": "NEURO_RAT",
    "RAT_IMMUNOTOXICITY": "IMMUNO_RAT",
    "HUMAN_HEPATOTOXICITY": "HEPATO_HUMAN",
    "RAT_HEPATOTOXICITY": "HEPATO_RAT",
    "HERG_BLOCK": "CARDIAC_HERG",
    "RAT_CARDIOTOXICITY": "CARDIAC_RAT",
    "ESTROGEN_RECEPTOR": "ENDO_ER",
    "ANDROGEN_RECEPTOR": "ENDO_AR",
    "THYROID_RECEPTOR": "ENDO_TR",
    "STEROIDOGENESIS": "ENDO_STEROID",
    "AMES_MUTAGENICITY": "GENO_AMES",
    "CHROMOSOMAL_ABERRATION": "GENO_CA",
    "MICRONUCLEUS": "GENO_MN",
    "DNA_DAMAGE": "GENO_DNA_DMG",
    "SKIN_SENSITIZATION": "SKIN_SENS",
    "EYE_IRRITATION": "EYE_IRRIT",
    "RESPIRATORY_SENSITIZATION": "RESP_SENS",
}


def standardize_smiles(smiles: str) -> str:
    """Standardize SMILES using RDKit MolStandardize."""
    if pd.isna(smiles) or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Standardize
        standardizer = MolStandardize.Standardizer()
        mol = standardizer.standardize(mol)
        # Get canonical SMILES
        canon_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canon_smiles
    except Exception:
        return None


def load_chemicals(chemicals_path: Path) -> pd.DataFrame:
    """Load and filter chemicals file."""
    print(f"Loading chemicals from {chemicals_path}...")
    df = pd.read_csv(chemicals_path, sep='\t', low_memory=False)
    print(f"  Total chemicals: {len(df)}")
    
    # Keep only essential columns
    keep_cols = ['DTXSID', 'PREFERRED_NAME', 'CASRN', 'SMILES', 'MOLECULAR_FORMULA', 'MOLECULAR_WEIGHT']
    df = df[keep_cols].copy()
    
    # Standardize SMILES
    print("  Standardizing SMILES...")
    df['SMILES_STD'] = df['SMILES'].apply(standardize_smiles)
    
    # Remove invalid SMILES
    before = len(df)
    df = df.dropna(subset=['SMILES_STD'])
    print(f"  Removed {before - len(df)} invalid SMILES")
    print(f"  Valid chemicals: {len(df)}")
    
    return df


def load_assay_results(results_path: Path) -> pd.DataFrame:
    """Load assay results."""
    print(f"Loading assay results from {results_path}...")
    df = pd.read_csv(results_path, sep='\t', low_memory=False)
    print(f"  Total assay results: {len(df)}")
    return df


def load_assay_info(info_path: Path) -> pd.DataFrame:
    """Load assay information."""
    print(f"Loading assay info from {info_path}...")
    df = pd.read_csv(info_path, sep='\t', low_memory=False)
    print(f"  Total assays: {len(df)}")
    return df


def filter_target_assays(assay_results: pd.DataFrame, assay_info: pd.DataFrame) -> pd.DataFrame:
    """Filter assay results to target endpoints only."""
    # Get assay IDs for target endpoints
    target_assay_names = list(TARGET_ENDPOINTS.keys())
    
    # Match by assay name
    target_assays = assay_info[assay_info['ASSAY_NAME'].isin(target_assay_names)]
    target_aids = target_assays['AID'].unique()
    
    print(f"Target assays found: {len(target_aids)} / {len(target_assay_names)}")
    
    # Filter results
    filtered = assay_results[assay_results['AID'].isin(target_aids)].copy()
    print(f"Filtered results: {len(filtered)}")
    
    return filtered, target_assays


def pivot_assay_results(filtered_results: pd.DataFrame, target_assays: pd.DataFrame, chemicals: pd.DataFrame) -> pd.DataFrame:
    """Pivot assay results to wide format (one row per chemical)."""
    # Merge assay names
    aid_to_name = dict(zip(target_assays['AID'], target_assays['ASSAY_NAME']))
    filtered_results['ASSAY_NAME'] = filtered_results['AID'].map(aid_to_name)
    filtered_results['ENDPOINT'] = filtered_results['ASSAY_NAME'].map(TARGET_ENDPOINTS)
    
    # Merge with chemicals to get SMILES
    merged = filtered_results.merge(chemicals[['DTXSID', 'SMILES_STD']], on='DTXSID', how='left')
    merged = merged.dropna(subset=['SMILES_STD'])
    
    # For classification: convert AC50 to binary (active/inactive)
    # AC50 < 10 uM = active (1), AC50 >= 10 uM or NaN = inactive (0)
    # For regression endpoints (LD50, NOAEL): keep numeric value
    
    classification_endpoints = [v for k, v in TARGET_ENDPOINTS.items() 
                                if not k.startswith(('RAT_LD50', 'MOUSE_LD50', 'RAT_NOAEL', 'DOG_NOAEL', 'RABBIT_LOAEL'))]
    regression_endpoints = [v for k, v in TARGET_ENDPOINTS.items() 
                            if k.startswith(('RAT_LD50', 'MOUSE_LD50', 'RAT_NOAEL', 'DOG_NOAEL', 'RABBIT_LOAEL'))]
    
    print(f"Classification endpoints: {len(classification_endpoints)}")
    print(f"Regression endpoints: {len(regression_endpoints)}")
    
    # Pivot classification endpoints
    class_data = merged[merged['ENDPOINT'].isin(classification_endpoints)].copy()
    class_data['ACTIVE'] = (class_data['AC50'] < 10).astype(int)  # AC50 in uM
    
    class_pivot = class_data.pivot_table(
        index='SMILES_STD',
        columns='ENDPOINT',
        values='ACTIVE',
        aggfunc='max'  # If multiple results, take max (most active)
    ).reset_index()
    
    # Pivot regression endpoints
    reg_data = merged[merged['ENDPOINT'].isin(regression_endpoints)].copy()
    # Convert to log scale for LD50/NOAEL
    reg_data['LOG_VALUE'] = np.log10(reg_data['AC50'].replace(0, np.nan))
    
    reg_pivot = reg_data.pivot_table(
        index='SMILES_STD',
        columns='ENDPOINT',
        values='LOG_VALUE',
        aggfunc='mean'
    ).reset_index()
    
    # Merge
    final = class_pivot.merge(reg_pivot, on='SMILES_STD', how='outer')
    final = final.rename(columns={'SMILES_STD': 'smiles'})
    
    print(f"Final curated compounds: {len(final)}")
    print(f"Total endpoints: {len(final.columns) - 1}")
    
    return final


def filter_by_coverage(df: pd.DataFrame, min_assays: int = 5) -> pd.DataFrame:
    """Filter compounds with minimum number of assay measurements."""
    endpoint_cols = [c for c in df.columns if c != 'smiles']
    df['num_measured'] = df[endpoint_cols].notna().sum(axis=1)
    
    print(f"Coverage distribution:")
    print(df['num_measured'].value_counts().sort_index())
    
    filtered = df[df['num_measured'] >= min_assays].copy()
    filtered = filtered.drop(columns=['num_measured'])
    
    print(f"After min {min_assays} assays filter: {len(filtered)} compounds")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="ETL CompTox v3.0 to curated dataset")
    parser.add_argument('--min-assays', type=int, default=5, help='Minimum assays per compound')
    parser.add_argument('--output', type=str, default='comptox_curated.csv', help='Output filename')
    args = parser.parse_args()
    
    # Check input files
    chemicals_file = DATA_DIR / "CompTox_Chemicals_v3.0.tsv"
    results_file = DATA_DIR / "CompTox_Assay_Results_v3.0.tsv"
    info_file = DATA_DIR / "CompTox_Assay_Information_v3.0.tsv"
    
    for f in [chemicals_file, results_file, info_file]:
        if not f.exists():
            print(f"Missing file: {f}")
            print("Run download_comptox.py first")
            return 1
    
    # Load data
    chemicals = load_chemicals(chemicals_file)
    assay_results = load_assay_results(results_file)
    assay_info = load_assay_info(info_file)
    
    # Filter to target assays
    filtered_results, target_assays = filter_target_assays(assay_results, assay_info)
    
    # Pivot to wide format
    curated = pivot_assay_results(filtered_results, target_assays, chemicals)
    
    # Filter by coverage
    curated = filter_by_coverage(curated, min_assays=args.min_assays)
    
    # Add source column
    curated['source'] = 'comptox'
    
    # Save
    output_path = OUTPUT_DIR / args.output
    curated.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Shape: {curated.shape}")
    
    # Print endpoint stats
    endpoint_cols = [c for c in curated.columns if c not in ['smiles', 'source']]
    print("\nEndpoint statistics:")
    for ep in endpoint_cols:
        non_null = curated[ep].notna().sum()
        if curated[ep].dtype in ['int64', 'float64']:
            if curated[ep].nunique() <= 2:
                pos = (curated[ep] == 1).sum()
                print(f"  {ep}: {non_null} measured, {pos} positive ({pos/non_null*100:.1f}%)")
            else:
                print(f"  {ep}: {non_null} measured, mean={curated[ep].mean():.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())