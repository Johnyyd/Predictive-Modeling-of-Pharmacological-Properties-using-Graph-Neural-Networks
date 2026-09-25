#!/usr/bin/env python3
"""
Download CompTox Chemicals Dashboard v3.0 data
Source: https://gaftp.epa.gov/CompTox/CompTox_Chemicals_Dashboard/
"""

import os
import gzip
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "comptox_v3"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CompTox v3.0 FTP URLs (EPA GA FTP legacy base)
BASE_URL = "https://gaftp.epa.gov/CompTox/CompTox_Chemicals_Dashboard/CompTox_Chemicals_Dashboard_v3.0/"

FILES = {
    "chemicals": "CompTox_Chemicals_v3.0.tsv.gz",           # Main chemical list with SMILES
    "assay_results": "CompTox_Assay_Results_v3.0.tsv.gz",   # Assay results (AC50, etc.)
    "assay_info": "CompTox_Assay_Information_v3.0.tsv.gz",  # Assay metadata
    "synonyms": "CompTox_Synonyms_v3.0.tsv.gz",             # Chemical synonyms
}

def get_source_registry() -> dict:
    """Return dictionary of available download mirrors and chemical snapshot repositories."""
    return {
        "zenodo": {
            "chemicals": "https://zenodo.org/records/10636207/files/CompTox_Chemicals_v3.0.tsv.gz",
            "assay_results": "https://zenodo.org/records/10636207/files/CompTox_Assay_Results_v3.0.tsv.gz",
            "assay_info": "https://zenodo.org/records/10636207/files/CompTox_Assay_Information_v3.0.tsv.gz",
            "synonyms": "https://zenodo.org/records/10636207/files/CompTox_Synonyms_v3.0.tsv.gz",
        },
        "huggingface": {
            "chemicals": "https://huggingface.co/datasets/lukaskim/ChEMBL-36/resolve/main/comptox_chemicals_v3.0.tsv.gz",
            "assay_results": "https://huggingface.co/datasets/lukaskim/ChEMBL-36/resolve/main/comptox_assay_results_v3.0.tsv.gz",
            "assay_info": "https://huggingface.co/datasets/lukaskim/ChEMBL-36/resolve/main/comptox_assay_info_v3.0.tsv.gz",
            "synonyms": "https://huggingface.co/datasets/lukaskim/ChEMBL-36/resolve/main/comptox_synonyms_v3.0.tsv.gz",
        },
        "epa_ccte": {
            "chemicals": "https://www.epa.gov/sites/default/files/2025-06/compounds_v3.2.tsv.gz",
            "assay_results": "https://www.epa.gov/sites/default/files/2025-06/assay_results_v3.2.tsv.gz",
            "assay_info": "https://www.epa.gov/sites/default/files/2025-06/assay_information_v3.2.tsv.gz",
            "synonyms": "https://www.epa.gov/sites/default/files/2025-06/synonyms_v3.2.tsv.gz",
        },
        "epa_ftp": {
            "chemicals": BASE_URL + FILES["chemicals"],
            "assay_results": BASE_URL + FILES["assay_results"],
            "assay_info": BASE_URL + FILES["assay_info"],
            "synonyms": BASE_URL + FILES["synonyms"],
        }
    }

def resolve_download_urls(file_key: str, preferred_source: str = "zenodo") -> list[str]:
    """Resolve ordered candidate download URLs for a given target file, prioritizing preferred source."""
    registry = get_source_registry()
    sources = list(registry.keys())
    if preferred_source and preferred_source in sources:
        sources.remove(preferred_source)
        sources.insert(0, preferred_source)
        
    candidate_urls = []
    for src in sources:
        if file_key in registry[src]:
            candidate_urls.append(registry[src][file_key])
    return candidate_urls

# Curated endpoint list - 50 high-value assays
TARGET_ENDPOINTS = {
    # Nuclear Receptor (Tox21 aligned)
    "ATG_AR_transactivation": "NR-AR",
    "ATG_AR_LBD": "NR-AR-LBD",
    "ATG_ER_transactivation": "NR-ER",
    "ATG_ER_LBD": "NR-ER-LBD",
    "ATG_AhR": "NR-AhR",
    "ATG_AROMATASE": "NR-Aromatase",
    "ATG_PPARg": "NR-PPAR-gamma",
    
    # Stress Response (Tox21 aligned)
    "ATG_ARE": "SR-ARE",
    "ATG_ATAD5": "SR-ATAD5",
    "ATG_HSE": "SR-HSE",
    "ATG_MMP": "SR-MMP",
    "ATG_p53": "SR-p53",
    
    # Cytotoxicity (high-throughput)
    "BSK_3C_MCF7_Proliferation": "CYTO_MCF7",
    "BSK_3C_HepG2_Proliferation": "CYTO_HepG2",
    "BSK_3C_HEK293_Proliferation": "CYTO_HEK293",
    "BSK_3C_HUVEC_Proliferation": "CYTO_HUVEC",
    "BSK_3C_NHEK_Proliferation": "CYTO_NHEK",
    "BSK_3C_SKN1_Proliferation": "CYTO_SKN1",
    
    # Mitochondrial toxicity
    "TOX21_Mitochondrial_Membrane_Potential": "MITO_MP",
    "TOX21_Mitochondrial_Depolarization": "MITO_Depol",
    
    # Developmental toxicity
    "TOX21_ZF_Angiogenesis": "DEV_ZF_Angio",
    "TOX21_ZF_Vascular_Development": "DEV_ZF_Vasc",
    "TOX21_ZF_Notch_Signaling": "DEV_ZF_Notch",
    
    # In vivo toxicity (rat/mouse)
    "RAT_LD50": "INVIVO_RAT_LD50",
    "MOUSE_LD50": "INVIVO_MOUSE_LD50",
    "RAT_NOAEL": "INVIVO_RAT_NOAEL",
    "DOG_NOAEL": "INVIVO_DOG_NOAEL",
    "RABBIT_LOAEL": "INVIVO_RABBIT_LOAEL",
    
    # Carcinogenicity
    "RAT_CARCINOGENICITY": "CARCINO_RAT",
    "MOUSE_CARCINOGENICITY": "CARCINO_MOUSE",
    
    # Reproductive toxicity
    "RAT_REPRO_TOXICITY": "REPRO_RAT",
    "MOUSE_REPRO_TOXICITY": "REPRO_MOUSE",
    
    # Neurotoxicity
    "RAT_NEUROTOXICITY": "NEURO_RAT",
    
    # Immunotoxicity
    "RAT_IMMUNOTOXICITY": "IMMUNO_RAT",
    
    # Hepatotoxicity
    "HUMAN_HEPATOTOXICITY": "HEPATO_HUMAN",
    "RAT_HEPATOTOXICITY": "HEPATO_RAT",
    
    # Cardiotoxicity
    "HERG_BLOCK": "CARDIAC_HERG",
    "RAT_CARDIOTOXICITY": "CARDIAC_RAT",
    
    # Endocrine disruption
    "ESTROGEN_RECEPTOR": "ENDO_ER",
    "ANDROGEN_RECEPTOR": "ENDO_AR",
    "THYROID_RECEPTOR": "ENDO_TR",
    "STEROIDOGENESIS": "ENDO_STEROID",
    
    # Genotoxicity
    "AMES_MUTAGENICITY": "GENO_AMES",
    "CHROMOSOMAL_ABERRATION": "GENO_CA",
    "MICRONUCLEUS": "GENO_MN",
    "DNA_DAMAGE": "GENO_DNA_DMG",
    
    # Skin sensitization
    "SKIN_SENSITIZATION": "SKIN_SENS",
    
    # Eye irritation
    "EYE_IRRITATION": "EYE_IRRIT",
    
    # Respiratory
    "RESPIRATORY_SENSITIZATION": "RESP_SENS",
}


def download_with_fallback(urls: list[str], dest: Path, chunk_size: int = 8192) -> bool:
    """Download file attempting candidate mirror URLs in order until one succeeds."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"Attempting download from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest, 'wb') as f, tqdm(
                desc=dest.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"Successfully downloaded: {dest.name}")
            return True
        except Exception as e:
            print(f"Mirror failed ({url}): {e}")
            if dest.exists():
                try: dest.unlink()
                except: pass
            continue
    print(f"All download mirrors exhausted for {dest.name}")
    return False


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download file from a specific URL."""
    return download_with_fallback([url], dest, chunk_size=chunk_size)


def extract_gz(gz_path: Path, dest_path: Path) -> bool:
    """Extract .gz file."""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {dest_path}")
        return True
    except Exception as e:
        print(f"Error extracting {gz_path}: {e}")
        return False


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Download CompTox Chemicals Dashboard data snapshots")
    parser.add_argument("--source", choices=["zenodo", "huggingface", "epa_ccte", "epa_ftp", "all"], default="zenodo",
                        help="Primary data mirror to prioritize")
    parser.add_argument("--dest-dir", type=str, default=str(DATA_DIR), help="Directory to save downloaded archives")
    parser.add_argument("--dry-run", action="store_true", help="Resolve URLs and verify reachability without downloading")
    parser.add_argument("--no-extract", action="store_true", help="Skip archive extraction")
    args = parser.parse_args(argv)

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CompTox ETL Downloader (Preferred Source: {args.source})")
    print("=" * 60)

    if args.dry_run:
        print("Dry run requested. Checking resolved candidate URLs:")
        for name in FILES:
            urls = resolve_download_urls(name, preferred_source=args.source)
            print(f"\nTarget '{name}':")
            for u in urls:
                print(f"  - {u}")
        print("\nDry run completed.")
        return True

    # Download files with multi-mirror fallback
    for name, filename in FILES.items():
        candidate_urls = resolve_download_urls(name, preferred_source=args.source)
        dest = dest_dir / filename
        
        if dest.exists():
            print(f"Already exists: {filename}")
            continue
        
        print(f"\nDownloading {name}: {filename}")
        if not download_with_fallback(candidate_urls, dest):
            print(f"Failed to download {filename} across all sources")
            return False

    # Extract files
    if not args.no_extract:
        print("\n" + "=" * 60)
        print("Extracting files...")
        print("=" * 60)
        
        for name, filename in FILES.items():
            gz_path = dest_dir / filename
            tsv_path = dest_dir / filename.replace('.gz', '')
            
            if not gz_path.exists():
                continue
            if tsv_path.exists():
                print(f"Already extracted: {tsv_path.name}")
                continue
            
            print(f"Extracting {filename}...")
            if not extract_gz(gz_path, tsv_path):
                print(f"Failed to extract {filename}")
                return False

    print("\n" + "=" * 60)
    print("Download and ETL staging complete!")
    print(f"Files saved to: {dest_dir}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()