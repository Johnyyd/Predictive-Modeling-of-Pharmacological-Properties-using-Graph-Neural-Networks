#!/usr/bin/env python3
"""
Alternative CompTox download using EPA's direct API and known URLs
"""

import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "comptox_v3"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Updated EPA CompTox URLs (as of 2024)
# Source: https://www.epa.gov/chemical-research/exploring-compounds-compound-explorer-tool
FILES = {
    "chemicals": "https://www.epa.gov/sites/default/files/2025-06/compounds_v3.2.tsv.gz",  # Updated
    "assay_results": "https://www.epa.gov/sites/default/files/2025-06/assay_results_v3.2.tsv.gz",
    "assay_information": "https://www.epa.gov/sites/default/files/2025-06/assay_information_v3.2.tsv.gz",
}

def download_file(url, path):
    print(f"Downloading {url}")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    for name, url in FILES.items():
        path = DATA_DIR / f"{name}.gz"
        download_file(url, path)