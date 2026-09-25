import pytest
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock

import scripts.download_comptox as dl

def test_source_registry_contains_modern_sources():
    """Verify that source registry provides Zenodo, HuggingFace, and EPA CCTE mirrors."""
    registry = dl.get_source_registry()
    assert "zenodo" in registry
    assert "huggingface" in registry
    assert "epa_ccte" in registry
    
    # Check that chemical files are mapped
    assert "chemicals" in registry["zenodo"]
    assert "chemicals" in registry["huggingface"]
    assert "chemicals" in registry["epa_ccte"]

def test_resolve_download_urls_priority():
    """Verify fallback list resolves with the user's preferred source first."""
    urls = dl.resolve_download_urls("chemicals", preferred_source="zenodo")
    assert len(urls) >= 2
    assert "zenodo" in urls[0].lower()

    hf_urls = dl.resolve_download_urls("chemicals", preferred_source="huggingface")
    assert "huggingface" in hf_urls[0].lower()

def test_download_with_fallback_recovers_from_primary_failure(tmp_path):
    """Verify that download_with_fallback continues to mirror when primary fails."""
    dest = tmp_path / "test_download.tsv.gz"
    
    mock_bad_resp = MagicMock()
    mock_bad_resp.raise_for_status.side_effect = Exception("404 Not Found")
    
    mock_good_resp = MagicMock()
    mock_good_resp.raise_for_status.return_value = None
    mock_good_resp.headers = {"content-length": "12"}
    mock_good_resp.iter_content.return_value = [b"mock payload"]
    
    with patch("requests.get", side_effect=[mock_bad_resp, mock_good_resp]) as mock_get:
        candidate_urls = ["https://broken.source.org/file.tsv.gz", "https://active.zenodo.org/file.tsv.gz"]
        success = dl.download_with_fallback(candidate_urls, dest)
        assert success is True
        assert dest.exists()
        assert dest.read_bytes() == b"mock payload"
        assert mock_get.call_count == 2

def test_extract_gz_utility(tmp_path):
    """Verify extract_gz decompresses gzip archives correctly."""
    gz_path = tmp_path / "sample.tsv.gz"
    tsv_path = tmp_path / "sample.tsv"
    
    sample_content = b"chemical_id\tsmiles\tname\n1\tCCO\tEthanol\n"
    with gzip.open(gz_path, "wb") as f:
        f.write(sample_content)
        
    assert dl.extract_gz(gz_path, tsv_path) is True
    assert tsv_path.exists()
    assert tsv_path.read_bytes() == sample_content
