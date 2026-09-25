import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from main import app
from scripts.phase5_deployment import export_model, load_calibration_temperature

def test_load_calibration_temperature():
    """Verify calibration temperature loading from JSON or default."""
    temp = load_calibration_temperature()
    assert isinstance(temp, float)
    assert temp > 0.0

def test_export_model_pipeline(tmp_path):
    """Verify export_model generates config, specs, and verifies serving."""
    success = export_model(
        weights_path="pharma_gnn_weights_universal.pt",
        hidden_channels=32,
        num_layers=2,
        heads=2,
        smoke_test=True
    )
    assert success is True
    
    # Check generated files
    assert Path("model_config.json").exists()
    assert Path("api_spec.json").exists()
    assert Path("monitoring_config.json").exists()
    assert Path("Dockerfile.production").exists()
    assert Path("deployment_summary.json").exists()
    
    # Verify model_config contents
    with open("model_config.json", "r") as f:
        cfg = json.load(f)
    assert cfg["hidden_channels"] == 32
    assert cfg["num_classes"] == 13
    assert len(cfg["tasks"]) == 13

def test_fastapi_endpoints_with_exported_config():
    """Verify live FastAPI service endpoints with updated configuration."""
    client = TestClient(app)
    
    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    
    # Predict endpoint
    res = client.post("/api/predict", json={"smiles": "CCO", "concentration_molar": 1e-5})
    assert res.status_code == 200
    data = res.json()
    assert "predictions" in data
    assert "toxicity_risk" in data["predictions"]
    assert "all_class_probs" in data["predictions"]
    assert len(data["predictions"]["all_class_probs"]) == 13
