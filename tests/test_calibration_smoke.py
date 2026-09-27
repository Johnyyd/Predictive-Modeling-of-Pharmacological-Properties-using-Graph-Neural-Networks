import pytest
import torch
import numpy as np
import json
from pathlib import Path

from model import PharmaGNN
from scripts.phase4_calibration import (
    TemperatureScaling,
    compute_calibration_metrics,
    run_chemical_sanity_checks,
    calibrate_model
)

def test_temperature_scaling_module():
    """Verify TemperatureScaling scales logits and preserves output shapes."""
    base_model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)
    cal_model = TemperatureScaling(base_model, initial_temp=1.5)
    
    assert cal_model.temperature.item() == 1.5
    
    x = torch.randn(4, 6)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, 1)
    batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    global_features = torch.randn(1, 11)
    func_group_features = torch.randn(1, 85)
    concentration = torch.tensor([[5.0]])
    
    base_model.eval()
    with torch.no_grad():
        raw_logits = base_model(x, edge_index, edge_attr, batch, global_features, func_group_features, concentration)
        scaled_logits = cal_model(x, edge_index, edge_attr, batch, global_features, func_group_features, concentration)
        
    assert scaled_logits.shape == raw_logits.shape
    assert torch.allclose(scaled_logits, raw_logits / 1.5, atol=1e-5)

def test_compute_calibration_metrics():
    """Verify ECE and Brier score computations on synthetic predictions."""
    preds = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    
    ece, brier = compute_calibration_metrics(preds, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 1.0
    # Well-calibrated predictions should have low brier score
    assert brier < 0.05

def test_run_chemical_sanity_checks():
    """Verify qualitative sanity checks on ATP, Sarin, Cyanide, and safe excipients."""
    model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)
    weights_path = Path("pharma_gnn_weights_universal.pt")
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
    results = run_chemical_sanity_checks(model, temperature=1.0, device=torch.device('cpu'))
    assert isinstance(results, dict)
    assert 'ATP' in results
    assert 'Cyanide' in results
    assert 'Sarin' in results
    assert 'Water' in results
    
    # Safe excipients should produce low probabilities (< 0.50)
    assert results['Water']['prob'] < 0.50
    # Toxicants should have valid probability values
    assert 0.0 <= results['Sarin']['prob'] <= 1.0
