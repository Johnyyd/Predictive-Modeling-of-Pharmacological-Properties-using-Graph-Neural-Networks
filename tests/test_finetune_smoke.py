import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from model import PharmaGNN
from scripts.phase3_finetune import (
    TASKS,
    load_pretrained_encoder,
    build_training_graphs,
    evaluate_model,
    create_fine_tune_optimizer,
    freeze_backbone,
    unfreeze_backbone
)

def test_tasks_alignment_with_system():
    """Verify that TASKS in phase3_finetune has exactly 13 classes aligning with model.py and main.py."""
    assert len(TASKS) == 13
    assert TASKS == [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 
        'CT_TOX'
    ]

def test_load_pretrained_encoder(tmp_path):
    """Verify weight transfer from pretrained encoder to scaled PharmaGNN model."""
    device = torch.device('cpu')
    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=128,
        num_classes=13,
        num_global_features=11,
        num_func_groups=85,
        num_layers=4,
        heads=4,
        residual=True,
        fg_embed_dim=16
    ).to(device)
    
    # Create synthetic pretrained encoder checkpoint
    encoder_state = {
        'conv1.lin_l.weight': torch.randn_like(model.conv1.lin_l.weight),
        'bn1.weight': torch.randn_like(model.bn1.weight),
        'bn1.bias': torch.randn_like(model.bn1.bias),
        'conv2.lin_l.weight': torch.randn_like(model.conv2.lin_l.weight),
        'bn2.weight': torch.randn_like(model.bn2.weight),
        'bn2.bias': torch.randn_like(model.bn2.bias),
    }
    ckpt_path = tmp_path / "mock_encoder.pt"
    torch.save(encoder_state, ckpt_path)
    
    loaded = load_pretrained_encoder(model, str(ckpt_path), device)
    assert loaded is True
    assert torch.equal(model.conv1.lin_l.weight, encoder_state['conv1.lin_l.weight'])

def test_two_stage_freeze_and_optimizer():
    """Verify backbone freezing in Stage 1 and discriminative learning rates in Stage 2."""
    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=64,
        num_classes=13,
        num_layers=2,
        heads=2
    )
    
    # Stage 1: Freeze backbone
    freeze_backbone(model)
    assert not model.conv1.lin_l.weight.requires_grad
    assert not model.conv2.lin_l.weight.requires_grad
    assert model.lin2.weight.requires_grad
    assert model.fg_interaction.embedding.weight.requires_grad
    
    # Stage 2: Unfreeze backbone
    unfreeze_backbone(model)
    assert model.conv1.lin_l.weight.requires_grad
    assert model.conv2.lin_l.weight.requires_grad
    assert model.lin2.weight.requires_grad
    
    # Optimizer with discriminative learning rates
    optimizer = create_fine_tune_optimizer(model, lr_backbone=1e-4, lr_head=1e-3)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['lr'] == 1e-4
    assert optimizer.param_groups[1]['lr'] == 1e-3

def test_build_training_graphs_and_evaluation():
    """Verify build_training_graphs builds valid Data objects and evaluate_model returns metrics."""
    data = {
        'smiles': ['CCO', 'c1ccccc1O', 'CC(=O)Oc1ccccc1C(=O)O', 'C#N'],
        'CT_TOX': [0.0, 1.0, 0.0, 1.0],
        'NR-AR': [0.0, 1.0, np.nan, np.nan],
        'NR-ER': [np.nan, 0.0, 1.0, np.nan]
    }
    df = pd.DataFrame(data)
    graphs = build_training_graphs(df)
    assert len(graphs) >= 3
    for g in graphs:
        assert g.y.shape == (1, 13)
        assert hasattr(g, 'global_features')
        assert hasattr(g, 'func_group_features')
        assert hasattr(g, 'concentration')
        
    model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)
    results = evaluate_model(model, graphs, torch.device('cpu'), min_samples=2)
    assert isinstance(results, dict)
