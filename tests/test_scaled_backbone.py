import pytest
import torch
from pathlib import Path
from model import PharmaGNN, FunctionalGroupInteraction

def test_pharma_gnn_scaled_v2_dimensions():
    """Verify that scaled 4-layer 128-channel PharmaGNN v2 runs and produces expected tensor shapes."""
    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=128,
        num_layers=4,
        heads=4,
        residual=True,
        fg_embed_dim=16,
        num_classes=13,
        num_global_features=11,
        num_func_groups=85
    )
    model.eval()

    num_nodes = 8
    x = torch.randn(num_nodes, 6)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
        [1, 0, 2, 1, 3, 2, 5, 4, 7, 6]
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    global_features = torch.randn(2, 11)
    func_group_features = torch.randn(2, 85)
    concentration = torch.tensor([[5.0], [6.0]], dtype=torch.float)

    logits, attn_weights = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        global_features=global_features,
        func_group_features=func_group_features,
        concentration=concentration,
        return_attention=True
    )

    assert logits.shape == (2, 13), f"Expected logits shape (2, 13), got {logits.shape}"
    assert attn_weights.shape == (2, 85, 85), f"Expected attention shape (2, 85, 85), got {attn_weights.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
    assert torch.isfinite(attn_weights).all(), "Attention weights contain NaN or Inf"

def test_pharma_gnn_legacy_weight_loading():
    """Verify that default PharmaGNN loads pharma_gnn_weights_universal.pt with 100% key compatibility."""
    weights_path = Path("pharma_gnn_weights_universal.pt")
    assert weights_path.exists(), "pharma_gnn_weights_universal.pt not found"

    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=32,
        num_layers=2,
        heads=2,
        fg_embed_dim=8,
        num_classes=13
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    load_res = model.load_state_dict(state_dict, strict=True)
    assert len(load_res.missing_keys) == 0, f"Missing keys when loading legacy checkpoint: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys when loading legacy checkpoint: {load_res.unexpected_keys}"

    # Verify forward pass with loaded weights
    model.eval()
    x = torch.randn(4, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_attr = torch.randn(4, 1)
    out = model(x, edge_index, edge_attr=edge_attr)
    assert out.shape == (1, 13)

def test_residual_gradient_flow():
    """Verify that gradients propagate cleanly through 4 layers with residual connections."""
    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=64,
        num_layers=4,
        heads=2,
        residual=True
    )
    model.train()

    x = torch.randn(6, 6, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    edge_attr = torch.randn(6, 1)

    out = model(x, edge_index, edge_attr=edge_attr)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input x received no gradients"
    assert torch.isfinite(x.grad).all(), "Gradients contain NaN or Inf"
    assert x.grad.abs().sum() > 0, "Gradients completely vanished to zero"
