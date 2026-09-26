import pytest
import torch
import numpy as np
from pathlib import Path
from rdkit import Chem

from scripts.phase2_pretrain import (
    PharmaGNN_Pretrain,
    smiles_to_pretrain_graph,
    pretrain_masking,
    pretrain_loss,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    NUM_MOTIFS
)

def test_pretrain_model_scaled_initialization():
    """Verify that PharmaGNN_Pretrain supports scaled 4-layer 128-channel backbone."""
    model = PharmaGNN_Pretrain(
        num_node_features=6,
        hidden_channels=128,
        num_layers=4,
        heads=4,
        residual=True,
        fg_embed_dim=16
    )
    assert model.hidden_channels == 128
    assert model.num_layers == 4
    assert model.atom_pred_head.in_features == 128
    assert model.atom_pred_head.out_features == NUM_ATOM_TYPES
    assert model.bond_pred_head.in_features == 128 * 2
    assert model.bond_pred_head.out_features == NUM_BOND_TYPES
    assert model.motif_head.in_features == 128
    assert model.motif_head.out_features == NUM_MOTIFS

def test_smiles_to_pretrain_graph_alignment():
    """Verify atom and bond labels align with explicit-H graph dimensions."""
    test_smiles = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O"]
    for smi in test_smiles:
        g = smiles_to_pretrain_graph(smi)
        assert g is not None
        assert hasattr(g, 'atom_types')
        assert hasattr(g, 'bond_types')
        assert g.atom_types.size(0) == g.num_nodes, f"Atom types mismatch for {smi}: {g.atom_types.size(0)} vs {g.num_nodes}"
        assert g.bond_types.size(0) == g.edge_index.size(1), f"Bond types mismatch for {smi}: {g.bond_types.size(0)} vs {g.edge_index.size(1)}"
        
        # Verify masking operates on valid indices
        g_masked = pretrain_masking(g, mask_rate=0.20)
        assert len(g_masked.masked_atom_indices) > 0
        assert len(g_masked.masked_atom_labels) == len(g_masked.masked_atom_indices)

def test_pretrain_loss_convergence():
    """Verify pretraining optimization loop converges and decreases loss on a small batch."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = PharmaGNN_Pretrain(
        num_node_features=6,
        hidden_channels=64,
        num_layers=2,
        heads=2,
        residual=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    test_smiles = [
        "CCO", "CO", "CC(=O)O", "CCCCO", "c1ccccc1", "c1ccccc1O",
        "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ]
    graphs = [smiles_to_pretrain_graph(s) for s in test_smiles if smiles_to_pretrain_graph(s) is not None]
    assert len(graphs) >= 5
    
    losses = []
    for epoch in range(5):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        for g in graphs:
            g = pretrain_masking(g)
            out = model.forward_pretrain(
                g.x, g.edge_index, g.edge_attr, g.batch,
                g.global_features, g.func_group_features, g.concentration,
                g.masked_atom_indices, g.masked_bond_indices
            )
            loss, _ = pretrain_loss(out, g)
            loss.backward()
            epoch_loss += loss.item()
        optimizer.step()
        losses.append(epoch_loss / len(graphs))
        
    assert losses[-1] < losses[0], f"Pretraining loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"


def test_pretrain_masking_epoch_independence():
    """Verify that multiple epochs of masking do not cumulatively zero out graph node features."""
    g = smiles_to_pretrain_graph("CCO")
    assert g is not None
    assert hasattr(g, 'raw_x')
    initial_nonzero = (g.raw_x != 0).sum().item()
    assert initial_nonzero > 0
    
    # Simulate 25 consecutive training epochs
    for _ in range(25):
        g = pretrain_masking(g, mask_rate=0.15)
        # Verify base features remain completely pristine
        assert (g.raw_x != 0).sum().item() == initial_nonzero
        # Verify masked g.x retains unmasked node features (~85%)
        unmasked_nonzero = (g.x != 0).sum().item()
        assert unmasked_nonzero >= int(initial_nonzero * 0.70)

