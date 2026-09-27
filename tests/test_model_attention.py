import torch
from model import PharmaGNN, FunctionalGroupInteraction

def test_functional_group_interaction_return_attention():
    fg = FunctionalGroupInteraction(num_groups=85, embed_dim=8)
    fg.eval()
    x = torch.zeros(1, 85, dtype=torch.float)
    x[0, 5] = 1.0
    x[0, 12] = 2.0
    
    # Without return_attention
    out = fg(x)
    assert out.shape == (1, 32)
    
    # With return_attention=True
    out, attn_weights = fg(x, return_attention=True)
    assert out.shape == (1, 32)
    assert attn_weights.shape == (1, 85, 85)
    assert torch.is_tensor(attn_weights)

def test_pharma_gnn_return_attention():
    model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)
    model.eval()
    
    x = torch.randn(4, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_attr = torch.randn(4, 1)
    batch = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    func_group_features = torch.zeros(1, 85)
    func_group_features[0, 10] = 1.0
    
    logits, attn_weights = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        func_group_features=func_group_features,
        return_attention=True
    )
    
    assert logits.shape == (1, 13)
    assert attn_weights.shape == (1, 85, 85)
