import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d

class FunctionalGroupInteraction(torch.nn.Module):
    def __init__(self, num_groups=85, embed_dim=8):
        super().__init__()
        self.embedding = torch.nn.Linear(1, embed_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * num_groups, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 32)
        )
        
    def forward(self, x, return_attention=False):
        # x shape: [batch_size, num_groups]
        x = x.unsqueeze(-1) # [batch_size, num_groups, 1]
        emb = self.embedding(x) # [batch_size, num_groups, embed_dim]
        # Compute functional group cross-attention interactions
        attn_out, attn_weights = self.attention(emb, emb, emb)
        # Aggregate all interaction representations
        out = attn_out.reshape(x.shape[0], -1)
        res = F.relu(self.fc(out))
        if return_attention:
            return res, attn_weights
        return res

class PharmaGNN(torch.nn.Module):
    # num_classes = 13, num_global_features = 11 (4 base descriptors + 7 toxicophore densities)
    def __init__(
        self,
        num_node_features,
        hidden_channels=32,
        num_classes=13,
        num_global_features=11,
        num_func_groups=85,
        num_layers=2,
        heads=2,
        residual=True,
        fg_embed_dim=8
    ):
        super(PharmaGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.residual = residual
        self.num_classes = num_classes
        self.num_global_features = num_global_features
        self.num_func_groups = num_func_groups
        self.fg_embed_dim = fg_embed_dim
        
        # Primary layers (exact names preserved for 100% backward compatibility with legacy weights)
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=heads, edge_dim=1, concat=False)
        self.bn1 = BatchNorm1d(hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, concat=False)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        # Additional layers for scaled foundation backbones (e.g. 4-layer PharmaGNN v2)
        self.extra_convs = torch.nn.ModuleList()
        self.extra_bns = torch.nn.ModuleList()
        for _ in range(num_layers - 2):
            self.extra_convs.append(
                GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, concat=False)
            )
            self.extra_bns.append(BatchNorm1d(hidden_channels))
        
        self.fg_interaction = FunctionalGroupInteraction(num_groups=num_func_groups, embed_dim=fg_embed_dim)
        
        # +1 for concentration feature (pIC50)
        self.lin1 = torch.nn.Linear(hidden_channels * 2 + num_global_features + 32 + 1, hidden_channels)
        # Final projection layer outputs logits for 13 multi-task endpoints
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None, global_features=None, func_group_features=None, concentration=None, return_attention=False):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        batch_size = batch.max().item() + 1
            
        if concentration is None:
            # If concentration unspecified, default to pIC50 = 0.0 (equivalent to 1.0 Molar)
            concentration = torch.zeros(batch_size, 1, dtype=torch.float, device=x.device)
            
        if global_features is None:
            global_features = torch.zeros(batch_size, self.num_global_features, dtype=torch.float, device=x.device)
            
        if func_group_features is None:
            func_group_features = torch.zeros(batch_size, self.num_func_groups, dtype=torch.float, device=x.device)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        # Layer 2 with residual skip connection if dimensions match
        h = self.conv2(x, edge_index, edge_attr=edge_attr)
        h = self.bn2(h)
        h = F.leaky_relu(h)
        if self.residual and x.shape == h.shape:
            x = x + h
        else:
            x = h
            
        # Deeper layers (for num_layers > 2)
        for conv, bn in zip(self.extra_convs, self.extra_bns):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = bn(h)
            h = F.leaky_relu(h)
            if self.residual and x.shape == h.shape:
                x = x + h
            else:
                x = h
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Functional group interaction learning
        if return_attention:
            fg_out, attn_weights = self.fg_interaction(func_group_features, return_attention=True)
        else:
            fg_out = self.fg_interaction(func_group_features)
        
        x = torch.cat([x_mean, x_max, global_features, fg_out, concentration], dim=1) 
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        # NOTE: Return raw logits (no sigmoid) so BCEWithLogitsLoss can handle missing label masks (NaN)
        if return_attention:
            return x, attn_weights
        return x