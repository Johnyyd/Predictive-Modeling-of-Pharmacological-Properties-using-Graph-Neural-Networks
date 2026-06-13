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
        
    def forward(self, x):
        # x shape: [batch_size, num_groups]
        x = x.unsqueeze(-1) # [batch_size, num_groups, 1]
        emb = self.embedding(x) # [batch_size, num_groups, embed_dim]
        # Tính toán tương tác giữa các gốc chức
        attn_out, _ = self.attention(emb, emb, emb)
        # Gộp tất cả tương tác lại
        out = attn_out.reshape(x.shape[0], -1)
        return F.relu(self.fc(out))

class PharmaGNN(torch.nn.Module):
    # num_classes bây giờ sẽ là 13, thêm num_global_features=11 (4 gốc + 7 mật độ độc tính)
    def __init__(self, num_node_features, hidden_channels, num_classes=13, num_global_features=11, num_func_groups=85):
        super(PharmaGNN, self).__init__()
        
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn1 = BatchNorm1d(hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        self.fg_interaction = FunctionalGroupInteraction(num_groups=num_func_groups, embed_dim=8)
        
        # +1 cho đặc trưng nồng độ (concentration - pIC50)
        self.lin1 = torch.nn.Linear(hidden_channels * 2 + num_global_features + 32 + 1, hidden_channels)
        # Lớp này sẽ xuất ra 12 giá trị (Logits)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr=None, batch=None, global_features=None, func_group_features=None, concentration=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        batch_size = batch.max().item() + 1
            
        if concentration is None:
            # Nếu không có nồng độ, giả định pIC50 = 0.0 (tương đương 1.0 Molar)
            concentration = torch.zeros(batch_size, 1, dtype=torch.float, device=x.device)
            
        if global_features is None:
            global_features = torch.zeros(batch_size, 11, dtype=torch.float, device=x.device)
            
        if func_group_features is None:
            func_group_features = torch.zeros(batch_size, 85, dtype=torch.float, device=x.device)

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Học tương tác nhóm chức
        fg_out = self.fg_interaction(func_group_features)
        
        x = torch.cat([x_mean, x_max, global_features, fg_out, concentration], dim=1) 
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        # CHÚ Ý: Bỏ hàm torch.sigmoid(x) ở đây. 
        # Chúng ta trả về giá trị thô (Logits) để hàm Loss xử lý các ô dữ liệu bị thiếu (NaN).
        return x