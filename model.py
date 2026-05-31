import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d

class PharmaGNN(torch.nn.Module):
    # num_classes bây giờ sẽ là 13, thêm num_global_features=4
    def __init__(self, num_node_features, hidden_channels, num_classes=13, num_global_features=4):
        super(PharmaGNN, self).__init__()
        
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn1 = BatchNorm1d(hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        self.lin1 = torch.nn.Linear(hidden_channels * 2 + num_global_features, hidden_channels)
        # Lớp này sẽ xuất ra 12 giá trị (Logits)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        global_features = data.global_features
        
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max, global_features], dim=1) 
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        # CHÚ Ý: Bỏ hàm torch.sigmoid(x) ở đây. 
        # Chúng ta trả về giá trị thô (Logits) để hàm Loss xử lý các ô dữ liệu bị thiếu (NaN).
        return x