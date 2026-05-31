import torch
import torch.nn.functional as F
# Đổi GATConv thành GATv2Conv
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d

class PharmaGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(PharmaGNN, self).__init__()
        
        # THÊM edge_dim=1 để AI bắt đầu đọc thông tin loại liên kết
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn1 = BatchNorm1d(hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # Lấy thêm edge_attr từ dữ liệu
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Bơm edge_attr vào quá trình Truyền thông điệp
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return torch.sigmoid(x)