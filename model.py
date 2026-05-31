import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PharmaGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(PharmaGNN, self).__init__()
        # Lớp Convolution 1: Nhận đặc trưng nguyên tử ban đầu
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        # Lớp Convolution 2: Học sâu hơn về ngữ cảnh lân cận
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Lớp Tuyến tính: Đưa ra dự đoán cuối cùng
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # x: Ma trận đỉnh, edge_index: Ma trận kết nối
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- 1. MESSAGE PASSING (Truyền thông điệp) ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # --- 2. READOUT / POOLING (Gom tụ toàn cục) ---
        # Gộp tất cả các nguyên tử lại thành 1 vector duy nhất đại diện cho phân tử
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        
        # --- 3. PREDICTION (Dự đoán) ---
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin(x)
        
        # Dùng Sigmoid để ép kết quả về dạng % (từ 0 đến 1)
        # Ví dụ: 0.85 = 85% khả năng có độc tính
        return torch.sigmoid(x)