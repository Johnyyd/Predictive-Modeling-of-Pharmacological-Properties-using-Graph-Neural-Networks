"""
model.py - Nơi định nghĩa Kiến trúc Trí tuệ Nhân tạo (PharmaGNN).
Bao gồm:
1. Mạng chú ý tương tác nhóm chức (FunctionalGroupInteraction)
2. Mạng tích chập đồ thị chính (PharmaGNN)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d

class FunctionalGroupInteraction(torch.nn.Module):
    """
    Mạng Nơ-ron phụ trách phân tích 85 Nhóm chức Hóa học.
    Thay vì chỉ đếm số lượng nhóm chức một cách thụ động, mạng này dùng cơ chế Multihead Attention
    để học cách các nhóm chức "tương tác" với nhau (vd: Có OH đi kèm với Vòng Benzen sẽ tạo ra hiệu ứng gì).
    """
    def __init__(self, num_groups=85, embed_dim=8):
        super().__init__()
        # Bước 1: Ánh xạ mỗi con số (số lượng của 1 nhóm chức) thành một vector 8 chiều (Embedding)
        self.embedding = torch.nn.Linear(1, embed_dim)
        
        # Bước 2: Dùng Multihead Attention để các nhóm chức "nhìn" lẫn nhau.
        # num_heads=4 nghĩa là có 4 góc nhìn khác nhau về sự tương tác này.
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Bước 3: Nén kết quả tương tác khổng lồ xuống còn một vector 32 chiều (để ghép vào mạng chính)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * num_groups, 128), # Phóng to ra để học các đặc trưng ẩn
            torch.nn.ReLU(),                              # Kích hoạt phi tuyến tính
            torch.nn.Dropout(0.3),                        # Tắt ngẫu nhiên 30% nơ-ron để chống học vẹt (Overfitting)
            torch.nn.Linear(128, 32)                      # Thu nhỏ lại chốt hạ ở 32 chiều
        )
        
    def forward(self, x):
        # Đầu vào x có hình dạng: [Kích_thước_lô, 85_nhóm_chức]
        x = x.unsqueeze(-1) # Thêm chiều giả: [Kích_thước_lô, 85, 1] để đưa vào Embedding
        emb = self.embedding(x) # Chuyển thành: [Kích_thước_lô, 85, 8_chiều]
        
        # Tính toán tương tác giữa các gốc chức bằng Attention
        # self.attention nhận vào 3 tham số Query, Key, Value. Ở đây dùng chung emb (Self-Attention)
        attn_out, _ = self.attention(emb, emb, emb)
        
        # Trải phẳng kết quả từ 3D [lô, 85, 8] thành 2D [lô, 85*8]
        out = attn_out.reshape(x.shape[0], -1)
        
        # Đưa qua các lớp nén (Linear) và trả về vector 32 chiều
        return F.relu(self.fc(out))

class PharmaGNN(torch.nn.Module):
    """
    Mạng Nơ-ron chính của hệ thống - Kết hợp Graph Neural Network (GNN) và Dữ liệu Hóa học.
    Nhiệm vụ: Đọc Đồ thị phân tử -> Kết hợp Đặc trưng Toàn cục -> Dự đoán 13 loại độc tính.
    """
    def __init__(self, num_node_features, hidden_channels, num_classes=13, num_global_features=11, num_func_groups=85):
        super(PharmaGNN, self).__init__()
        
        # Giai đoạn 1: Mạng Tích chập Đồ thị (Graph Convolution)
        # GATv2Conv cho phép mỗi nguyên tử chú ý (attention) đến láng giềng của nó với các trọng số khác nhau.
        # heads=2: Chạy 2 luồng chú ý song song.
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn1 = BatchNorm1d(hidden_channels) # Chuẩn hóa lô (Batch Normalization) giúp huấn luyện nhanh và ổn định hơn
        
        # Lớp chập thứ 2: Học các tương tác xa hơn (láng giềng của láng giềng)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=2, edge_dim=1, concat=False)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        # Khởi tạo Mạng phụ phân tích Nhóm chức (Đã định nghĩa ở trên)
        self.fg_interaction = FunctionalGroupInteraction(num_groups=num_func_groups, embed_dim=8)
        
        # Giai đoạn 2: Lớp Tuyến tính cuối cùng (Classifier Head)
        # Đầu vào của lin1 là sự gộp lại của 4 luồng kiến thức:
        # 1. hidden_channels: Đại diện đồ thị (Trung bình - Mean)
        # 2. hidden_channels: Đại diện đồ thị (Cực đại - Max)
        # 3. num_global_features: 11 đặc trưng toàn cục (MolWt, LogP... + Mật độ độc tính)
        # 4. 32: Kết quả từ mạng phụ fg_interaction
        self.lin1 = torch.nn.Linear(hidden_channels * 2 + num_global_features + 32, hidden_channels)
        
        # lin2 là chốt chặn cuối cùng, xuất ra đúng 13 con số (tương ứng 13 bài test độc tính)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # Tách các thành phần từ đối tượng Data của PyTorch Geometric
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        global_features = data.global_features
        fg_features = data.func_group_features
        
        # --- BƯỚC 1: XỬ LÝ ĐỒ THỊ (MESSAGE PASSING) ---
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x) # Hàm kích hoạt cho phép giá trị âm nhỏ đi qua (chống nghẽn nơ-ron)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        # --- BƯỚC 2: GỘP CỤM (READOUT / POOLING) ---
        # Gộp tất cả nguyên tử của 1 phân tử thành 1 vector duy nhất
        x_mean = global_mean_pool(x, batch) # Lấy trung bình (Thể hiện bức tranh tổng thể)
        x_max = global_max_pool(x, batch)   # Lấy cực đại (Bắt lấy những nguyên tử nổi bật/bất thường nhất)
        
        # --- BƯỚC 3: XỬ LÝ NHÓM CHỨC BẰNG MẠNG PHỤ ---
        fg_out = self.fg_interaction(fg_features)
        
        # --- BƯỚC 4: LAI TẠO KIẾN THỨC (FUSION) ---
        # Ghép nối (Concat) tất cả các vector lại với nhau theo chiều ngang
        x = torch.cat([x_mean, x_max, global_features, fg_out], dim=1) 
        
        # --- BƯỚC 5: PHÁN QUYẾT (CLASSIFICATION) ---
        x = F.dropout(x, p=0.5, training=self.training) # Tắt 50% nơ-ron để ép mô hình học thuộc bản chất, không học mẹo
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        # CHÚ Ý QUAN TRỌNG: KHÔNG sử dụng hàm torch.sigmoid(x) ở đây. 
        # Trả về nguyên giá trị thô (Logits) để hàm BCEWithLogitsLoss ở file train.py xử lý.
        # Điều này giúp hệ thống học ổn định hơn rất nhiều khi gặp các ô dữ liệu bị bỏ trống (NaN).
        return x