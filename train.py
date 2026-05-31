import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

# Import mô hình và hàm xử lý từ các file bạn đã viết
from model import PharmaGNN
from main import smiles_to_graph 

# --- 1. CHUẨN BỊ DỮ LIỆU (DATASET) ---
# Quy ước nhãn: 1.0 = Có độc tính/Nguy hiểm, 0.0 = An toàn
mock_dataset = [
    ("CC(=O)Oc1ccccc1C(=O)O", 0.0), # Aspirin (An toàn)
    ("CCO", 0.0),                  # Ethanol (An toàn)
    ("C(C(=O)O)N", 0.0),           # Glycine (An toàn)
    ("c1ccccc1", 1.0),             # Benzene (Gây ung thư)
    ("C#N", 1.0),                  # Cyanide (Cực độc)
    ("C1=CC=C(C=C1)O", 1.0)        # Phenol (Độc)
]

print("[1] Đang tiền xử lý chuỗi SMILES thành Đồ thị (Graphs)...")
graph_list = []
for smiles, label in mock_dataset:
    graph = smiles_to_graph(smiles)
    if graph is not None:
        # Gắn thêm nhãn (target) vào đồ thị để AI lấy làm mốc so sánh
        graph.y = torch.tensor([[label]], dtype=torch.float)
        graph_list.append(graph)

# DataLoader giúp chia nhỏ dữ liệu thành các lô (batch) để RAM không bị quá tải
loader = DataLoader(graph_list, batch_size=2, shuffle=True)

# --- 2. KHỞI TẠO MÔ HÌNH & HÀM MẤT MÁT ---
print("[2] Khởi tạo mạng GNN...")
# Lưu ý: hidden_channels=32 phải khớp với file main.py lúc bạn dùng để dự đoán
model = PharmaGNN(num_node_features=3, hidden_channels=32, num_classes=1)

# Sử dụng Binary Cross Entropy Loss (Hàm chuẩn cho bài toán phân loại Nhị phân)
criterion = nn.BCELoss() 
# Bộ tối ưu Adam giúp AI điều chỉnh trọng số (learning rate = 0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 3. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ---
epochs = 50
print(f"\n[3] 🚀 Bắt đầu ép xung học tập ({epochs} vòng)...")

model.train() # Chuyển mô hình sang chế độ học
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()          # 1. Xóa bộ nhớ gradient cũ
        predictions = model(batch)     # 2. AI thử dự đoán (Forward Pass)
        loss = criterion(predictions, batch.y) # 3. Chấm điểm độ sai lệch
        loss.backward()                # 4. Học ngược để rút kinh nghiệm (Backprop)
        optimizer.step()               # 5. Cập nhật lại nơ-ron
        
        total_loss += loss.item()
        
    # In báo cáo mỗi 10 vòng
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Mức độ sai lệch (Loss): {total_loss/len(loader):.4f}")

# --- 4. LƯU THÀNH QUẢ ---
# Xuất trọng số ra file .pt (PyTorch Tensor)
torch.save(model.state_dict(), "pharma_gnn_weights.pt")
print("\n✅ HOÀN TẤT! Đã đóng gói bộ não AI vào file 'pharma_gnn_weights.pt'.")