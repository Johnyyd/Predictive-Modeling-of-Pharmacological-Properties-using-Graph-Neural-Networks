import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pandas as pd
import random
import os

from model import PharmaGNN
from main import smiles_to_graph

# --- 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU TỪ INTERNET ---
print("[1] Đang tải bộ dữ liệu ClinTox (FDA)...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
df = pd.read_csv(url, compression='gzip')

# SỬ DỤNG ĐÚNG TÊN CỘT 'CT_TOX'
df = df[['smiles', 'CT_TOX']].dropna()
print(f"-> Đã tải {len(df)} phân tử. Đang dựng Ma trận Đồ thị (Sẽ mất khoảng 10-20 giây)...")

graph_list = []
for index, row in df.iterrows():
    graph = smiles_to_graph(row['smiles'])
    if graph is not None:
        # Cập nhật tên cột ở đây
        graph.y = torch.tensor([[row['CT_TOX']]], dtype=torch.float)
        graph_list.append(graph)

random.shuffle(graph_list)

split_idx = int(len(graph_list) * 0.8)
train_data = graph_list[:split_idx]
test_data = graph_list[split_idx:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- 2. KHỞI TẠO MÔ HÌNH VÀ THUẬT TOÁN ---
model = PharmaGNN(num_node_features=3, hidden_channels=32, num_classes=1)

# Nếu đã có não cũ từ bài test trước, nạp vào để học tiếp (Transfer Learning)
weights_path = "pharma_gnn_weights.pt"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print("✅ Đã nạp thành công trọng số cũ, AI đang học nâng cao thêm!")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# --- 3. VÒNG LẶP HUẤN LUYỆN (EPOCHS) ---
epochs = 50
print(f"\n[2] Bắt đầu Deep Learning với {len(train_data)} mẫu ({epochs} vòng)...")

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch)
        # 1. Tính toán độ sai lệch cơ bản (chưa trung bình hóa)
        criterion = nn.BCELoss(reduction='none') 
        base_loss = criterion(predictions, batch.y)

        # 2. Tạo ma trận Trọng số: Gán hệ số 10.0 cho chất Độc (1.0), và 1.0 cho chất An toàn (0.0)
        weights = torch.where(batch.y == 1.0, 10.0, 1.0)

        # 3. Nhân phạt và lấy trung bình
        loss = (base_loss * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval() 
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch)
                predicted_labels = (preds > 0.5).float()
                correct += (predicted_labels == batch.y).sum().item()
                
        accuracy = (correct / len(test_data)) * 100
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Độ chính xác (Test): {accuracy:.2f}%")

# --- 4. LƯU THÀNH QUẢ ---
torch.save(model.state_dict(), "pharma_gnn_weights.pt")
print("\n✅ HOÀN TẤT! File 'pharma_gnn_weights.pt' đã được cập nhật với tri thức từ FDA.")