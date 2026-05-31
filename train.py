import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import random
import os
from sklearn.metrics import roc_auc_score

from model import PharmaGNN
from main import smiles_to_graph

# --- 1. TẢI DỮ LIỆU CLINTOX (ĐỘC TÍNH LÂM SÀNG CHUNG) ---
print("[1] Đang tải bộ dữ liệu ClinTox (FDA)...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
df = pd.read_csv(url, compression='gzip')

df = df[['smiles', 'CT_TOX']].dropna()
print(f"-> Đã tải {len(df)} phân tử. Đang dựng Ma trận Đồ thị (GATv2)...")

graph_list = []
num_safe = 0
num_toxic = 0

for index, row in df.iterrows():
    graph = smiles_to_graph(row['smiles'])
    if graph is not None:
        label = float(row['CT_TOX'])
        graph.y = torch.tensor([[label]], dtype=torch.float)
        graph_list.append(graph)
        
        if label == 1.0: num_toxic += 1
        else: num_safe += 1

# Tính trọng số phạt động (giới hạn trần để AI không bị hoảng loạn)
dynamic_penalty = min(num_safe / num_toxic if num_toxic > 0 else 1.0, 5.0)
print(f"-> Phân bố: {num_safe} An toàn | {num_toxic} Độc hại.")
print(f"-> Trọng số phạt Focal Loss: {dynamic_penalty:.2f}")

random.shuffle(graph_list)
split_idx = int(len(graph_list) * 0.8)
train_data = graph_list[:split_idx]
test_data = graph_list[split_idx:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# --- 2. KHỞI TẠO MÔ HÌNH ---
model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=1)

# BẮT BUỘC TỰ ĐỘNG XÓA TRỌNG SỐ CŨ (Để AI học lại tư duy ClinTox)
weights_path = "pharma_gnn_weights.pt"
if os.path.exists(weights_path):
    print("🧹 Tự động xóa ký ức Tox21 cũ để học dữ liệu ClinTox...")
    os.remove(weights_path)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Định nghĩa Focal Loss
def focal_loss(predictions, targets, pos_weight, gamma=2.0):
    bce = F.binary_cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-bce)
    alpha_t = torch.where(targets == 1.0, pos_weight, 1.0)
    loss = alpha_t * (1 - pt) ** gamma * bce
    return loss.mean()

# --- 3. VÒNG LẶP HUẤN LUYỆN ---
epochs = 60 # ClinTox ít dữ liệu hơn, tăng vòng lặp lên 60 để AI ngấm sâu
print(f"\n[2] Bắt đầu ép xung học tập với {len(train_data)} mẫu ({epochs} vòng)...")

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch)
        loss = focal_loss(predictions, batch.y, pos_weight=dynamic_penalty, gamma=2.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval() 
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                
        try:
            auc_score = roc_auc_score(all_labels, all_preds)
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Điểm ROC-AUC: {auc_score:.4f}")
        except ValueError:
            pass

# --- 4. LƯU THÀNH QUẢ ---
torch.save(model.state_dict(), weights_path)
print("\n✅ HOÀN TẤT! AI đã trở thành chuyên gia FDA.")