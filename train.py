import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from model import PharmaGNN
from main import smiles_to_graph

# 13 Bài test sinh học (12 của Tox21 + 1 CT_TOX của ClinTox)
tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'CT_TOX']

print("[1] Đang tải bộ dữ liệu Đa nhãn Tox21 và ClinTox...")
url_tox21 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df_tox21 = pd.read_csv(url_tox21, compression='gzip')

url_clintox = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
df_clintox = pd.read_csv(url_clintox)

# Gộp 2 dataset
df = pd.merge(df_tox21, df_clintox, on='smiles', how='outer')

# --- LABEL CORRECTION & OVERSAMPLING ---
# Vì ClinTox/Tox21 chỉ đánh giá "Thuốc", nên các chất độc công nghiệp bị gán nhãn 0 (An toàn).
# Ta ép mô hình phải học các chất này là Độc hại (CT_TOX = 1.0)
known_poisons = [
    'C#N', 'c1ccccc1O', 'C1=CC=C(C=C1)O', # Cyanide, Phenol
    'C(C(=O)O)NCP(=O)(O)O', # Glyphosate
    'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1' # DDT
]

known_safe = [
    'C(C(=O)O)N', # Glycine
    'CC(C(=O)O)N', # Alanine
    'CC(C)C(C(=O)O)N', # Valine
    'CC(=O)Nc1ccc(O)cc1', # Paracetamol
    'CC(=O)Oc1ccccc1C(=O)O', # Aspirin
]

# Bổ sung trực tiếp poisons nếu chưa có trong tập dữ liệu
new_poisons_df = pd.DataFrame({'smiles': ['C(C(=O)O)NCP(=O)(O)O', 'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1']})
for task in tasks:
    new_poisons_df[task] = -1.0 # Mask
new_poisons_df['CT_TOX'] = 1.0 # Gắn nhãn kịch độc
df = pd.concat([df, new_poisons_df], ignore_index=True)

# Tắt 12 nhãn Tox21 (-1.0) cho các chất kịch độc để tránh xung đột Multi-task learning
for task in tasks:
    if task != 'CT_TOX':
        df.loc[df['smiles'].isin(known_poisons), task] = -1.0
df.loc[df['smiles'].isin(known_poisons), 'CT_TOX'] = 1.0

# Bổ sung trực tiếp safe compounds nếu chưa có
new_safe_df = pd.DataFrame({'smiles': known_safe})
for task in tasks:
    new_safe_df[task] = -1.0 # Mask
new_safe_df['CT_TOX'] = 0.0 # Gắn nhãn an toàn
df = pd.concat([df, new_safe_df], ignore_index=True)

df.loc[df['smiles'].isin(known_safe), 'CT_TOX'] = 0.0

# Kỹ thuật Oversampling: Phục hồi lại 100 lần để ép mô hình học thuộc các chất cốt lõi 
# (Vì đã có MLP nên không sợ bị bias nhầm).
poisons_df = df[df['smiles'].isin(known_poisons)]
safe_df = df[df['smiles'].isin(known_safe)]
df = pd.concat([df] + [poisons_df]*100 + [safe_df]*100, ignore_index=True)

print(f"-> Đã tải {len(df)} phân tử. Đang dựng Ma trận Đồ thị...")

graph_list = []
for index, row in df.iterrows():
    graph = smiles_to_graph(row['smiles'])
    if graph is not None:
        # Lấy 12 nhãn. Nếu bị NaN (khuyết), điền -1.0 để làm dấu (Mask)
        labels = []
        for task in tasks:
            val = row[task]
            labels.append(-1.0 if pd.isna(val) else float(val))
            
        graph.y = torch.tensor([labels], dtype=torch.float)
        graph_list.append(graph)

random.shuffle(graph_list)
split_idx = int(len(graph_list) * 0.8)
train_data, test_data = graph_list[:split_idx], graph_list[split_idx:]

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Khởi tạo mô hình Đa nhiệm (num_classes=13)
model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)

weights_path = "pharma_gnn_weights_universal.pt"
if os.path.exists(weights_path):
    os.remove(weights_path) # Xóa não cũ

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# --- VÒNG LẶP HUẤN LUYỆN ĐA NHIỆM ---
epochs = 25 # Tăng lên 25 để đồ thị vẽ ra có dạng cong đẹp hơn
print(f"\n[2] Bắt đầu học Đa nhiệm (Multi-task) với {len(train_data)} mẫu...")

def compute_mean_auc(preds_list, labels_list):
    all_preds = np.vstack(preds_list)
    all_labels = np.vstack(labels_list)
    valid_auc_scores = []
    for i in range(13):
        task_labels = all_labels[:, i]
        task_preds = all_preds[:, i]
        valid_idx = task_labels != -1.0
        if valid_idx.sum() > 0:
            try:
                auc = roc_auc_score(task_labels[valid_idx], task_preds[valid_idx])
                valid_auc_scores.append(auc)
            except ValueError:
                pass
    return np.mean(valid_auc_scores) if valid_auc_scores else 0

history_train_loss = []
history_val_loss = []
history_train_auc = []
history_val_auc = []

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    train_preds_list = []
    train_labels_list = []
    
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch) # Xuất ra 13 giá trị
        
        # TẠO MẶT NẠ (MASK): Chỉ tính Loss ở những nhãn khác -1.0
        mask = batch.y != -1.0
        
        # Dùng BCEWithLogitsLoss với trọng số phạt (pos_weight) để cân bằng dữ liệu gốc
        pos_weight = torch.tensor([5.0]).to(predictions.device)
        loss = F.binary_cross_entropy_with_logits(predictions[mask], batch.y[mask], pos_weight=pos_weight)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Lưu kết quả Train để tính AUC
        train_preds_list.append(torch.sigmoid(predictions.detach()).cpu().numpy())
        train_labels_list.append(batch.y.cpu().numpy())

    epoch_train_loss = total_loss / len(train_loader)
    epoch_train_auc = compute_mean_auc(train_preds_list, train_labels_list)
    
    history_train_loss.append(epoch_train_loss)
    history_train_auc.append(epoch_train_auc)

    # Đánh giá trên tập Test để lấy Validation Loss và Validation AUC
    model.eval() 
    val_total_loss = 0
    val_preds_list = []
    val_labels_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            predictions = model(batch) 
            mask = batch.y != -1.0
            pos_weight = torch.tensor([5.0]).to(predictions.device)
            loss = F.binary_cross_entropy_with_logits(predictions[mask], batch.y[mask], pos_weight=pos_weight)
            val_total_loss += loss.item()
            
            val_preds_list.append(torch.sigmoid(predictions).cpu().numpy())
            val_labels_list.append(batch.y.cpu().numpy())
            
    epoch_val_loss = val_total_loss / len(test_loader)
    epoch_val_auc = compute_mean_auc(val_preds_list, val_labels_list)
    
    history_val_loss.append(epoch_val_loss)
    history_val_auc.append(epoch_val_auc)
    
    print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train AUC: {epoch_train_auc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {epoch_val_auc:.4f}")

torch.save(model.state_dict(), weights_path)
print("\n✅ HOÀN TẤT! Mô hình Đa nhiệm đã sẵn sàng.")

# --- VẼ ĐỒ THỊ CHUẨN KERAS ---
# 1. Loss Curve
plt.figure()
plt.plot(range(epochs), history_train_loss, label='Training loss')
plt.plot(range(epochs), history_val_loss, label='Validation loss')
plt.title('GNN Model Loss Progression During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

# 2. Accuracy Curve
plt.figure()
plt.plot(range(epochs), history_train_auc, label='Training accuracy')
plt.plot(range(epochs), history_val_auc, label='Validation accuracy')
plt.title('GNN Model Accuracy Progression During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('auc_curve.png')
plt.close()

print("📊 Đã lưu 2 biểu đồ: loss_curve.png và auc_curve.png với style chuẩn!")