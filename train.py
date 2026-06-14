"""
train.py - Kịch bản huấn luyện (Training Script) cho mô hình PharmaGNN.
Nhiệm vụ: Tải dữ liệu, tiền xử lý (cân bằng nhãn, xử lý giá trị khuyết), 
dựng đồ thị, huấn luyện mô hình đa nhiệm, và vẽ biểu đồ đánh giá.
"""

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

# 13 Bài test sinh học (12 của Tox21 đo lường tương tác protein + 1 CT_TOX của ClinTox đo độc tính lâm sàng)
tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'CT_TOX']

print("[1] Đang tải bộ dữ liệu Đa nhãn Tox21 và ClinTox...")
# Tải bộ dữ liệu Tox21 (chứa hàng ngàn phân tử test trên 12 mục tiêu sinh học)
url_tox21 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df_tox21 = pd.read_csv(url_tox21, compression='gzip')

# Tải bộ dữ liệu ClinTox (chứa kết quả lâm sàng: thuốc được FDA duyệt hay thất bại do độc)
url_clintox = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
df_clintox = pd.read_csv(url_clintox)

# Gộp 2 dataset lại với nhau dựa trên chuỗi SMILES
df = pd.merge(df_tox21, df_clintox, on='smiles', how='outer')

# --- LABEL CORRECTION & OVERSAMPLING (Sửa nhãn & Cân bằng dữ liệu) ---
# Vì ClinTox/Tox21 chủ yếu chứa "Thuốc", nên các chất độc công nghiệp bị thiếu hoặc gán nhãn 0 (An toàn).
# Ta ép mô hình phải học các chất kịch độc này (gán CT_TOX = 1.0) để nó không bị "ngây thơ"
known_poisons = [
    'C#N', 'Oc1ccccc1', # Cyanide, Phenol (Sử dụng chuẩn SMILES 'Oc1ccccc1' có trong bộ dữ liệu)
    'C(C(=O)O)NCP(=O)(O)O', # Glyphosate (Thuốc diệt cỏ)
    'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1' # DDT (Thuốc trừ sâu kịch độc)
]

# Danh sách các chất chắc chắn an toàn (Axit amin, thuốc giảm đau phổ biến)
known_safe = [
    'C(C(=O)O)N', # Glycine
    'CC(C(=O)O)N', # Alanine
    'CC(C)C(C(=O)O)N', # Valine
    'CC(=O)Nc1ccc(O)cc1', # Paracetamol
    'CC(=O)Oc1ccccc1C(=O)O', # Aspirin
]

# Bổ sung trực tiếp TẤT CẢ poisons vào tập dữ liệu (để đảm bảo không bị sót do khác biệt format SMILES)
new_poisons_df = pd.DataFrame({'smiles': known_poisons})
for task in tasks:
    new_poisons_df[task] = -1.0 # Mask (-1.0 là dấu hiệu bỏ qua lúc tính Loss, vì ta không có data Tox21 cho các chất này)
new_poisons_df['CT_TOX'] = 1.0 # Gắn nhãn kịch độc (1.0) cho bài test lâm sàng
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

# Kỹ thuật Oversampling: Nhân bản 100 lần các chất cốt lõi (Poison/Safe)
# Việc này ép mô hình phải "học thuộc lòng" các mầm mống độc hại cơ bản trước khi xử lý đồ thị phức tạp.
poisons_df = df[df['smiles'].isin(known_poisons)]
safe_df = df[df['smiles'].isin(known_safe)]
df = pd.concat([df] + [poisons_df]*100 + [safe_df]*100, ignore_index=True)

print(f"-> Đã tải {len(df)} phân tử. Đang dựng Ma trận Đồ thị...")

# --- DỰNG ĐỒ THỊ VÀ XỬ LÝ NHÃN (LABEL MASKING) ---
graph_list = []
for index, row in df.iterrows():
    graph = smiles_to_graph(row['smiles'])
    if graph is not None:
        labels = []
        for task in tasks:
            val = row[task]
            # Nếu bộ dữ liệu không có thông tin (NaN) cho bài test này, điền -1.0 làm mặt nạ (Mask)
            labels.append(-1.0 if pd.isna(val) else float(val))
            
        graph.y = torch.tensor([labels], dtype=torch.float)
        graph_list.append(graph)

# Xáo trộn dữ liệu và chia tập Train/Test theo tỷ lệ 80-20
random.shuffle(graph_list)
split_idx = int(len(graph_list) * 0.8)
train_data, test_data = graph_list[:split_idx], graph_list[split_idx:]

# Đóng gói dữ liệu thành các Lô (Batch) kích thước 64 để đưa vào GPU/CPU
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Khởi tạo mô hình Đa nhiệm (num_classes=13)
model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)

# Đường dẫn lưu file trọng số duy nhất của hệ thống
weights_path = "pharma_gnn_weights_universal.pt"
if os.path.exists(weights_path):
    os.remove(weights_path) # Xóa não cũ trước khi train lại

# Bộ tối ưu hóa Adam (giúp mô hình cập nhật kiến thức)
# Bổ sung L2 Regularization (weight_decay=1e-4) để phạt các trọng số quá lớn, chống Overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# --- VÒNG LẶP HUẤN LUYỆN ĐA NHIỆM ---
epochs = 25 # Số vòng lặp huấn luyện (Đủ lớn để đồ thị có đường cong rõ rệt)
print(f"\n[2] Bắt đầu học Đa nhiệm (Multi-task) với {len(train_data)} mẫu...")

def compute_mean_auc(preds_list, labels_list):
    """
    Hàm tính điểm số ROC-AUC trung bình cho cả 13 bài test.
    AUC đánh giá khả năng mô hình phân định ranh giới giữa Độc và Không Độc (tốt hơn dùng Accuracy).
    """
    all_preds = np.vstack(preds_list)
    all_labels = np.vstack(labels_list)
    valid_auc_scores = []
    for i in range(13):
        task_labels = all_labels[:, i]
        task_preds = all_preds[:, i]
        # BỎ QUA CÁC Ô KHUYẾT DỮ LIỆU: Chỉ tính AUC trên các mẫu có nhãn khác -1.0
        valid_idx = task_labels != -1.0
        if valid_idx.sum() > 0:
            try:
                auc = roc_auc_score(task_labels[valid_idx], task_preds[valid_idx])
                valid_auc_scores.append(auc)
            except ValueError:
                pass # Bỏ qua nếu cột đó chỉ toàn 1 class (vd: toàn an toàn)
    return np.mean(valid_auc_scores) if valid_auc_scores else 0

# Các mảng lưu trữ lịch sử để vẽ đồ thị
history_train_loss = []
history_val_loss = []
history_train_auc = []
history_val_auc = []

# Khởi tạo biến để theo dõi và thực hiện Early Stopping (Dừng sớm / Lưu model tốt nhất)
best_val_auc = 0.0
best_epoch = 0

for epoch in range(epochs):
    # Kích hoạt chế độ Huấn luyện (Bật Dropout)
    model.train() 
    total_loss = 0
    train_preds_list = []
    train_labels_list = []
    
    for batch in train_loader:
        optimizer.zero_grad() # Xóa gradient cũ
        predictions = model(batch) # Suy luận: Xuất ra 13 điểm số thô (Logits)
        
        # TẠO MẶT NẠ (MASK): Lọc ra các điểm dữ liệu hợp lệ (nhãn khác -1.0)
        mask = batch.y != -1.0
        
        # Trọng số phạt (Class Imbalance Penalty): Đặt 5.0
        # Lý do: Đoán sai 1 chất ĐỘC thành AN TOÀN nguy hiểm hơn đoán sai chất AN TOÀN thành ĐỘC.
        # Do đó mô hình bị phạt gấp 5 lần nếu bỏ lọt chất độc.
        pos_weight = torch.tensor([5.0]).to(predictions.device)
        
        # Hàm Loss: BCEWithLogitsLoss tính toán sai số. Chỉ tính trên phần dữ liệu không bị Mask.
        loss = F.binary_cross_entropy_with_logits(predictions[mask], batch.y[mask], pos_weight=pos_weight)
        
        loss.backward() # Lan truyền ngược tính đạo hàm
        optimizer.step() # Cập nhật trọng số
        total_loss += loss.item()
        
        # Lưu kết quả Train (đã qua Sigmoid) để tính AUC
        train_preds_list.append(torch.sigmoid(predictions.detach()).cpu().numpy())
        train_labels_list.append(batch.y.cpu().numpy())

    epoch_train_loss = total_loss / len(train_loader)
    epoch_train_auc = compute_mean_auc(train_preds_list, train_labels_list)
    
    history_train_loss.append(epoch_train_loss)
    history_train_auc.append(epoch_train_auc)

    # Kích hoạt chế độ Đánh giá trên tập Test (Tắt Dropout)
    model.eval() 
    val_total_loss = 0
    val_preds_list = []
    val_labels_list = []
    
    with torch.no_grad(): # Không tính đạo hàm lúc Test để chạy nhanh hơn
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
    
    # EARLY STOPPING & CHECKPOINTING: Chỉ lưu mô hình khi có kết quả trên tập Validation tốt hơn
    if epoch_val_auc > best_val_auc:
        best_val_auc = epoch_val_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), weights_path)
        print(f"   => Đã lưu mô hình tốt nhất mới tại Epoch {best_epoch} (Val AUC: {best_val_auc:.4f})")

print(f"\n✅ HOÀN TẤT! Mô hình Đa nhiệm đã sẵn sàng (Đã lưu bản tốt nhất ở Epoch {best_epoch}).")

# --- VẼ ĐỒ THỊ CHUẨN KERAS ---
# 1. Biểu đồ Loss Curve (Mức độ sai lệch)
plt.figure()
plt.plot(range(epochs), history_train_loss, label='Training loss')
plt.plot(range(epochs), history_val_loss, label='Validation loss')
plt.title('GNN Model Loss Progression During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

# 2. Biểu đồ Accuracy/AUC Curve (Mức độ chính xác phân loại)
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