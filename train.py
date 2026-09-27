import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import random
import os
import numpy as np
import math
from sklearn.metrics import roc_auc_score

from model import PharmaGNN
from main import smiles_to_graph

# 13 biological assay tasks (12 Tox21 targets + 1 ClinTox CT_TOX)
tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'CT_TOX']

print("[1] Loading multi-task datasets: Tox21 and ClinTox...")
url_tox21 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df_tox21 = pd.read_csv(url_tox21, compression='gzip')

url_clintox = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
df_clintox = pd.read_csv(url_clintox)

# Merge datasets
df = pd.merge(df_tox21, df_clintox, on='smiles', how='outer')

# --- LABEL CORRECTION & OVERSAMPLING ---
# ClinTox/Tox21 predominantly evaluate approved drugs; certain industrial toxicants are unassigned or zero.
# We ensure known industrial poisons are explicitly labeled as toxic (CT_TOX = 1.0).
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
    # Additional safe compounds - common salts, sugars, food additives
    '[Na+].[Cl-]', # Sodium chloride (table salt)
    '[Na+].[F-]', # Sodium fluoride
    '[K+].[Cl-]', # Potassium chloride
    '[Ca+2].[Cl-].[Cl-]', # Calcium chloride
    '[NH4+].[Cl-]', # Ammonium chloride
    '[Na+][O-]S(=O)=O', # Sodium sulfate
    '[Na+][O-]S([O-])=O', # Sodium sulfite
    'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O', # Glucose
    'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O', # Fructose
    'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O', # Galactose
    'CC(=O)O', # Acetic acid (vinegar)
    'CCC(=O)O', # Propionic acid
    'CCCC(=O)O', # Butyric acid
    'CC(O)=O', # Lactic acid
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    'Cn1cnc2c(N=c1ncn2c3N(CH3)CH2CH3)N=C3', # Theophylline
    'COc1ccc2nc(nc2c1)N', # Adenine
    'O', # Water
    'CO', # Methanol
    'CCO', # Ethanol
    'CCOCCO', # Ethylene glycol
    'CCCCO', # Butanol
    'CC(=O)OCC(=O)O', # Ethyl acetate
    'CC(=O)O', # Acetic acid
    'CC(=O)OCC(=O)OCC(=O)O', # Triacetin
    'C1CCOCC1', # Tetrahydrofuran
    'C1COCCO1', # 1,4-dioxane
    'CC(=O)NC1=CC=CC=C1', # Acetanilide
    'CN1C(=O)NC2=NC=NC=C21', # Theophylline variant
    'OC[C@H]1OP(O)(O)=O[C@@H]2O[C@H](O)[C@@H](O)[C@H]2O', # Glucose-6-phosphate
    'CC(N)C(=O)O', # Alanine (alternate)
    'CC[C@H](N)C(=O)O', # Valine (alternate)
    'CC[C@@H](N)C(=O)O', # Valine (alternate)
]

# Inject reference poisons if missing from dataset
new_poisons_df = pd.DataFrame({'smiles': ['C(C(=O)O)NCP(=O)(O)O', 'Clc1ccc(C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl)cc1']})
for task in tasks:
    new_poisons_df[task] = -1.0 # Mask
new_poisons_df['CT_TOX'] = 1.0 # Toxic label
df = pd.concat([df, new_poisons_df], ignore_index=True)

# Mask out 12 Tox21 targets (-1.0) for pure chemical toxicants to prevent multi-task conflict
for task in tasks:
    if task != 'CT_TOX':
        df.loc[df['smiles'].isin(known_poisons), task] = -1.0
df.loc[df['smiles'].isin(known_poisons), 'CT_TOX'] = 1.0

# Inject safe reference compounds if missing
new_safe_df = pd.DataFrame({'smiles': known_safe})
for task in tasks:
    new_safe_df[task] = -1.0 # Mask
new_safe_df['CT_TOX'] = 0.0 # Safe label
df = pd.concat([df, new_safe_df], ignore_index=True)

df.loc[df['smiles'].isin(known_safe), 'CT_TOX'] = 0.0

# Add dietary, low-toxicity compounds to counterbalance toxicant oversampling
extra_safe_smiles = [
    '[Na+].[Cl-]',       # NaCl
    '[Na+].[F-]',       # NaF
    '[K+].[Cl-]',       # KCl
    '[Ca+2].[Cl-].[Cl-]', # CaCl2
    '[NH4+].[Cl-]',     # NH4Cl
    '[Na+][O-]S(=O)=O', # Na2SO4
    '[Na+][O-]S([O-])=O', # Na2SO3
    'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O', # Glucose
    'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O', # Fructose
    'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O', # Galactose
    'CC(=O)O',          # Acetic acid
    'CCC(=O)O',         # Propionic acid
    'CCCC(=O)O',        # Butyric acid
    'CC(O)=O',          # Lactic acid
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
    'Cn1cnc2c(N=c1ncn2c3N(CH3)CH2CH3)N=C3', # Theophylline
]
new_extra_safe_df = pd.DataFrame({'smiles': extra_safe_smiles})
for task in tasks:
    new_extra_safe_df[task] = -1.0
new_extra_safe_df['CT_TOX'] = 0.0
df = pd.concat([df, new_extra_safe_df], ignore_index=True)
df.loc[df['smiles'].isin(extra_safe_smiles), 'CT_TOX'] = 0.0

# Oversampling strategy: repeat critical reference compounds to ensure robust baseline calibration
poisons_df = df[df['smiles'].isin(known_poisons)]
safe_df = df[df['smiles'].isin(known_safe)]
df = pd.concat([df] + [poisons_df]*100 + [safe_df]*100, ignore_index=True)

print(f"-> Loaded {len(df)} compounds. Constructing molecular graph matrices...")

graph_list = []
for index, row in df.iterrows():
    graph = smiles_to_graph(row['smiles'], concentration_molar=1e-5)  # 10 µM screening concentration
    if graph is not None:
        # Extract 13 task labels; missing (NaN) values marked with -1.0 as loss mask
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

# Initialize multi-task model (num_classes=13)
model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)

weights_path = "pharma_gnn_weights_universal.pt"
if os.path.exists(weights_path):
    os.remove(weights_path)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# --- MULTI-TASK TRAINING LOOP ---
epochs = 10
print(f"\n[2] Starting multi-task training with {len(train_data)} samples...")

for epoch in range(epochs):
    model.train() 
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            edge_attr=getattr(batch, 'edge_attr', None),
            batch=batch.batch, 
            global_features=getattr(batch, 'global_features', None), 
            func_group_features=getattr(batch, 'func_group_features', None),
            concentration=getattr(batch, 'concentration', None)
        )
        
        # LOSS MASK: Only compute loss on valid labels (!= -1.0)
        mask = batch.y != -1.0
        
        # BCEWithLogitsLoss with positive class weighting (pos_weight) for class imbalance
        pos_weight = torch.tensor([5.0]).to(predictions.device)
        loss = F.binary_cross_entropy_with_logits(predictions[mask], batch.y[mask], pos_weight=pos_weight)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval() 
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                preds = torch.sigmoid(model(
                    x=batch.x, 
                    edge_index=batch.edge_index, 
                    edge_attr=getattr(batch, 'edge_attr', None),
                    batch=batch.batch, 
                    global_features=getattr(batch, 'global_features', None), 
                    func_group_features=getattr(batch, 'func_group_features', None),
                    concentration=getattr(batch, 'concentration', None)
                ))
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Compute mean ROC-AUC across all 13 evaluation tasks
        valid_auc_scores = []
        for i in range(13):
            task_labels = all_labels[:, i]
            task_preds = all_preds[:, i]
            # Exclusively compute on unmasked entries (!= -1)
            valid_idx = task_labels != -1.0
            if valid_idx.sum() > 0:
                try:
                    auc = roc_auc_score(task_labels[valid_idx], task_preds[valid_idx])
                    valid_auc_scores.append(auc)
                except ValueError:
                    pass
                    
        mean_auc = np.mean(valid_auc_scores) if valid_auc_scores else 0
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Mean ROC-AUC (13 Tasks): {mean_auc:.4f}")

torch.save(model.state_dict(), weights_path)
print("\n✅ Training complete! Multi-task GNN weights saved successfully.")