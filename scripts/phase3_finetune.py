#!/usr/bin/env python3
"""
Phase 3: Multi-task Fine-tuning on expanded dataset
Uses the current model weights and trains on all available data
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, MolStandardize
from sklearn.metrics import roc_auc_score
import math
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import PharmaGNN
from main import smiles_to_graph

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 
         'CT_TOX', 'FDA_APPROVED']

def standardize_smiles(smiles):
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def build_training_graphs(df, smiles_col='smiles'):
    """Build graph dataset from DataFrame"""
    graphs = []
    labels_list = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles):
            continue
        
        std_smiles = standardize_smiles(smiles)
        if std_smiles is None:
            continue
        
        g = smiles_to_graph(std_smiles, concentration_molar=1e-5)
        if g is None:
            continue
        
        # Extract labels for all tasks
        task_labels = []
        for task in TASKS:
            if task in row and not pd.isna(row[task]):
                task_labels.append(float(row[task]))
            else:
                task_labels.append(np.nan)
        
        g.y = torch.tensor(task_labels, dtype=torch.float).unsqueeze(0)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
        graphs.append(g)
    
    return graphs

def evaluate_model(model, graphs, device):
    """Evaluate model on graph dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            logits = model(
                x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, batch=g.batch,
                global_features=g.global_features, func_group_features=g.func_group_features,
                concentration=g.concentration
            )
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(g.y.cpu().numpy())
    
    if not all_preds:
        return {}
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    results = {}
    for i, task in enumerate(TASKS):
        mask = ~np.isnan(all_labels[:, i])
        if mask.sum() > 10:  # Need enough samples
            auc = roc_auc_score(all_labels[mask, i], all_preds[mask, i])
            results[task] = auc
    
    return results

def build_training_graphs(df, smiles_col='smiles'):
    """Build graph dataset from DataFrame"""
    graphs = []
    labels_list = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles):
            continue
        
        std_smiles = standardize_smiles(smiles)
        if std_smiles is None:
            continue
        
        g = smiles_to_graph(std_smiles, concentration_molar=1e-5)
        if g is None:
            continue
        
        # Extract labels for all tasks
        task_labels = []
        for task in TASKS:
            if task in row and not pd.isna(row[task]):
                task_labels.append(float(row[task]))
            else:
                task_labels.append(np.nan)
        
        g.y = torch.tensor(task_labels, dtype=torch.float).unsqueeze(0)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
        graphs.append(g)
    
    return graphs

def main():
    print("=" * 70)
    print("Phase 3: Multi-task Fine-tuning")
    print("=" * 70)
    
    # Load training dataset
    print("\n[1] Loading training data...")
    train_df = pd.read_csv(PROCESSED_DIR / "training_dataset.csv")
    print(f"Total: {len(train_df)} compounds")
    
    # Filter by split
    train_data = train_df[train_df['split'] == 'train']
    val_data = train_df[train_df['split'] == 'val']
    test_data = train_df[train_df['split'] == 'test']
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Build graphs
    print("\n[2] Building graphs...")
    train_graphs = build_training_graphs(train_data)
    val_graphs = build_training_graphs(val_data)
    test_graphs = build_training_graphs(test_data)
    
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Val graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    if len(train_graphs) < 100:
        print("Not enough training graphs!")
        return
    
    # Initialize model
    print("\n[3] Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=32,
        num_classes=len(TASKS),
        num_global_features=11,
        num_func_groups=85
    ).to(device)
    
    # Load existing weights
    weights_path = 'pharma_gnn_weights_universal.pt'
    if Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded weights from {weights_path}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    
    # Training loop
    print("\n[4] Starting fine-tuning...")
    epochs = 50
    best_val_auc = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(
                x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, 
                batch=batch.batch, global_features=batch.global_features,
                func_group_features=batch.func_group_features, concentration=batch.concentration
            )
            
            # Multi-task BCE with NaN masking
            target = batch.y
            mask = ~torch.isnan(target)
            
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        
        # Validation
        val_aucs = evaluate_model(model, val_graphs, device)
        avg_val_auc = np.mean(list(val_aucs.values())) if val_aucs else 0
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val AUC: {avg_val_auc:.4f}")
        for task, auc in sorted(val_aucs.items()):
            print(f"  {task}: {auc:.4f}")
        
        scheduler.step(avg_val_auc)
        
        if avg_val_auc > best_val_auc:
            best_val_auc = avg_val_auc
            torch.save(model.state_dict(), 'pharma_gnn_finetuned.pt')
            print(f"  ✓ Saved best model (AUC={avg_val_auc:.4f})")
    
    # Final test evaluation
    print("\n[5] Final test evaluation...")
    test_aucs = evaluate_model(model, test_graphs, device)
    print("Test AUCs:")
    for task, auc in sorted(test_aucs.items()):
        print(f"  {task}: {auc:.4f}")
    
    avg_test_auc = np.mean(list(test_aucs.values()))
    print(f"\nAverage Test AUC: {avg_test_auc:.4f}")
    
    print("\n" + "=" * 70)
    print("Fine-tuning complete!")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Avg Test AUC: {avg_test_auc:.4f}")
    print("Saved: pharma_gnn_finetuned.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()