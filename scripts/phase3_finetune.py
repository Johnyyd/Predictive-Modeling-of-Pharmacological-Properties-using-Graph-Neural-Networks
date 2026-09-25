#!/usr/bin/env python3
"""
Phase 3: Multi-task Fine-Tuning for PharmaGNN v2 Foundation Architecture
Transfers self-supervised pretrained encoder weights and conducts two-stage
fine-tuning across 13 biological and pharmacological toxicity endpoints.
"""

import sys
import math
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MolStandardize
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import PharmaGNN
from main import smiles_to_graph

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# 13 standard biological and pharmacological endpoints
TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 
    'CT_TOX'
]


def standardize_smiles(smiles: str) -> str:
    """Standardize SMILES into canonical form."""
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_backbone_param(name: str) -> bool:
    """Determine whether a parameter belongs to the GNN feature extraction backbone."""
    backbone_prefixes = ('conv1', 'bn1', 'conv2', 'bn2', 'extra_convs', 'extra_bns')
    return any(name.startswith(p) for p in backbone_prefixes)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters for Stage 1 head warmup."""
    for name, param in model.named_parameters():
        if is_backbone_param(name):
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters for Stage 2 end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def create_fine_tune_optimizer(model: nn.Module, lr_backbone: float = 1e-4, 
                               lr_head: float = 1e-3, weight_decay: float = 1e-4):
    """Create optimizer with discriminative learning rates for backbone vs downstream heads."""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_backbone_param(name):
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay}
    ]
    return torch.optim.AdamW(param_groups)


def load_pretrained_encoder(model: nn.Module, weights_path: str, device: torch.device) -> bool:
    """Load matching encoder weights from Phase 2 SSL pretraining checkpoint or legacy weights."""
    if not weights_path:
        return False
    p = Path(weights_path)
    if not p.exists():
        return False
        
    try:
        checkpoint = torch.load(p, map_location=device)
        model_dict = model.state_dict()
        
        # Match only encoder/backbone layers where shapes match exactly
        matched_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                # Do not overwrite prediction heads if shape doesn't match task count
                if any(head in k for head in ['pred_head', 'motif_head', 'context_head']):
                    continue
                matched_dict[k] = v
                
        if matched_dict:
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            print(f"✓ Successfully transferred {len(matched_dict)} tensors from {weights_path}")
            return True
        else:
            print(f"[!] No compatible shape-matched tensors found in {weights_path}")
            return False
    except Exception as e:
        print(f"[!] Warning: Failed loading pretrained encoder from {weights_path}: {e}")
        return False


def build_training_graphs(df: pd.DataFrame, smiles_col: str = 'smiles') -> list:
    """Convert dataframe rows to PyTorch Geometric Data instances with 13-task label vectors."""
    graphs = []
    
    for idx, row in df.iterrows():
        smiles = row.get(smiles_col)
        if pd.isna(smiles):
            continue
        
        std_smiles = standardize_smiles(smiles)
        if std_smiles is None:
            continue
        
        g = smiles_to_graph(std_smiles, concentration_molar=1e-5)
        if g is None:
            continue
        
        # Extract labels for all 13 tasks
        task_labels = []
        for task in TASKS:
            val = row.get(task, np.nan)
            if not pd.isna(val):
                try:
                    task_labels.append(float(val))
                except (ValueError, TypeError):
                    task_labels.append(np.nan)
            else:
                task_labels.append(np.nan)
        
        g.y = torch.tensor(task_labels, dtype=torch.float).unsqueeze(0)
        graphs.append(g)
    
    return graphs


def evaluate_model(model: nn.Module, graphs: list, device: torch.device, min_samples: int = 10) -> dict:
    """Evaluate model on graph list and calculate ROC-AUC per task."""
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
        if mask.sum() >= min_samples:
            targets = all_labels[mask, i]
            # ROC-AUC requires both positive and negative classes
            if len(np.unique(targets)) > 1:
                try:
                    auc = roc_auc_score(targets, all_preds[mask, i])
                    results[task] = float(auc)
                except Exception:
                    pass
    
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3: Multi-task Fine-Tuning for PharmaGNN v2 Foundation")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification run on small subset")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of dataset rows to use")
    parser.add_argument("--epochs", type=int, default=30, help="Total fine-tuning epochs")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Backbone freeze warmup epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden channels (128 for v2 foundation)")
    parser.add_argument("--num-layers", type=int, default=4, help="GATv2 layers (4 for v2 foundation)")
    parser.add_argument("--heads", type=int, default=4, help="GATv2 attention heads")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head & interaction learning rate")
    parser.add_argument("--pretrained-weights", type=str, default="pharma_gnn_pretrained_encoder.pt", help="Pretrained encoder checkpoint")
    parser.add_argument("--output-weights", type=str, default="pharma_gnn_finetuned.pt", help="Output model checkpoint")
    args = parser.parse_args(argv)

    print("=" * 70)
    print("Phase 3: Multi-Task Downstream Fine-Tuning (PharmaGNN v2 Foundation)")
    print("=" * 70)

    if args.smoke_test:
        print("[!] Smoke-test mode active: configuring fast verification run")
        args.limit = min(args.limit or 200, 200)
        args.epochs = min(args.epochs, 3)
        args.warmup_epochs = min(args.warmup_epochs, 1)

    # 1. Load training dataset
    print("\n[1] Loading dataset...")
    train_path = PROCESSED_DIR / "training_dataset_v2.csv"
    if not train_path.exists():
        train_path = PROCESSED_DIR / "training_dataset.csv"
    if not train_path.exists():
        train_path = PROCESSED_DIR / "master_toxicity_dataset.csv"

    print(f"Reading from: {train_path}")
    df = pd.read_csv(train_path)
    if args.limit:
        df = df.iloc[:args.limit].copy()
    print(f"Loaded {len(df)} compounds")

    # Splits
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
    else:
        np.random.seed(42)
        perm = np.random.permutation(len(df))
        n_train = int(0.8 * len(df))
        n_val = int(0.1 * len(df))
        train_df = df.iloc[perm[:n_train]]
        val_df = df.iloc[perm[n_train:n_train+n_val]]
        test_df = df.iloc[perm[n_train+n_val:]]

    print(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Build graph datasets
    print("\n[2] Building graph representations...")
    train_graphs = build_training_graphs(train_df)
    val_graphs = build_training_graphs(val_df)
    test_graphs = build_training_graphs(test_df)
    print(f"Graphs -> Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    if len(train_graphs) < 5:
        print("Error: insufficient valid graphs constructed.")
        return False

    # 3. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[3] Initializing model on device {device} (hidden={args.hidden_channels}, layers={args.num_layers}, heads={args.heads})...")
    model = PharmaGNN(
        num_node_features=6,
        hidden_channels=args.hidden_channels,
        num_classes=len(TASKS),
        num_global_features=11,
        num_func_groups=85,
        num_layers=args.num_layers,
        heads=args.heads,
        residual=True,
        fg_embed_dim=16 if args.hidden_channels > 32 else 8
    ).to(device)

    # Transfer pretrained encoder weights if available
    transferred = load_pretrained_encoder(model, args.pretrained_weights, device)
    if not transferred and Path('pharma_gnn_weights_universal.pt').exists():
        load_pretrained_encoder(model, 'pharma_gnn_weights_universal.pt', device)

    # Dataloaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    # Positive class weighting to balance rare toxicity positives
    pos_weight = torch.tensor([5.0], device=device)

    # 4. Two-Stage Fine-Tuning Loop
    print(f"\n[4] Starting Fine-Tuning ({args.epochs} total epochs, {args.warmup_epochs} warmup epochs)...")
    best_val_auc = 0.0
    history = []

    for epoch in range(args.epochs):
        # Stage 1 vs Stage 2 switching
        if epoch < args.warmup_epochs:
            stage_name = "Stage 1 (Head Warmup)"
            freeze_backbone(model)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr_head,
                weight_decay=1e-4
            )
        else:
            stage_name = "Stage 2 (End-to-End Fine-Tuning)"
            unfreeze_backbone(model)
            optimizer = create_fine_tune_optimizer(
                model,
                lr_backbone=args.lr_backbone,
                lr_head=args.lr_head
            )

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

            targets = batch.y
            mask = ~torch.isnan(targets)

            if mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask], targets[mask], pos_weight=pos_weight
                )
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0

        # Evaluate validation
        val_aucs = evaluate_model(model, val_graphs, device, min_samples=2 if args.smoke_test else 10)
        avg_val_auc = np.mean(list(val_aucs.values())) if val_aucs else 0.0
        history.append((avg_train_loss, avg_val_auc))

        print(f"Epoch {epoch+1:2d}/{args.epochs} [{stage_name}] | Train Loss: {avg_train_loss:.4f} | Val AUC: {avg_val_auc:.4f}")

        if avg_val_auc >= best_val_auc:
            best_val_auc = avg_val_auc
            torch.save(model.state_dict(), args.output_weights)
            print(f"  ✓ Saved best checkpoint ({args.output_weights})")

    # 5. Final Test Evaluation
    print("\n[5] Final Test Evaluation...")
    test_aucs = evaluate_model(model, test_graphs, device, min_samples=2 if args.smoke_test else 10)
    print("Test AUCs per task:")
    for t, score in sorted(test_aucs.items()):
        print(f"  {t:15s}: {score:.4f}")
    avg_test_auc = np.mean(list(test_aucs.values())) if test_aucs else 0.0
    print(f"Average Test AUC: {avg_test_auc:.4f}")

    print("\n" + "=" * 70)
    print("Phase 3 Fine-Tuning Execution Complete!")
    print(f"Best Val AUC: {best_val_auc:.4f} | Final Test AUC: {avg_test_auc:.4f}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()