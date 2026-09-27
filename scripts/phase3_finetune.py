#!/usr/bin/env python3
"""
Phase 3: Multi-task Fine-Tuning for PharmaGNN v2 Foundation Architecture
Transfers self-supervised pretrained encoder weights and conducts two-stage
fine-tuning across 13 biological and pharmacological toxicity endpoints.
"""

import sys
import os
import math
import time
import datetime
import csv
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
CACHE_PATH = PROCESSED_DIR / "finetune_graphs_cache.pt"

# 13 standard biological and pharmacological endpoints
TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 
    'CT_TOX'
]


def format_duration(seconds: float) -> str:
    """Format duration into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem_sec = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {rem_sec:02d}s"
    hours = int(minutes // 60)
    rem_min = int(minutes % 60)
    return f"{hours}h {rem_min:02d}m {rem_sec:02d}s"


class FinetuneLogger:
    """Dual console and file logger with structured metrics streaming to CSV."""

    def __init__(self, log_file: str = "logs/phase3_finetune.log", metrics_file: str = "logs/finetune_metrics.csv"):
        self.log_file = Path(log_file) if log_file else None
        self.metrics_file = Path(metrics_file) if metrics_file else None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*75}\n[FINE-TUNING SESSION START] {ts}\n{'='*75}\n")

        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.metrics_file.exists() or self.metrics_file.stat().st_size == 0:
                with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "epoch", "stage", "train_loss", "val_auc",
                        "lr_backbone", "lr_head", "epoch_time_sec", "best_val_auc", "is_best"
                    ])

    def log(self, msg: str = "", to_file: bool = True):
        """Print to stdout and append to persistent log file."""
        print(msg, flush=True)
        if to_file and self.log_file:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    if msg.strip() == "" or msg.startswith("=") or msg.startswith("-"):
                        f.write(f"{msg}\n")
                    else:
                        f.write(f"[{ts}] {msg}\n")
            except Exception:
                pass

    def log_metrics(self, epoch: int, stage: str, train_loss: float, val_auc: float,
                    lr_backbone: float, lr_head: float, epoch_time: float,
                    best_val_auc: float, is_best: bool):
        """Append epoch summary metrics to CSV."""
        if not self.metrics_file:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.metrics_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts,
                    epoch,
                    stage,
                    f"{train_loss:.4f}",
                    f"{val_auc:.4f}",
                    f"{lr_backbone:.6e}",
                    f"{lr_head:.6e}",
                    f"{epoch_time:.2f}",
                    f"{best_val_auc:.4f}",
                    int(is_best)
                ])
        except Exception:
            pass


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


def get_or_build_graphs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                        use_cache: bool = True, logger: FinetuneLogger = None) -> tuple:
    """Load preprocessed graphs from disk cache or construct and cache them."""
    if use_cache and CACHE_PATH.exists():
        try:
            cached = torch.load(CACHE_PATH, weights_only=False)
            if isinstance(cached, dict) and 'train' in cached and 'val' in cached and 'test' in cached:
                if (len(cached['train']) == len(train_df) and 
                    len(cached['val']) == len(val_df) and 
                    len(cached['test']) == len(test_df)):
                    if logger:
                        logger.log(f"✓ Loaded {len(cached['train']) + len(cached['val']) + len(cached['test'])} pre-featurized graphs from cache ({CACHE_PATH})")
                    return cached['train'], cached['val'], cached['test']
        except Exception as e:
            if logger:
                logger.log(f"[!] Graph cache read error ({e}); rebuilding graphs...")

    if logger:
        logger.log("Building graph representations from SMILES...")
    t0 = time.time()
    train_graphs = build_training_graphs(train_df)
    val_graphs = build_training_graphs(val_df)
    test_graphs = build_training_graphs(test_df)
    elapsed = max(time.time() - t0, 1e-4)
    total_graphs = len(train_graphs) + len(val_graphs) + len(test_graphs)
    if logger:
        logger.log(f"✓ Featurized {total_graphs} graphs in {elapsed:.2f}s ({total_graphs/elapsed:.1f} graphs/sec)")

    if use_cache:
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'train': train_graphs, 'val': val_graphs, 'test': test_graphs}, CACHE_PATH)
            if logger:
                logger.log(f"✓ Saved graph cache to {CACHE_PATH}")
        except Exception as e:
            if logger:
                logger.log(f"[!] Warning: failed saving graph cache: {e}")

    return train_graphs, val_graphs, test_graphs


def evaluate_model(model: nn.Module, graphs: list, device: torch.device, 
                   min_samples: int = 10, batch_size: int = 64,
                   return_loss: bool = False, pos_weight: torch.Tensor = None):
    """Evaluate model on graph list and calculate ROC-AUC per task using batched inference."""
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0.0
    val_batches_count = 0
    
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False) if isinstance(graphs, list) else graphs

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch,
                global_features=batch.global_features, func_group_features=batch.func_group_features,
                concentration=batch.concentration
            )
            if return_loss:
                mask = ~torch.isnan(batch.y)
                if mask.any():
                    pw = pos_weight if pos_weight is not None else torch.tensor([5.0], device=device)
                    l = F.binary_cross_entropy_with_logits(logits[mask], batch.y[mask], pos_weight=pw)
                    total_val_loss += l.item()
                    val_batches_count += 1

            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    
    avg_loss = (total_val_loss / val_batches_count) if val_batches_count > 0 else 0.0

    if not all_preds:
        return ({}, avg_loss) if return_loss else {}
    
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
    
    return (results, avg_loss) if return_loss else results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3: Multi-task Fine-Tuning for PharmaGNN v2 Foundation")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast verification run on small subset")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of dataset rows to use")
    parser.add_argument("--epochs", type=int, default=30, help="Total fine-tuning epochs")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Backbone freeze warmup epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden channels (128 for v2 foundation)")
    parser.add_argument("--num-layers", type=int, default=4, help="GATv2 layers (4 for v2 foundation)")
    parser.add_argument("--heads", type=int, default=4, help="GATv2 attention heads")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head & interaction learning rate")
    parser.add_argument("--pretrained-weights", type=str, default="pharma_gnn_pretrained_encoder.pt", help="Pretrained encoder checkpoint")
    parser.add_argument("--output-weights", type=str, default="pharma_gnn_finetuned.pt", help="Output model checkpoint")
    parser.add_argument("--output-loss-plot", type=str, default="finetune_loss_curve.png", help="Path to save fine-tuning loss plot")
    parser.add_argument("--output-auc-plot", type=str, default="finetune_auc_curve.png", help="Path to save fine-tuning ROC-AUC plot")
    parser.add_argument("--log-file", type=str, default="logs/phase3_finetune.log", help="Path to text log file")
    parser.add_argument("--metrics-file", type=str, default="logs/finetune_metrics.csv", help="Path to CSV metrics file")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk graph caching")
    args = parser.parse_args(argv)

    logger = FinetuneLogger(log_file=args.log_file, metrics_file=args.metrics_file)

    logger.log("=" * 75)
    logger.log("Phase 3: Multi-Task Downstream Fine-Tuning (PharmaGNN v2 Foundation)")
    logger.log("=" * 75)

    if args.smoke_test:
        logger.log("[!] Smoke-test mode active: configuring fast verification run")
        args.limit = min(args.limit or 200, 200)
        args.epochs = min(args.epochs, 3)
        args.warmup_epochs = min(args.warmup_epochs, 1)
        args.patience = 3

    # 1. Load training dataset
    logger.log("\n[1] Loading dataset...")
    train_path = PROCESSED_DIR / "training_dataset_v2.csv"
    if not train_path.exists():
        train_path = PROCESSED_DIR / "training_dataset.csv"
    if not train_path.exists():
        train_path = PROCESSED_DIR / "master_toxicity_dataset.csv"

    logger.log(f"Reading from: {train_path}")
    df = pd.read_csv(train_path)
    if args.limit:
        df = df.iloc[:args.limit].copy()
    logger.log(f"Loaded {len(df)} compounds across 13 target endpoints")

    # Splits
    if 'split' in df.columns and not args.limit:
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
    else:
        np.random.seed(42)
        perm = np.random.permutation(len(df))
        n_train = int(0.8 * len(df))
        n_val = int(0.1 * len(df))
        train_df = df.iloc[perm[:n_train]].reset_index(drop=True)
        val_df = df.iloc[perm[n_train:n_train+n_val]].reset_index(drop=True)
        test_df = df.iloc[perm[n_train+n_val:]].reset_index(drop=True)

    logger.log(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Build or load graph datasets
    logger.log("\n[2] Preparing graph representations...")
    use_cache = (not args.no_cache) and (args.limit is None)
    train_graphs, val_graphs, test_graphs = get_or_build_graphs(
        train_df, val_df, test_df, use_cache=use_cache, logger=logger
    )
    logger.log(f"Graphs ready -> Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    if len(train_graphs) < 5:
        logger.log("Error: insufficient valid graphs constructed.")
        return False

    # 3. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"\n[3] Initializing model on device {device} (hidden={args.hidden_channels}, layers={args.num_layers}, heads={args.heads})...")
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
    if transferred:
        logger.log(f"✓ Transferred pretrained backbone encoder from {args.pretrained_weights}")
    elif Path('pharma_gnn_weights_universal.pt').exists():
        loaded_legacy = load_pretrained_encoder(model, 'pharma_gnn_weights_universal.pt', device)
        if loaded_legacy:
            logger.log("✓ Transferred baseline weights from pharma_gnn_weights_universal.pt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"PharmaGNN Architecture -> Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Dataloaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)

    # Positive class weighting to balance rare toxicity positives
    pos_weight = torch.tensor([5.0], device=device)

    # 4. Two-Stage Fine-Tuning Loop
    logger.log(f"\n[4] Starting Fine-Tuning ({args.epochs} total epochs, {args.warmup_epochs} warmup epochs, patience={args.patience})...")
    best_val_auc = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    history_epochs = []
    history_train_losses = []
    history_val_losses = []
    history_val_aucs = []

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Stage 1 vs Stage 2 switching
        if epoch < args.warmup_epochs:
            stage_name = "Stage 1 (Head Warmup)"
            freeze_backbone(model)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr_head,
                weight_decay=1e-4
            )
            cur_lr_backbone = 0.0
            cur_lr_head = args.lr_head
        else:
            stage_name = "Stage 2 (End-to-End Fine-Tuning)"
            unfreeze_backbone(model)
            optimizer = create_fine_tune_optimizer(
                model,
                lr_backbone=args.lr_backbone,
                lr_head=args.lr_head
            )
            cur_lr_backbone = args.lr_backbone
            cur_lr_head = args.lr_head

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

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Evaluate validation
        val_aucs, avg_val_loss = evaluate_model(
            model, val_graphs, device,
            min_samples=2 if args.smoke_test else 10,
            batch_size=args.eval_batch_size,
            return_loss=True,
            pos_weight=pos_weight
        )
        avg_val_auc = float(np.mean(list(val_aucs.values()))) if val_aucs else 0.0
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        history_epochs.append(epoch + 1)
        history_train_losses.append(avg_train_loss)
        history_val_losses.append(avg_val_loss)
        history_val_aucs.append(avg_val_auc)

        is_best = avg_val_auc > best_val_auc
        if is_best:
            best_val_auc = avg_val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.output_weights)
            best_mark = f" [★ New Best Checkpoint -> {args.output_weights}]"
        else:
            epochs_no_improve += 1
            best_mark = ""

        logger.log(
            f"Epoch {epoch+1:2d}/{args.epochs} [{stage_name}] | "
            f"Loss: {avg_train_loss:.4f} (Val: {avg_val_loss:.4f}) | Val AUC: {avg_val_auc:.4f} "
            f"| Time: {format_duration(epoch_time)} | Total: {format_duration(total_elapsed)}"
            f"{best_mark}"
        )

        logger.log_metrics(
            epoch=epoch + 1,
            stage=stage_name,
            train_loss=avg_train_loss,
            val_auc=avg_val_auc,
            lr_backbone=cur_lr_backbone,
            lr_head=cur_lr_head,
            epoch_time=epoch_time,
            best_val_auc=best_val_auc,
            is_best=is_best
        )

        # Early stopping check (only in Stage 2)
        if epoch >= args.warmup_epochs and epochs_no_improve >= args.patience:
            logger.log(f"\n[!] Early stopping triggered: no validation AUC improvement for {args.patience} epochs.")
            break

    total_duration = time.time() - start_time

    # 5. Final Test Evaluation using best weights
    logger.log("\n[5] Final Test Evaluation...")
    if Path(args.output_weights).exists():
        model.load_state_dict(torch.load(args.output_weights, map_location=device))
        logger.log(f"✓ Loaded best model checkpoint from {args.output_weights}")

    test_aucs = evaluate_model(
        model, test_graphs, device,
        min_samples=2 if args.smoke_test else 10,
        batch_size=args.eval_batch_size
    )
    logger.log("\nTest ROC-AUC per Target Endpoint:")
    logger.log("-" * 45)
    for t in TASKS:
        score = test_aucs.get(t, None)
        if score is not None:
            logger.log(f"  • {t:15s}: {score:.4f}")
        else:
            logger.log(f"  • {t:15s}: N/A (< min test samples)")
    logger.log("-" * 45)
    avg_test_auc = float(np.mean(list(test_aucs.values()))) if test_aucs else 0.0
    logger.log(f"Average Test ROC-AUC across all tasks: {avg_test_auc:.4f}")

    logger.log("\n" + "=" * 75)
    logger.log("Phase 3 Multi-Task Fine-Tuning Completed Successfully!")
    logger.log(f"  • Total Epochs Run     : {epoch + 1}/{args.epochs}")
    logger.log(f"  • Best Validation AUC  : {best_val_auc:.4f}")
    logger.log(f"  • Final Test AUC (Mean): {avg_test_auc:.4f}")
    logger.log(f"  • Total Training Time  : {format_duration(total_duration)}")
    logger.log(f"  • Checkpoint Saved     : {args.output_weights}")
    logger.log(f"  • Detailed Text Log    : {args.log_file}")
    logger.log(f"  • CSV Metrics Log      : {args.metrics_file}")

    # Render and save Phase 3 downstream fine-tuning plots
    try:
        from pharma_gnn.visualization import plot_finetune_curves
        plot_finetune_curves(
            epochs=history_epochs,
            train_losses=history_train_losses,
            val_losses=history_val_losses,
            val_aucs=history_val_aucs,
            loss_path=args.output_loss_plot,
            auc_path=args.output_auc_plot
        )
        logger.log(f"  • Finetune Loss Plot   : {args.output_loss_plot}")
        logger.log(f"  • Finetune AUC Plot    : {args.output_auc_plot}")
    except Exception as e:
        logger.log(f"  [-] Warning saving fine-tuning plots: {e}")

    logger.log("=" * 75)
    return True


if __name__ == "__main__":
    main()