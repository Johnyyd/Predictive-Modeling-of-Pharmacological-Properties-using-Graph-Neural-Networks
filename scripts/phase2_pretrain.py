#!/usr/bin/env python3
"""
Phase 2: Self-Supervised Pretraining on large compound set
Adds pretraining heads to PharmaGNN and trains on ~100K+ SMILES
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch.nn import BatchNorm1d, Linear
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, MolStandardize
import math
import sys
import time
import datetime
import csv
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from model import PharmaGNN
from main import smiles_to_graph

# Pretraining tasks
NUM_ATOM_TYPES = 119  # Periodic table elements
NUM_BOND_TYPES = 4    # Single, Double, Triple, Aromatic
NUM_MOTIFS = 85       # RDKit functional groups

class PharmaGNN_Pretrain(PharmaGNN):
    """PharmaGNN with self-supervised pretraining heads"""
    
    def __init__(self, num_node_features=6, hidden_channels=128, num_classes=13, 
                 num_global_features=11, num_func_groups=85, num_layers=4,
                 heads=4, residual=True, fg_embed_dim=16):
        super().__init__(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_global_features=num_global_features,
            num_func_groups=num_func_groups,
            num_layers=num_layers,
            heads=heads,
            residual=residual,
            fg_embed_dim=fg_embed_dim
        )
        
        # Pretraining heads
        self.atom_pred_head = Linear(hidden_channels, NUM_ATOM_TYPES)
        self.bond_pred_head = Linear(hidden_channels * 2, NUM_BOND_TYPES)
        self.motif_head = Linear(hidden_channels, NUM_MOTIFS)
        
        # Context prediction head (graph-level representation projection)
        self.context_head = Linear(hidden_channels * 2, hidden_channels * 2)
        
    def forward_pretrain(self, x, edge_index, edge_attr, batch, 
                         global_features, func_group_features, concentration,
                         masked_atom_indices=None, masked_bond_indices=None):
        """Forward pass with pretraining outputs"""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        batch_size = batch.max().item() + 1
            
        if concentration is None:
            concentration = torch.zeros(batch_size, 1, dtype=torch.float, device=x.device)
            
        if global_features is None:
            global_features = torch.zeros(batch_size, self.num_global_features, dtype=torch.float, device=x.device)
            
        if func_group_features is None:
            func_group_features = torch.zeros(batch_size, self.num_func_groups, dtype=torch.float, device=x.device)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.leaky_relu(x)
        
        # Layer 2 with residual skip connection
        h = self.conv2(x, edge_index, edge_attr=edge_attr)
        if h.size(0) > 1:
            h = self.bn2(h)
        h = F.leaky_relu(h)
        if self.residual and x.shape == h.shape:
            x = x + h
        else:
            x = h
            
        # Extra layers (num_layers > 2)
        for conv, bn in zip(self.extra_convs, self.extra_bns):
            h = conv(x, edge_index, edge_attr=edge_attr)
            if h.size(0) > 1:
                h = bn(h)
            h = F.leaky_relu(h)
            if self.residual and x.shape == h.shape:
                x = x + h
            else:
                x = h
        
        # Node-level representations for atom masking
        node_repr = x
        
        # Graph-level representations
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        fg_out = self.fg_interaction(func_group_features)
        
        graph_repr = torch.cat([x_mean, x_max, global_features, fg_out, concentration], dim=1)
        
        # Supervised head (for fine-tuning later)
        x_sup = F.dropout(graph_repr, p=0.5, training=self.training)
        x_sup = self.lin1(x_sup)
        x_sup = F.relu(x_sup)
        supervised_logits = self.lin2(x_sup)
        
        # Pretraining outputs
        pretrain_outputs = {}
        
        # 1. Atom masking prediction
        if masked_atom_indices is not None and len(masked_atom_indices) > 0:
            masked_node_repr = node_repr[masked_atom_indices]
            atom_logits = self.atom_pred_head(masked_node_repr)
            pretrain_outputs['atom_logits'] = atom_logits
        
        # 2. Bond type prediction (for masked edges)
        if masked_bond_indices is not None and len(masked_bond_indices) > 0:
            src, dst = edge_index[:, masked_bond_indices]
            bond_repr = torch.cat([node_repr[src], node_repr[dst]], dim=1)
            bond_logits = self.bond_pred_head(bond_repr)
            pretrain_outputs['bond_logits'] = bond_logits
        
        # 3. Functional group / motif prediction
        motif_logits = self.motif_head(x_mean)
        pretrain_outputs['motif_logits'] = motif_logits
        
        # 4. Context prediction (graph representation)
        pooled_context = torch.cat([x_mean, x_max], dim=1)
        context_repr = self.context_head(pooled_context)
        pretrain_outputs['context_repr'] = context_repr
        pretrain_outputs['context_target'] = pooled_context.detach()
        
        return {
            'supervised_logits': supervised_logits,
            'pretrain_outputs': pretrain_outputs,
            'node_repr': node_repr,
            'graph_repr': graph_repr
        }
    
    def forward(self, x, edge_index, edge_attr=None, batch=None, 
                global_features=None, func_group_features=None, concentration=None):
        """Standard forward for fine-tuning"""
        return super().forward(x, edge_index, edge_attr, batch, 
                               global_features, func_group_features, concentration)


def compute_motif_features(mol):
    """Compute RDKit functional group counts for a molecule"""
    frag_funcs = [func for name, func in Descriptors.descList if name.startswith('fr_')]
    features = []
    for func in frag_funcs:
        try:
            count = func(mol)
            features.append(float(count))
        except:
            features.append(0.0)
    return features


def smiles_to_pretrain_graph(smiles):
    """Convert SMILES to graph with pretraining targets aligned to explicit-H graph."""
    g = smiles_to_graph(smiles, concentration_molar=1e-5)
    if g is None:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add motif features
    motif_features = compute_motif_features(mol)
    g.func_group_features = torch.tensor([motif_features], dtype=torch.float)
    
    # Convert to explicit H to match smiles_to_graph node count
    mol_hs = Chem.AddHs(mol)
    
    atom_types = [atom.GetAtomicNum() for atom in mol_hs.GetAtoms()]
    g.atom_types = torch.tensor(atom_types, dtype=torch.long)
    
    # Bond types for both directed edges (i->j and j->i)
    bond_types = []
    for bond in mol_hs.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE: b_code = 0
        elif bt == Chem.BondType.DOUBLE: b_code = 1
        elif bt == Chem.BondType.TRIPLE: b_code = 2
        elif bt == Chem.BondType.AROMATIC: b_code = 3
        else: b_code = 0
        bond_types.extend([b_code, b_code])
    g.bond_types = torch.tensor(bond_types, dtype=torch.long)
    
    # Store clean unmasked base node features for epoch-independent masking
    g.raw_x = g.x.clone()
    
    return g


def pretrain_masking(g, mask_rate=0.15):
    """Apply random masking for atom and bond prediction, replacing masked atom features with mask token."""
    num_nodes = g.num_nodes
    num_edges = g.edge_index.size(1)
    
    # Ensure atom_types matches graph node count
    if not hasattr(g, 'atom_types') or g.atom_types.size(0) != num_nodes:
        g.masked_atom_indices = torch.tensor([], dtype=torch.long)
        g.masked_atom_labels = torch.tensor([], dtype=torch.long)
        g.masked_bond_indices = torch.tensor([], dtype=torch.long)
        g.masked_bond_labels = torch.tensor([], dtype=torch.long)
        return g
    
    # Mask atoms
    num_mask_atoms = max(1, int(num_nodes * mask_rate))
    perm = torch.randperm(num_nodes)
    masked_atom_indices = perm[:num_mask_atoms]
    
    g.masked_atom_indices = masked_atom_indices
    g.masked_atom_labels = g.atom_types[masked_atom_indices].clone()
    
    # Preserve pristine base features and mask a clean copy each epoch
    if not hasattr(g, 'raw_x'):
        g.raw_x = g.x.clone()
    masked_x = g.raw_x.clone()
    masked_x[masked_atom_indices] = 0.0
    g.x = masked_x
    
    # Mask bonds
    if num_edges > 0 and hasattr(g, 'bond_types') and g.bond_types.size(0) == num_edges:
        num_mask_bonds = max(1, int(num_edges * mask_rate))
        perm_bonds = torch.randperm(num_edges)
        masked_bond_indices = perm_bonds[:num_mask_bonds]
        g.masked_bond_indices = masked_bond_indices
        g.masked_bond_labels = g.bond_types[masked_bond_indices].clone()
    else:
        g.masked_bond_indices = torch.tensor([], dtype=torch.long)
        g.masked_bond_labels = torch.tensor([], dtype=torch.long)
    
    return g


def collate_pretrain_batch(graph_list):
    """Collate a list of individual pretrain graphs into a single unified PyG Batch with global masking indices."""
    batch_pyg = Batch.from_data_list(graph_list)
    all_masked_atom_idx = []
    all_masked_atom_lbl = []
    all_masked_bond_idx = []
    all_masked_bond_lbl = []
    edge_offset = 0

    for i, g in enumerate(graph_list):
        node_offset = batch_pyg.ptr[i].item()
        if hasattr(g, "masked_atom_indices") and len(g.masked_atom_indices) > 0:
            all_masked_atom_idx.append(g.masked_atom_indices + node_offset)
            all_masked_atom_lbl.append(g.masked_atom_labels)
        if hasattr(g, "masked_bond_indices") and len(g.masked_bond_indices) > 0:
            all_masked_bond_idx.append(g.masked_bond_indices + edge_offset)
            all_masked_bond_lbl.append(g.masked_bond_labels)
        edge_offset += g.edge_index.size(1)

    batch_pyg.masked_atom_indices = torch.cat(all_masked_atom_idx) if all_masked_atom_idx else torch.tensor([], dtype=torch.long)
    batch_pyg.masked_atom_labels = torch.cat(all_masked_atom_lbl) if all_masked_atom_lbl else torch.tensor([], dtype=torch.long)
    batch_pyg.masked_bond_indices = torch.cat(all_masked_bond_idx) if all_masked_bond_idx else torch.tensor([], dtype=torch.long)
    batch_pyg.masked_bond_labels = torch.cat(all_masked_bond_lbl) if all_masked_bond_lbl else torch.tensor([], dtype=torch.long)
    return batch_pyg


def apply_vectorized_batch_masking(batch_pyg, mask_rate=0.15):
    """
    Apply fast vectorized random masking directly to an entire collated PyG Batch in native PyTorch C++/AVX.
    Eliminates Python-level loop overhead and reduces batch preparation latency by ~4x.
    """
    num_nodes = batch_pyg.num_nodes
    if hasattr(batch_pyg, 'atom_types') and batch_pyg.atom_types.size(0) == num_nodes:
        num_mask_atoms = max(1, int(num_nodes * mask_rate))
        perm = torch.randperm(num_nodes)
        masked_atom_indices = perm[:num_mask_atoms]
        batch_pyg.masked_atom_indices = masked_atom_indices
        batch_pyg.masked_atom_labels = batch_pyg.atom_types[masked_atom_indices].clone()
        
        # Reset node features from raw_x if present, then mask
        if hasattr(batch_pyg, 'raw_x'):
            batch_pyg.x = batch_pyg.raw_x.clone()
        else:
            batch_pyg.x = batch_pyg.x.clone()
        batch_pyg.x[masked_atom_indices] = 0.0
    else:
        batch_pyg.masked_atom_indices = torch.tensor([], dtype=torch.long)
        batch_pyg.masked_atom_labels = torch.tensor([], dtype=torch.long)

    num_edges = batch_pyg.edge_index.size(1)
    if num_edges > 0 and hasattr(batch_pyg, 'bond_types') and batch_pyg.bond_types.size(0) == num_edges:
        num_mask_bonds = max(1, int(num_edges * mask_rate))
        perm_bonds = torch.randperm(num_edges)
        masked_bond_indices = perm_bonds[:num_mask_bonds]
        batch_pyg.masked_bond_indices = masked_bond_indices
        batch_pyg.masked_bond_labels = batch_pyg.bond_types[masked_bond_indices].clone()
    else:
        batch_pyg.masked_bond_indices = torch.tensor([], dtype=torch.long)
        batch_pyg.masked_bond_labels = torch.tensor([], dtype=torch.long)

    return batch_pyg


def pretrain_loss(outputs, g, supervised_labels=None):
    """Compute combined pretraining loss across atom, bond, motif, and context tasks."""
    losses = {}
    
    # 1. Atom masking loss (CrossEntropy)
    if 'atom_logits' in outputs['pretrain_outputs'] and hasattr(g, 'masked_atom_indices') and len(g.masked_atom_indices) > 0:
        atom_logits = outputs['pretrain_outputs']['atom_logits']
        atom_labels = g.masked_atom_labels
        atom_labels = torch.clamp(atom_labels, 0, NUM_ATOM_TYPES - 1)
        losses['atom'] = F.cross_entropy(atom_logits, atom_labels)
    
    # 2. Bond type prediction loss
    if 'bond_logits' in outputs['pretrain_outputs'] and hasattr(g, 'masked_bond_indices') and len(g.masked_bond_indices) > 0:
        bond_logits = outputs['pretrain_outputs']['bond_logits']
        bond_labels = g.masked_bond_labels
        bond_labels = torch.clamp(bond_labels, 0, NUM_BOND_TYPES - 1)
        losses['bond'] = F.cross_entropy(bond_logits, bond_labels)
    
    # 3. Motif prediction loss (BCE multi-label)
    if 'motif_logits' in outputs['pretrain_outputs']:
        motif_logits = outputs['pretrain_outputs']['motif_logits']
        motif_targets = (g.func_group_features > 0).float()
        if motif_logits.shape != motif_targets.shape:
            motif_logits = motif_logits.view(motif_targets.shape)
        losses['motif'] = F.binary_cross_entropy_with_logits(motif_logits, motif_targets)
    
    # 4. Context prediction loss (MSE)
    if 'context_repr' in outputs['pretrain_outputs']:
        context_repr = outputs['pretrain_outputs']['context_repr']
        if 'context_target' in outputs['pretrain_outputs']:
            target_repr = outputs['pretrain_outputs']['context_target']
        else:
            target_repr = outputs['graph_repr'].detach()
        if context_repr.shape == target_repr.shape:
            losses['context'] = F.mse_loss(context_repr, target_repr)
    
    # 5. Supervised loss (if labels available)
    if supervised_labels is not None:
        sup_logits = outputs['supervised_logits']
        mask = ~torch.isnan(supervised_labels)
        if mask.any():
            losses['supervised'] = F.binary_cross_entropy_with_logits(
                sup_logits[mask], supervised_labels[mask])
    
    # Weighted sum
    weights = {'atom': 1.0, 'bond': 1.0, 'motif': 0.5, 'context': 0.5, 'supervised': 2.0}
    total_loss = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    
    return total_loss, losses


def load_pretrain_smiles(limit=None):
    """Load SMILES for pretraining from 760K universe or local master dataset."""
    all_smiles = []
    
    # 1. Primary: 760K Pretraining Universe
    universe_path = Path('data/comptox_v3/pretrain_760k_smiles.csv')
    if universe_path.exists():
        print(f"Loading pretraining universe from {universe_path}...")
        try:
            u_df = pd.read_csv(universe_path)
            col = 'smiles' if 'smiles' in u_df.columns else u_df.columns[0]
            u_smiles = u_df[col].dropna().unique().tolist()
            all_smiles.extend(u_smiles)
            print(f"Pretraining universe loaded: {len(u_smiles):,} SMILES")
        except Exception as e:
            print(f"[-] Warning loading pretraining universe: {e}")
            
    # 2. Secondary: From existing master dataset if universe not present
    if not all_smiles:
        master_path = Path('data/processed/master_toxicity_dataset_expanded.csv')
        if master_path.exists():
            master = pd.read_csv(master_path)
            master_smiles = master['smiles'].dropna().unique().tolist()
            all_smiles.extend(master_smiles)
            print(f"Master dataset: {len(master_smiles)} SMILES")
        
        train_path = Path('data/processed/training_dataset_v2.csv')
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            train_smiles = train_df['smiles'].dropna().unique().tolist()
            all_smiles.extend(train_smiles)
    
    # Deduplicate
    all_smiles = list(dict.fromkeys(all_smiles))
    print(f"Total unique SMILES loaded: {len(all_smiles)}")
    
    if limit:
        all_smiles = all_smiles[:limit]
    
    return all_smiles


def format_time(seconds: float) -> str:
    """Format seconds into human-readable duration (e.g. '14.2s', '2m 15s', '1h 04m 20s')."""
    if seconds is None or seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem_sec = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {rem_sec:02d}s"
    hours = int(minutes // 60)
    rem_min = int(minutes % 60)
    return f"{hours}h {rem_min:02d}m {rem_sec:02d}s"


class PretrainLogger:
    """Dual console and file logger with structured metrics streaming to CSV."""

    def __init__(self, log_file: str = "logs/phase2_pretrain.log", metrics_file: str = "logs/pretrain_metrics.csv"):
        self.log_file = Path(log_file) if log_file else None
        self.metrics_file = Path(metrics_file) if metrics_file else None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*75}\n[PRETRAINING SESSION START] {ts}\n{'='*75}\n")

        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.metrics_file.exists() or self.metrics_file.stat().st_size == 0:
                with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "epoch", "train_loss", "val_loss",
                        "atom_loss", "bond_loss", "motif_loss", "context_loss",
                        "lr", "epoch_time_sec", "best_val_loss", "is_best"
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

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float,
                    sub_losses: dict, lr: float, epoch_time: float,
                    best_val_loss: float, is_best: bool):
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
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{sub_losses.get('atom', 0.0):.4f}",
                    f"{sub_losses.get('bond', 0.0):.4f}",
                    f"{sub_losses.get('motif', 0.0):.4f}",
                    f"{sub_losses.get('context', 0.0):.4f}",
                    f"{lr:.6e}",
                    f"{epoch_time:.2f}",
                    f"{best_val_loss:.4f}",
                    int(is_best)
                ])
        except Exception:
            pass


def _process_smiles_chunk_to_disk(args):
    """Worker task: converts a slice of SMILES into graphs and saves directly to disk to prevent IPC memory exhaustion."""
    chunk_idx, smiles_chunk, temp_dir = args
    graphs = []
    for s in smiles_chunk:
        try:
            g = smiles_to_pretrain_graph(s)
            if g is not None:
                graphs.append(g)
        except Exception:
            pass
    out_file = Path(temp_dir) / f"chunk_{chunk_idx:05d}.pt"
    torch.save(graphs, out_file)
    return chunk_idx, str(out_file), len(graphs), len(smiles_chunk)


def build_pretrain_graphs(smiles_list, batch_size=32, logger=None, cache_path=None, force_rebuild=False, num_workers=4):
    """Build graph dataset for pretraining with multiprocessing and disk caching."""
    graphs = []
    total = len(smiles_list)
    log_fn = logger.log if logger else print

    # 1. Try loading from disk cache
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists() and not force_rebuild:
            try:
                log_fn(f"  ⚡ Loading preprocessed graphs from cache: {cache_file}...")
                cached = torch.load(cache_file, weights_only=False)
                log_fn(f"  ✓ Loaded {len(cached):,} precomputed graphs from cache.")
                if len(cached) >= total:
                    return cached[:total]
                return cached
            except Exception as e:
                log_fn(f"  [-] Cache read failed ({e}), rebuilding graphs from scratch...")

    # 2. Build graphs with ProcessPoolExecutor using disk-buffered chunks
    t0 = time.time()
    effective_workers = min(num_workers, 4)
    if total <= 200 or effective_workers <= 1:
        log_fn(f"  Converting {total:,} SMILES into PyTorch Geometric graph structures...")
        for idx, smiles in enumerate(smiles_list, 1):
            g = smiles_to_pretrain_graph(smiles)
            if g is not None:
                graphs.append(g)
            if idx % max(50, total // 5) == 0 or idx == total:
                elapsed = time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (total - idx) / rate if rate > 0 else 0
                pct = (idx / total) * 100
                log_fn(
                    f"  [Progress] {idx:6,d}/{total:6,d} ({pct:5.1f}%) | "
                    f"Valid: {len(graphs):6,d} | Rate: {rate:6.0f} mol/s | "
                    f"Elapsed: {format_time(elapsed):>8s} | ETA: {format_time(eta):>8s}"
                )
    else:
        import tempfile
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Sub-divide SMILES into disk-buffered slices (e.g., 2,500 compounds each)
        chunk_size = max(500, min(2500, total // (effective_workers * 4)))
        chunks = [smiles_list[i:i + chunk_size] for i in range(0, total, chunk_size)]
        total_chunks = len(chunks)

        log_fn(f"  Converting {total:,} SMILES across {effective_workers} CPU workers ({total_chunks} disk-buffered chunks)...")

        with tempfile.TemporaryDirectory() as temp_dir:
            task_args = [(i, c, temp_dir) for i, c in enumerate(chunks)]
            chunk_results = [None] * total_chunks
            processed_smiles = 0
            valid_graphs_count = 0

            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = {executor.submit(_process_smiles_chunk_to_disk, t): t[0] for t in task_args}
                for fut in as_completed(futures):
                    chunk_idx, fpath, n_valid, n_total_chunk = fut.result()
                    chunk_results[chunk_idx] = (fpath, n_valid)
                    processed_smiles += n_total_chunk
                    valid_graphs_count += n_valid

                    elapsed = time.time() - t0
                    rate = processed_smiles / elapsed if elapsed > 0 else 0
                    eta = (total - processed_smiles) / rate if rate > 0 else 0
                    pct = (processed_smiles / total) * 100
                    log_fn(
                        f"  [Progress] {processed_smiles:6,d}/{total:6,d} ({pct:5.1f}%) | "
                        f"Valid: {valid_graphs_count:6,d} | Rate: {rate:6.0f} mol/s | "
                        f"Elapsed: {format_time(elapsed):>8s} | ETA: {format_time(eta):>8s}"
                    )

            # Assemble merged graphs in deterministic chunk order
            log_fn(f"  Assembling and validating {valid_graphs_count:,} graphs from disk chunks...")
            for fpath, _ in chunk_results:
                if fpath and Path(fpath).exists():
                    sub_graphs = torch.load(fpath, weights_only=False)
                    graphs.extend(sub_graphs)

    total_time = time.time() - t0
    success_rate = (len(graphs) / total * 100) if total > 0 else 0
    log_fn(f"  ✓ Built {len(graphs):,}/{total:,} valid graphs in {format_time(total_time)} ({success_rate:.2f}% conversion rate)")

    # 3. Save to disk cache for instantaneous reloads in subsequent runs
    if cache_path and len(graphs) > 0:
        try:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            log_fn(f"  💾 Caching {len(graphs):,} graphs to {cache_file}...")
            torch.save(graphs, cache_file)
            log_fn("  ✓ Graphs cache successfully written.")
        except Exception as e:
            log_fn(f"  [-] Warning: Could not write graphs cache: {e}")

    return graphs


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: Self-Supervised Pretraining for PharmaGNN v2 Foundation Model")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test to verify loss convergence")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum number of SMILES to load")
    parser.add_argument("--epochs", type=int, default=20, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Graph batch size")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden dimension (128 for v2 foundation)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GATv2 layers (4 for v2 foundation)")
    parser.add_argument("--heads", type=int, default=4, help="Number of GATv2 attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--log-interval", type=int, default=50, help="Batch logging frequency inside each epoch")
    parser.add_argument("--log-file", type=str, default="logs/phase2_pretrain.log", help="Path to text log file")
    parser.add_argument("--metrics-file", type=str, default="logs/pretrain_metrics.csv", help="Path to CSV metrics file")
    parser.add_argument("--num-threads", type=int, default=4, help="PyTorch intra-op CPU threads (default: 4 for physical cores)")
    parser.add_argument("--cache-graphs", type=str, default="data/processed/pretrain_graphs_cache.pt", help="Path to cache prebuilt graph dataset")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding graphs ignoring cache")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for SMILES graph conversion")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(model, dynamic=True) for PyTorch 2.x kernel fusion")
    parser.add_argument("--output-weights", type=str, default="pharma_gnn_pretrained_encoder.pt", help="Path to save pretrained encoder weights")
    parser.add_argument("--output-plot", type=str, default="pretrain_loss_curve.png", help="Path to save pretraining progression plot")
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

    logger = PretrainLogger(log_file=args.log_file, metrics_file=args.metrics_file)

    logger.log("=" * 75)
    logger.log("Phase 2: Self-Supervised Pretraining (PharmaGNN v2 Foundation - High-Perf)")
    logger.log("=" * 75)

    if args.smoke_test:
        logger.log("[!] Smoke-test mode active: configuring fast verification run")
        args.limit = min(args.limit, 1000)
        args.epochs = min(args.epochs, 3)
        args.log_interval = max(1, min(args.log_interval, 5))

    total_start_time = time.time()

    # 1. Load SMILES
    logger.log(f"\n[1] Loading SMILES for pretraining (limit={args.limit:,})...")
    smiles_list = load_pretrain_smiles(limit=args.limit)

    # 2. Build graphs
    logger.log(f"\n[2] Building graph dataset from {len(smiles_list):,} compounds...")
    graphs = build_pretrain_graphs(
        smiles_list,
        batch_size=args.batch_size,
        logger=logger,
        cache_path=None if args.smoke_test else args.cache_graphs,
        force_rebuild=args.force_rebuild,
        num_workers=min(args.workers, 2 if args.smoke_test else args.workers)
    )

    if len(graphs) < 10:
        logger.log("[-] Error: Not enough valid graphs generated!")
        return False

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    train_split = int(0.9 * len(graphs))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]

    total_batches = (len(train_graphs) + args.batch_size - 1) // args.batch_size
    log_interval = max(1, min(args.log_interval, total_batches))

    logger.log(f"\n[Dataset Split] Train graphs: {len(train_graphs):,d} ({total_batches:,d} batches) | Val graphs: {len(val_graphs):,d}")

    # 3. Initialize model
    logger.log(f"\n[3] Initializing model architecture...")
    logger.log(f"    • Hidden channels : {args.hidden_channels}")
    logger.log(f"    • GATv2 layers    : {args.num_layers}")
    logger.log(f"    • Attention heads : {args.heads}")
    logger.log(f"    • Residual skips  : Enabled")
    logger.log(f"    • Learning rate   : {args.lr:.4e}")
    logger.log(f"    • Early stopping  : {args.patience} epochs patience")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"    • Hardware device : {device} (CPU threads: {torch.get_num_threads()})")

    model = PharmaGNN_Pretrain(
        num_node_features=6,
        hidden_channels=args.hidden_channels,
        num_classes=13,
        num_global_features=11,
        num_func_groups=85,
        num_layers=args.num_layers,
        heads=args.heads,
        residual=True,
        fg_embed_dim=16 if args.hidden_channels > 32 else 8
    ).to(device)

    if getattr(args, 'compile', False):
        try:
            logger.log("    • Torch compile   : Enabled (dynamic=True)")
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            logger.log(f"    [-] torch.compile unavailable: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"    • Total parameters: {total_params:,d} ({trainable_params:,d} trainable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training loop
    logger.log(f"\n[4] Starting pretraining ({args.epochs} epochs, {total_batches:,d} batches/epoch, logging every {log_interval} batches)...")
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    train_history = []
    sub_losses_history = {'atom': [], 'bond': [], 'motif': [], 'context': []}

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        running_sub_losses = {'atom': [], 'bond': [], 'motif': [], 'context': []}
        epoch_start_time = time.time()
        batch_idx = 0

        # Fast in-memory epoch permutation
        perm = np.random.permutation(len(train_graphs))

        # Train mini-batches with fast vectorized batch masking
        for i in range(0, len(train_graphs), args.batch_size):
            batch_idx += 1
            batch_indices = perm[i:i+args.batch_size]
            batch_graphs = [train_graphs[idx] for idx in batch_indices]

            # Vectorized batch collation and masking in PyTorch C++/AVX
            collated = Batch.from_data_list(batch_graphs).to(device)
            collated = apply_vectorized_batch_masking(collated, mask_rate=0.15)

            optimizer.zero_grad()
            outputs = model.forward_pretrain(
                collated.x, collated.edge_index, collated.edge_attr, collated.batch,
                collated.global_features, collated.func_group_features, collated.concentration,
                collated.masked_atom_indices, collated.masked_bond_indices
            )
            loss, loss_dict = pretrain_loss(outputs, collated)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_losses.append(batch_loss)
            for k in running_sub_losses:
                if k in loss_dict:
                    running_sub_losses[k].append(loss_dict[k].item())

            # Periodic batch-level progress logging
            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                elapsed_epoch = time.time() - epoch_start_time
                batches_per_sec = batch_idx / elapsed_epoch if elapsed_epoch > 0 else 0
                graphs_per_sec = (batch_idx * args.batch_size) / elapsed_epoch if elapsed_epoch > 0 else 0
                eta_epoch = (total_batches - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
                pct = (batch_idx / total_batches) * 100

                window = min(len(train_losses), log_interval)
                cur_loss = np.mean(train_losses[-window:])
                cur_atom = np.mean(running_sub_losses['atom'][-window:]) if running_sub_losses['atom'] else 0.0
                cur_bond = np.mean(running_sub_losses['bond'][-window:]) if running_sub_losses['bond'] else 0.0
                cur_motif = np.mean(running_sub_losses['motif'][-window:]) if running_sub_losses['motif'] else 0.0
                cur_ctx = np.mean(running_sub_losses['context'][-window:]) if running_sub_losses['context'] else 0.0

                logger.log(
                    f"  [Epoch {epoch+1:2d}/{args.epochs}] Batch {batch_idx:4d}/{total_batches:4d} ({pct:5.1f}%) | "
                    f"Loss: {cur_loss:.4f} [Atom: {cur_atom:.3f}, Bond: {cur_bond:.3f}, Motif: {cur_motif:.3f}, Ctx: {cur_ctx:.3f}] | "
                    f"Speed: {batches_per_sec:.1f} b/s ({graphs_per_sec:.0f} g/s) | "
                    f"Elapsed: {format_time(elapsed_epoch):>7s} | ETA: {format_time(eta_epoch):>7s}"
                )

        # Validation
        model.eval()
        val_losses = []
        val_t0 = time.time()
        val_sample_size = min(len(val_graphs), 50)
        logger.log(f"  [Epoch {epoch+1:2d}/{args.epochs}] Evaluating {val_sample_size} validation graphs...")

        with torch.no_grad():
            sample_val = [val_graphs[k] for k in range(val_sample_size)]
            collated_val = Batch.from_data_list(sample_val).to(device)
            collated_val = apply_vectorized_batch_masking(collated_val, mask_rate=0.15)
            outputs = model.forward_pretrain(
                collated_val.x, collated_val.edge_index, collated_val.edge_attr, collated_val.batch,
                collated_val.global_features, collated_val.func_group_features, collated_val.concentration,
                collated_val.masked_atom_indices, collated_val.masked_bond_indices
            )
            val_loss, _ = pretrain_loss(outputs, collated_val)
            val_losses.append(val_loss.item())

        val_duration = time.time() - val_t0
        avg_train = np.mean(train_losses) if train_losses else 0.0
        avg_val = np.mean(val_losses) if val_losses else 0.0
        epoch_time = time.time() - epoch_start_time
        total_run_time = time.time() - total_start_time
        train_history.append((avg_train, avg_val))

        cur_lr = optimizer.param_groups[0]['lr']
        is_best = avg_val < best_val_loss

        epoch_sub = {
            'atom': np.mean(running_sub_losses['atom']) if running_sub_losses['atom'] else 0.0,
            'bond': np.mean(running_sub_losses['bond']) if running_sub_losses['bond'] else 0.0,
            'motif': np.mean(running_sub_losses['motif']) if running_sub_losses['motif'] else 0.0,
            'context': np.mean(running_sub_losses['context']) if running_sub_losses['context'] else 0.0,
        }

        # Epoch Summary Banner
        logger.log("-" * 75)
        logger.log(
            f"Epoch {epoch+1:2d}/{args.epochs} Summary [Duration: {format_time(epoch_time)} | Total Run: {format_time(total_run_time)}]\n"
            f"  • Train Loss : {avg_train:.4f} [Atom: {epoch_sub['atom']:.4f}, Bond: {epoch_sub['bond']:.4f}, Motif: {epoch_sub['motif']:.4f}, Ctx: {epoch_sub['context']:.4f}]\n"
            f"  • Val Loss   : {avg_val:.4f} (evaluated in {format_time(val_duration)})\n"
            f"  • LR         : {cur_lr:.4e}"
        )

        # Stream structured metrics to CSV & history
        logger.log_metrics(
            epoch=epoch+1,
            train_loss=avg_train,
            val_loss=avg_val,
            sub_losses=epoch_sub,
            lr=cur_lr,
            epoch_time=epoch_time,
            best_val_loss=best_val_loss,
            is_best=is_best
        )
        for k in sub_losses_history:
            sub_losses_history[k].append(epoch_sub.get(k, 0.0))

        if is_best:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            epochs_no_improve = 0
            encoder_state = {k: v for k, v in model.state_dict().items() 
                           if 'pred_head' not in k and 'motif_head' not in k and 'context_head' not in k}
            torch.save(encoder_state, args.output_weights)
            logger.log(f"  ★ Checkpoint : New best validation loss ({avg_val:.4f}) -> Saved to {args.output_weights}")
        else:
            epochs_no_improve += 1
            logger.log(f"  • Patience   : {epochs_no_improve}/{args.patience} epochs without improvement (Best: {best_val_loss:.4f} at Epoch {best_epoch})")
            if epochs_no_improve >= args.patience:
                logger.log(f"\n[!] Early stopping triggered: no validation loss improvement for {args.patience} epochs.")
                logger.log("-" * 75)
                break

        logger.log("-" * 75)
        scheduler.step()

    total_training_duration = time.time() - total_start_time

    logger.log("\n" + "=" * 75)
    logger.log("Pretraining Completed Successfully!")
    logger.log(f"  • Total Epochs Run     : {len(train_history)}/{args.epochs}")
    logger.log(f"  • Best Validation Loss : {best_val_loss:.4f} (Epoch {best_epoch})")
    logger.log(f"  • Total Training Time  : {format_time(total_training_duration)}")
    logger.log(f"  • Checkpoint Saved     : {args.output_weights}")
    logger.log(f"  • Detailed Text Log    : {args.log_file}")
    logger.log(f"  • CSV Metrics Log      : {args.metrics_file}")

    # Render and save Pretraining progression plot
    try:
        from pharma_gnn.visualization import plot_pretrain_progression
        output_plot = getattr(args, 'output_plot', 'pretrain_loss_curve.png')
        plot_pretrain_progression(
            epochs=list(range(1, len(train_history) + 1)),
            train_losses=[x[0] for x in train_history],
            val_losses=[x[1] for x in train_history],
            sub_losses=sub_losses_history,
            output_path=output_plot
        )
        logger.log(f"  • Pretrain Plot Saved  : {output_plot}")
    except Exception as e:
        logger.log(f"  [-] Warning saving pretraining plot: {e}")

    logger.log("=" * 75)

    if args.smoke_test and len(train_history) >= 2:
        initial_train = train_history[0][0]
        final_train = train_history[-1][0]
        assert final_train <= initial_train, f"Smoke test failed: loss did not decrease ({initial_train:.4f} -> {final_train:.4f})"
        logger.log(f"✓ Smoke test convergence verified: initial={initial_train:.4f} -> final={final_train:.4f}")

    return True


if __name__ == "__main__":
    main()