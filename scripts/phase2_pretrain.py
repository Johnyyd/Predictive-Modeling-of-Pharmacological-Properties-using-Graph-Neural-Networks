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
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    
    # Replace masked atom features with mask token (zero vector)
    g.x = g.x.clone()
    g.x[masked_atom_indices] = 0.0
    
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


def build_pretrain_graphs(smiles_list, batch_size=32):
    """Build graph dataset for pretraining."""
    graphs = []
    for smiles in smiles_list:
        g = smiles_to_pretrain_graph(smiles)
        if g is not None:
            graphs.append(g)
        if len(graphs) % 1000 == 0 and len(graphs) > 0:
            print(f"  Built {len(graphs)} graphs...")
    print(f"Total valid graphs: {len(graphs)}")
    return graphs


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: Self-Supervised Pretraining for PharmaGNN v2 Foundation Model")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test to verify loss convergence")
    parser.add_argument("--limit", type=int, default=50000, help="Maximum number of SMILES to load")
    parser.add_argument("--epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Graph batch size")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden dimension (128 for v2 foundation)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GATv2 layers (4 for v2 foundation)")
    parser.add_argument("--heads", type=int, default=4, help="Number of GATv2 attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-weights", type=str, default="pharma_gnn_pretrained_encoder.pt", help="Path to save pretrained encoder weights")
    args = parser.parse_args(argv)

    print("=" * 70)
    print("Phase 2: Self-Supervised Pretraining (PharmaGNN v2 Foundation)")
    print("=" * 70)
    
    if args.smoke_test:
        print("[!] Smoke-test mode active: configuring fast verification run")
        args.limit = min(args.limit, 1000)
        args.epochs = min(args.epochs, 3)

    # Load SMILES
    print(f"\n[1] Loading SMILES for pretraining (limit={args.limit})...")
    smiles_list = load_pretrain_smiles(limit=args.limit)
    
    # Build graphs
    print("\n[2] Building graph dataset...")
    graphs = build_pretrain_graphs(smiles_list)
    
    if len(graphs) < 10:
        print("Not enough valid graphs!")
        return False
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    train_split = int(0.9 * len(graphs))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Initialize model
    print(f"\n[3] Initializing model (hidden={args.hidden_channels}, layers={args.num_layers}, heads={args.heads})...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n[4] Starting pretraining ({args.epochs} epochs)...")
    best_val_loss = float('inf')
    train_history = []
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        # Train mini-batches
        for i in range(0, len(train_graphs), args.batch_size):
            batch_graphs = train_graphs[i:i+args.batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for g in batch_graphs:
                g = g.to(device)
                g = pretrain_masking(g)
                outputs = model.forward_pretrain(
                    g.x, g.edge_index, g.edge_attr, g.batch,
                    g.global_features, g.func_group_features, g.concentration,
                    g.masked_atom_indices, g.masked_bond_indices
                )
                loss, loss_dict = pretrain_loss(outputs, g)
                (loss / len(batch_graphs)).backward()
                batch_loss += loss.item()
            
            optimizer.step()
            train_losses.append(batch_loss / len(batch_graphs))
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            sample_val = val_graphs[:min(len(val_graphs), 50)]
            for g in sample_val:
                g = g.to(device)
                g = pretrain_masking(g)
                outputs = model.forward_pretrain(
                    g.x, g.edge_index, g.edge_attr, g.batch,
                    g.global_features, g.func_group_features, g.concentration,
                    g.masked_atom_indices, g.masked_bond_indices
                )
                loss, _ = pretrain_loss(outputs, g)
                val_losses.append(loss.item())
        
        avg_train = np.mean(train_losses) if train_losses else 0
        avg_val = np.mean(val_losses) if val_losses else 0
        train_history.append((avg_train, avg_val))
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            encoder_state = {k: v for k, v in model.state_dict().items() 
                           if 'pred_head' not in k and 'motif_head' not in k and 'context_head' not in k}
            torch.save(encoder_state, args.output_weights)
            print(f"  ✓ Saved best encoder (val_loss={avg_val:.4f})")
        
        scheduler.step()
    
    print("\n" + "=" * 70)
    print("Pretraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved: {args.output_weights}")
    print("=" * 70)

    if args.smoke_test and len(train_history) >= 2:
        initial_train = train_history[0][0]
        final_train = train_history[-1][0]
        assert final_train <= initial_train, f"Smoke test failed: loss did not decrease ({initial_train:.4f} -> {final_train:.4f})"
        print(f"✓ Smoke test convergence verified: initial={initial_train:.4f} -> final={final_train:.4f}")
        
    return True


if __name__ == "__main__":
    main()