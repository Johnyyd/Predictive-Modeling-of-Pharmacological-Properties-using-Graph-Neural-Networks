# Hybrid Training Plan: Multi-Source Toxicity Prediction Model

## Overview
Train a production-ready PharmaGNN on ~200K high-quality compounds from multiple sources, with self-supervised pretraining on 760K CompTox SMILES.

---

## Phase 1: Data Curation & ETL (Week 1-2)

### 1.1 CompTox 3.0 Curated Subset (~50-100K compounds)
**Source**: EPA CompTox Chemicals Dashboard (v3.0)
- Download: `wget https://gaftp.epa.gov/CompTox/.../compounds_v3.txt.gz`
- Download assay results: `assay_results_v3.txt.gz`
- **Target endpoints (50 assay)**: 
  - Nuclear receptor: AR, AR-LBD, ER, ER-LBD, AhR, Aromatase, PPAR-gamma
  - Stress response: ARE, ATAD5, HSE, MMP, p53
  - Developmental: ZF, etc.
  - Cytotoxicity: HepG2, HEK293, etc.
  - In vivo: Rat LD50, Mouse LD50, Rat NOAEL
- **Filter**: Keep compounds with ≥5 assay measurements
- **Output**: `comptox_curated.csv` (SMILES, endpoint_matrix)

### 1.2 ChEMBL Bioactivity Data (~50K compounds)
**Source**: ChEMBL 33 (or latest)
- Target: Human protein targets with IC50/Ki/EC50
- Filter: Standard type = IC50, standard unit = nM, confidence ≥ 7
- Convert to pIC50 = -log10(IC50 × 1e-9)
- **Output**: `chembl_bioactivity.csv` (SMILES, target, pIC50)

### 1.3 Existing Datasets (Already integrated)
- Tox21: 12 endpoints, ~12K compounds
- ClinTox: 2 endpoints (CT_TOX, FDA_APPROVED), ~1.5K compounds

### 1.4 Data Merging Strategy
```python
# Master table schema
master_df = pd.DataFrame({
    'smiles': [...],           # Canonical SMILES (RDKit standardized)
    'source': [...],           # 'comptox' | 'chembl' | 'tox21' | 'clintox'
    'split': [...],            # 'train' | 'val' | 'test' (scaffold split)
    # Multi-task labels (NaN = missing)
    'NR-AR': [...],
    'NR-AR-LBD': [...],
    # ... all 50+ endpoints
    # Regression targets (ChEMBL)
    'CHEMBL_TARGET_1_pIC50': [...],
    'CHEMBL_TARGET_2_pIC50': [...],
})
```

### 1.5 Scaffold Split (Critical for generalization)
- Use `deepchem.splits.ScaffoldSplitter` or RDKit Murcko scaffolds
- 80/10/10 split ensuring no scaffold leakage
- **Must do BEFORE any feature engineering**

---

## Phase 2: Self-Supervised Pretraining (Week 2-3)

### 2.1 Pretraining Objectives (on 760K CompTox SMILES)
| Task | Description | Loss |
|------|-------------|------|
| **Atom Masking** | Mask 15% atoms, predict atom type + formal charge | CrossEntropy |
| **Bond Prediction** | Predict bond type between masked atoms | CrossEntropy |
| **Context Prediction** | Subgraph → global context embedding | MSE / InfoNCE |
| **Motif Prediction** | Predict functional group presence | BCE |

### 2.2 Architecture Modifications
```python
class PharmaGNN_Pretrain(PharmaGNN):
    def __init__(self, ...):
        super().__init__(...)
        self.atom_pred_head = nn.Linear(hidden_dim, num_atom_types)
        self.bond_pred_head = nn.Linear(hidden_dim * 2, num_bond_types)
        self.motif_head = nn.Linear(hidden_dim, num_motifs)
    
    def forward_pretrain(self, x, edge_index, edge_attr, batch, mask):
        # ... encode graph
        # Atom masking loss
        # Bond prediction loss
        # Motif prediction loss
        return total_loss
```

### 2.3 Training Config
- **Epochs**: 50-100
- **Batch**: 256 (gradient accumulation if OOM)
- **LR**: 1e-3 with cosine annealing
- **GPU**: 1-2× V100/A100 (or 4× T4)
- **Checkpoint**: Save best by validation loss

---

## Phase 3: Multi-Task Fine-Tuning (Week 3-4)

### 3.1 Task Formulation
| Task Type | Endpoints | Loss | Weight |
|-----------|-----------|------|--------|
| **Classification** | 50 Tox21/CompTox binary | BCEWithLogits | 1.0 |
| **Regression** | ChEMBL pIC50 (20 targets) | MSE / Huber | 0.5 |
| **Multi-label** | ClinTox (CT_TOX, FDA) | BCE | 2.0 (oversample) |

### 3.2 Loss Function
```python
def multitask_loss(pred_dict, target_dict, task_weights):
    loss = 0
    for task_name, pred in pred_dict.items():
        target = target_dict[task_name]
        mask = ~torch.isnan(target)
        if task_name in CLASSIFICATION_TASKS:
            loss += task_weights[task_name] * BCEWithLogitsLoss()(
                pred[mask], target[mask])
        else:  # regression
            loss += task_weights[task_name] * HuberLoss()(
                pred[mask], target[mask])
    return loss
```

### 3.3 Training Strategy
1. **Freeze encoder** (pretrained GNN) → train heads only (5 epochs)
2. **Unfreeze all** → full fine-tuning (30-50 epochs)
3. **Learning rates**: encoder=1e-4, heads=1e-3
4. **Early stopping**: patience=10 on validation AUC (classification) / RMSE (regression)

### 3.4 Concentration Integration
- Add `concentration_molar` as global feature (already implemented)
- For ChEMBL: use assay concentration if available, else 10 µM default
- For Tox21/CompTox: 10 µM (standard screening concentration)

---

## Phase 4: Evaluation & Calibration (Week 4)

### 4.1 Metrics
| Task | Metric | Target |
|------|--------|--------|
| Classification | ROC-AUC (per endpoint) | > 0.80 |
| Classification | PR-AUC (imbalanced) | > 0.50 |
| Regression | RMSE (pIC50) | < 0.8 log units |
| Regression | Pearson R | > 0.6 |
| Overall | Balanced accuracy | > 0.75 |

### 4.2 Calibration
- **Temperature scaling** on validation set
- **Conformal prediction** for uncertainty intervals
- Reliability diagrams per endpoint

### 4.3 Interpretability Validation
- GNNExplainer on test set
- Compare attention weights with known toxicophores
- Sanity check: known toxicophores → high attention

---

## Phase 5: Deployment & Monitoring (Week 5)

### 5.1 Model Export
```python
# Export to TorchScript for production
scripted = torch.jit.script(model.eval())
scripted.save("pharma_gnn_production.pt")
```

### 5.2 API Updates
- Add `/api/predict_batch` endpoint
- Return uncertainty estimates (MC dropout or ensemble)
- Add model versioning header

### 5.3 Monitoring
- Track prediction drift (input distribution shift)
- Log prediction confidence distributions
- A/B test against current model

---

## Compute Requirements

| Phase | GPU Hours | Est. Cost (A100 80GB @ $2/hr) |
|-------|-----------|-------------------------------|
| Pretraining (760K) | 100-200 | $200-400 |
| Fine-tuning (200K) | 50-100 | $100-200 |
| Hyperparameter tuning | 100-200 | $200-400 |
| **Total** | **250-500** | **$500-1000** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CompTox ETL fails | Fallback: Use pre-curated MoleculeNet datasets |
| Pretraining doesn't help | Skip to supervised only; pretrain is bonus |
| GPU OOM | Gradient accumulation, mixed precision, smaller batch |
| Label noise in CompTox | Aggressive filtering, ensemble of annotators |
| Scaffold split too hard | Try random split as baseline, report both |

---

## Deliverables Timeline

| Week | Deliverable |
|------|-------------|
| 1 | `comptox_curated.csv`, `chembl_bioactivity.csv`, merged master dataset |
| 2 | Pretrained model checkpoint (`pretrain_best.pt`) |
| 3 | Fine-tuned model (`finetune_best.pt`), evaluation report |
| 4 | Calibrated production model (`pharma_gnn_production.pt`) |
| 5 | Deployed API with new model, monitoring dashboard |

---

## Next Steps (Immediate)

1. **Download CompTox v3.0 data** (run tonight)
2. **Write ETL scripts** for CompTox + ChEMBL
3. **Implement scaffold split** utility
4. **Modify PharmaGNN** for pretraining heads
5. **Setup training pipeline** with config management (Hydra/YAML)

---

*Plan created: 2026-09-25*
*Status: Ready for execution*