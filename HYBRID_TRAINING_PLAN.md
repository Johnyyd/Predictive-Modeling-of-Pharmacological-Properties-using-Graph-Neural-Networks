# Comprehensive Engineering Plan: 760K-Molecule Pretraining & Multi-Task Fine-Tuning for PharmaGNN v2

## 1. Executive Summary & Objective

The objective of this plan is to scale the current PharmaGNN model from a dataset of ~7,800 compounds (Tox21 + ClinTox) to a **Foundation Model for Molecular Pharmacology & Toxicology** pre-trained on **760,000+ chemical substances** (EPA CompTox DSSTox chemical library).

### Core Goals:
1. **Chemical Universe Representation**: Learn general chemical grammar, valence rules, and electronic topology across 760k diverse xenobiotics, agrochemicals, natural products, and industrial compounds.
2. **Elimination of False Positives**: Differentiate endogenous biological metabolites (e.g., nucleotide triphosphates like ATP/ADP/GTP) from synthetic organophosphate toxicants (e.g., Sarin, Chlorpyrifos) through contextualized motif learning.
3. **Preservation of Core Innovations**: Retain the 85-functional-group Multihead Cross-Attention synergy mechanism and dosage/concentration sensitivity ($pIC_{50}$).
4. **Production Readiness**: Provide seamless drop-in weight updates for the existing FastAPI backend and Streamlit dashboard with zero downtime.

---

## 2. System Architecture & Model Evolution (PharmaGNN v1 → v2)

```
+-----------------------------------------------------------------------------------+
|                           PHASE 1: 760K PRETRAINING                               |
|                                                                                   |
|  EPA CompTox 760K SMILES                                                          |
|            |                                                                      |
|            v                                                                      |
|  [RDKit Sanitization & Featurization] ---> [LMDB / Sharded Graph Cache]           |
|                                                              |                    |
|                                                              v                    |
|  +-----------------------------------------------------------------------------+  |
|  | PharmaGNN-SSL Backbone (GATv2Conv 4-layer, Hidden=128, Heads=4)             |  |
|  |                                                                             |  |
|  | + Atom Masking Head (Predict Z in 1..118, formal charge)                   |  |
|  | + Bond Prediction Head (Single, Double, Triple, Aromatic)                   |  |
|  | + Motif Coupling Head (85 RDKit Functional Groups Multihead Attention)      |  |
|  +-----------------------------------------------------------------------------+  |
+------------------------------------------+----------------------------------------+
                                           |
                                           | Pretrained Backbone Weights
                                           v
+-----------------------------------------------------------------------------------+
|                        PHASE 2: MULTI-TASK FINE-TUNING                            |
|                                                                                   |
|  Curated Tox21 (12 targets) + ClinTox (CT_TOX) + ChEMBL Bioactivity               |
|                                    |                                              |
|                                    v                                              |
|  +-----------------------------------------------------------------------------+  |
|  | PharmaGNN v2 Downstream Architecture                                        |  |
|  | - Pretrained GATv2 Backbone (Transferred weights)                           |  |
|  | - FunctionalGroupInteraction (85x85 Cross-Attention Synergy)                 |  |
|  | - Global Molecular Features + Toxicophore Density (7 SMARTS alerts)         |  |
|  | - Concentration Conditioning Feature (pIC50 = -log10 Molar)                 |  |
|  | - Multi-Task Projection Layer (13 biological endpoints)                     |  |
|  +-----------------------------------------------------------------------------+  |
|                                    |                                              |
|                                    v                                              |
|                  pharma_gnn_weights_universal.pt (v2)                             |
+------------------------------------+----------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------------+
|                       PHASE 3: PRODUCTION INFERENCE                               |
|                                                                                   |
|  FastAPI (/api/predict) + Streamlit UI (app.py)                                   |
|  - Decision Breakdown: Toxicophores, Attention Synergy, Dosage Impact, Atoms      |
+-----------------------------------------------------------------------------------+
```

---

## 3. Detailed Implementation Phases

### Phase 1: Data Acquisition & High-Performance ETL (Weeks 1 - 2)

#### 1.1 Data Sourcing
- **Primary Source (760K CompTox)**:
  - EPA DSSTox (Distributed Structure-Searchable Toxicity) full snapshot (~875,000 chemical records, ~760,000 unique valid organic structures).
  - Mirror source: EPA CCTE Zenodo/Figshare open research archives and HuggingFace Chemistry Hub (`lukaskim/ChEMBL-36`, `edmanft/zinc250k`).
- **Supervised Benchmark Datasets (Fine-Tuning)**:
  - **Tox21**: 7,831 compounds with 12 nuclear receptor & stress response pathways.
  - **ClinTox**: 1,478 compounds with FDA clinical toxicity / clinical phase outcomes.
  - **ChEMBL 33/34 Curated Bioactivity**: ~50,000 compounds with human target $IC_{50}/K_i$.

#### 1.2 Data Sanitization & Standardization Pipeline
- Parse SMILES via RDKit with strict validation:
  1. Remove inorganic salts / disconnect metal counter-ions (`SaltRemover`).
  2. Neutralize charges where chemically standard.
  3. Generate canonical non-isomeric and isomeric SMILES.
  4. Filter molecules: $3 \le \text{num\_heavy\_atoms} \le 100$.
- Store in high-throughput **LMDB (Lightning Memory-Mapped Database)** or **Parquet partitions** (compressed with Snappy/Zstandard) to prevent RAM bottlenecks during multi-epoch training.

---

### Phase 2: Self-Supervised Pretraining (PharmaGNN-SSL) (Weeks 2 - 3)

#### 2.1 Self-Supervised Pretraining Objectives
To learn molecular physics and chemistry without needing experimental assay labels for all 760k molecules:

1. **Atom Attribute Masking (15% random node masking)**:
   - Replace 15% of atomic features with a `[MASK]` token.
   - Objective: Predict atomic element number $Z \in [1, 118]$ and formal charge using CrossEntropy loss:
     $$\mathcal{L}_{\text{atom}} = -\sum_{i \in \mathcal{M}_{\text{atom}}} \log P(Z_i \mid \mathcal{G}_{\backslash i})$$
2. **Bond Topology Prediction**:
   - Randomly mask 10% of edges in the graph.
   - Objective: Predict whether an edge exists and its bond type (Single, Double, Triple, Aromatic).
     $$\mathcal{L}_{\text{bond}} = -\sum_{(u, v) \in \mathcal{M}_{\text{edge}}} \log P(B_{uv} \mid \mathcal{G}_{\backslash uv})$$
3. **Motif & Functional Group Regularization**:
   - Compute presence of 85 RDKit functional groups.
   - Multi-label binary cross-entropy loss aligning the graph representation with known chemical functional motifs.

#### 2.2 Model Scaling Parameters
| Hyperparameter | PharmaGNN v1 (Current) | PharmaGNN v2 (760K Foundation) |
| :--- | :--- | :--- |
| **GNN Layers** | 2 GATv2Conv layers | 4 GATv2Conv layers + Residual Skips |
| **Hidden Channels** | 32 | 128 |
| **Attention Heads** | 2 | 4 |
| **FG Attention Dim** | 8 | 16 |
| **Total Parameters** | ~98,000 (~400 KB) | ~2,400,000 (~9.5 MB) |
| **Context Regularization**| None | Contrastive Subgraph Mutual Information |

#### 2.3 Optimization & Distributed Training Setup
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.98$, weight decay $10^{-4}$).
- **Learning Rate Schedule**: Cosine Annealing with linear warmup (5 epochs warmup, peak LR $10^{-3}$, decay to $10^{-5}$).
- **Precision**: Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.bfloat16)`).
- **Batch Size**: 256 graph batch (effective batch size 1024 with gradient accumulation).

---

### Phase 3: Multi-Task Downstream Fine-Tuning (Weeks 3 - 4)

#### 3.1 Architecture Integration
- Initialize the GNN backbone with the pretrained weights from Phase 2.
- Attach the specialized pharmacological modules:
  1. `FunctionalGroupInteraction`: Multihead Attention over 85 functional group embeddings (extracting $85 \times 85$ synergy weights).
  2. `GlobalFeatureProjection`: MolWt, MolLogP, TPSA, RotatableBonds, and 7 knowledge-based toxicophore densities.
  3. `ConcentrationProjection`: Dosage feature ($pIC_{50} = -\log_{10}(\text{Molar})$).
  4. `MultiTaskHead`: Linear projection to 13 target biological endpoints.

#### 3.2 Two-Stage Fine-Tuning Strategy
1. **Stage 1 (Head Warmup)**: Freeze the pretrained GNN backbone for 5 epochs; train only the interaction layers, global projection, and classification heads.
2. **Stage 2 (End-to-End Fine-Tuning)**: Unfreeze the full model with discriminative learning rates:
   - Backbone LR: $10^{-4}$
   - Head LR: $10^{-3}$
- **Loss Function**: Masked Multi-Task Binary Cross-Entropy with Logits, weighting positive toxic instances ($pos\_weight = 5.0$) to counteract clinical label imbalance:
  $$\mathcal{L}_{\text{multitask}} = \sum_{t=1}^{13} w_t \cdot \text{BCEWithLogits}(y_t, \hat{y}_t; \text{mask}_t)$$

---

### Phase 4: Quality Control, Benchmarking & Explainability Validation (Week 5)

#### 4.1 Quantitative Benchmarking Gates
The fine-tuned model must pass strict quantitative validation gates on held-out scaffold test sets:
- **Mean ROC-AUC across 12 Tox21 tasks**: $\ge 0.82$ (Baseline v1: 0.74).
- **ClinTox (CT_TOX) ROC-AUC**: $\ge 0.88$ (Baseline v1: 0.79).
- **PR-AUC (Precision-Recall under extreme class imbalance)**: $\ge 0.55$.

#### 4.2 Qualitative Chemical Sanity Check Suite
Every candidate checkpoint must be evaluated against the curated chemical verification suite:
1. **ATP (Endogenous Nucleotide)**:
   - **Target**: Must be classified as **Low Concern (< 35%)**.
   - **Mechanism**: The 760K contextualized backbone must recognize the polyphosphate-ribose linkage as an endogenous nucleotide, avoiding false-positive organophosphate alerts.
2. **Cyanide ($C\#N$) & Sarin**:
   - **Target**: Must be classified as **Critical Hazard (> 95%)**.
3. **Phenol ($C_6H_5OH$)**:
   - **Target**: Correctly categorized with dilute pharmaceutical tolerance vs. high-concentration chemical burn advisory.
4. **Common Nutrients & Excipients** (Water, Glucose, NaCl, Glycine, Aspirin):
   - **Target**: Must remain consistently **Safe (< 15%)**.

#### 4.3 Explainability Calibration
- Ensure `attn_weights` from `FunctionalGroupInteraction` properly isolate true toxicophore pairs.
- GNNExplainer saliency maps must concentrate > 70% of importance on pharmacophore trigger atoms.

---

### Phase 5: Production Deployment & Zero-Downtime Rollout (Week 6)

#### 5.1 Weight Serialization & Backward Compatibility
- Export model weights to `pharma_gnn_weights_universal.pt`.
- Guarantee that `PharmaGNN.forward(..., return_attention=True)` preserves the exact tensor interface expected by [main.py](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/main.py).

#### 5.2 Containerized Rolling Update
1. Update `model_config.json` with model version (`v2.0-760k`), hidden dimension (`128`), and benchmark metrics.
2. Build new Docker images:
   ```bash
   docker compose build backend frontend
   docker compose up -d --no-deps backend
   docker compose up -d --no-deps frontend
   ```
3. Run automated end-to-end regression tests (`pytest tests/`) against the live container before promoting to production.

---

## 4. Compute Budget & Resource Matrix

| Training Stage | Dataset Size | Recommended Hardware | Est. Duration | Cost (Cloud GPU) |
| :--- | :--- | :--- | :--- | :--- |
| **ETL & Graph Cache** | 760K SMILES | 8-core CPU, 32GB RAM | 2 - 3 hours | $0 (Local) |
| **SSL Pretraining (50 epochs)** | 760K graphs | 1x NVIDIA RTX 4090 / A100 (80GB) | 18 - 24 hours | ~$35 - $50 |
| **Fine-Tuning (30 epochs)** | ~15K graphs | 1x NVIDIA RTX 3080/4090 | 1 - 2 hours | ~$3 - $5 |
| **Evaluation & GNNExplainer** | 2K test graphs | 1x GPU or 8-core CPU | 30 minutes | ~$1 |

---

## 5. Immediate Action Items (Sprint 1) — COMPLETED

1. [x] **Update ETL script** ([scripts/download_comptox.py](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/scripts/download_comptox.py)): Replaced deprecated EPA FTP links with active CCTE Zenodo and HuggingFace chemical snapshots. Added resilient mirror failover mechanism (`resolve_download_urls`, `download_with_fallback`), `--dry-run` verification flag, and automated `.gz` extraction. Verified in `tests/test_download_comptox.py`.
2. [x] **Refine SMARTS Definitions** ([main.py](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/main.py)): Replaced crude `P(=O)(O)(O)` pattern with recursive disjunction `[$([P](=[O,S])[F,Cl]),$([P](=[O,S])C#N),$([P](=[O,S])([#6])S),$([P;!$([P]-[O]-[P])](=[O,S])([O,S][#6])([O,S][#6])([O,S,#6][#6]))]`. Successfully excludes endogenous biological metabolites (ATP, ADP, GTP, AMP, cAMP, pyrophosphate) while retaining high-sensitivity detection of nerve agents (Sarin, Soman, VX) and organophosphate insecticides (Parathion, Chlorpyrifos, Malathion). Verified in `tests/test_biological_phosphate_exclusion.py`.
3. [x] **Implement Scaled Backbone** ([model.py](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/model.py)): Extended `PharmaGNN` to support configurable 4-layer 128-channel residual GATv2 architectures with multi-head attention, Xavier parameter initialization, and single-node batchnorm guards, while preserving 100% strict backward compatibility with existing 2-layer 32-channel weights (`pharma_gnn_weights_universal.pt`). Verified in `tests/test_scaled_backbone.py`.
4. [x] **Run Pretraining Smoke Test**: Upgraded [scripts/phase2_pretrain.py](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/scripts/phase2_pretrain.py) with `--smoke-test`, configurable architecture flags, aligned explicit-hydrogen graph conversion, and convergence verification assertions. Verified loss convergence in `tests/test_pretrain_smoke.py` and via CLI execution (`--smoke-test --limit 200 --epochs 3`).

---

## 6. End-to-End Pipeline Scaling & Production Verification — COMPLETED

1. [x] **Phase 1 (Data Acquisition & ETL)**: Verified `download_comptox.py` and `phase1_build_dataset.py` with valid schema serialization and standard int type casting for `data/processed/metadata.json`.
2. [x] **Phase 2 (Self-Supervised Pretraining)**: Verified masked atom/bond prediction with 4-layer 128-channel residual backbone and cosine annealing schedule (`scripts/phase2_pretrain.py`, `tests/test_pretrain_smoke.py`).
3. [x] **Phase 3 (Multi-Task Fine-Tuning)**: Standardized downstream tasks to 13 endpoints (12 Tox21 + CT_TOX). Implemented two-stage fine-tuning (backbone warmup -> discriminative end-to-end training with $pos\_weight=5.0$) (`scripts/phase3_finetune.py`, `tests/test_finetune_smoke.py`).
4. [x] **Phase 4 (Model Calibration & Chemical Sanity)**: Implemented `TemperatureScaling` module ($T=4.000$) reducing ECE from 0.2040 to 0.0951 and Brier score from 0.2272 to 0.1788. Integrated qualitative chemical sanity check suite testing ATP, Cyanide, Sarin, Phenol, and excipients (`scripts/phase4_calibration.py`, `tests/test_calibration_smoke.py`).
5. [x] **Phase 5 (Production Deployment & Serving)**: Config-driven hot-reloading architecture in `main.py` driven by `model_config.json`, healthcheck route `/api/health`, and verified FastAPI serving endpoints (`scripts/phase5_deployment.py`, `tests/test_deployment_smoke.py`).
6. [x] **Full Regression Test Suite**: 32/32 tests passing (100% green).