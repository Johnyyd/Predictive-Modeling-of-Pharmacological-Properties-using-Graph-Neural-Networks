# PharmaGraph: AI Pharmacological Properties and Toxicity Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.4%2B-EE4C2C.svg?logo=pytorch)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09%2B-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PharmaGraph** is an end-to-end deep learning platform designed for computational chemistry, toxicology, and drug discovery. It predicts compound pharmacological properties and biological toxicity risks using **Graph Attention Networks (GATv2)** enhanced with **Functional Group Cross-Attention**, **Knowledge-Based Toxicophore Alert Densities**, and **Concentration/Dosage Awareness**.

---

## 📑 Table of Contents

- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Model Details](#model-details)
- [Installation & Environment Setup](#installation--environment-setup)
- [Running the Application](#running-the-application)
  - [Local Development Mode](#1-local-development-mode)
  - [Docker & Docker Compose](#2-docker--docker-compose)
- [Web Dashboard Guide](#web-dashboard-guide)
- [REST API Reference](#rest-api-reference)
- [Model Training & Dataset Pipeline](#model-training--dataset-pipeline)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🌟 Key Capabilities

1. **Multi-Task Pharmacological Profiling**:
   Jointly predicts 13 biological assay endpoints, including 12 nuclear receptor (NR) and stress response (SR) pathways from the NIH Tox21 initiative, alongside clinical toxicity (`CT_TOX`) from the ClinTox benchmark.
2. **Concentration & Dosage Conditioning**:
   Toxicity is fundamentally dose-dependent. PharmaGraph accepts compound dosage inputs (Molar concentration) and projects it into a normalized affinity space ($pIC_{50} = -\log_{10}(\text{Molar})$) to condition predictions dynamically.
3. **Functional Group Cross-Attention Mechanism**:
   Extracts 85 chemical functional group fragment counts via RDKit and processes them through multi-head self-attention to capture non-linear group synergies and reactive chemical interactions.
4. **Knowledge-Driven Toxicophore Densities**:
   Encodes structural alert densities for recognized toxicophores (Cyanides, Organophosphates, Benzene rings, Phenols, Aldehydes, Sulfides, and Halogenated aromatics).
5. **Explainable AI (XAI) with 3D Visualization**:
   Integrates **GNNExplainer** to assign importance scores to individual atoms. Identified toxicophores are mapped onto interactive 3D conformations (`py3Dmol`), highlighting high-risk regions in red.
6. **Flexible Input Modalities**:
   Supports four distinct, exclusive input methods:
   - Curated reference presets (Aspirin, Paracetamol, Caffeine, Nicotine, Cyanide, Phenol, etc.)
   - Chemical name search with local caching and automated PubChem lookup
   - Direct SMILES string specification with RDKit chemical syntax validation
   - Interactive 2D chemical structure sketcher (Ketcher)

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([User / Chemist]) -->|Web Interface| UI[Streamlit Frontend :8501]
    User -->|HTTP Requests| API[FastAPI Backend :1234]

    subgraph Frontend [Streamlit Presentation Layer]
        UI --> M1[Method 1: Presets]
        UI --> M2[Method 2: Name Search]
        UI --> M3[Method 3: Direct SMILES]
        UI --> M4[Method 4: 2D Sketcher]
        M1 & M2 & M3 & M4 --> Validation[RDKit SMILES Validation]
        Validation --> StateMgr[State Manager & Reset Controller]
    end

    StateMgr -->|JSON Payload| API

    subgraph Backend [FastAPI Core Engine]
        API --> GraphBuilder[smiles_to_graph Converter]
        GraphBuilder --> MolGraph[Atom Features + Bond Edges]
        GraphBuilder --> GlobDesc[Global Descriptors: MolWt, LogP, TPSA, RotBonds]
        GraphBuilder --> ToxDensity[7 Toxicophore Density Metrics]
        GraphBuilder --> FuncGroups[85 RDKit Functional Groups]
        GraphBuilder --> DoseCond[Dosage Conditioning: pIC50]

        MolGraph & GlobDesc & ToxDensity & FuncGroups & DoseCond --> GNN[PharmaGNN Architecture]
        GNN --> Preds[13 Multi-Task Logits / Probabilities]
        GNN --> Explainer[GNNExplainer Node Attribution]
        API --> PubChemCache[Local Cache / PubChem Reverse Resolver]
    end

    Preds --> Visualizer[3D Py3Dmol Conformation & Metrics Display]
    Explainer --> Visualizer
    Visualizer --> UI
```

---

## 🧠 Model Details

The **PharmaGNN** model is an attentive graph neural network specifically tailored for small-molecule chemical graphs:

- **Node Features (6 dimensions per atom)**:
  1. Atomic number
  2. Degree (connectivity)
  3. Aromaticity indicator (0 or 1)
  4. Implicit valence
  5. Formal charge
  6. Number of radical electrons
- **Edge Attributes**: Bond type multiplicity (single, double, triple, aromatic).
- **Message Passing Backbone**: 2-layer **GATv2Conv** (Graph Attention Network v2) with multi-head attention and batch normalization.
- **Pooling**: Hybrid dual pooling combining both `global_mean_pool` and `global_max_pool`.
- **Functional Group Module**: Multi-head self-attention module (`embed_dim=8`, 4 heads) operating over 85 RDKit functional fragments, followed by a multi-layer perceptron.
- **Conditioning**: Concatenation of pooled atom representations, 11 global molecular descriptors (MW, LogP, TPSA, Rotatable Bonds + 7 toxicophore densities), 32-dim functional group embeddings, and scalar dosage $pIC_{50}$.
- **Output Head**: Multi-task classification producing independent logits across 13 pharmacological endpoints.

---

## 💻 Installation & Environment Setup

### Prerequisites

- Linux, macOS, or Windows (WSL2 recommended for Windows)
- Python **3.10** or higher
- System libraries for RDKit rendering: `libxrender1`, `libxext6`, `libexpat1`

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y libxrender1 libxext6 libexpat1
```

### 1. Clone the Repository

```bash
git clone https://github.com/Johnyyd/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks.git
cd Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install "numpy<2"
pip install -r requirements.txt
```

*(Optional: Install `streamlit-ketcher` if you wish to enable the 2D molecule sketcher):*

```bash
pip install streamlit-ketcher
```

---

## 🚀 Running the Application

### 1. Local Development Mode

To run both services locally, launch the backend server first, followed by the Streamlit frontend.

#### Step 1: Start the FastAPI AI Backend

```bash
# In Terminal 1
uvicorn main:app --host 0.0.0.0 --port 1234 --reload
```

The API service will start on `http://localhost:1234`. Access interactive Swagger documentation at `http://localhost:1234/docs`.

#### Step 2: Start the Streamlit Web Application

```bash
# In Terminal 2
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Open your browser and navigate to `http://localhost:8501`.

---

### 2. Docker & Docker Compose

Run the entire production-grade stack with a single command:

```bash
docker-compose up -d --build
```

- **Frontend Dashboard**: `http://localhost:8501`
- **FastAPI Backend**: `http://localhost:1234`
- **Swagger Documentation**: `http://localhost:1234/docs`

To view container logs or stop services:

```bash
# Check service logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## 🖥️ Web Dashboard Guide

The PharmaGraph dashboard provides an intuitive, streamlined interface for molecular evaluation:

### 1. Selecting an Input Method
Use the top horizontal selector to choose your preferred input method. Switching methods automatically clears previous selections to prevent conflicting parameters:

- **Method 1: Curated Presets**  
  Select from pre-configured chemical agents (e.g., *Aspirin*, *Paracetamol*, *Caffeine*, *Ethanol*, *Nicotine*, *Hydrogen Cyanide*, *Phenol*).
- **Method 2: Search by Name**  
  Type a common or IUPAC drug/chemical name (e.g., `Ibuprofen`, `Penicillin`, `Glucose`). The system queries local caches first, falling back to the PubChem REST API.
- **Method 3: Direct SMILES**  
  Input any standard SMILES chemical string (e.g., `CC(=O)Oc1ccccc1C(=O)O`). Structures are validated using RDKit before processing.
- **Method 4: 2D Sketcher** *(if installed)*  
  Interactively draw molecular structures and click **Apply** to convert to SMILES.

### 2. Setting Concentration
Enter the target concentration in Molar units (default: `1.0 M`). For high dilution screening (e.g., $10\ \mu\text{M}$), enter `0.00001`.

### 3. Interpreting Results
Click **🚀 Run AI GNN Pharmacological Analysis**:
- **3D Interactive Molecular Viewer**: Rotate, zoom, and inspect molecular conformations in real time. Atoms contributing to toxic risk are highlighted with red spheres.
- **Toxicity Risk Indicator**:
  - `Safe / Low Concern (< 50%)`: Stable, non-toxic profile.
  - `Moderate Concern (50% - 79%)`: Contains potential structural alerts or moderate receptor activity.
  - `Critical Risk (≥ 80%)`: Severe biological risk; strong toxicophore presence.
- **Chemical Properties**: Identified compound name, molecular formula, molecular weight, atom breakdown, and detected functional groups.
- **Graph Metrics**: Total atoms (graph nodes) and covalent bonds (graph edges).

---

## 📡 REST API Reference

### Health Check / Documentation
- **Interactive Swagger UI**: `http://localhost:1234/docs`
- **OpenAPI Schema**: `http://localhost:1234/openapi.json`

---

### Predict Toxicity Endpoint

- **Route**: `POST /api/predict`
- **Content-Type**: `application/json`

#### Request Schema

```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "concentration_molar": 1.0
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `smiles` | `string` | **Yes** | — | Valid SMILES string representing the molecule |
| `concentration_molar` | `float` | No | `1.0` | Target concentration in Molar units (M) |

#### Example Request (`curl`)

```bash
curl -X POST http://localhost:1234/api/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "concentration_molar": 1.0}'
```

#### Example Response

```json
{
  "smiles": "CCO",
  "compound_name": "Ethanol",
  "concentration_molar": 1.0,
  "pIC50": 0.0,
  "graph_info": {
    "atoms_count": 9,
    "bonds_count": 8
  },
  "predictions": {
    "toxicity_risk": "0.00%",
    "target_class": 4,
    "ct_tox_class": 12,
    "all_class_probs": [
      "0.0175", "0.0013", "0.0017", "0.0007", "0.0543", 
      "0.0003", "0.0021", "0.0182", "0.0004", "0.0134", 
      "0.0007", "0.0001", "0.0000"
    ]
  },
  "explanation": {
    "node_importance": [
      0.1998, 0.2250, 0.1811, 0.1321, 0.1364, 
      0.1357, 0.1264, 0.1268, 0.1279
    ]
  },
  "status": "Success"
}
```

---

## 🧪 Model Training & Dataset Pipeline

The repository includes a complete training pipeline supporting multi-task loss masking and dataset balancing:

### 1. Training Datasets
- **Tox21**: 12 in-vitro assays targeting endocrine disruption and cell viability pathways (7,831 compounds).
- **ClinTox**: Clinical trial failure records and FDA-approved status (1,478 compounds).
- **Curated Reference Toxins & Benign Compounds**: Explicit balance sets ensuring critical industrial toxicophores (cyanides, phenols, halogenated aromatics) and benign food-grade compounds are accurately learned.

### 2. Running Model Training

To retrain the multi-task GNN model from scratch:

```bash
python train.py
```

The script will:
1. Download and merge the Tox21 and ClinTox multi-task datasets.
2. Apply scaffold split and positive class re-weighting (`pos_weight=5.0`).
3. Train for 10 epochs using `BCEWithLogitsLoss` with loss masking for missing assay values.
4. Calculate and report mean ROC-AUC across all valid tasks.
5. Save the updated network weights to `pharma_gnn_weights_universal.pt`.

### 3. Extended Training Pipeline (`scripts/` & `run_pipeline.py`)
For larger-scale foundation-tier workflows, the multi-phase training pipeline automates self-supervised pretraining, multi-task transfer learning, calibration, and production packaging:

| Giai đoạn (Phase) | Script thực thi | Trọng số / Artifact sinh ra | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| **Phase 1: Dataset Build** | `scripts/phase1_build_dataset.py` | `data/comptox_pretrain.csv` | Thu thập và chuẩn hóa 100.000 hợp chất hữu cơ từ CompTox 3.0. |
| **Phase 2: Pretraining** | `scripts/phase2_pretrain.py` | `pharma_gnn_pretrained_encoder.pt` | Self-supervised masked atom & functional group pretraining (6h runtime). |
| **Phase 3: Fine-tuning** | `scripts/phase3_finetune.py` | `pharma_gnn_finetuned.pt` | Huấn luyện đa nhiệm 30 epochs trên 13 tasks (Tox21 + ClinTox), đạt **Macro Test ROC-AUC = 0.8146 (81.46%)**. |
| **Phase 4: Calibration** | `scripts/phase4_calibration.py` | `calibration_info.json` | Tối ưu nhiệt độ $T=0.5$, giảm sai số hiệu chuẩn ECE từ 18.0% xuống **11.5%**. |
| **Phase 5: Deployment** | `scripts/phase5_deployment.py` | 🎯 `pharma_gnn_production_state_dict.pt` | **Mô hình chính thức hoàn thành toàn bộ pipeline**, đi kèm `model_config.json`. |

> [!NOTE]
> **Lưu ý về Giới hạn Phần cứng (Hardware Constraints & Reproducibility Note):**
> Do điều kiện và tài nguyên phần cứng tính toán có giới hạn (môi trường máy tính cá nhân/GPU đơn lẻ thay vì hạ tầng cụm GPU/TPU cluster công nghiệp), tác giả đã tối ưu hóa pipeline huấn luyện tốt nhất có thể trong phạm vi tài nguyên hiện có:
> - Áp dụng tập con **100.000 hợp chất đại diện** từ CompTox 3.0 cho Phase 2 Pretraining (thay vì toàn bộ 760.000 hợp chất) để hoàn thành trong ~6 giờ tính toán.
> - Huấn luyện mô hình đạt **Macro Test ROC-AUC = 0.8146 (81.46%)** trên 13 biological tasks với kiến trúc 128 hidden channels, 4 GATv2 layers và 4 attention heads.
> - Kết quả này phản ánh năng lực dự đoán tối ưu nhất đạt được trong điều kiện giới hạn phần cứng hiện tại của tác giả, đồng thời cung cấp kiến trúc mã nguồn mở hoàn chỉnh để cộng đồng có thể dễ dàng scale up trên hạ tầng mạnh mẽ hơn.

---

## 📁 Project Structure & Model Checkpoints

```text
.
├── app.py                             # Streamlit interactive Apple-grade web dashboard
├── main.py                            # FastAPI backend REST service, DoS protection & GNNExplainer
├── model.py                           # PyTorch Geometric PharmaGNN entry point (re-exports pharma_gnn.model)
├── train.py                           # Multi-task GNN baseline training script
├── run_pipeline.py                    # Orchestrator pipeline tự động chạy từ Phase 1 đến Phase 5
│
├── 📦 pharma_gnn/                      # Core modular Python package
│   ├── __init__.py                    # Public package exports
│   ├── model.py                       # GATv2Conv + FunctionalGroupInteraction architecture
│   ├── chemistry.py                   # Atom features, graph construction, toxicophores & descriptors
│   ├── security.py                    # Sliding window rate limiter & OWASP security middlewares
│   └── config.py                      # Centralized configuration loader
│
├── 🏆 MODEL CHECKPOINTS:
│   ├── pharma_gnn_production_state_dict.pt  # [CHÍNH THỨC] Mô hình hoàn thành toàn bộ 5 Phase (128d, 4 layers, 4 heads)
│   ├── pharma_gnn_finetuned.pt              # Checkpoint sau Phase 3 Fine-tuning (ROC-AUC 81.46%)
│   ├── pharma_gnn_pretrained_encoder.pt     # Checkpoint sau Phase 2 Pretraining (100k hợp chất CompTox)
│   └── pharma_gnn_weights_universal.pt      # Bản baseline cũ v1 (32d, 2 layers, ~9k mẫu)
│
├── ⚙️ CONFIGURATION & CALIBRATION (configs/ & root):
│   ├── configs/                             # Centralized configs directory
│   ├── model_config.json                    # Cấu hình hyperparameters & đường dẫn weights production
│   ├── calibration_info.json                # Thông số Temperature Scaling & kiểm định hóa học
│   ├── deployment_summary.json              # Tổng kết triển khai Phase 5
│   └── monitoring_config.json               # Cấu hình giám sát độ trôi dữ liệu (Data drift)
│
├── requirements.txt                   # Python dependencies specification
├── Dockerfile.backend                 # Docker container for FastAPI backend
├── Dockerfile.frontend                # Docker container for Streamlit frontend
├── Dockerfile.production              # Docker container for production deployment
├── docker-compose.yml                 # Multi-container orchestration config
├── HYBRID_TRAINING_PLAN.md            # Comprehensive multi-source training roadmap
├── data/                              # Local dataset cache and splits
├── scripts/                           # Multi-phase pretraining & calibration pipeline
│   ├── phase1_build_dataset.py
│   ├── phase2_pretrain.py
│   ├── phase3_finetune.py
│   ├── phase4_calibration.py
│   └── phase5_deployment.py
└── README.md                          # Project documentation
```

---

## 🛡️ Production Security & DoS/DDoS Hardening (OWASP Top 10)

The production API in [`main.py`](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/main.py) incorporates defense-in-depth protections engineered specifically against volumetric DoS, algorithmic complexity attacks, and data leakage:

### 1. In-Memory Sliding-Window Rate Limiting
- **Client IP Tracking**: Inspects proxy headers (`X-Forwarded-For`, `X-Real-IP`) and native socket connections with microsecond timestamp deque pruning.
- **Differentiated Endpoint Quotas**:
  - `POST /api/predict`: Capped at **20 requests/minute** per IP (prevents CPU exhaustion on compute-heavy GNN forward passes & GNNExplainer saliency extraction).
  - General API: **120 requests/minute** per IP.
- **RFC 6585 Compliance**: Exceeded quotas return HTTP `429 Too Many Requests` with a dynamic `Retry-After: <seconds>` header.
- **Probe Immunity**: Orchestrator readiness and liveness checks (`/api/health`, `/health`) bypass rate limiting to prevent Kubernetes flapping.

### 2. Payload Safety & Memory Exhaustion Shield
- **Strict Body Capping**: [`PayloadSizeLimitMiddleware`](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/main.py) rejects any incoming request body exceeding **64 KB** with HTTP `413 Payload Too Large` before buffer allocation.
- **SMILES Length & Geometry Bounds**: Maximum chemical SMILES length is restricted to **500 characters** with Pydantic validation (HTTP `422`), preventing RDKit parser denial-of-service from cyclomatic explosion.
- **Concentration Clamping**: Concentration is strictly bounded to physiological and experimental ranges ($10^{-12} \le C \le 10.0$ M).

### 3. OWASP Top 10 Response Headers
All HTTP responses are armored with secure headers:
- `X-Content-Type-Options: nosniff` (A02 Security Misconfiguration)
- `X-Frame-Options: DENY` (Clickjacking defense)
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`
- `Permissions-Policy: accelerometer=(), camera=(), geolocation=(), microphone=()`
- `X-XSS-Protection: 1; mode=block`

### 4. Traceback & Internal Path Masking
- Custom global exception handlers intercept unhandled runtime errors, logging tracebacks securely on the server while returning generic sanitized messages (`HTTP 500`) without exposing internal server paths or stack traces.

---

## 🎨 Apple-Tier Responsive Design System

The frontend in [`app.py`](file:///home/tringuyen/Documents/GitHub/Predictive-Modeling-of-Pharmacological-Properties-using-Graph-Neural-Networks/app.py) was built adhering to Apple WWDC fluid interface principles and Emil Kowalski design engineering:
- **Optical Typography**: Integrated `Plus Jakarta Sans` with tight negative tracking (`-0.025em`) on display headings and `JetBrains Mono` for molecular formulas.
- **Double-Bezel Glassmorphism**: Cards feature concentric translucent backdrops (`backdrop-filter: blur(24px) saturate(180%)`), inner 1px highlight rims (`box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16)`), and deep space gradients (`#080c14`).
- **Tactile Micro-Interactions**: Active press feedback (`transform: scale(0.97)` on `:active`), 150ms `cubic-bezier(0.23, 1, 0.32, 1)` transitions, and glowing hazard indicator badges.
- **Fluid Responsiveness**: Adaptive layouts optimized from mobile phones (`min-h-[100dvh]`) to ultra-wide displays.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

