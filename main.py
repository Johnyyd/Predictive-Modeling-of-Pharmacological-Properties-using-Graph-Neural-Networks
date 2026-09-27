import os
import math
import logging
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.explain import Explainer, GNNExplainer

# Modular PharmaGNN package imports
from pharma_gnn.model import PharmaGNN
from pharma_gnn.chemistry import (
    frag_names,
    frag_funcs,
    TOXICOPHORE_DEFINITIONS,
    get_toxicophore_density,
    analyze_toxicophores,
    get_compound_name,
    smiles_to_graph
)
from pharma_gnn.security import (
    SlidingWindowRateLimiter,
    RateLimitMiddleware,
    PayloadSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    rate_limiter,
    DEFAULT_RATE_LIMIT,
    PREDICT_RATE_LIMIT
)
from pharma_gnn.config import load_model_config

RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger("pharmagnn_api")

app = FastAPI(
    title="PharmaGraph GNN Service",
    description="High-performance Pharmacological GNN API with multi-layered DoS defense, sliding-window rate limiting, and OWASP hardening.",
    version="2.1.0"
)

# Register security & defense middlewares (innermost to outermost)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(PayloadSizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Safe global exception handler preventing internal traceback / path disclosure (OWASP A02/A10)
@app.exception_handler(Exception)
async def safe_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers
        )
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred while processing chemical graph."}
    )

# 1. Initialize model architecture and weights
model_cfg = load_model_config("model_config.json")
hidden_channels = model_cfg.get('hidden_channels', 32)
num_layers = model_cfg.get('num_layers', 2)
heads = model_cfg.get('heads', 2)
residual = model_cfg.get('residual', num_layers > 2)
fg_embed_dim = model_cfg.get('fg_embed_dim', 16 if hidden_channels > 32 else 8)

ai_model = PharmaGNN(
    num_node_features=6, 
    hidden_channels=hidden_channels, 
    num_classes=13,
    num_layers=num_layers,
    heads=heads,
    residual=residual,
    fg_embed_dim=fg_embed_dim
)

weights_path = model_cfg.get(
    'weights_path', 
    'pharma_gnn_production_state_dict.pt' if os.path.exists('pharma_gnn_production_state_dict.pt') and hidden_channels > 32 else 'pharma_gnn_weights_universal.pt'
)
if os.path.exists(weights_path):
    try:
        ai_model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"[+] Successfully loaded trained model weights from {weights_path}.")
    except Exception as e:
        print(f"[-] Warning: Could not load model weights from {weights_path}: {e}")
else:
    print("[-] No weights file detected. Initializing random weights.")

ai_model.eval()

class MoleculeRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=500, description="SMILES chemical structure (max 500 chars)")
    concentration_molar: float = Field(default=1.0, ge=1e-12, le=10.0, description="Concentration in Molar units (10^-12 to 10.0 M)")

@app.get("/api/health")
@app.get("/health")
async def health_check():
    """Health and readiness probe endpoint (exempt from rate limiting)."""
    return {
        "status": "ok",
        "service": "PharmaGraph GNN Service",
        "version": "2.1.0",
        "hidden_channels": hidden_channels,
        "layers": num_layers
    }

@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
    """
    Predict 13 pharmacological and biological toxicity endpoints using PharmaGNN.
    Includes attention synergy extraction, GNNExplainer atom attribution, and dosage sensitivity.
    """
    raw_mol = Chem.MolFromSmiles(request.smiles)
    if raw_mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
    graph = smiles_to_graph(request.smiles, request.concentration_molar)
    if graph is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
    # Compute pIC50 from concentration (Molar)
    if request.concentration_molar <= 0:
        pIC50 = 0.0
    else:
        pIC50 = -math.log10(request.concentration_molar)
        
    concentration_tensor = torch.tensor([[pIC50]], dtype=torch.float)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    
    with torch.no_grad():
        prediction_tensor, attn_weights = ai_model(
            x=graph.x, 
            edge_index=graph.edge_index, 
            edge_attr=graph.edge_attr, 
            batch=graph.batch, 
            global_features=graph.global_features, 
            func_group_features=graph.func_group_features,
            concentration=concentration_tensor,
            return_attention=True
        )
        
        # Apply Sigmoid to convert raw logits to probabilities in [0, 1]
        probabilities = torch.sigmoid(prediction_tensor)
        
        # CT_TOX is class index 12 (ClinTox toxicity) - primary toxicity_risk
        CT_TOX_IDX = 12
        ct_tox_prob = probabilities[0, CT_TOX_IDX].item()
        toxicity_score = ct_tox_prob * 100
        
        # Report most probable target class for reference
        max_prob, target_class = torch.max(probabilities, dim=1)
        target_class_idx = target_class.item()
        all_probs = probabilities[0].tolist()
        
        # Baseline screening prediction at 10 µM (pIC50 = 5.0) for dosage sensitivity analysis
        baseline_pic50 = 5.0
        baseline_conc_tensor = torch.tensor([[baseline_pic50]], dtype=torch.float)
        baseline_pred = ai_model(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
            global_features=graph.global_features,
            func_group_features=graph.func_group_features,
            concentration=baseline_conc_tensor
        )
        baseline_risk = torch.sigmoid(baseline_pred)[0, CT_TOX_IDX].item() * 100.0
        delta_risk = toxicity_score - baseline_risk
        
    # Extract functional groups present and pairwise cross-attention synergy
    fg_tensor = graph.func_group_features[0]
    active_indices = [i for i, cnt in enumerate(fg_tensor) if cnt > 0]
    functional_groups_present = [
        {"name": frag_names[i], "count": int(fg_tensor[i].item())}
        for i in active_indices
    ]
    
    synergy_pairs = []
    for idx_a, i in enumerate(active_indices):
        for j in active_indices[idx_a + 1:]:
            score = (attn_weights[0, i, j].item() + attn_weights[0, j, i].item()) / 2.0
            synergy_pairs.append({
                "group_a": frag_names[i],
                "group_b": frag_names[j],
                "count_a": int(fg_tensor[i].item()),
                "count_b": int(fg_tensor[j].item()),
                "synergy_score": round(float(score), 4),
                "description": f"Attention coupling between {frag_names[i]} and {frag_names[j]}"
            })
    synergy_pairs.sort(key=lambda s: s["synergy_score"], reverse=True)
    
    # Analyze knowledge-based toxicophore alerts
    toxicophore_alerts = analyze_toxicophores(raw_mol)
    
    # Assess dosage impact
    if request.concentration_molar >= 1e-3 and delta_risk > 10.0:
        dosage_assessment = (
            f"Elevated concentration ({request.concentration_molar:.2g} M) amplifies predicted toxicity "
            f"by +{delta_risk:.1f}% compared to standard baseline screening (10 µM)."
        )
    elif request.concentration_molar <= 1e-6 and delta_risk < -10.0:
        dosage_assessment = (
            f"Sub-micromolar dilution mitigates predicted toxicity by {delta_risk:.1f}% "
            f"compared to standard baseline screening (10 µM)."
        )
    elif baseline_risk >= 50.0:
        dosage_assessment = (
            f"Intrinsic baseline toxicity is high ({baseline_risk:.1f}%), indicating intrinsic structural hazard "
            f"independent of dosage variations (current delta: {delta_risk:+.1f}%)."
        )
    else:
        dosage_assessment = (
            f"Dosage response remains stable around baseline screening levels (current delta: {delta_risk:+.1f}%)."
        )
        
    dosage_effect = {
        "user_concentration_molar": request.concentration_molar,
        "baseline_concentration_molar": 1e-5,
        "user_toxicity_risk": round(toxicity_score, 2),
        "baseline_toxicity_risk": round(baseline_risk, 2),
        "delta_risk": round(delta_risk, 2),
        "assessment": dosage_assessment
    }
        
    # Interpretability with GNNExplainer
    ai_model.eval()
    explainer = Explainer(
        model=ai_model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    explanation = explainer(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.batch,
        global_features=graph.global_features,
        func_group_features=graph.func_group_features,
        concentration=concentration_tensor
    )
    
    # Extract atom (node) importance scores
    if explanation.node_mask is not None:
        node_importance = explanation.node_mask.mean(dim=1).tolist()
    else:
        node_importance = [0.0] * graph.num_nodes
        
    # Identify top contributing atoms (prioritizing heavy atoms)
    mol_hs = Chem.AddHs(raw_mol)
    atom_attributions = []
    for idx, imp in enumerate(node_importance):
        if idx < mol_hs.GetNumAtoms():
            symbol = mol_hs.GetAtomWithIdx(idx).GetSymbol()
            atom_attributions.append({
                "atom_index": idx,
                "element": symbol,
                "importance_score": round(float(imp), 4)
            })
    atom_attributions.sort(key=lambda a: a["importance_score"], reverse=True)
    heavy_atoms = [a for a in atom_attributions if a["element"] != "H"]
    top_contributing_atoms = heavy_atoms[:5] if heavy_atoms else atom_attributions[:5]
    
    # Determine primary driving factor & synthesis
    if toxicophore_alerts:
        primary_driver = "Intrinsic Structural Alerts & Toxicophores"
        names_str = ", ".join([a["name"] for a in toxicophore_alerts[:3]])
        summary_text = f"Predicted toxicity ({toxicity_score:.1f}%) is predominantly driven by structural alerts: {names_str}. "
        if synergy_pairs and synergy_pairs[0]["synergy_score"] > 0.05:
            summary_text += f"The GNN attention mechanism detected synergy between {synergy_pairs[0]['group_a']} and {synergy_pairs[0]['group_b']}. "
        if delta_risk > 10.0:
            summary_text += f"High concentration ({request.concentration_molar:.2g} M) further amplifies risk by +{delta_risk:.1f}%."
    elif toxicity_score >= 50.0:
        if delta_risk > 15.0:
            primary_driver = "High Concentration / Dosage Amplification"
            summary_text = (
                f"Predicted toxicity ({toxicity_score:.1f}%) is primarily driven by elevated dosage/concentration "
                f"({request.concentration_molar:.2g} M vs 10 µM screening baseline, +{delta_risk:.1f}% shift)."
            )
        elif synergy_pairs:
            primary_driver = "Functional Group Synergy"
            top_syn = synergy_pairs[0]
            summary_text = (
                f"Predicted toxicity ({toxicity_score:.1f}%) is driven by synergistic interaction between "
                f"{top_syn['group_a']} and {top_syn['group_b']} (attention coupling: {top_syn['synergy_score']})."
            )
        else:
            primary_driver = "Graph Topology & Electronic Distribution"
            summary_text = (
                f"Predicted toxicity ({toxicity_score:.1f}%) is attributed to specific atom connectivity "
                f"and electron charge distribution across the graph network."
            )
    else:
        primary_driver = "Benign / Safe Molecular Profile"
        summary_text = (
            f"Low predicted toxicity risk ({toxicity_score:.1f}%). No high-hazard structural alerts detected, "
            f"and functional group interactions remain within safe physiological thresholds."
        )
    
    # Lookup compound name
    compound_name = get_compound_name(request.smiles)
    
    return {
        "smiles": request.smiles,
        "compound_name": compound_name,
        "concentration_molar": request.concentration_molar,
        "pIC50": pIC50,
        "graph_info": {
            "atoms_count": graph.num_nodes,
            "bonds_count": graph.num_edges // 2
        },
        "predictions": {
            "toxicity_risk": f"{toxicity_score:.2f}%",
            "target_class": target_class_idx,
            "ct_tox_class": CT_TOX_IDX,
            "all_class_probs": [f"{p:.4f}" for p in all_probs]
        },
        "decision_attribution": {
            "primary_driver": primary_driver,
            "summary_text": summary_text.strip(),
            "toxicophores": toxicophore_alerts,
            "functional_groups_present": functional_groups_present,
            "functional_group_synergy": synergy_pairs,
            "dosage_effect": dosage_effect,
            "top_contributing_atoms": top_contributing_atoms
        },
        "explanation": {
            "node_importance": node_importance
        },
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
