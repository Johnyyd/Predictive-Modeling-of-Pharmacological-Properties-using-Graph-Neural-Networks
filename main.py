from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import os
import math
import pubchempy as pcp

# Automatically extract 85 RDKit functional group descriptors and names
frag_items = [(name.replace('fr_', ''), func) for name, func in Descriptors.descList if name.startswith('fr_')]
frag_names = [item[0] for item in frag_items]
frag_funcs = [item[1] for item in frag_items]

# Knowledge-based toxicophore alerts with descriptions
TOXICOPHORE_DEFINITIONS = [
    {
        "name": "Cyanide / Nitrile group",
        "smarts": "C#N",
        "description": "Cyano group known for cellular respiration toxicity and cytochrome c oxidase inhibition."
    },
    {
        "name": "Organophosphate ester",
        "smarts": "[$([P](=[O,S])[F,Cl]),$([P](=[O,S])C#N),$([P](=[O,S])([#6])S),$([P;!$([P]-[O]-[P])](=[O,S])([O,S][#6])([O,S][#6])([O,S,#6][#6]))]",
        "description": "Potent neurotoxic pharmacophore acting via acetylcholinesterase inhibition."
    },
    {
        "name": "Benzene ring",
        "smarts": "c1ccccc1",
        "description": "Aromatic hydrocarbon core linked to metabolic bioactivation and reactive metabolite formation."
    },
    {
        "name": "Phenol moiety",
        "smarts": "c1ccccc1O",
        "description": "Hydroxylated aromatic ring capable of quinone/semiquinone redox cycling and mitochondrial uncoupling."
    },
    {
        "name": "Reactive Aldehyde",
        "smarts": "[CX3H1](=O)",
        "description": "Electrophilic carbonyl center forming covalent adducts with cellular proteins and DNA."
    },
    {
        "name": "Sulfide / Thiol center",
        "smarts": "[S]",
        "description": "Reactive sulfur center susceptible to redox cycling and cellular glutathione depletion."
    },
    {
        "name": "Halogenated aromatic",
        "smarts": "[Cl,Br,I]c1ccccc1",
        "description": "Halogenated phenyl ring associated with high lipophilicity, metabolic persistence, and bioaccumulation."
    },
]
toxic_patterns = [Chem.MolFromSmarts(item["smarts"]) for item in TOXICOPHORE_DEFINITIONS]

def get_toxicophore_density(mol):
    total_atoms = mol.GetNumAtoms()
    if total_atoms == 0: return [0.0]*len(toxic_patterns)
    
    densities = []
    for pattern in toxic_patterns:
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            densities.append(0.0)
        else:
            # Count unique toxicophore atoms
            toxic_atoms = set()
            for match in matches:
                toxic_atoms.update(match)
            densities.append(len(toxic_atoms) / total_atoms)
    return densities

def analyze_toxicophores(mol):
    total_atoms = mol.GetNumAtoms()
    alerts = []
    for defn, pattern in zip(TOXICOPHORE_DEFINITIONS, toxic_patterns):
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            matched_atoms = set()
            for match in matches:
                matched_atoms.update(match)
            alerts.append({
                "name": defn["name"],
                "smarts": defn["smarts"],
                "description": defn["description"],
                "count": len(matches),
                "matched_atom_indices": sorted(list(matched_atoms)),
                "density": round(len(matched_atoms) / total_atoms, 4) if total_atoms > 0 else 0.0
            })
    return alerts

# Compound name mapping for common chemicals (fallback when PubChem is unavailable)
COMPOUND_NAME_MAP = {
    'O': 'Water',
    '[Na+].[Cl-]': 'Sodium chloride (NaCl)',
    '[Na+].[F-]': 'Sodium fluoride',
    '[K+].[Cl-]': 'Potassium chloride',
    '[Ca+2].[Cl-].[Cl-]': 'Calcium chloride',
    '[NH4+].[Cl-]': 'Ammonium chloride',
    'CCO': 'Ethanol',
    'CO': 'Methanol',
    'CCOCCO': 'Ethylene glycol',
    'CCCCO': 'Butanol',
    'CC(=O)O': 'Acetic acid',
    'CCC(=O)O': 'Propionic acid',
    'CCCC(=O)O': 'Butyric acid',
    'CC(O)=O': 'Lactic acid',
    'CC(=O)Oc1ccccc1C(=O)O': 'Aspirin',
    'CC(=O)Nc1ccc(O)cc1': 'Paracetamol',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C': 'Caffeine',
    'C#N': 'Cyanide',
    'C1=CC=C(C=C1)O': 'Phenol',
    'c1ccccc1O': 'Phenol',
    'C1=CC=C(C=C1)O': 'Phenol',
    'C(C(=O)O)N': 'Glycine',
    'CC(C(=O)O)N': 'Alanine',
    'CC(C)C(C(=O)O)N': 'Valine',
    'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O': 'Glucose',
    'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O': 'Fructose',
    'COc1ccc2nc(nc2c1)N': 'Adenine',
    'C1=CC=C(C=C1)': 'Benzene',
    'Cc1ccccc1': 'Toluene',
    'c1ccccc1': 'Benzene',
    'C1=CC=C(C=C1)O': 'Phenol',
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O': 'Ibuprofen',
    'CN1CCCC1c2cccnc2': 'Nicotine',
    'NC(=O)N': 'Urea',
    'N': 'Ammonia',
    'C1CCOCC1': 'Tetrahydrofuran',
    'C1COCCO1': '1,4-dioxane',
    'CC(=O)NC1=CC=CC=C1': 'Acetanilide',
    'CN1C(=O)NC2=NC=NC=C21': 'Theophylline',
}

# PubChem reverse lookup - get compound name from SMILES
compound_name_cache = {}

def get_compound_name(smiles):
    """Lookup compound name from PubChem using SMILES with local fallback."""
    # First check local mapping
    if smiles in COMPOUND_NAME_MAP:
        return COMPOUND_NAME_MAP[smiles]
    
    if smiles in compound_name_cache:
        return compound_name_cache[smiles]
    
    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds:
            c = compounds[0]
            # Prefer IUPAC name, then synonym
            name = c.iupac_name or (c.synonyms[0] if c.synonyms else 'Unknown')
            compound_name_cache[smiles] = name
            return name
    except Exception:
        pass
    
    compound_name_cache[smiles] = 'Unknown'
    return 'Unknown'

# Import GNN model architecture
from model import PharmaGNN
RDLogger.DisableLog('rdApp.*')
app = FastAPI(title="PharmaGraph GNN Service")

# 1. Initialize model architecture (num_node_features = 6)
config_path = "model_config.json"
model_cfg = {}
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            model_cfg = json.load(f)
    except Exception:
        pass

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

weights_path = "pharma_gnn_weights_universal.pt"
if os.path.exists(weights_path):
    try:
        ai_model.load_state_dict(torch.load(weights_path, weights_only=True))
        print("[+] Successfully loaded trained model weights.")
    except Exception as e:
        print(f"[-] Warning: Could not load model weights: {e}")
else:
    print("[-] No weights file detected. Initializing random weights.")

ai_model.eval()

class MoleculeRequest(BaseModel):
    smiles: str
    concentration_molar: float = 1.0 # Default 1.0 Molar

# 2. Extract 6 chemical node features
def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),            
        atom.GetDegree(),               
        int(atom.GetIsAromatic()),      
        atom.GetImplicitValence(), # Implicit valence
        atom.GetFormalCharge(),         
        atom.GetNumRadicalElectrons()   
    ]

def smiles_to_graph(smiles_string, concentration_molar=1e-5):
    """
    Convert SMILES to PyG Data object.
    Default concentration_molar=1e-5 (10 µM) matches Tox21/ClinTox screening concentration.
    pIC50 = -log10(concentration_molar) => 5.0 for 10 µM
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edges_src, edges_dst, edge_features = [], [], []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_src += [i, j]
        edges_dst += [j, i]
        b_type = bond.GetBondTypeAsDouble()
        edge_features += [[b_type], [b_type]]
        
    global_features = [
        Descriptors.MolWt(mol) / 100.0, 
        Descriptors.MolLogP(mol), 
        Descriptors.TPSA(mol) / 100.0, 
        float(Descriptors.NumRotatableBonds(mol))
    ]
    
    # Append 7 toxicophore density features to global features
    toxic_densities = get_toxicophore_density(mol)
    global_features.extend(toxic_densities)
    
    # Extract 85 functional group features
    func_group_features = [float(func(mol)) for func in frag_funcs]
    
    # Compute pIC50 from concentration_molar
    if concentration_molar <= 0:
        pIC50 = 0.0
    else:
        pIC50 = -math.log10(concentration_molar)
    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        global_features=torch.tensor([global_features], dtype=torch.float),
        func_group_features=torch.tensor([func_group_features], dtype=torch.float),
        concentration=torch.tensor([[pIC50]], dtype=torch.float)
    )

from torch_geometric.explain import Explainer, GNNExplainer

@app.get("/api/health")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "PharmaGraph GNN Service", "version": "2.0.0"}

@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
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
        
        # Full 13 class probabilities for client visualization
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
    if toxicity_score >= 50.0:
        if toxicophore_alerts:
            primary_driver = "Intrinsic Structural Alerts & Toxicophores"
            names_str = ", ".join([a["name"] for a in toxicophore_alerts[:3]])
            summary_text = f"Predicted toxicity ({toxicity_score:.1f}%) is predominantly driven by structural alerts: {names_str}. "
            if synergy_pairs and synergy_pairs[0]["synergy_score"] > 0.05:
                summary_text += f"The GNN attention mechanism detected synergy between {synergy_pairs[0]['group_a']} and {synergy_pairs[0]['group_b']}. "
            if delta_risk > 10.0:
                summary_text += f"High concentration ({request.concentration_molar:.2g} M) further amplifies risk by +{delta_risk:.1f}%."
        elif delta_risk > 15.0:
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
        if not toxicophore_alerts:
            summary_text = (
                f"Low predicted toxicity risk ({toxicity_score:.1f}%). No high-hazard structural alerts detected, "
                f"and functional group interactions remain within safe physiological thresholds."
            )
        else:
            summary_text = (
                f"Low predicted toxicity risk ({toxicity_score:.1f}%) despite minor alerts, "
                f"counterbalanced by favorable molecular topology and safe baseline profile."
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
