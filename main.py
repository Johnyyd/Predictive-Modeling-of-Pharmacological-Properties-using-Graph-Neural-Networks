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

# Automatically extract 85 RDKit functional group descriptors
frag_funcs = [func for name, func in Descriptors.descList if name.startswith('fr_')]

# Knowledge-based toxicophore alerts
toxic_smarts = [
    'C#N', # Cyanide
    'P(=O)(O)(O)', # Organophosphates
    'c1ccccc1', # Benzene ring
    'c1ccccc1O', # Phenol
    '[CX3H1](=O)', # Aldehyde (e.g., Formaldehyde)
    '[S]', # Sulfide (H2S, thiols)
    '[Cl,Br,I]c1ccccc1', # Halogenated aromatics (PCB, DDT)
]
toxic_patterns = [Chem.MolFromSmarts(sm) for sm in toxic_smarts]

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
ai_model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)

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

@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
    graph = smiles_to_graph(request.smiles)
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
        prediction_tensor = ai_model(
            x=graph.x, 
            edge_index=graph.edge_index, 
            edge_attr=graph.edge_attr, 
            batch=graph.batch, 
            global_features=graph.global_features, 
            func_group_features=graph.func_group_features,
            concentration=concentration_tensor
        )
        
        # Apply Sigmoid to convert raw logits to probabilities in [0, 1]
        probabilities = torch.sigmoid(prediction_tensor)
        
        # CT_TOX is class index 12 (ClinTox toxicity) - primary toxicity_risk
        # 13 classes: [NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, 
        #              NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53, CT_TOX]
        CT_TOX_IDX = 12
        ct_tox_prob = probabilities[0, CT_TOX_IDX].item()
        toxicity_score = ct_tox_prob * 100
        
        # Report most probable target class for reference
        max_prob, target_class = torch.max(probabilities, dim=1)
        target_class_idx = target_class.item()
        
        # Full 13 class probabilities for client visualization
        all_probs = probabilities[0].tolist()
        
    # Interpretability with GNNExplainer
    # Enable gradients temporarily for GNNExplainer backward pass
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
        "explanation": {
            "node_importance": node_importance
        },
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
