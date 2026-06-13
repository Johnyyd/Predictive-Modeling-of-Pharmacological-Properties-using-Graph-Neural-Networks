from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import os

# Tự động lấy danh sách 85 hàm đếm nhóm chức từ RDKit
frag_funcs = [func for name, func in Descriptors.descList if name.startswith('fr_')]

# Danh sách cảnh báo độc tính tiên nghiệm (Knowledge-based Toxicophores)
toxic_smarts = [
    'C#N', # Cyanide
    'P(=O)(O)(O)', # Organophosphates
    'c1ccccc1', # Benzene ring
    'c1ccccc1O', # Phenol
    '[CX3H1](=O)', # Aldehyde (như Formaldehyde)
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
            # Đếm số nguyên tử độc duy nhất
            toxic_atoms = set()
            for match in matches:
                toxic_atoms.update(match)
            densities.append(len(toxic_atoms) / total_atoms)
    return densities

# Import mô hình GNN
from model import PharmaGNN
RDLogger.DisableLog('rdApp.*')
app = FastAPI(title="PharmaGraph GNN Service")

# 1. KHỞI TẠO NÃO MỚI (Lưu ý: num_node_features đã tăng lên 6)
ai_model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)

weights_path = "pharma_gnn_weights_universal.pt"
if os.path.exists(weights_path):
    try:
        ai_model.load_state_dict(torch.load(weights_path, weights_only=True))
        print("[+] Da nap thanh cong trong so huan luyen!")
    except Exception as e:
        print(f"[-] Canh bao: Khong the nap trong so (co the do sai lech kien truc). Chi tiet: {e}")
else:
    print("[-] Chua co file trong so, AI dang dung nao ngau nhien.")

ai_model.eval()

class MoleculeRequest(BaseModel):
    smiles: str
    concentration_molar: float = 1.0 # Default 1.0 Molar

# 2. HÀM BÓC TÁCH 6 ĐẶC TRƯNG HÓA HỌC (Đã nâng cấp)
def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),            
        atom.GetDegree(),               
        int(atom.GetIsAromatic()),      
        atom.GetImplicitValence(), # Hóa trị ẩn
        atom.GetFormalCharge(),         
        atom.GetNumRadicalElectrons()   
    ]

def smiles_to_graph(smiles_string):
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
    
    # Cộng gộp 7 đặc trưng Mật độ Độc tính vào Global Features
    toxic_densities = get_toxicophore_density(mol)
    global_features.extend(toxic_densities)
    
    # Trích xuất 85 đặc trưng nhóm chức (Functional Groups)
    func_group_features = [float(func(mol)) for func in frag_funcs]
        
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        global_features=torch.tensor([global_features], dtype=torch.float),
        func_group_features=torch.tensor([func_group_features], dtype=torch.float)
    )

from torch_geometric.explain import Explainer, GNNExplainer
import math

@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
    graph = smiles_to_graph(request.smiles)
    if graph is None:
        raise HTTPException(status_code=400, detail="Chuỗi SMILES không hợp lệ")
        
    # Tính pIC50 từ nồng độ (Molar)
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
        
        # Áp dụng Sigmoid để đưa raw logits về khoảng [0, 1]
        probabilities = torch.sigmoid(prediction_tensor)
        
        # Lấy giá trị độc tính cao nhất trong 13 bài test (13 classes bao gồm cả ClinTox)
        max_prob, target_class = torch.max(probabilities, dim=1)
        toxicity_score = max_prob.item() * 100 
        target_class_idx = target_class.item()
        
    # Giải thích bằng GNNExplainer
    # Bật gradient cho các features tạm thời (vì GNNExplainer cần backward pass)
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
    
    # Lấy trọng số các node (nguyên tử)
    if explanation.node_mask is not None:
        node_importance = explanation.node_mask.mean(dim=1).tolist()
    else:
        node_importance = [0.0] * graph.num_nodes
    
    return {
        "smiles": request.smiles,
        "concentration_molar": request.concentration_molar,
        "pIC50": pIC50,
        "graph_info": {
            "atoms_count": graph.num_nodes,
            "bonds_count": graph.num_edges // 2
        },
        "predictions": {
            "toxicity_risk": f"{toxicity_score:.2f}%",
            "target_class": target_class_idx
        },
        "explanation": {
            "node_importance": node_importance
        },
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)