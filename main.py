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

# 2. HÀM BÓC TÁCH 6 ĐẶC TRƯNG HÓA HỌC (Đã nâng cấp)
def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),            
        atom.GetDegree(),               
        int(atom.GetIsAromatic()),      
        atom.GetValence(Chem.ValenceType.IMPLICIT), # SỬA DÒNG NÀY (Hóa trị ẩn)
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
    
    # Trích xuất 85 đặc trưng nhóm chức (Functional Groups)
    func_group_features = [float(func(mol)) for func in frag_funcs]
        
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        global_features=torch.tensor([global_features], dtype=torch.float),
        func_group_features=torch.tensor([func_group_features], dtype=torch.float)
    )

@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
    graph = smiles_to_graph(request.smiles)
    if graph is None:
        raise HTTPException(status_code=400, detail="Chuỗi SMILES không hợp lệ")
    
    with torch.no_grad():
        # Thêm batch size giả định = 0 (vì mạng GAT sử dụng Pooling cần có batch)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        prediction_tensor = ai_model(graph)
        
        # Áp dụng Sigmoid để đưa raw logits về khoảng [0, 1]
        probabilities = torch.sigmoid(prediction_tensor)
        
        # Lấy giá trị độc tính cao nhất trong 13 bài test (13 classes bao gồm cả ClinTox)
        toxicity_score = torch.max(probabilities).item() * 100 
    
    return {
        "smiles": request.smiles,
        "graph_info": {
            "atoms_count": graph.num_nodes,
            "bonds_count": graph.num_edges // 2
        },
        "predictions": {
            "toxicity_risk": f"{toxicity_score:.2f}%"
        },
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)