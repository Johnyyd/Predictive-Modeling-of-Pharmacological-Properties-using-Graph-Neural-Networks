from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from rdkit import Chem

# Import mô hình GNN vừa tạo
from model import PharmaGNN

app = FastAPI(title="PharmaGraph GNN Service")

# Khởi tạo Bộ não AI (Với đầu vào là 3 đặc trưng nguyên tử chúng ta đã trích xuất)
# (Trong thực tế, bạn sẽ dùng lệnh torch.load() để tải file .pt đã được train)
ai_model = PharmaGNN(num_node_features=3, hidden_channels=64, num_classes=1)
ai_model.eval() # Bật chế độ suy luận (tắt Dropout/BatchNorm)

class MoleculeRequest(BaseModel):
    smiles: str

def get_atom_features(atom):
    return [atom.GetAtomicNum(), atom.GetDegree(), int(atom.GetIsAromatic())]

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
        
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float)
    )

@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
    graph = smiles_to_graph(request.smiles)
    if graph is None:
        raise HTTPException(status_code=400, detail="Chuỗi SMILES không hợp lệ")
    
    # --- GỌI AI ĐỂ SUY LUẬN ---
    with torch.no_grad(): # Tắt tính toán đạo hàm để chạy nhanh hơn
        prediction_tensor = ai_model(graph)
        toxicity_score = prediction_tensor.item() * 100 # Chuyển thành %
    
    # Trả về kết quả JSON cho ứng dụng Frontend (React/Flutter/C#)
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
    uvicorn.run(app, host="localhost", port=8000)