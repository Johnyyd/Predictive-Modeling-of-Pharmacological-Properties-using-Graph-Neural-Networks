"""
main.py - File Backend API chính của hệ thống PharmaGraph.
Nhiệm vụ: Cung cấp API FastAPI để nhận chuỗi SMILES từ giao diện (app.py),
chuyển đổi chuỗi đó thành dữ liệu Đồ thị (Graph), và đưa qua mô hình AI (PharmaGNN) để dự đoán độc tính.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import os

# Tự động lấy danh sách 85 hàm đếm nhóm chức từ RDKit
# frag_funcs sẽ chứa các hàm kiểm tra xem phân tử có bao nhiêu nhóm Ancol, nhóm Axit, v.v.
frag_funcs = [func for name, func in Descriptors.descList if name.startswith('fr_')]

# Danh sách cảnh báo độc tính tiên nghiệm (Knowledge-based Toxicophores)
# Đây là những cấu trúc hóa học được con người đúc kết là mang sẵn mầm mống độc hại.
toxic_smarts = [
    'C#N', # Cyanide (Cực độc)
    'P(=O)(O)(O)', # Organophosphates (Thuốc trừ sâu)
    'c1ccccc1', # Benzene ring (Vòng Benzen - Dễ gây ung thư nếu tích tụ)
    'c1ccccc1O', # Phenol (Chất độc ăn mòn)
    '[CX3H1](=O)', # Aldehyde (như Formaldehyde - Gây kích ứng)
    '[S]', # Sulfide (H2S, thiols - Có mùi thối, thường độc)
    '[Cl,Br,I]c1ccccc1', # Halogenated aromatics (Chất ô nhiễm hữu cơ khó phân hủy như PCB, DDT)
]
# Biên dịch các chuỗi SMARTS trên thành mẫu cấu trúc (patterns) để RDKit tìm kiếm nhanh hơn
toxic_patterns = [Chem.MolFromSmarts(sm) for sm in toxic_smarts]

def get_toxicophore_density(mol):
    """
    Hàm tính mật độ các gốc kịch độc trong phân tử.
    Trả về tỷ lệ % nguyên tử thuộc về các nhóm độc này so với tổng số nguyên tử.
    """
    total_atoms = mol.GetNumAtoms()
    # Nếu phân tử rỗng, trả về mảng 0
    if total_atoms == 0: return [0.0]*len(toxic_patterns)
    
    densities = []
    for pattern in toxic_patterns:
        # Tìm xem phân tử có chứa gốc độc này không
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            densities.append(0.0)
        else:
            # Đếm số nguyên tử độc duy nhất (dùng set để loại bỏ trùng lặp nếu có 2 nhóm độc dính nhau)
            toxic_atoms = set()
            for match in matches:
                toxic_atoms.update(match)
            # Tính tỷ lệ: (Số nguyên tử độc) / (Tổng số nguyên tử)
            densities.append(len(toxic_atoms) / total_atoms)
    return densities

# Import mô hình AI chính từ file model.py
from model import PharmaGNN

# Tắt các cảnh báo (warnings) lặt vặt của RDKit trên Terminal để giao diện console sạch hơn
RDLogger.DisableLog('rdApp.*')

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="PharmaGraph GNN Service")

# 1. KHỞI TẠO MÔ HÌNH VÀ NẠP TRỌNG SỐ (KIẾN THỨC)
# num_node_features=6: Mỗi nguyên tử có 6 chỉ số (nguyên tử khối, hóa trị, điện tích...)
# num_classes=13: Dự đoán 13 loại độc tính cùng lúc
ai_model = PharmaGNN(num_node_features=6, hidden_channels=32, num_classes=13)

weights_path = "pharma_gnn_weights_universal.pt"
if os.path.exists(weights_path):
    try:
        # Tải kiến thức đã học từ file .pt vào mô hình
        ai_model.load_state_dict(torch.load(weights_path, weights_only=True))
        print("[+] Da nap thanh cong trong so huan luyen!")
    except Exception as e:
        print(f"[-] Canh bao: Khong the nap trong so (co the do sai lech kien truc). Chi tiet: {e}")
else:
    print("[-] Chua co file trong so, AI dang dung nao ngau nhien.")

# Chuyển mô hình sang chế độ Evaluation (Đánh giá). 
# Tắt Dropout và BatchNorm để cho kết quả dự đoán nhất quán, không bị thay đổi ngẫu nhiên.
ai_model.eval()

class MoleculeRequest(BaseModel):
    """Khai báo cấu trúc dữ liệu đầu vào cho API (nhận chuỗi SMILES)"""
    smiles: str

# 2. HÀM BÓC TÁCH 6 ĐẶC TRƯNG CỦA TỪNG NGUYÊN TỬ (NODE FEATURES)
def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),            # Số hiệu nguyên tử (Oxy=8, Carbon=6, Nitrogen=7)
        atom.GetDegree(),               # Bậc của nguyên tử (Số liên kết xung quanh nó)
        int(atom.GetIsAromatic()),      # Có nằm trong vòng thơm (Aromatic) hay không (0 hoặc 1)
        atom.GetValence(Chem.ValenceType.IMPLICIT), # Hóa trị ẩn (Số lượng liên kết với Hydro bị ẩn đi)
        atom.GetFormalCharge(),         # Điện tích chính thức (ion âm hay dương)
        atom.GetNumRadicalElectrons()   # Số electron gốc tự do (rất quan trọng vì gốc tự do thường gây độc)
    ]

# 3. HÀM CHUYỂN ĐỔI CHUỖI SMILES THÀNH MA TRẬN ĐỒ THỊ (GRAPH)
def smiles_to_graph(smiles_string):
    # Dựng phân tử từ chuỗi SMILES
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    
    # BẮT BUỘC: Thêm các nguyên tử Hydro vào cấu trúc 3D (vì SMILES thường ẩn Hydro đi).
    # Hydro quyết định liên kết hydro - sống còn để biết thuốc có tương tác với protein người hay không.
    mol = Chem.AddHs(mol)
    
    # Quét từng nguyên tử để lấy Node Features
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    
    # Chuẩn bị danh sách lưu thông tin liên kết (Cạnh - Edges)
    edges_src, edges_dst, edge_features = [], [], []
    
    for bond in mol.GetBonds():
        # Lấy ID của 2 nguyên tử được nối với nhau
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Mạng GNN của PyTorch yêu cầu đồ thị vô hướng phải được khai báo bằng 2 cạnh có hướng ngược nhau
        edges_src += [i, j]
        edges_dst += [j, i]
        
        # Lấy loại liên kết (Đơn=1.0, Đôi=2.0, Thơm=1.5...)
        b_type = bond.GetBondTypeAsDouble()
        edge_features += [[b_type], [b_type]]
        
    # Tính toán 4 Đặc trưng Toàn cục (Global Features - Các chỉ số vật lý của toàn bộ phân tử)
    global_features = [
        Descriptors.MolWt(mol) / 100.0,             # Khối lượng phân tử (chia 100 để thu nhỏ giá trị)
        Descriptors.MolLogP(mol),                   # Độ ưa mỡ (Lipophilicity - Thuốc quá ưa mỡ dễ tích tụ trong gan)
        Descriptors.TPSA(mol) / 100.0,              # Diện tích bề mặt phân cực (Quyết định khả năng qua hàng rào máu não)
        float(Descriptors.NumRotatableBonds(mol))   # Số liên kết có thể xoay (Độ mềm dẻo của thuốc)
    ]
    
    # Cộng gộp thêm 7 đặc trưng Mật độ Độc tính (Toxicophores) vào Global Features
    toxic_densities = get_toxicophore_density(mol)
    global_features.extend(toxic_densities)
    
    # Trích xuất 85 đặc trưng nhóm chức (Functional Groups) để đưa vào mạng Attention
    func_group_features = [float(func(mol)) for func in frag_funcs]
        
    # Đóng gói tất cả thành đối tượng Data chuẩn của thư viện PyTorch Geometric
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),                      # Đặc trưng Đỉnh (Node)
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),     # Bản đồ kết nối
        edge_attr=torch.tensor(edge_features, dtype=torch.float),              # Đặc trưng Cạnh (Edge)
        global_features=torch.tensor([global_features], dtype=torch.float),    # Đặc trưng Toàn cục (Vật lý + Kịch độc)
        func_group_features=torch.tensor([func_group_features], dtype=torch.float) # Đặc trưng 85 Nhóm chức
    )

# 4. ĐIỂM KẾT NỐI API (ENDPOINT)
@app.post("/api/predict")
async def predict_molecule(request: MoleculeRequest):
    """Nhận yêu cầu dự đoán từ web, chạy qua AI và trả về kết quả"""
    
    # Chuyển đổi chuỗi nhận được sang Đồ thị
    graph = smiles_to_graph(request.smiles)
    if graph is None:
        raise HTTPException(status_code=400, detail="Chuỗi SMILES không hợp lệ")
    
    # Quá trình suy luận (Inference) không cần tính đạo hàm (no_grad) để tiết kiệm RAM và tăng tốc
    with torch.no_grad():
        # Thêm batch size giả định = 0 (Bắt buộc phải có vì mạng GNN dùng Pooling theo từng lô - batch)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        
        # Đưa vào "não" AI để tính toán
        prediction_tensor = ai_model(graph)
        
        # Lớp cuối cùng của AI trả về Logits (số thô). Dùng Sigmoid để ép về khoảng % (từ 0.0 đến 1.0)
        probabilities = torch.sigmoid(prediction_tensor)
        
        # Vì AI dự đoán cùng lúc 13 loại bệnh/độc tính, ta lấy điểm số rủi ro CAO NHẤT trong 13 loại đó
        # làm điểm rủi ro tổng thể để hiển thị cho người dùng.
        toxicity_score = torch.max(probabilities).item() * 100 
    
    return {
        "smiles": request.smiles,
        "graph_info": {
            "atoms_count": graph.num_nodes,
            "bonds_count": graph.num_edges // 2 # Chia 2 vì lúc tạo chúng ta nhân đôi cạnh 2 chiều
        },
        "predictions": {
            "toxicity_risk": f"{toxicity_score:.2f}%"
        },
        "status": "Success"
    }

# Khởi chạy server FastAPI trên cổng 7077
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7077)