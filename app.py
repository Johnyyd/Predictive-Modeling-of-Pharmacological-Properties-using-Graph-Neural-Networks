import streamlit as st
import requests
import pubchempy as pcp  # Thư viện tra cứu hóa học tự động
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, AllChem, Fragments, Descriptors
from collections import Counter
import py3Dmol
from stmol import showmol
# Cấu hình trang Dashboard
st.set_page_config(page_title="PharmaGraph AI", page_icon="🧬", layout="wide")

st.title("🧬 Hệ thống Phân tích Dược phẩm PharmaGraph")
st.markdown("Dự đoán rủi ro độc tính phân tử bằng GNN kết hợp **Cơ chế Attention học Tương tác Nhóm chức**, dành cho Chuyên viên và Nhà nghiên cứu.")

API_URL = "http://localhost:8000/api/predict"

# --- GIẢI PHÁP 1: DANH SÁCH MẪU CÓ SẴN (PRESETS) ---
st.markdown("### 🗂️ Cách 1: Chọn nhanh dược chất từ danh sách mẫu")
presets = {
    "Chưa chọn...": "",
    "Aspirin (Thuốc giảm đau quen thuộc)": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol (Thuốc hạ sốt Efferalgan/Hapacol)": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine (Chất kích thích trong Cà phê)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol (Cồn y tế / Rượu bia)": "CCO",
    "Nicotine (Chất gây nghiện trong Thuốc lá)": "CN1CCCC1c2cccnc2",
    "Cyanide (Chất kịch độc nổi tiếng)": "C#N",
    "Phenol (Chất độc công nghiệp gây bỏng)": "C1=CC=C(C=C1)O"
}
selected_preset = st.selectbox("Bấm vào đây để chọn nhanh một chất để thử nghiệm:", list(presets.keys()))

# --- GIẢI PHÁP 2: TRA CỨU TỰ ĐỘNG BẰNG TÊN (SEARCH ENGINE) ---
st.markdown("### 🔍 Cách 2: Tìm kiếm bằng Tên thông thường (Tiếng Anh)")
search_name = st.text_input("Nhập tên thuốc hoặc hóa chất bằng tiếng Anh:", placeholder="Ví dụ: Ibuprofen, Penicillin, Nicotine, Water...")

# Biến trung gian để chốt chuỗi SMILES cuối cùng đẩy vào AI
final_smiles = ""

# Ưu tiên lấy từ ô tìm kiếm trước, nếu trống thì lấy từ danh sách mẫu
if search_name:
    with st.spinner(f"🔍 Đang tìm cấu trúc của '{search_name}' trên kho dữ liệu quốc tế PubChem..."):
        try:
            # Tìm kiếm bằng Tên trước
            compounds = pcp.get_compounds(search_name, 'name')
            if not compounds:
                # Nếu không thấy tên, tìm bằng Công thức hóa học
                compounds = pcp.get_compounds(search_name, 'formula')
                
            if compounds:
                # Lấy ra chuỗi SMILES tiêu chuẩn của chất tìm thấy
                final_smiles = compounds[0].isomeric_smiles
                st.info(f"🧬 Đã tìm thấy cấu trúc SMILES phù hợp: `{final_smiles}` (Khớp với: {compounds[0].synonyms[0] if compounds[0].synonyms else 'Chất vô danh'})")
            else:
                st.error(f"❌ Không tìm thấy chất nào khớp với Tên hoặc Công thức '{search_name}'. Vui lòng kiểm tra lại!")
        except Exception as e:
            error_msg = str(e)
            if "PUGREST.BadRequest" in error_msg:
                st.error(f"❌ '{search_name}' có thể là một danh mục quá rộng (như Thuốc trừ sâu, Nhựa...) hoặc không phải là tên một hóa chất cụ thể. Vui lòng nhập tên một chất chính xác hơn (ví dụ: Glyphosate thay vì Pesticide).")
            else:
                st.error(f"Lỗi kết nối mạng hoặc lỗi từ PubChem: {e}")
elif selected_preset and selected_preset != "Chưa chọn...":
    final_smiles = presets[selected_preset]

# --- KHỐI XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
if final_smiles:
    st.markdown("---")
    if st.button("🚀 Chạy phân tích AI bằng mạng GNN"):
        with st.spinner("Mạng GNN đang bóc tách đồ thị phân tử và tính toán..."):
            try:
                # 1. Phân tích chi tiết và vẽ 3D bằng RDKit & py3Dmol
                mol = Chem.MolFromSmiles(final_smiles)
                if mol is not None:
                    # Trích xuất thông tin
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    mw = rdMolDescriptors.CalcExactMolWt(mol)
                    
                    atoms_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    atoms_count = dict(Counter(atoms_list))
                    atoms_str = ", ".join([f"{k}: {v}" for k, v in atoms_count.items()])
                    
                    func_groups = []
                    for name, func in Descriptors.descList:
                        if name.startswith('fr_'):
                            count = func(mol)
                            if count > 0:
                                friendly_name = name.replace('fr_', '')
                                func_groups.append(f"{friendly_name} ({count})")
                    if not func_groups: func_groups.append("Không có nhóm chức đặc biệt")
                    
                    # Sinh tọa độ 3D
                    mol_3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                    mol_block = Chem.MolToMolBlock(mol_3d)
                    
                    # Chia màn hình làm 2 cột
                    left_col, right_col = st.columns([1, 1.5])
                    
                    with left_col:
                        st.markdown("### 🔬 Cấu trúc 3D Interactive")
                        viewer = py3Dmol.view(width=400, height=350)
                        viewer.addModel(mol_block, "mol")
                        viewer.setStyle({'stick': {}, 'sphere': {'radius': 0.4}})
                        viewer.setBackgroundColor('#0e1117') # Khớp với theme tối
                        viewer.zoomTo()
                        showmol(viewer, height=350, width=400)
                        
                        st.markdown(f"**Công thức:** {formula}")
                        st.markdown(f"**Khối lượng:** {mw:.2f} g/mol")
                        st.markdown(f"**Cấu tạo:** {atoms_str}")
                        st.markdown(f"**Nhóm chức:** {', '.join(func_groups)}")
                
                # 2. Gọi API để AI GNN dự đoán
                response = requests.post(API_URL, json={"smiles": final_smiles})
                if response.status_code == 200:
                    data = response.json()
                    
                    with right_col:
                        st.success("✅ Phân tích AI hoàn tất!")
                        st.markdown("### 📊 Chỉ số Đồ thị & Dự đoán")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(label="Số lượng Nguyên tử (Nodes)", value=data["graph_info"]["atoms_count"])
                        with col_b:
                            st.metric(label="Số lượng Liên kết (Edges)", value=data["graph_info"]["bonds_count"])
                        
                        st.markdown("<br>", unsafe_allow_html=True) # Tạo khoảng trống
                        
                        toxicity_str = data["predictions"]["toxicity_risk"]
                        toxicity_float = float(toxicity_str.replace("%", ""))
                        
                        # ... (Bên trong khối hiển thị kết quả của app.py) ...
                        if toxicity_float >= 80.0:
                            st.metric(label="Rủi ro Độc tính 🛑", value=toxicity_str, delta="KỊCH ĐỘC (Nguy cơ cực cao)", delta_color="inverse")
                            st.error("Cảnh báo: Hợp chất này có mức rủi ro sinh học cực cao. Phát hiện cấu trúc đặc biệt nguy hiểm.")
                        elif toxicity_float >= 50.0:
                            st.metric(label="Rủi ro Độc tính ⚠️", value=toxicity_str, delta="Cần lưu ý", delta_color="off")
                            st.warning("Cấu trúc chứa liên kết hoặc đặc trưng có thể gây độc. Cần đánh giá thêm bằng lâm sàng.")
                        else:
                            st.metric(label="Rủi ro Độc tính ✅", value=toxicity_str, delta="An toàn", delta_color="normal")
                            st.success("Cấu trúc ổn định, không phát hiện rủi ro nghiêm trọng theo cơ sở dữ liệu.")
                            
                        st.progress(int(toxicity_float))
                else:
                    st.error(f"Lỗi từ máy chủ AI: {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến Lõi AI FastAPI. Chi tiết lỗi: {e}")