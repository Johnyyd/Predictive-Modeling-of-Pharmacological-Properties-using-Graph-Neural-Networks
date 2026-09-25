import streamlit as st
import requests
import pubchempy as pcp  # Thư viện tra cứu hóa học tự động
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, AllChem, Fragments, Descriptors
from collections import Counter
import py3Dmol
from stmol import showmol
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    st_ketcher = None
# Cấu hình trang Dashboard
st.set_page_config(page_title="PharmaGraph AI", page_icon="🧬", layout="wide")

st.title("🧬 Hệ thống Phân tích Dược phẩm PharmaGraph")
st.markdown("Dự đoán rủi ro độc tính phân tử bằng GNN kết hợp **Cơ chế Attention học Tương tác Nhóm chức**, dành cho Chuyên viên và Nhà nghiên cứu.")

import os
API_URL = os.environ.get("API_URL", "http://localhost:1234/api/predict")

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

# Local cache for common compounds to avoid PubChem failures
COMMON_COMPOUNDS = {
    'water': 'O',
    'sugar': 'CC(=O)O',  # glucose - will be overridden
    'glucose': 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O',
    'fructose': 'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O',
    'sucrose': 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O[C@H]2O[C@@H](O)[C@H](O)[C@@H](O)[C@H]2O',
    'salt': '[Na+].[Cl-]',
    'sodium chloride': '[Na+].[Cl-]',
    'nacl': '[Na+].[Cl-]',
    'ethanol': 'CCO',
    'alcohol': 'CCO',
    'methanol': 'CO',
    'acetic acid': 'CC(=O)O',
    'vinegar': 'CC(=O)O',
    'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'paracetamol': 'CC(=O)Nc1ccc(O)cc1',
    'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'nicotine': 'CN1CCCC1c2cccnc2',
    'cyanide': 'C#N',
    'phenol': 'C1=CC=C(C=C1)O',
    'benzene': 'c1ccccc1',
    'toluene': 'Cc1ccccc1',
    'ammonia': 'N',
    'urea': 'NC(=O)N',
    'glycine': 'C(C(=O)O)N',
    'alanine': 'CC(C(=O)O)N',
    'valine': 'CC(C)C(C(=O)O)N',
}

# --- GIẢI PHÁP 3: NHẬP TRỰC TIẾP CHUỖI SMILES ---
st.markdown("### 🧪 Cách 3: Nhập trực tiếp cấu trúc phân tử (Chuỗi SMILES)")
raw_smiles = st.text_input("Dành cho hóa chất mới tự tổng hợp hoặc không có trong cơ sở dữ liệu:", placeholder="Ví dụ: C1=CC=C(C=C1)O")

# --- GIẢI PHÁP 4: VẼ TRỰC TIẾP ---
drawn_smiles = ""
if st_ketcher is not None:
    st.markdown("### 🖌️ Cách 4: Vẽ phân tử trực tiếp (Dành cho người không chuyên)")
    st.markdown("Sử dụng công cụ dưới đây để vẽ. Hệ thống sẽ tự động dịch hình vẽ thành mã hóa học. Nhấn **Apply** sau khi vẽ xong.")
    drawn_smiles = st_ketcher(key="ketcher_editor")

# Biến trung gian để chốt chuỗi SMILES cuối cùng đẩy vào AI
final_smiles = ""

# Ưu tiên: Vẽ tay -> SMILES nhập tay -> Tìm kiếm PubChem -> Danh sách mẫu
if drawn_smiles and drawn_smiles != "":
    final_smiles = drawn_smiles.strip()
    st.info(f"🎨 Đang sử dụng hình vẽ phân tử (SMILES: `{final_smiles}`)")
elif raw_smiles:
    final_smiles = raw_smiles.strip()
    st.info(f"🧬 Đang sử dụng cấu trúc SMILES nhập tay: `{final_smiles}`")
elif search_name:
    # Normalize search name
    search_name_norm = search_name.strip().lower()
    
    # First try local cache for common compounds
    if search_name_norm in COMMON_COMPOUNDS:
        final_smiles = COMMON_COMPOUNDS[search_name_norm]
        st.info(f"🧬 Đã tìm thấy trong cache nội bộ: `{final_smiles}` (Tên: {search_name})")
    else:
        # Fallback to PubChem
        with st.spinner(f"🔍 Đang tìm cấu trúc của '{search_name}' trên kho dữ liệu PubChem..."):
            try:
                # Try exact name match first
                compounds = pcp.get_compounds(search_name, 'name')
                if not compounds:
                    # Try lowercase
                    compounds = pcp.get_compounds(search_name.lower(), 'name')
                if not compounds:
                    # Try formula search
                    compounds = pcp.get_compounds(search_name, 'formula')
                
                if compounds:
                    final_smiles = compounds[0].isomeric_smiles
                    compound_display_name = compounds[0].synonyms[0] if compounds[0].synonyms else compounds[0].iupac_name or search_name
                    st.info(f"🧬 Đã tìm thấy trên PubChem: `{final_smiles}` (Tên: {compound_display_name})")
                else:
                    st.error(f"❌ Không tìm thấy '{search_name}' trong cache hoặc PubChem. Vui lòng thử tên khác hoặc dùng Cách 3 nhập SMILES trực tiếp!")
            except Exception as e:
                error_msg = str(e).lower()
                if "bad request" in error_msg or "expecting value" in error_msg or "json decode" in error_msg:
                    # PubChem error - suggest alternatives
                    st.error(f"⚠️ PubChem trả về lỗi cho '{search_name}'. Thử với tên tiếng Anh chuẩn hoặc dùng cache nội bộ. Gợi ý: water, ethanol, caffeine, aspirin, paracetamol")
                elif "timeout" in error_msg or "connection" in error_msg:
                    st.error(f"⚠️ Mất kết nối PubChem. Thử lại hoặc dùng cache nội bộ.")
                else:
                    st.error(f"⚠️ Lỗi tìm kiếm: {e}. Thử lại hoặc dùng SMILES trực tiếp.")
elif selected_preset and selected_preset != "Chưa chọn...":
    final_smiles = presets[selected_preset]

# --- KHỐI XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
if final_smiles:
    st.markdown("---")
    concentration_input = st.number_input("🧪 Nhập Nồng độ / Liều lượng (Molar) - Để 1.0 M nếu không có dữ liệu:", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%f")
    if st.button("🚀 Chạy phân tích AI bằng mạng GNN"):
        with st.spinner("Mạng GNN đang bóc tách đồ thị phân tử và tính toán..."):
            try:
                # 1. Gọi API để AI GNN dự đoán
                response = requests.post(API_URL, json={"smiles": final_smiles, "concentration_molar": concentration_input})
                if response.status_code == 200:
                    data = response.json()
                    
                    # 2. Phân tích chi tiết và vẽ 3D bằng RDKit & py3Dmol
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
                        
                        # Truy xuất ngược tên Hóa chất từ SMILES bằng PubChem
                        compound_name = "Chất vô danh (Không có trong dữ liệu quốc tế)"
                        try:
                            c = pcp.get_compounds(final_smiles, 'smiles')
                            if c:
                                if c[0].synonyms:
                                    compound_name = c[0].synonyms[0]
                                elif c[0].iupac_name:
                                    compound_name = c[0].iupac_name
                        except Exception:
                            pass
                        
                        # Chia màn hình làm 2 cột
                        left_col, right_col = st.columns([1, 1.5])
                        
                        with left_col:
                            st.markdown("### 🔬 Cấu trúc 3D Interactive")
                            viewer = py3Dmol.view(width=400, height=350)
                            viewer.addModel(mol_block, "mol")
                            
                            # Tô màu nguyên tử độc tính (khoanh vùng đỏ)
                            node_importance = data.get("explanation", {}).get("node_importance", [])
                            risk_str = data.get("predictions", {}).get("toxicity_risk", "0%")
                            try:
                                risk_value = float(risk_str.replace('%', ''))
                            except:
                                risk_value = 0.0

                            if node_importance and risk_value > 50.0:
                                max_imp = max(node_importance) if max(node_importance) > 0 else 1.0
                                
                                # Thiết lập hiển thị liên kết (stick) cho toàn bộ phân tử trước
                                viewer.setStyle({'stick': {'radius': 0.15}})
                                
                                for i, imp in enumerate(node_importance):
                                    norm_imp = imp / max_imp
                                    # Chỉ highlight nguyên tử có trọng số > 85% so với max (chỉ nguyên tử độc nhất)
                                    if norm_imp > 0.85:
                                        viewer.addStyle({'serial': i+1}, {'sphere': {'color': 'red', 'radius': 0.5}})
                                    else:
                                        viewer.addStyle({'serial': i+1}, {'sphere': {'color': 'white', 'radius': 0.3}})
                            else:
                                # Nếu không độc, vẽ toàn bộ màu mặc định
                                viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.3}})
                            
                            viewer.setBackgroundColor('#0e1117') # Khớp với theme tối
                            viewer.zoomTo()
                            showmol(viewer, height=350, width=400)
                            
                            st.markdown(f"**Tên định danh:** {compound_name}")
                            st.markdown(f"**Công thức:** {formula}")
                            st.markdown(f"**Khối lượng:** {mw:.2f} g/mol")
                            st.markdown(f"**Cấu tạo:** {atoms_str}")
                            st.markdown(f"**Nhóm chức:** {', '.join(func_groups)}")
                    
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
