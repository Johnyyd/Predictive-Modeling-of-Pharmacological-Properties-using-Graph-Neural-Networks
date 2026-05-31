import streamlit as st
import requests
import pubchempy as pcp  # Thư viện tra cứu hóa học tự động
from rdkit import Chem
from rdkit.Chem import Draw

# Cấu hình trang Dashboard
st.set_page_config(page_title="PharmaGraph AI", page_icon="🧬", layout="wide")

st.title("🧬 Hệ thống Phân tích Dược phẩm PharmaGraph")
st.markdown("Dự đoán rủi ro độc tính phân tử bằng GNN dành cho Chuyên viên và Nhà nghiên cứu.")

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
            # Gọi API của PubChem để tìm kiếm phân tử qua tên gọi
            compounds = pcp.get_compounds(search_name, 'name')
            if compounds:
                # Lấy ra chuỗi SMILES tiêu chuẩn của chất tìm thấy
                final_smiles = compounds[0].isomeric_smiles
                st.info(f"🧬 Đã tìm thấy cấu trúc SMILES phù hợp: `{final_smiles}`")
            else:
                st.error(f"❌ Không tìm thấy chất nào có tên '{search_name}' trên hệ thống PubChem. Vui lòng kiểm tra lại chính tả!")
        except Exception as e:
            st.error(f"Lỗi kết nối mạng khi tra cứu: {e}")
elif selected_preset and selected_preset != "Chưa chọn...":
    final_smiles = presets[selected_preset]

# --- KHỐI XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---
if final_smiles:
    st.markdown("---")
    if st.button("🚀 Chạy phân tích AI bằng mạng GNN"):
        with st.spinner("Mạng GNN đang bóc tách đồ thị phân tử và tính toán..."):
            try:
                # 1. Vẽ hình cấu trúc Phân tử ngay lập tức bằng RDKit
                mol = Chem.MolFromSmiles(final_smiles)
                if mol is not None:
                    # Tạo ảnh 2D của phân tử
                    img = Draw.MolToImage(mol, size=(400, 400), fitImage=True)
                    
                    # Chia màn hình làm 2 cột: 1 bên hình ảnh, 1 bên số liệu AI
                    left_col, right_col = st.columns([1, 2])
                    
                    with left_col:
                        st.markdown("### 🔬 Cấu trúc Hóa học")
                        st.image(img, use_container_width=True)
                
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
                        
                        if toxicity_float >= 50.0:
                            st.metric(label="Rủi ro Độc tính ⚠️", value=toxicity_str, delta="Cảnh báo: Nguy hiểm", delta_color="inverse")
                            st.error("Cấu trúc đồ thị chứa các mẫu hình liên kết có rủi ro độc tính cao theo FDA.")
                        else:
                            st.metric(label="Rủi ro Độc tính ✅", value=toxicity_str, delta="An toàn", delta_color="normal")
                            st.success("Cấu trúc đồ thị ổn định, chưa phát hiện rủi ro nghiêm trọng.")
                            
                        st.progress(int(toxicity_float))
                else:
                    st.error(f"Lỗi từ máy chủ AI: {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến Lõi AI FastAPI. Chi tiết lỗi: {e}")