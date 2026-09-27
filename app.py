import streamlit as st
import requests
import pubchempy as pcp  # Automated chemical lookup library
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, AllChem, Fragments, Descriptors
from collections import Counter
import py3Dmol
from stmol import showmol
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    st_ketcher = None

# Input method constants
METHOD_PRESET = "🗂️ Method 1: Curated Presets"
METHOD_SEARCH = "🔍 Method 2: Search by Name"
METHOD_SMILES = "🧪 Method 3: Direct SMILES"
METHOD_DRAW = "🖌️ Method 4: 2D Sketcher"

available_methods = [METHOD_PRESET, METHOD_SEARCH, METHOD_SMILES]
if st_ketcher is not None:
    available_methods.append(METHOD_DRAW)

# Initialize session state for input management and data caching
if 'input_method' not in st.session_state:
    st.session_state.input_method = METHOD_PRESET
if 'final_smiles' not in st.session_state:
    st.session_state.final_smiles = ""
if 'search_name_input' not in st.session_state:
    st.session_state.search_name_input = ""
if 'raw_smiles_input' not in st.session_state:
    st.session_state.raw_smiles_input = ""
if 'selected_preset_input' not in st.session_state:
    st.session_state.selected_preset_input = "None selected..."
if 'drawn_smiles_input' not in st.session_state:
    st.session_state.drawn_smiles_input = ""
if 'ketcher_key_idx' not in st.session_state:
    st.session_state.ketcher_key_idx = 0
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'last_analyzed_smiles' not in st.session_state:
    st.session_state.last_analyzed_smiles = ""

def reset_all_inputs():
    """Automatically clear all previous inputs and cached analysis results."""
    st.session_state.selected_preset_input = "None selected..."
    st.session_state.search_name_input = ""
    st.session_state.raw_smiles_input = ""
    st.session_state.drawn_smiles_input = ""
    st.session_state.ketcher_key_idx = st.session_state.get('ketcher_key_idx', 0) + 1
    st.session_state.final_smiles = ""
    st.session_state.current_analysis = None
    st.session_state.last_analyzed_smiles = ""

def on_method_change():
    """Callback when switching input method: clears prior method state."""
    reset_all_inputs()

# Dashboard page configuration
st.set_page_config(page_title="PharmaGraph AI", page_icon="🧬", layout="wide")

# --- HIGH-END APPLE / LINEAR DESIGN SYSTEM INJECTION ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-canvas: #080c14;
    --surface-glass: rgba(15, 23, 42, 0.72);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-highlight: rgba(255, 255, 255, 0.16);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-cyan: #38bdf8;
    --ease-apple: cubic-bezier(0.23, 1, 0.32, 1);
}

.stApp {
    background-color: var(--bg-canvas) !important;
    background-image: 
        radial-gradient(ellipse 90% 55% at 50% -15%, rgba(56, 189, 248, 0.14), transparent),
        radial-gradient(ellipse 70% 45% at 85% 85%, rgba(99, 102, 241, 0.08), transparent) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--text-primary) !important;
    min-height: 100dvh;
}

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.025em !important;
    font-weight: 700 !important;
}

.hero-container {
    margin-bottom: 2rem;
    padding: 0.5rem 0 1rem 0;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.10);
    border: 1px solid rgba(56, 189, 248, 0.28);
    color: #38bdf8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
}

.hero-badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #38bdf8;
    box-shadow: 0 0 10px #38bdf8;
    animation: pulse-dot 2s infinite var(--ease-apple);
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.85); }
}

.hero-title {
    font-size: 2.35rem;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 60%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.35rem 0 0.65rem 0;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 860px;
}

/* Apple Concentric Glassmorphic Cards */
.pg-glass-card {
    background: var(--surface-glass);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid var(--border-subtle);
    border-top: 1px solid var(--border-highlight);
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 
        0 12px 32px -8px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: border-color 200ms var(--ease-apple), transform 200ms var(--ease-apple);
}

.pg-glass-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: 1px solid var(--border-highlight) !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.15rem !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3) !important;
    transition: transform 150ms var(--ease-apple), border-color 150ms var(--ease-apple) !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.3) !important;
}

div[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Emil Kowalski Buttons */
button[kind="primary"], .stButton > button {
    background: linear-gradient(180deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    transition: all 150ms var(--ease-apple) !important;
}

button[kind="primary"]:hover, .stButton > button:hover {
    background: linear-gradient(180deg, #38bdf8 0%, #0284c7 100%) !important;
    box-shadow: 0 6px 22px rgba(2, 132, 199, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

button[kind="primary"]:active, .stButton > button:active {
    transform: scale(0.97) !important;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3) !important;
    transition: transform 100ms ease-out !important;
}

/* Radio button pills */
div[role="radiogroup"] {
    gap: 8px !important;
}

div[role="radiogroup"] > label {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    transition: all 160ms var(--ease-apple) !important;
}

div[role="radiogroup"] > label:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    background: rgba(30, 41, 59, 0.7) !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 6px !important;
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border: none !important;
    background: transparent !important;
    transition: all 180ms var(--ease-apple) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* Glassmorphic Callout Banners */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3) !important;
    transition: all 180ms var(--ease-apple) !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    background: rgba(16, 185, 129, 0.12) !important;
    border: 1px solid rgba(16, 185, 129, 0.35) !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
    background: rgba(245, 158, 11, 0.12) !important;
    border: 1px solid rgba(245, 158, 11, 0.35) !important;
}

div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
    background: rgba(239, 68, 68, 0.14) !important;
    border: 1px solid rgba(239, 68, 68, 0.38) !important;
}

/* Responsive mobile */
@media (max-width: 768px) {
    .hero-title {
        font-size: 1.75rem !important;
    }
    .hero-subtitle {
        font-size: 0.95rem !important;
    }
    .pg-glass-card {
        padding: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        GNN Production Engine v2.1 • 81.5% ROC-AUC • DoS Hardened
    </div>
    <h1 class="hero-title">🧬 PharmaGraph: AI Pharmacological Analysis Platform</h1>
    <p class="hero-subtitle">
        Predict molecular toxicity and pharmacological risks using GNN with <strong>Functional Group Attention & Chemical Interaction Learning</strong>, engineered for researchers and toxicologists.
    </p>
</div>
""", unsafe_allow_html=True)

import os
API_URL = os.environ.get("API_URL", "http://localhost:1234/api/predict")

# Curated reference presets
presets = {
    "None selected...": "",
    "Aspirin (Common analgesic & anti-inflammatory)": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol / Acetaminophen (Analgesic & antipyretic)": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine (Central nervous system stimulant)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol (Medical alcohol / solvent)": "CCO",
    "Nicotine (Tobacco alkaloid & stimulant)": "CN1CCCC1c2cccnc2",
    "Cyanide / Hydrogen Cyanide (Potent respiratory toxicant)": "C#N",
    "Phenol (Industrial toxicant & chemical cauterant)": "C1=CC=C(C=C1)O"
}

# Local cache for common compounds to avoid network latency / PubChem rate limits
COMMON_COMPOUNDS = {
    'water': 'O',
    'sugar': 'CC(=O)O',  # glucose fallback
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

# --- METHOD SELECTOR ---
st.markdown("### 🎯 Select Molecular Input Method")
method_col, reset_col = st.columns([4.2, 1.2])

with method_col:
    chosen_method = st.radio(
        "Select input method:",
        options=available_methods,
        key="input_method",
        horizontal=True,
        on_change=on_method_change,
        label_visibility="collapsed"
    )

with reset_col:
    if st.button("🔄 Reset All", use_container_width=True, help="Clear all inputs and existing analysis results"):
        reset_all_inputs()
        st.rerun()

final_smiles = ""

# --- RENDER ACTIVE METHOD INTERFACE ---
if chosen_method == METHOD_PRESET:
    st.markdown("#### 🗂️ Method 1: Select Sample Compound from Presets")
    selected_preset = st.selectbox(
        "Select a sample compound to test:",
        options=list(presets.keys()),
        key="selected_preset_input"
    )
    if selected_preset and selected_preset != "None selected...":
        final_smiles = presets[selected_preset]
        st.info(f"🧬 Selected preset: **{selected_preset}** (SMILES: `{final_smiles}`)")

elif chosen_method == METHOD_SEARCH:
    st.markdown("#### 🔍 Method 2: Search by Compound Name")
    search_name = st.text_input(
        "Enter drug or chemical compound name:",
        placeholder="e.g., Ibuprofen, Penicillin, Nicotine, Water, Aspirin...",
        key="search_name_input"
    )
    if search_name and search_name.strip():
        search_name_norm = search_name.strip().lower()
        if search_name_norm in COMMON_COMPOUNDS:
            final_smiles = COMMON_COMPOUNDS[search_name_norm]
            st.info(f"🧬 Found in local cache: `{final_smiles}` (Query: {search_name})")
        else:
            with st.spinner(f"🔍 Searching PubChem database for '{search_name}'..."):
                try:
                    compounds = pcp.get_compounds(search_name.strip(), 'name')
                    if not compounds:
                        compounds = pcp.get_compounds(search_name.strip().lower(), 'name')
                    if not compounds:
                        compounds = pcp.get_compounds(search_name.strip(), 'formula')
                    
                    if compounds:
                        final_smiles = compounds[0].isomeric_smiles
                        compound_display_name = compounds[0].synonyms[0] if compounds[0].synonyms else compounds[0].iupac_name or search_name
                        st.info(f"🧬 Found on PubChem: `{final_smiles}` (Name: {compound_display_name})")
                    else:
                        st.error(f"❌ '{search_name}' was not found in local cache or PubChem. Please try another name or use Method 3 (Direct SMILES)!")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "bad request" in error_msg or "expecting value" in error_msg or "json decode" in error_msg:
                        st.error(f"⚠️ PubChem returned an error for '{search_name}'. Try standard chemical names or local cache. Suggestions: water, ethanol, caffeine, aspirin, paracetamol")
                    elif "timeout" in error_msg or "connection" in error_msg:
                        st.error(f"⚠️ PubChem connection timed out. Please try again or use direct SMILES.")
                    else:
                        st.error(f"⚠️ Search error: {e}. Please retry or enter SMILES directly.")

elif chosen_method == METHOD_SMILES:
    st.markdown("#### 🧪 Method 3: Direct Molecular SMILES String Input")
    raw_smiles = st.text_input(
        "For novel, synthesized, or unindexed chemical structures:",
        placeholder="e.g., C1=CC=C(C=C1)O",
        key="raw_smiles_input"
    )
    if raw_smiles and raw_smiles.strip():
        candidate_smiles = raw_smiles.strip()
        mol_check = Chem.MolFromSmiles(candidate_smiles)
        if mol_check is not None:
            final_smiles = candidate_smiles
            st.info(f"🧬 Using custom SMILES structure: `{final_smiles}`")
        else:
            st.error("❌ Invalid SMILES string according to RDKit chemical validation standards! Please verify the molecular structure.")

elif chosen_method == METHOD_DRAW and st_ketcher is not None:
    st.markdown("#### 🖌️ Method 4: Draw Molecule (Interactive 2D Sketcher)")
    st.markdown("Draw your structure using the chemical editor below. The system automatically translates your drawing to SMILES. Click **Apply** when done.")
    drawn_smiles = st_ketcher(key=f"ketcher_editor_{st.session_state.ketcher_key_idx}")
    if drawn_smiles and drawn_smiles.strip():
        final_smiles = drawn_smiles.strip()
        st.info(f"🎨 Using drawn molecular structure (SMILES: `{final_smiles}`)")

# Persist final_smiles in session_state
st.session_state.final_smiles = final_smiles

# Automatically invalidate previous analysis if target molecule changed
if st.session_state.get('last_analyzed_smiles') != final_smiles:
    st.session_state.current_analysis = None

# --- ANALYSIS AND RESULTS BLOCK ---
if final_smiles:
    st.markdown("---")
    concentration_input = st.number_input(
        "🧪 Concentration / Dosage (Molar) - Default 1.0 M if unspecified:",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%f"
    )
    if st.button("🚀 Run AI GNN Pharmacological Analysis"):
        with st.spinner("GNN model is extracting molecular graph features and computing predictions..."):
            try:
                # 1. Query FastAPI backend
                response = requests.post(API_URL, json={"smiles": final_smiles, "concentration_molar": concentration_input})
                if response.status_code == 200:
                    st.session_state.current_analysis = {
                        "smiles": final_smiles,
                        "data": response.json()
                    }
                    st.session_state.last_analyzed_smiles = final_smiles
                elif response.status_code == 429:
                    err_json = response.json()
                    retry_sec = err_json.get("retry_after", 3)
                    st.warning(f"⏳ **Rate Limiting Active (HTTP 429)**: {err_json.get('detail', 'Too many requests. Please wait a few moments before retrying.')}")
                elif response.status_code == 413:
                    st.error("🛑 **Payload Too Large (HTTP 413)**: Request body exceeds the 64 KB safety limit.")
                elif response.status_code == 422:
                    st.error("⚠️ **Validation Error (HTTP 422)**: Molecule structure exceeds 500 characters or concentration is out of safe range (10⁻¹² to 10.0 M).")
                else:
                    st.error(f"AI Server Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI AI Backend. Error details: {e}")

    # Display analysis results if available for current molecule
    if (
        st.session_state.get("current_analysis") is not None
        and st.session_state.current_analysis.get("smiles") == final_smiles
    ):
        data = st.session_state.current_analysis["data"]
        
        # 2. Detailed RDKit conformation and 3D visualization with py3Dmol
        mol = Chem.MolFromSmiles(final_smiles)
        if mol is not None:
            # Extract molecular descriptors
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
            if not func_groups:
                func_groups.append("No specific functional groups detected")
            
            # Generate 3D coordinates
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            mol_block = Chem.MolToMolBlock(mol_3d)
            
            # Reverse lookup compound name from SMILES via PubChem
            compound_name = "Unindexed / Custom Compound"
            try:
                c = pcp.get_compounds(final_smiles, 'smiles')
                if c:
                    if c[0].synonyms:
                        compound_name = c[0].synonyms[0]
                    elif c[0].iupac_name:
                        compound_name = c[0].iupac_name
            except Exception:
                pass
            
            # Two-column layout
            left_col, right_col = st.columns([1, 1.5])
            
            with left_col:
                st.markdown("### 🔬 Interactive 3D Conformation")
                viewer = py3Dmol.view(width=400, height=350)
                viewer.addModel(mol_block, "mol")
                
                # Highlight toxic atoms (red spheres) if risk is high
                node_importance = data.get("explanation", {}).get("node_importance", [])
                risk_str = data.get("predictions", {}).get("toxicity_risk", "0%")
                try:
                    risk_value = float(risk_str.replace('%', ''))
                except:
                    risk_value = 0.0

                if node_importance and risk_value > 50.0:
                    max_imp = max(node_importance) if max(node_importance) > 0 else 1.0
                    
                    # Base stick style for all bonds
                    viewer.setStyle({'stick': {'radius': 0.15}})
                    
                    for i, imp in enumerate(node_importance):
                        norm_imp = imp / max_imp
                        # Highlight critical toxicophore atoms
                        if norm_imp > 0.85:
                            viewer.addStyle({'serial': i+1}, {'sphere': {'color': 'red', 'radius': 0.5}})
                        else:
                            viewer.addStyle({'serial': i+1}, {'sphere': {'color': 'white', 'radius': 0.3}})
                else:
                    # Default coloring if non-toxic
                    viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.3}})
                
                viewer.setBackgroundColor('#080c14') # Match Apple dark canvas
                viewer.zoomTo()
                showmol(viewer, height=350, width=400)
                
                st.markdown(f"""
                <div class="pg-glass-card" style="padding: 1.15rem; margin-top: 0.85rem;">
                    <div style="font-weight: 700; font-size: 1.15rem; color: #f8fafc; margin-bottom: 0.4rem;">{compound_name}</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 0.75rem;">
                        <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.28); border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;">{formula}</span>
                        <span style="background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.28); border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;">{mw:.2f} g/mol</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5; margin-bottom: 0.35rem;"><strong>Composition:</strong> {atoms_str}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;"><strong>Functional Groups:</strong> {', '.join(func_groups)}</div>
                </div>
                """, unsafe_allow_html=True)
        
            with right_col:
                st.success("✅ AI Analysis Complete!")
                st.markdown("### 📊 Graph Metrics & Predictions")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(label="Atoms Count (Nodes)", value=data["graph_info"]["atoms_count"])
                with col_b:
                    st.metric(label="Bonds Count (Edges)", value=data["graph_info"]["bonds_count"])
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                toxicity_str = data["predictions"]["toxicity_risk"]
                toxicity_float = float(toxicity_str.replace("%", ""))
                
                if toxicity_float >= 65.0:
                    st.metric(label="Toxicity Risk 🛑", value=toxicity_str, delta="CRITICAL RISK (Nguy hiểm)", delta_color="inverse")
                    st.error("Warning: This compound presents a severe biological toxicity risk. Structural toxicophores detected.")
                elif toxicity_float >= 35.0:
                    st.metric(label="Toxicity Risk ⚠️", value=toxicity_str, delta="Moderate Concern (Cần kiểm chứng)", delta_color="off")
                    st.warning("Structure contains potential toxicophores or alerts. Experimental / clinical assessment advised.")
                else:
                    st.metric(label="Toxicity Risk ✅", value=toxicity_str, delta="Low Concern / Safe (An toàn)", delta_color="normal")
                    st.success("Stable structure. No critical toxicity hazards detected based on reference database.")
                    
                st.progress(int(toxicity_float))

            # --- MODEL DECISION BREAKDOWN & INTERPRETABILITY ---
            attribution = data.get("decision_attribution")
            if attribution:
                st.markdown("---")
                st.markdown("### 🧠 Model Decision Breakdown (Why this prediction?)")
                st.caption("Comprehensive pharmacological attribution explaining model reasoning: driving structural alerts, functional group synergy, dosage sensitivity, and atomic saliency.")

                primary_driver = attribution.get("primary_driver", "General Molecular Topology")
                summary_text = attribution.get("summary_text", "")
                toxicophores = attribution.get("toxicophores", [])
                func_groups_present = attribution.get("functional_groups_present", [])
                synergy_pairs = attribution.get("functional_group_synergy", [])
                dosage_effect = attribution.get("dosage_effect", {})
                top_atoms = attribution.get("top_contributing_atoms", [])

                # 1. Primary Driver Callout Card (Dynamic risk-tier background: Green / Yellow / Red)
                driver_col1, driver_col2 = st.columns([1, 2.2])
                with driver_col1:
                    if toxicity_float >= 65.0:
                        # Nguy hiểm -> Nền đỏ (Red)
                        st.error(f"**Primary Driving Factor:**\n\n🚨 {primary_driver}\n\n*(Mức độ: **Nguy hiểm** — {toxicity_float:.1f}% risk)*")
                    elif toxicity_float >= 35.0:
                        # Cần kiểm chứng -> Nền vàng (Yellow)
                        icon = "🧪" if "Concentration" in primary_driver or "Dosage" in primary_driver else ("⚛️" if "Synergy" in primary_driver else "⚠️")
                        st.warning(f"**Primary Driving Factor:**\n\n{icon} {primary_driver}\n\n*(Mức độ: **Cần kiểm chứng** — {toxicity_float:.1f}% risk)*")
                    else:
                        # An toàn -> Nền xanh lá (Green)
                        st.success(f"**Primary Driving Factor:**\n\n✅ {primary_driver}\n\n*(Mức độ: **An toàn** — {toxicity_float:.1f}% risk)*")

                with driver_col2:
                    st.markdown(f"**Reasoning Summary:**\n\n{summary_text}")

                # 2. Interactive Feature Attribution Tabs
                tab_alerts, tab_synergy, tab_dosage, tab_atoms = st.tabs([
                    "🚨 Structural Alerts & Toxicophores",
                    "⚛️ Functional Group Synergy",
                    "🧪 Dosage & Concentration Impact",
                    "🎯 Atomic Attribution"
                ])

                with tab_alerts:
                    st.markdown("#### 🚨 Knowledge-Based Structural Toxicophore Alerts")
                    if toxicophores:
                        for alert in toxicophores:
                            st.markdown(f"##### • **{alert['name']}** (`{alert['smarts']}`)")
                            st.markdown(f"- **Matches in molecule:** {alert['count']} occurrence(s) (Atom indices: `{alert.get('matched_atom_indices', [])}`)")
                            st.markdown(f"- **Biological Mechanism:** {alert['description']}")
                            st.markdown("---")
                    else:
                        st.success("✅ **No hazardous structural alerts or reactive toxicophore patterns detected.**")

                    st.markdown("#### 🧬 Detected Functional Groups (RDKit)")
                    if func_groups_present:
                        cols = st.columns(min(len(func_groups_present), 4))
                        for idx, fg in enumerate(func_groups_present):
                            col_target = cols[idx % len(cols)]
                            col_target.info(f"**{fg['name']}**\nCount: {fg['count']}")
                    else:
                        st.caption("No specific fragments from the 85-functional group catalog detected.")

                with tab_synergy:
                    st.markdown("#### ⚛️ Functional Group Cross-Attention Interactions")
                    st.caption("Learned cross-attention coupling from the model's `FunctionalGroupInteraction` Multihead Attention layer. High attention coupling indicates synergistic contribution to pharmacological liability.")
                    
                    if synergy_pairs:
                        st.markdown(f"**Top Interacting Pairs ({len(synergy_pairs)} total pairs):**")
                        for idx, syn in enumerate(synergy_pairs[:6]):
                            col_s1, col_s2 = st.columns([1.5, 2.5])
                            with col_s1:
                                st.markdown(f"**{idx+1}. {syn['group_a']} ↔ {syn['group_b']}**")
                                st.caption(f"Counts: {syn.get('count_a', 1)} × {syn.get('count_b', 1)}")
                            with col_s2:
                                score_val = syn['synergy_score']
                                st.progress(min(max(score_val * 4.0, 0.05), 1.0))
                                st.caption(f"Attention Coupling Score: **{score_val:.4f}**")
                    else:
                        st.info("Fewer than 2 active functional groups detected in this structure; inter-group cross-attention synergy is not applicable.")

                with tab_dosage:
                    st.markdown("#### 🧪 Dosage & Concentration Sensitivity")
                    st.caption("Comparison between user-defined concentration and standard in vitro Tox21/ClinTox screening baseline (10 µM / pIC50 = 5.0).")
                    
                    d_col1, d_col2, d_col3 = st.columns(3)
                    with d_col1:
                        user_conc = dosage_effect.get("user_concentration_molar", 1.0)
                        st.metric(
                            label="User Dosage",
                            value=f"{user_conc:.2g} M",
                            help="Input concentration specified for this evaluation"
                        )
                        st.caption(f"Predicted Risk: **{dosage_effect.get('user_toxicity_risk', 0.0):.1f}%**")
                    with d_col2:
                        st.metric(
                            label="Baseline Screening",
                            value="10 µM (1e-5 M)",
                            help="Standard assay screening concentration (Tox21 / ClinTox)"
                        )
                        st.caption(f"Baseline Risk: **{dosage_effect.get('baseline_toxicity_risk', 0.0):.1f}%**")
                    with d_col3:
                        delta = dosage_effect.get("delta_risk", 0.0)
                        st.metric(
                            label="Dosage Impact (ΔRisk)",
                            value=f"{delta:+.1f}%",
                            delta=f"{delta:+.1f}%",
                            delta_color="inverse" if delta > 0 else "normal"
                        )
                    
                    st.info(f"💡 **Dosage Analysis:** {dosage_effect.get('assessment', '')}")

                with tab_atoms:
                    st.markdown("#### 🎯 Key Influential Atoms (GNNExplainer)")
                    st.caption("Top atomic centers contributing most strongly to the GNN prediction. High-saliency atoms correspond to the red sphere highlights in the 3D conformation view above.")
                    
                    if top_atoms:
                        atom_cols = st.columns(min(len(top_atoms), 5))
                        for idx, atom_info in enumerate(top_atoms):
                            col_a = atom_cols[idx % len(atom_cols)]
                            col_a.metric(
                                label=f"Atom #{atom_info['atom_index']} ({atom_info['element']})",
                                value=f"{atom_info['importance_score']:.3f}"
                            )
                    else:
                        st.caption("No individual atom saliency highlights available.")


