from streamlit.testing.v1 import AppTest
import pytest

def test_app_renders_without_exceptions():
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App had exceptions on initial load: {at.exception}"

def test_app_shows_decision_attribution_when_data_present():
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()
    
    # Mock analysis data with decision_attribution
    mock_data = {
        "smiles": "c1ccccc1O",
        "compound_name": "Phenol",
        "concentration_molar": 1.0,
        "pIC50": 0.0,
        "graph_info": {"atoms_count": 7, "bonds_count": 7},
        "predictions": {
            "toxicity_risk": "85.20%",
            "target_class": 12,
            "ct_tox_class": 12,
            "all_class_probs": ["0.8520"] * 13
        },
        "decision_attribution": {
            "primary_driver": "Intrinsic Structural Alerts & Toxicophores",
            "summary_text": "Predicted toxicity (85.2%) is predominantly driven by structural alerts: Phenol moiety, Benzene ring.",
            "toxicophores": [
                {
                    "name": "Phenol moiety",
                    "smarts": "c1ccccc1O",
                    "description": "Hydroxylated aromatic ring capable of redox cycling.",
                    "count": 1,
                    "matched_atom_indices": [0, 1, 2, 3, 4, 5, 6],
                    "density": 1.0
                }
            ],
            "functional_groups_present": [
                {"name": "phenol", "count": 1},
                {"name": "benzene", "count": 1}
            ],
            "functional_group_synergy": [
                {
                    "group_a": "phenol",
                    "group_b": "benzene",
                    "count_a": 1,
                    "count_b": 1,
                    "synergy_score": 0.125,
                    "description": "Attention coupling between phenol and benzene"
                }
            ],
            "dosage_effect": {
                "user_concentration_molar": 1.0,
                "baseline_concentration_molar": 1e-5,
                "user_toxicity_risk": 85.20,
                "baseline_toxicity_risk": 72.10,
                "delta_risk": 13.10,
                "assessment": "Elevated concentration (1.0 M) amplifies predicted toxicity by +13.1% compared to standard baseline screening (10 µM)."
            },
            "top_contributing_atoms": [
                {"atom_index": 6, "element": "O", "importance_score": 0.95}
            ]
        },
        "explanation": {"node_importance": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.95]}
    }
    
    smiles_val = "C1=CC=C(C=C1)O"
    mock_data["smiles"] = smiles_val
    
    # Select Phenol preset to set final_smiles to C1=CC=C(C=C1)O
    at.selectbox(key="selected_preset_input").select("Phenol (Industrial toxicant & chemical cauterant)")
    at.session_state["current_analysis"] = {
        "smiles": smiles_val,
        "data": mock_data
    }
    at.session_state["last_analyzed_smiles"] = smiles_val
    at.run()
    
    assert not at.exception, f"App threw exception when rendering decision attribution: {at.exception}"
    
    # Verify that the decision attribution section headers and tabs are rendered
    markdown_texts = [m.value for m in at.markdown]
    combined_markdown = " ".join(markdown_texts)
    error_texts = [e.value for e in at.error]
    combined_errors = " ".join(error_texts)
    
    assert "Decision Breakdown" in combined_markdown, "Should render Model Decision Breakdown header"
    assert any("Primary Driving Factor" in err for err in error_texts) or "Primary Driving Factor" in combined_markdown
    assert "Functional Group Cross-Attention Interactions" in combined_markdown
    assert "Dosage & Concentration Sensitivity" in combined_markdown
    assert "Key Influential Atoms" in combined_markdown
