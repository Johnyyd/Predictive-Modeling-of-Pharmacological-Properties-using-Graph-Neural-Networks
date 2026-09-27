from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_decision_attribution_phenol():
    # Phenol (c1ccccc1O): Contains benzene ring and phenol toxicophores
    response = client.post("/api/predict", json={"smiles": "c1ccccc1O", "concentration_molar": 1.0})
    assert response.status_code == 200
    data = response.json()
    
    assert "decision_attribution" in data, "decision_attribution must be present in response"
    attribution = data["decision_attribution"]
    
    # 1. Primary driver & summary
    assert "primary_driver" in attribution
    assert "summary_text" in attribution
    assert len(attribution["summary_text"]) > 0
    
    # 2. Toxicophores
    assert "toxicophores" in attribution
    assert isinstance(attribution["toxicophores"], list)
    toxicophore_names = [t["name"] for t in attribution["toxicophores"]]
    assert any("Phenol" in name or "Benzene" in name for name in toxicophore_names)
    
    # 3. Functional groups present
    assert "functional_groups_present" in attribution
    assert isinstance(attribution["functional_groups_present"], list)
    
    # 4. Synergy
    assert "functional_group_synergy" in attribution
    assert isinstance(attribution["functional_group_synergy"], list)
    
    # 5. Dosage effect
    assert "dosage_effect" in attribution
    dosage = attribution["dosage_effect"]
    assert dosage["user_concentration_molar"] == 1.0
    assert dosage["baseline_concentration_molar"] == 1e-5
    assert isinstance(dosage["user_toxicity_risk"], float)
    assert isinstance(dosage["baseline_toxicity_risk"], float)
    assert isinstance(dosage["delta_risk"], float)
    assert isinstance(dosage["assessment"], str)
    
    # 6. Top contributing atoms
    assert "top_contributing_atoms" in attribution
    assert isinstance(attribution["top_contributing_atoms"], list)
    if attribution["top_contributing_atoms"]:
        first_atom = attribution["top_contributing_atoms"][0]
        assert "atom_index" in first_atom
        assert "element" in first_atom
        assert "importance_score" in first_atom

def test_predict_decision_attribution_cyanide():
    # Cyanide (C#N): Acute toxicophore
    response = client.post("/api/predict", json={"smiles": "C#N", "concentration_molar": 1.0})
    assert response.status_code == 200
    data = response.json()
    attribution = data["decision_attribution"]
    
    # Should identify Cyanide alert
    alert_names = [a["name"] for a in attribution["toxicophores"]]
    assert any("Cyanide" in name for name in alert_names)
    assert "Intrinsic Structural Alerts" in attribution["primary_driver"]

def test_predict_decision_attribution_aspirin_synergy():
    # Aspirin: CC(=O)Oc1ccccc1C(=O)O has multiple functional groups (ester, carboxylic acid, aromatic)
    response = client.post("/api/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O", "concentration_molar": 0.001})
    assert response.status_code == 200
    data = response.json()
    attribution = data["decision_attribution"]
    
    # Should have multiple functional groups and computed synergy pairs
    assert len(attribution["functional_groups_present"]) >= 2
    assert len(attribution["functional_group_synergy"]) >= 1
    top_pair = attribution["functional_group_synergy"][0]
    assert "group_a" in top_pair
    assert "group_b" in top_pair
    assert "synergy_score" in top_pair
    assert top_pair["synergy_score"] >= 0.0

def test_predict_decision_attribution_water_safe():
    # Water (O): Clean, safe, no toxicophores
    response = client.post("/api/predict", json={"smiles": "O", "concentration_molar": 1.0})
    assert response.status_code == 200
    data = response.json()
    attribution = data["decision_attribution"]
    assert len(attribution["toxicophores"]) == 0
    assert "Safe" in attribution["primary_driver"] or "Benign" in attribution["primary_driver"]
