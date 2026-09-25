import pytest
from rdkit import Chem
from fastapi.testclient import TestClient
from main import app, analyze_toxicophores, get_toxicophore_density, TOXICOPHORE_DEFINITIONS

BIOLOGICAL_PHOSPHATES = {
    "ATP": "NC1=NC=NC2=C1N=CN2[C@@H]3O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O",
    "ADP": "NC1=NC=NC2=C1N=CN2[C@@H]3O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O",
    "AMP": "NC1=NC=NC2=C1N=CN2[C@@H]3O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]3O",
    "GTP": "NC1=NC2=C(N=CN2[C@@H]3O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O)C(=O)N1",
    "cAMP": "C1C2C(C(O1)N3C=NC4=C3N=CN=C4N)OP(=O)(O2)O",
    "Inorganic Phosphate": "OP(=O)(O)O",
    "Pyrophosphate": "OP(=O)(O)OP(=O)(O)O",
    "Glucose-6-phosphate": "C(C1C(C(C(C(O1)O)O)O)O)OP(=O)(O)O",
}

SYNTHETIC_ORGANOPHOSPHATE_TOXICANTS = {
    "Sarin": "CC(C)OP(=O)(C)F",
    "Soman": "CC(C(C)(C)C)OP(=O)(C)F",
    "Tabun": "CCOP(=O)(C#N)N(C)C",
    "VX": "CCOP(=O)(C)SCCN(C(C)C)C(C)C",
    "Chlorpyrifos": "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl",
    "Malathion": "CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC",
    "Paraoxon": "CCOP(=O)(OCC)Oc1ccc([N+](=O)[O-])cc1",
    "Parathion": "CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1",
    "Diazinon": "CCOP(=S)(OCC)Oc1nc(C)cc(C(C)C)n1",
    "DFP": "CC(C)OP(=O)(OC(C)C)F",
    "Dichlorvos": "COP(=O)(OC)OC=C(Cl)Cl",
}

def test_biological_phosphates_no_organophosphate_alert():
    """Verify that endogenous metabolites and nucleotides do not trigger organophosphate false positives."""
    for name, smiles in BIOLOGICAL_PHOSPHATES.items():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES for {name}"
        alerts = analyze_toxicophores(mol)
        op_alerts = [a for a in alerts if a["name"] == "Organophosphate ester"]
        assert len(op_alerts) == 0, f"False positive organophosphate alert triggered for biological molecule: {name}"

def test_synthetic_organophosphate_toxicants_matched():
    """Verify that genuine nerve agents and OP pesticides trigger the organophosphate alert."""
    for name, smiles in SYNTHETIC_ORGANOPHOSPHATE_TOXICANTS.items():
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Failed to parse SMILES for {name}"
        alerts = analyze_toxicophores(mol)
        op_alerts = [a for a in alerts if a["name"] == "Organophosphate ester"]
        assert len(op_alerts) >= 1, f"Failed to detect organophosphate toxicophore in: {name}"
        assert op_alerts[0]["count"] >= 1
        assert op_alerts[0]["density"] > 0

def test_atp_api_predict_excludes_op_and_low_concern():
    """Verify that ATP in /api/predict is not flagged as an organophosphate pesticide and has low risk."""
    client = TestClient(app)
    response = client.post("/api/predict", json={
        "smiles": BIOLOGICAL_PHOSPHATES["ATP"],
        "concentration_molar": 1e-6
    })
    assert response.status_code == 200
    data = response.json()
    attribution = data.get("decision_attribution", {})
    toxicophores = attribution.get("toxicophores", [])
    op_names = [t["name"] for t in toxicophores if "Organophosphate" in t["name"]]
    assert len(op_names) == 0, f"ATP unexpectedly received organophosphate toxicophore alert: {op_names}"
    assert "Organophosphate" not in attribution.get("summary_text", ""), "Summary text should not mention organophosphates for ATP"
    assert attribution.get("primary_driver") != "Intrinsic Structural Alerts & Toxicophores", "ATP should not be flagged as structural toxicophore hazard"

def test_sarin_api_predict_detects_organophosphate():
    """Verify that Sarin in /api/predict is correctly flagged with the organophosphate toxicophore."""
    client = TestClient(app)
    response = client.post("/api/predict", json={
        "smiles": SYNTHETIC_ORGANOPHOSPHATE_TOXICANTS["Sarin"],
        "concentration_molar": 1e-6
    })
    assert response.status_code == 200
    data = response.json()
    toxicophores = data.get("decision_attribution", {}).get("toxicophores", [])
    op_names = [t["name"] for t in toxicophores if "Organophosphate" in t["name"]]
    assert len(op_names) == 1, f"Expected 1 organophosphate alert for Sarin, found: {op_names}"
