import math
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp

# Automatically extract 85 RDKit functional group descriptors and names
frag_items = [(name.replace('fr_', ''), func) for name, func in Descriptors.descList if name.startswith('fr_')]
frag_names = [item[0] for item in frag_items]
frag_funcs = [item[1] for item in frag_items]

# Knowledge-based toxicophore alerts with biological mechanism descriptions
TOXICOPHORE_DEFINITIONS = [
    {
        "name": "Cyanide / Nitrile group",
        "smarts": "C#N",
        "description": "Cyano group known for cellular respiration toxicity and cytochrome c oxidase inhibition."
    },
    {
        "name": "Organophosphate ester",
        "smarts": "[$([P](=[O,S])[F,Cl]),$([P](=[O,S])C#N),$([P](=[O,S])([#6])S),$([P;!$([P]-[O]-[P])](=[O,S])([O,S][#6])([O,S][#6])([O,S,#6][#6]))]",
        "description": "Potent neurotoxic pharmacophore acting via acetylcholinesterase inhibition."
    },
    {
        "name": "Benzene ring",
        "smarts": "c1ccccc1",
        "description": "Aromatic hydrocarbon core linked to metabolic bioactivation and reactive metabolite formation."
    },
    {
        "name": "Phenol moiety",
        "smarts": "c1ccccc1O",
        "description": "Hydroxylated aromatic ring capable of quinone/semiquinone redox cycling and mitochondrial uncoupling."
    },
    {
        "name": "Reactive Aldehyde",
        "smarts": "[CX3H1](=O)",
        "description": "Electrophilic carbonyl center forming covalent adducts with cellular proteins and DNA."
    },
    {
        "name": "Sulfide / Thiol center",
        "smarts": "[S]",
        "description": "Reactive sulfur center susceptible to redox cycling and cellular glutathione depletion."
    },
    {
        "name": "Halogenated aromatic",
        "smarts": "[Cl,Br,I]c1ccccc1",
        "description": "Halogenated phenyl ring associated with high lipophilicity, metabolic persistence, and bioaccumulation."
    },
]
toxic_patterns = [Chem.MolFromSmarts(item["smarts"]) for item in TOXICOPHORE_DEFINITIONS]

def get_toxicophore_density(mol):
    """Calculate relative atom density for each recognized toxicophore pattern."""
    total_atoms = mol.GetNumAtoms()
    if total_atoms == 0:
        return [0.0] * len(toxic_patterns)
    
    densities = []
    for pattern in toxic_patterns:
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            densities.append(0.0)
        else:
            toxic_atoms = set()
            for match in matches:
                toxic_atoms.update(match)
            densities.append(len(toxic_atoms) / total_atoms)
    return densities

def analyze_toxicophores(mol):
    """Detailed structural toxicophore analysis with matched atom indices and descriptions."""
    total_atoms = mol.GetNumAtoms()
    alerts = []
    for defn, pattern in zip(TOXICOPHORE_DEFINITIONS, toxic_patterns):
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            matched_atoms = set()
            for match in matches:
                matched_atoms.update(match)
            alerts.append({
                "name": defn["name"],
                "smarts": defn["smarts"],
                "description": defn["description"],
                "count": len(matches),
                "matched_atom_indices": sorted(list(matched_atoms)),
                "density": round(len(matched_atoms) / total_atoms, 4) if total_atoms > 0 else 0.0
            })
    return alerts

# Common chemicals dictionary for instant local lookup
COMPOUND_NAME_MAP = {
    'O': 'Water',
    '[Na+].[Cl-]': 'Sodium chloride (NaCl)',
    '[Na+].[F-]': 'Sodium fluoride',
    '[K+].[Cl-]': 'Potassium chloride',
    '[Ca+2].[Cl-].[Cl-]': 'Calcium chloride',
    '[NH4+].[Cl-]': 'Ammonium chloride',
    'CCO': 'Ethanol',
    'CO': 'Methanol',
    'CCOCCO': 'Ethylene glycol',
    'CCCCO': 'Butanol',
    'CC(=O)O': 'Acetic acid',
    'CCC(=O)O': 'Propionic acid',
    'CCCC(=O)O': 'Butyric acid',
    'CC(O)=O': 'Lactic acid',
    'CC(=O)Oc1ccccc1C(=O)O': 'Aspirin',
    'CC(=O)Nc1ccc(O)cc1': 'Paracetamol',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C': 'Caffeine',
    'C#N': 'Cyanide',
    'C1=CC=C(C=C1)O': 'Phenol',
    'c1ccccc1O': 'Phenol',
    'C(C(=O)O)N': 'Glycine',
    'CC(C(=O)O)N': 'Alanine',
    'CC(C)C(C(=O)O)N': 'Valine',
    'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O': 'Glucose',
    'OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@H]1O': 'Fructose',
    'COc1ccc2nc(nc2c1)N': 'Adenine',
    'C1=CC=C(C=C1)': 'Benzene',
    'Cc1ccccc1': 'Toluene',
    'c1ccccc1': 'Benzene',
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O': 'Ibuprofen',
    'CN1CCCC1c2cccnc2': 'Nicotine',
    'NC(=O)N': 'Urea',
    'N': 'Ammonia',
    'C1CCOCC1': 'Tetrahydrofuran',
    'C1COCCO1': '1,4-dioxane',
    'CC(=O)NC1=CC=CC=C1': 'Acetanilide',
    'CN1C(=O)NC2=NC=NC=C21': 'Theophylline',
}

compound_name_cache = {}

def get_compound_name(smiles: str) -> str:
    """Lookup compound name from PubChem using SMILES with local fallback."""
    if smiles in COMPOUND_NAME_MAP:
        return COMPOUND_NAME_MAP[smiles]
    
    if smiles in compound_name_cache:
        return compound_name_cache[smiles]
    
    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds:
            c = compounds[0]
            name = c.iupac_name or (c.synonyms[0] if c.synonyms else 'Unknown')
            compound_name_cache[smiles] = name
            return name
    except Exception:
        pass
    
    compound_name_cache[smiles] = 'Unknown'
    return 'Unknown'

def get_atom_features(atom):
    """Extract 6 chemical node features per atom."""
    return [
        atom.GetAtomicNum(),            
        atom.GetDegree(),               
        int(atom.GetIsAromatic()),      
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),         
        atom.GetNumRadicalElectrons()   
    ]

def smiles_to_graph(smiles_string: str, concentration_molar: float = 1e-5):
    """
    Convert SMILES to PyTorch Geometric Data object with node, edge, global,
    toxicophore density, functional group, and dosage features.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
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
    
    toxic_densities = get_toxicophore_density(mol)
    global_features.extend(toxic_densities)
    
    func_group_features = [float(func(mol)) for func in frag_funcs]
    
    if concentration_molar <= 0:
        pIC50 = 0.0
    else:
        pIC50 = -math.log10(concentration_molar)
    
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        global_features=torch.tensor([global_features], dtype=torch.float),
        func_group_features=torch.tensor([func_group_features], dtype=torch.float),
        concentration=torch.tensor([[pIC50]], dtype=torch.float)
    )
