"""
Molecular Feature Extraction Module
====================================
Converts SMILES strings into numerical feature vectors suitable for
quantum circuit encoding. Uses RDKit descriptors.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from sklearn.preprocessing import MinMaxScaler


# Core physicochemical descriptors we'll encode onto qubits.
# We keep the count manageable since each feature = 1 qubit.
DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "RotBonds": Descriptors.NumRotatableBonds,
    "AromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol object with validation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def compute_descriptors(smiles: str) -> dict:
    """Compute physicochemical descriptors for a single molecule."""
    mol = smiles_to_mol(smiles)
    return {name: func(mol) for name, func in DESCRIPTOR_FUNCTIONS.items()}


def compute_descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    """
    Compute descriptor matrix for a list of molecules.
    Returns: np.ndarray of shape (n_molecules, n_descriptors)
    """
    all_descs = []
    for smi in smiles_list:
        descs = compute_descriptors(smi)
        all_descs.append(list(descs.values()))
    return np.array(all_descs)


def normalize_features(features: np.ndarray, scaler=None) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize features to [0, pi] range for quantum circuit encoding.
    Rotation gates use angles, so we map features to [0, pi].
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        normalized = scaler.fit_transform(features)
    else:
        normalized = scaler.transform(features)
    return normalized, scaler


def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint (classical baseline for comparison)."""
    mol = smiles_to_mol(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Classical Tanimoto similarity (baseline comparison)."""
    mol_a = smiles_to_mol(smiles_a)
    mol_b = smiles_to_mol(smiles_b)
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ---- Quick test ----
if __name__ == "__main__":
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",      # Aspirin
        "CC(=O)Nc1ccc(O)cc1",           # Acetaminophen
        "CC12CCC3C(CCC4CC(=O)CCC34C)C1CCC2O",  # Testosterone
        "c1ccc2[nH]c(-c3ccccn3)nc2c1",  # Pyridine-benzimidazole
    ]

    print("=== Molecular Descriptors ===")
    for smi in test_smiles:
        descs = compute_descriptors(smi)
        print(f"\n{smi}")
        for k, v in descs.items():
            print(f"  {k}: {v:.2f}")

    print("\n=== Descriptor Matrix (normalized for quantum encoding) ===")
    matrix = compute_descriptor_matrix(test_smiles)
    normalized, scaler = normalize_features(matrix)
    print(f"Shape: {normalized.shape}")
    print(f"Range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    print("\n=== Classical Tanimoto Similarities ===")
    for i in range(len(test_smiles)):
        for j in range(i + 1, len(test_smiles)):
            sim = tanimoto_similarity(test_smiles[i], test_smiles[j])
            print(f"  {test_smiles[i][:20]}... vs {test_smiles[j][:20]}...: {sim:.3f}")
