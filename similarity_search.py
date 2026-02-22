"""
Quantum Molecular Similarity Search Pipeline
==============================================
End-to-end pipeline: takes a query molecule (SMILES) and a library of
compounds, computes quantum kernel similarities, and returns ranked results
alongside classical Tanimoto baseline for comparison.
"""

import numpy as np
import pandas as pd
from molecular_features import (
    compute_descriptor_matrix,
    normalize_features,
    tanimoto_similarity,
    compute_descriptors,
    DESCRIPTOR_FUNCTIONS,
)
from quantum_kernel import create_zz_feature_map, create_quantum_kernel, compute_kernel_matrix


# ---------------------------------------------------------------------------
# Sample compound library (replace with ChEMBL pull for production)
# ---------------------------------------------------------------------------
SAMPLE_LIBRARY = {
    "Aspirin":           "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen":         "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Acetaminophen":     "CC(=O)Nc1ccc(O)cc1",
    "Naproxen":          "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "Celecoxib":         "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Diclofenac":        "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Indomethacin":      "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
    "Piroxicam":         "OC1=C(C(=O)N2CCCCN2C)N(C)S(=O)(=O)c2ccccc21",
    "Meloxicam":         "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
    "Sulfasalazine":     "OC(=O)c1cc(/N=N/c2ccc(cc2)S(=O)(=O)Nc2ccccn2)ccc1O",
    "Methotrexate":      "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "Dexamethasone":     "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "Caffeine":          "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Theophylline":      "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "Theobromine":       "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "Quercetin":         "OC1=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c2C1=O",
    "Resveratrol":       "OC1=CC=C(/C=C/C2=CC(O)=CC(O)=C2)C=C1",
    "Curcumin":          "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
    "Sorafenib":         "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Imatinib":          "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}


def run_similarity_search(
    query_smiles: str,
    library: dict[str, str] = None,
    n_qubits: int = None,
    n_layers: int = 2,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Run quantum similarity search against a compound library.
    
    Args:
        query_smiles: SMILES string of the query/lead molecule
        library: Dict of {name: SMILES}. Uses sample library if None.
        n_qubits: Number of qubits (= number of descriptors to use).
                  Defaults to all available descriptors.
        n_layers: Quantum circuit depth
        top_k: Number of top results to return
    
    Returns:
        DataFrame with columns: Name, SMILES, QuantumSimilarity, 
        TanimotoSimilarity, and molecular descriptors
    """
    if library is None:
        library = SAMPLE_LIBRARY

    if n_qubits is None:
        n_qubits = len(DESCRIPTOR_FUNCTIONS)

    lib_names = list(library.keys())
    lib_smiles = list(library.values())

    print(f"[Pipeline] Query: {query_smiles}")
    print(f"[Pipeline] Library size: {len(lib_smiles)} compounds")
    print(f"[Pipeline] Qubits: {n_qubits}, Layers: {n_layers}")

    # --- Step 1: Compute descriptors ---
    print("[Step 1] Computing molecular descriptors...")
    all_smiles = [query_smiles] + lib_smiles
    descriptor_matrix = compute_descriptor_matrix(all_smiles)

    # Use only first n_qubits descriptors
    descriptor_matrix = descriptor_matrix[:, :n_qubits]

    # --- Step 2: Normalize for quantum encoding ---
    print("[Step 2] Normalizing features to [0, pi]...")
    normalized, scaler = normalize_features(descriptor_matrix)

    query_features = normalized[0:1]      # Shape (1, n_qubits)
    library_features = normalized[1:]     # Shape (n_lib, n_qubits)

    # --- Step 3: Quantum kernel computation (IBM Qiskit) ---
    print("[Step 3] Computing quantum kernel similarities (Qiskit ZZFeatureMap)...")
    feature_map = create_zz_feature_map(n_qubits, reps=n_layers, entanglement="linear")
    kernel = create_quantum_kernel(feature_map=feature_map)
    quantum_sims = compute_kernel_matrix(query_features, kernel, Y=library_features)
    quantum_sims = quantum_sims.flatten()  # Shape (n_lib,)

    # --- Step 4: Classical Tanimoto baseline ---
    print("[Step 4] Computing classical Tanimoto similarities...")
    tanimoto_sims = [tanimoto_similarity(query_smiles, smi) for smi in lib_smiles]

    # --- Step 5: Compile results ---
    print("[Step 5] Compiling results...")
    descs_df = pd.DataFrame(
        descriptor_matrix[1:, :],
        columns=list(DESCRIPTOR_FUNCTIONS.keys())[:n_qubits],
    )

    results = pd.DataFrame({
        "Name": lib_names,
        "SMILES": lib_smiles,
        "Quantum_Similarity": quantum_sims,
        "Tanimoto_Similarity": tanimoto_sims,
    })
    results = pd.concat([results, descs_df], axis=1)
    results = results.sort_values("Quantum_Similarity", ascending=False)

    return results.head(top_k).reset_index(drop=True)


def print_results(results: pd.DataFrame, query_smiles: str):
    """Pretty-print similarity search results."""
    print("\n" + "=" * 80)
    print(f"QLEADFINDER - QUANTUM MOLECULAR SIMILARITY SEARCH RESULTS")
    print(f"Query: {query_smiles}")
    print("=" * 80)

    for idx, row in results.iterrows():
        print(f"\n#{idx + 1}: {row['Name']}")
        print(f"    SMILES:    {row['SMILES']}")
        print(f"    Quantum:   {row['Quantum_Similarity']:.4f}")
        print(f"    Tanimoto:  {row['Tanimoto_Similarity']:.4f}")
        q_rank = idx + 1
        t_rank = (
            results.sort_values("Tanimoto_Similarity", ascending=False)
            .reset_index(drop=True)
            .index[results.sort_values("Tanimoto_Similarity", ascending=False)
            .reset_index(drop=True)["Name"] == row["Name"]]
            .tolist()
        )
        t_rank = t_rank[0] + 1 if t_rank else "?"
        if q_rank != t_rank:
            print(f"    ** Rank differs: Quantum=#{q_rank}, Tanimoto=#{t_rank}")

    # Summary statistics
    print("\n" + "-" * 80)
    corr = results["Quantum_Similarity"].corr(results["Tanimoto_Similarity"])
    print(f"Quantum vs Tanimoto rank correlation: {corr:.3f}")
    disagreements = sum(
        results.sort_values("Quantum_Similarity", ascending=False)
        .reset_index(drop=True)["Name"].values
        != results.sort_values("Tanimoto_Similarity", ascending=False)
        .reset_index(drop=True)["Name"].values
    )
    print(f"Ranking disagreements in top {len(results)}: {disagreements}")
    print("-" * 80)


# ---- Main demo ----
if __name__ == "__main__":
    # Query: Find molecules similar to Aspirin
    query = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

    print("=" * 80)
    print("QLEADFINDER - QUANTUM MOLECULAR SIMILARITY SEARCH")
    print("QLeadFinder - Quantum-Enhanced Lead Discovery")
    print("=" * 80)

    results = run_similarity_search(
        query_smiles=query,
        n_qubits=8,    # Use all 8 descriptors = 8 qubits
        n_layers=2,     # 2 encoding layers
        top_k=10,       # Return top 10 similar
    )

    print_results(results, query)

    # Also show query descriptors
    print("\nQuery Molecule Descriptors:")
    descs = compute_descriptors(query)
    for k, v in descs.items():
        print(f"  {k}: {v:.2f}")
