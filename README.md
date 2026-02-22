# QLeadFinder

**Quantum-Enhanced Lead Discovery, Powered by IBM Qiskit**

## What It Does

Given a lead molecule (as a SMILES string), this tool finds the most similar compounds in a library using a **quantum kernel** - a similarity measure computed in the exponentially large Hilbert space of a quantum system. Results are compared against classical Tanimoto fingerprint similarity to highlight where quantum approaches surface different (potentially more meaningful) rankings.

## Architecture

```
SMILES Input
    |
    v
[Classical Preprocessing] -- RDKit descriptors (MolWt, LogP, TPSA, etc.)
    |
    v
[Feature Normalization] --- Scale to [0, pi] for quantum gate angles
    |
    v
[Quantum Feature Map] ----- Qiskit ZZFeatureMap: Hadamard + Z rotations + ZZ entanglement
    |
    v
[Quantum Kernel] ---------- FidelityQuantumKernel: K(A,B) = |<psi(A)|psi(B)>|^2
    |
    v
[Ranked Results] ---------- Top-K similar molecules + comparison with Tanimoto baseline
```

## Files

| File | Purpose |
|------|---------|
| `src/molecular_features.py` | RDKit descriptor computation, normalization, classical fingerprints |
| `src/quantum_kernel.py` | IBM Qiskit ZZFeatureMap, FidelityQuantumKernel, hardware integration |
| `src/similarity_search.py` | End-to-end pipeline with sample compound library |

## Quick Start

```bash
# Install dependencies
pip install rdkit qiskit qiskit-machine-learning scikit-learn numpy pandas

# Run the demo (searches for molecules similar to Aspirin)
cd src
python similarity_search.py
```

## Key Concepts

- **ZZFeatureMap (Qiskit)**: Encodes molecular descriptors onto qubits using Z rotations and ZZ entangling gates. The ZZ interaction captures pairwise feature correlations (e.g., LogP x TPSA).
- **FidelityQuantumKernel**: IBM's implementation of quantum kernel similarity via state fidelity. K(A,B) = |<psi(A)|psi(B)>|^2.
- **Hybrid Approach**: Classical preprocessing (RDKit) + quantum similarity (Qiskit) + classical ranking.

## Running on IBM Quantum Hardware

```python
# 1. Sign up at https://quantum.ibm.com (free tier available)
# 2. pip install qiskit-ibm-runtime
# 3. Use the hardware kernel:

from quantum_kernel import create_zz_feature_map, create_hardware_kernel

feature_map = create_zz_feature_map(n_features=8, reps=2)
kernel = create_hardware_kernel(feature_map, ibm_token="YOUR_TOKEN", backend_name="ibm_brisbane")
```

## Extending This

- **Connect to ChEMBL**: Pull real compound libraries via the ChEMBL API
- **Build a frontend**: Streamlit or React app with SMILES input + results table
- **Run on IBM Quantum hardware**: Use `create_hardware_kernel()` with your IBM Quantum token
- **Activity prediction**: Use `QSVC` from qiskit-machine-learning for bioactivity classification
- **Experiment with feature maps**: Try PauliFeatureMap or custom circuits for different encoding strategies
- **Scale features**: Experiment with different descriptor subsets and circuit depths

## Dependencies

- Python 3.10+
- Qiskit 2.x (quantum circuit SDK)
- qiskit-machine-learning (FidelityQuantumKernel, QSVC)
- RDKit (molecular chemistry toolkit)
- scikit-learn, numpy, pandas
- Optional: qiskit-ibm-runtime (for IBM Quantum hardware access)
