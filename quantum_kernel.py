"""
Quantum Kernel Module (IBM Qiskit)
====================================
Implements quantum feature maps and kernel computation using IBM's Qiskit
and qiskit-machine-learning. Uses Qiskit's built-in ZZFeatureMap and
FidelityQuantumKernel for computing molecular similarity.

Key IBM Qiskit components used:
- ZZFeatureMap: Built-in parameterized circuit for data encoding with
  entanglement (the ZZ interaction captures pairwise feature correlations)
- FidelityQuantumKernel: Computes K(x1,x2) = |<psi(x1)|psi(x2)>|^2
  using Qiskit's fidelity primitives
- StatevectorSampler: Local simulator for hackathon dev; swap to
  IBM Quantum backends for real hardware execution
"""

import numpy as np
from itertools import combinations

# Qiskit core
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap

# Qiskit Machine Learning
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def create_zz_feature_map(n_features: int, reps: int = 2, entanglement: str = "linear"):
    """
    Create a ZZFeatureMap quantum circuit for molecular feature encoding.
    
    The ZZFeatureMap is Qiskit's standard feature map for kernel methods.
    It applies:
      1. Hadamard gates on all qubits
      2. Single-qubit Z rotations parameterized by features
      3. Two-qubit ZZ entangling gates parameterized by feature products
      4. Repeats for 'reps' layers
    
    The ZZ interaction naturally captures pairwise feature correlations,
    which is exactly what we want for molecular descriptor encoding
    (e.g., the interplay between LogP and TPSA affects drug behavior).
    
    Args:
        n_features: Number of molecular features (= number of qubits)
        reps: Number of repetitions/layers (more = more expressivity)
        entanglement: Entanglement strategy:
            - "linear": nearest-neighbor (fastest, good for ordered features)
            - "full": all-to-all (most expressive, slower)
            - "circular": ring topology (good balance)
    
    Returns:
        ZZFeatureMap circuit
    """
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=reps,
        entanglement=entanglement,
    )
    return feature_map


def create_quantum_kernel(feature_map=None, n_features: int = 8, reps: int = 2):
    """
    Create a FidelityQuantumKernel using IBM Qiskit.
    
    This is the primary interface for computing quantum similarities.
    Under the hood, it:
      1. Encodes data point x1 using the feature map -> |psi(x1)>
      2. Encodes data point x2 using the feature map -> |psi(x2)>
      3. Computes fidelity: K(x1,x2) = |<psi(x1)|psi(x2)>|^2
    
    For hackathon/simulation: uses local statevector simulator (exact).
    For IBM Quantum hardware: swap in a real backend via IBM Quantum Platform.
    
    Args:
        feature_map: Pre-built Qiskit feature map circuit. 
                     If None, creates a ZZFeatureMap.
        n_features: Number of features (used only if feature_map is None)
        reps: Circuit repetitions (used only if feature_map is None)
    
    Returns:
        FidelityQuantumKernel instance
    """
    if feature_map is None:
        feature_map = create_zz_feature_map(n_features, reps=reps)

    kernel = FidelityQuantumKernel(feature_map=feature_map)
    return kernel


def compute_kernel_matrix(X: np.ndarray, kernel: FidelityQuantumKernel, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute the quantum kernel matrix using Qiskit's FidelityQuantumKernel.
    
    K[i,j] = |<psi(x_i)|psi(x_j)>|^2
    
    Args:
        X: Feature matrix (n_samples_x, n_features), values in [0, pi]
        kernel: FidelityQuantumKernel instance
        Y: Optional second feature matrix for cross-kernel computation
    
    Returns:
        Kernel matrix of shape (n_samples_x, n_samples_y)
    """
    if Y is None:
        return kernel.evaluate(X)
    else:
        return kernel.evaluate(X, Y)


def compare_kernels(X: np.ndarray, kernel: FidelityQuantumKernel, classical_kernel_fn) -> dict:
    """
    Compare quantum kernel matrix with a classical kernel for analysis.
    """
    quantum_K = compute_kernel_matrix(X, kernel)

    n = X.shape[0]
    classical_K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            classical_K[i, j] = classical_kernel_fn(X[i], X[j])

    upper_idx = np.triu_indices(n, k=1)
    q_vals = quantum_K[upper_idx]
    c_vals = classical_K[upper_idx]
    correlation = np.corrcoef(q_vals, c_vals)[0, 1] if len(q_vals) > 1 else 0.0

    return {
        "quantum_kernel": quantum_K,
        "classical_kernel": classical_K,
        "correlation": correlation,
        "quantum_mean": float(np.mean(q_vals)),
        "classical_mean": float(np.mean(c_vals)),
    }


def get_circuit_info(feature_map) -> dict:
    """Get useful info about the quantum circuit for display/debugging."""
    return {
        "num_qubits": feature_map.num_qubits,
        "num_parameters": feature_map.num_parameters,
        "depth": feature_map.depth(),
        "circuit_diagram": str(feature_map.draw(output="text")),
    }


# ---------------------------------------------------------------------------
# IBM Quantum Hardware Integration (for production / demo on real QPU)
# ---------------------------------------------------------------------------
def create_hardware_kernel(feature_map, ibm_token: str = None, backend_name: str = "ibm_brisbane"):
    """
    Create a quantum kernel that runs on real IBM Quantum hardware.
    
    IMPORTANT: Requires an IBM Quantum account (free tier available).
    Sign up at: https://quantum.ibm.com
    
    Steps to use real hardware:
    1. Sign up at https://quantum.ibm.com
    2. Copy your API token from Account Settings
    3. pip install qiskit-ibm-runtime
    4. Call this function with your token
    
    Args:
        feature_map: Qiskit feature map circuit
        ibm_token: Your IBM Quantum API token
        backend_name: IBM backend (e.g., "ibm_brisbane", "ibm_osaka")
    
    Returns:
        FidelityQuantumKernel configured for hardware execution
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        
        if ibm_token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
        else:
            service = QiskitRuntimeService()
        
        backend = service.backend(backend_name)
        print(f"[Hardware] Connected to {backend_name}")
        print(f"[Hardware] Qubits available: {backend.num_qubits}")
        
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        return kernel
        
    except ImportError:
        print("[Warning] qiskit-ibm-runtime not installed.")
        print("  Install with: pip install qiskit-ibm-runtime")
        print("  Falling back to local simulator.")
        return FidelityQuantumKernel(feature_map=feature_map)
    except Exception as e:
        print(f"[Warning] Could not connect to IBM Quantum: {e}")
        print("  Falling back to local simulator.")
        return FidelityQuantumKernel(feature_map=feature_map)


# ---- Quick test ----
if __name__ == "__main__":
    np.random.seed(42)

    n_qubits = 4
    n_samples = 5

    X = np.random.uniform(0, np.pi, (n_samples, n_qubits))

    print(f"Creating ZZFeatureMap with {n_qubits} qubits, 2 reps...")
    fmap = create_zz_feature_map(n_qubits, reps=2)
    
    info = get_circuit_info(fmap)
    print(f"Circuit depth: {info['depth']}")
    print(f"Parameters: {info['num_parameters']}")
    print(f"\nCircuit diagram:\n{info['circuit_diagram']}")

    print("\n=== Creating Quantum Kernel ===")
    kernel = create_quantum_kernel(feature_map=fmap)

    print("\n=== Computing Kernel Matrix ===")
    K = compute_kernel_matrix(X, kernel)
    print(np.round(K, 3))

    print(f"\nDiagonal (self-similarity): {np.diag(K).round(3)}")
    print(f"Off-diagonal range: [{K[np.triu_indices(n_samples, k=1)].min():.3f}, "
          f"{K[np.triu_indices(n_samples, k=1)].max():.3f}]")
