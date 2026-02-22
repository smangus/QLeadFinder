# QLeadFinder

**Quantum-Enhanced Lead Discovery, Powered by IBM Qiskit + ChEMBL**

QLeadFinder searches ChEMBL's 2.4M+ compound database and re-ranks candidates using a quantum kernel computed via IBM Qiskit. Traditional molecular similarity (Tanimoto fingerprinting) compares molecules like a structural checklist. Quantum kernels measure similarity in an exponentially larger mathematical space, capturing subtle interactions between molecular properties that classical methods miss, potentially surfacing better drug candidates faster.

**Live app:** [qleadfinder.vercel.app](https://qleadfinder.vercel.app) (or your Vercel URL)

Built by [Scriptome.AI](https://scriptome.ai)

---

## How It Works

```
User enters SMILES query
        |
        v
[1. ChEMBL API] ---------- Classical Tanimoto pre-filter across 2.4M+ compounds
        |                   Returns top 50 structurally similar candidates
        v
[2. RDKit Descriptors] ---- MolWt, LogP, HBD, HBA, TPSA, RotBonds, AromaticRings, HeavyAtomCount
        |
        v
[3. Normalize to [0, pi]] - Scale features for quantum gate rotation angles
        |
        v
[4. Qiskit ZZFeatureMap] -- Encode descriptors onto qubits with ZZ entanglement
        |                   Captures pairwise property interactions (e.g., LogP x TPSA)
        v
[5. Quantum Kernel] ------- FidelityQuantumKernel: K(A,B) = |<psi(A)|psi(B)>|^2
        |
        v
[6. Re-ranked Results] ---- Quantum vs Tanimoto rankings side by side
```

The key insight: ChEMBL handles the heavy lifting of searching millions of compounds (fast classical Tanimoto), then the quantum kernel re-ranks just the shortlist. This hybrid approach is both practical and powerful.

## Architecture

```
┌─────────────────────────┐       ┌─────────────────────────────────┐
│   Frontend (Vercel)     │       │   Backend API (Railway)         │
│                         │       │                                 │
│   Next.js / React       │──────>│   FastAPI + Python 3.11         │
│   Tailwind-style CSS    │<──────│   ├── molecular_features.py     │
│                         │       │   ├── quantum_kernel.py         │
│   Features:             │       │   └── similarity_search.py      │
│   - SMILES input        │       │                                 │
│   - Qubit/Layer config  │       │   Calls:                        │
│   - ChEMBL cutoff       │       │   ├── ChEMBL REST API (2.4M+)  │
│   - Results table       │       │   ├── RDKit (descriptors)       │
│   - Rank comparison     │       │   └── Qiskit (quantum kernel)   │
│   - ChEMBL report links │       │                                 │
└─────────────────────────┘       └─────────────────────────────────┘
```

## Project Structure

```
QLeadFinder/
├── api/
│   ├── main.py                 # FastAPI backend (ChEMBL + quantum pipeline)
│   └── requirements.txt        # Python API dependencies
├── frontend/
│   ├── src/app/
│   │   ├── layout.tsx          # App layout with fonts
│   │   ├── page.tsx            # Main search UI component
│   │   └── globals.css         # Dark theme styling
│   ├── public/
│   │   └── scriptome_logo.png  # Scriptome.AI branding
│   ├── package.json
│   └── next.config.js
├── molecular_features.py       # RDKit descriptor extraction + Tanimoto baseline
├── quantum_kernel.py           # Qiskit ZZFeatureMap + FidelityQuantumKernel
├── similarity_search.py        # Standalone CLI pipeline
├── Procfile                    # Railway deployment config
├── runtime.txt                 # Python version for Railway
├── requirements.txt            # Core Python dependencies
└── run_api.sh                  # Local API dev server script
```

## User-Configurable Parameters

| Parameter | Range | What It Does |
|-----------|-------|-------------|
| **Qubits** | 2-8 | Number of molecular descriptors encoded onto the quantum circuit. Each qubit = one property (MolWt, LogP, TPSA, etc.). More qubits = richer comparison. |
| **Layers** | 1-4 | Depth of the quantum circuit. Each layer applies rotation and entanglement gates. More layers = more expressive similarity measure. |
| **ChEMBL Cutoff** | 20-90% | Minimum Tanimoto similarity for the classical pre-filter. Lower = wider net (more diverse candidates); higher = only close structural analogs. |

## Local Development

**Prerequisites:** Python 3.10+, Node.js 18+

### Backend (Terminal 1)

```bash
cd QLeadFinder
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
bash run_api.sh
# API running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Frontend (Terminal 2)

```bash
cd QLeadFinder/frontend
npm install
npm run dev
# App running at http://localhost:3000
```

## Deployment

### Backend on Railway

1. Push repo to GitHub
2. Create a new project at [railway.com](https://railway.com)
3. Deploy from GitHub repo
4. Set Build Command: `pip install -r api/requirements.txt`
5. Set Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. Add variable: `PYTHON_VERSION` = `3.11`

### Frontend on Vercel

1. Import repo at [vercel.com](https://vercel.com)
2. Set Root Directory to `frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL` = your Railway URL
4. Deploy

## Key Technical Details

- **ZZFeatureMap (Qiskit):** Encodes molecular descriptors onto qubits using Hadamard gates, single-qubit Z rotations, and two-qubit ZZ entangling gates. The ZZ interaction captures pairwise feature correlations (e.g., how lipophilicity and polar surface area interact), which is exactly what matters for predicting molecular behavior.

- **FidelityQuantumKernel:** IBM's implementation of quantum kernel similarity via state fidelity. K(A,B) = |<psi(A)|psi(B)>|^2. This measures overlap in Hilbert space, a much richer similarity metric than fingerprint-based Tanimoto.

- **Simulation vs Hardware:** The app runs exact statevector simulation (mathematically identical to real quantum hardware at this qubit count). IBM Quantum hardware becomes necessary beyond ~20-25 qubits where classical simulation becomes intractable.

- **ChEMBL Integration:** Uses the ChEMBL REST API similarity endpoint to pre-filter candidates from 2.4M+ compounds before quantum re-ranking. No local database required.

## Extending QLeadFinder

- **IBM Quantum hardware:** Use `create_hardware_kernel()` with an IBM Quantum token for real QPU execution
- **Custom compound libraries:** Swap the ChEMBL API call for internal screening libraries
- **Activity prediction:** Use `QSVC` from qiskit-machine-learning for quantum-enhanced bioactivity classification
- **Alternative feature maps:** Try PauliFeatureMap or custom circuits for different encoding strategies
- **Larger candidate pools:** Increase `chembl_limit` to 100+ and test scaling behavior

## Dependencies

**Backend:** Python 3.11, FastAPI, Qiskit 2.x, qiskit-machine-learning, RDKit, httpx, scikit-learn, numpy, pandas

**Frontend:** Next.js 14, React 18, TypeScript

## License

MIT
