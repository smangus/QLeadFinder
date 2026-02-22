"""
QLeadFinder API
================
FastAPI backend wrapping the quantum molecular similarity search pipeline.
Deployed on Railway, called by the Next.js frontend on Vercel.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from molecular_features import (
    compute_descriptor_matrix,
    normalize_features,
    tanimoto_similarity,
    compute_descriptors,
    smiles_to_mol,
    DESCRIPTOR_FUNCTIONS,
)
from quantum_kernel import (
    create_zz_feature_map,
    create_quantum_kernel,
    compute_kernel_matrix,
    get_circuit_info,
)
from similarity_search import SAMPLE_LIBRARY

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QLeadFinder API",
    description="Quantum-Enhanced Molecular Similarity Search powered by IBM Qiskit",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query_smiles: str = Field(..., description="SMILES string of the query molecule")
    n_qubits: int = Field(default=8, ge=2, le=8, description="Number of qubits (2-8)")
    n_layers: int = Field(default=2, ge=1, le=4, description="Circuit depth (1-4)")
    top_k: int = Field(default=10, ge=1, le=20, description="Number of results")


class CompoundResult(BaseModel):
    rank: int
    name: str
    smiles: str
    quantum_similarity: float
    tanimoto_similarity: float
    quantum_rank: int
    tanimoto_rank: int
    rank_changed: bool
    descriptors: dict


class SearchResponse(BaseModel):
    query_smiles: str
    query_name: str | None
    query_descriptors: dict
    results: list[CompoundResult]
    stats: dict
    circuit_info: dict
    computation_time_ms: float


class ValidateRequest(BaseModel):
    smiles: str


class ValidateResponse(BaseModel):
    valid: bool
    name: str | None = None
    descriptors: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Reverse lookup: SMILES -> name
# ---------------------------------------------------------------------------
SMILES_TO_NAME = {v: k for k, v in SAMPLE_LIBRARY.items()}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "qleadfinder-api"}


@app.post("/validate", response_model=ValidateResponse)
def validate_smiles(req: ValidateRequest):
    """Validate a SMILES string and return its descriptors."""
    try:
        mol = smiles_to_mol(req.smiles)
        descs = compute_descriptors(req.smiles)
        name = SMILES_TO_NAME.get(req.smiles)
        return ValidateResponse(
            valid=True,
            name=name,
            descriptors={k: round(v, 2) for k, v in descs.items()},
        )
    except Exception as e:
        return ValidateResponse(valid=False, error=str(e))


@app.post("/search", response_model=SearchResponse)
def similarity_search(req: SearchRequest):
    """Run quantum similarity search against the compound library."""
    start = time.time()

    # Validate query
    try:
        smiles_to_mol(req.query_smiles)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {req.query_smiles}")

    library = SAMPLE_LIBRARY
    lib_names = list(library.keys())
    lib_smiles = list(library.values())

    # Step 1: Compute descriptors
    all_smiles = [req.query_smiles] + lib_smiles
    descriptor_matrix = compute_descriptor_matrix(all_smiles)
    descriptor_matrix = descriptor_matrix[:, : req.n_qubits]

    # Step 2: Normalize
    normalized, scaler = normalize_features(descriptor_matrix)
    query_features = normalized[0:1]
    library_features = normalized[1:]

    # Step 3: Quantum kernel
    feature_map = create_zz_feature_map(
        req.n_qubits, reps=req.n_layers, entanglement="linear"
    )
    kernel = create_quantum_kernel(feature_map=feature_map)
    quantum_sims = compute_kernel_matrix(query_features, kernel, Y=library_features).flatten()

    # Step 4: Tanimoto baseline
    tanimoto_sims = [tanimoto_similarity(req.query_smiles, smi) for smi in lib_smiles]

    # Step 5: Build ranked results
    desc_names = list(DESCRIPTOR_FUNCTIONS.keys())[: req.n_qubits]

    # Quantum ranking
    q_order = np.argsort(-quantum_sims)
    # Tanimoto ranking
    t_order = np.argsort(-np.array(tanimoto_sims))

    q_rank_map = {idx: rank + 1 for rank, idx in enumerate(q_order)}
    t_rank_map = {idx: rank + 1 for rank, idx in enumerate(t_order)}

    results = []
    for rank, idx in enumerate(q_order[: req.top_k]):
        results.append(
            CompoundResult(
                rank=rank + 1,
                name=lib_names[idx],
                smiles=lib_smiles[idx],
                quantum_similarity=round(float(quantum_sims[idx]), 4),
                tanimoto_similarity=round(float(tanimoto_sims[idx]), 4),
                quantum_rank=q_rank_map[idx],
                tanimoto_rank=t_rank_map[idx],
                rank_changed=q_rank_map[idx] != t_rank_map[idx],
                descriptors={
                    desc_names[j]: round(float(descriptor_matrix[idx + 1, j]), 2)
                    for j in range(req.n_qubits)
                },
            )
        )

    # Stats
    all_q_ranks = [q_rank_map[i] for i in range(len(lib_names))]
    all_t_ranks = [t_rank_map[i] for i in range(len(lib_names))]
    rank_corr = float(np.corrcoef(all_q_ranks, all_t_ranks)[0, 1])
    disagreements = sum(1 for i in range(min(req.top_k, len(lib_names))) if q_order[i] != t_order[i])

    # Circuit info
    circuit_info = get_circuit_info(feature_map)
    circuit_info.pop("circuit_diagram", None)  # Too large for JSON

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return SearchResponse(
        query_smiles=req.query_smiles,
        query_name=SMILES_TO_NAME.get(req.query_smiles),
        query_descriptors={
            k: round(float(v), 2)
            for k, v in compute_descriptors(req.query_smiles).items()
        },
        results=results,
        stats={
            "library_size": len(lib_names),
            "rank_correlation": round(rank_corr, 3),
            "disagreements": disagreements,
            "n_qubits": req.n_qubits,
            "n_layers": req.n_layers,
        },
        circuit_info=circuit_info,
        computation_time_ms=elapsed_ms,
    )


@app.get("/library")
def get_library():
    """Return the compound library with names and SMILES."""
    compounds = []
    for name, smiles in SAMPLE_LIBRARY.items():
        try:
            descs = compute_descriptors(smiles)
            compounds.append({
                "name": name,
                "smiles": smiles,
                "descriptors": {k: round(v, 2) for k, v in descs.items()},
            })
        except Exception:
            compounds.append({"name": name, "smiles": smiles, "descriptors": {}})
    return {"compounds": compounds, "count": len(compounds)}
