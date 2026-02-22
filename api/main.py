"""
QLeadFinder API
================
FastAPI backend wrapping the quantum molecular similarity search pipeline.
Searches ChEMBL's 2.4M+ compounds via their REST API, then re-ranks
the top candidates using a quantum kernel (IBM Qiskit).
"""

import os
import sys
import time
import numpy as np
from urllib.parse import quote
import httpx
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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QLeadFinder API",
    description="Quantum-Enhanced Molecular Similarity Search powered by IBM Qiskit + ChEMBL",
    version="0.2.0",
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

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


# ---------------------------------------------------------------------------
# ChEMBL integration
# ---------------------------------------------------------------------------
async def chembl_similarity_search(
    smiles: str,
    similarity_cutoff: int = 70,
    limit: int = 50,
) -> list[dict]:
    """
    Query ChEMBL similarity endpoint to find structurally similar compounds.
    Returns list of {chembl_id, name, smiles, similarity} dicts.
    """
    encoded = quote(smiles, safe="")
    url = f"{CHEMBL_BASE}/similarity/{encoded}/{similarity_cutoff}.json"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()

    results = []
    seen_smiles = set()
    for mol in data.get("molecules", []):
        structures = mol.get("molecule_structures") or {}
        canonical = structures.get("canonical_smiles")
        chembl_id = mol.get("molecule_chembl_id")
        pref_name = mol.get("pref_name")
        sim_score = mol.get("similarity")

        if not canonical or not chembl_id:
            continue
        # Skip the query molecule itself
        if canonical == smiles:
            continue
        # Deduplicate
        if canonical in seen_smiles:
            continue
        seen_smiles.add(canonical)

        results.append({
            "chembl_id": chembl_id,
            "name": pref_name or chembl_id,
            "smiles": canonical,
            "chembl_similarity": float(sim_score) if sim_score else None,
        })

    return results


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query_smiles: str = Field(..., description="SMILES string of the query molecule")
    n_qubits: int = Field(default=8, ge=2, le=8, description="Number of qubits (2-8)")
    n_layers: int = Field(default=2, ge=1, le=4, description="Circuit depth (1-4)")
    top_k: int = Field(default=20, ge=1, le=50, description="Number of results to return")
    chembl_cutoff: int = Field(default=70, ge=20, le=95, description="ChEMBL Tanimoto similarity cutoff (%)")
    chembl_limit: int = Field(default=50, ge=10, le=100, description="Max compounds to pull from ChEMBL")


class CompoundResult(BaseModel):
    rank: int
    name: str
    smiles: str
    chembl_id: str | None = None
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
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "qleadfinder-api", "version": "0.2.0", "database": "ChEMBL"}


@app.post("/validate", response_model=ValidateResponse)
def validate_smiles(req: ValidateRequest):
    """Validate a SMILES string and return its descriptors."""
    try:
        mol = smiles_to_mol(req.smiles)
        descs = compute_descriptors(req.smiles)
        return ValidateResponse(
            valid=True,
            descriptors={k: round(v, 2) for k, v in descs.items()},
        )
    except Exception as e:
        return ValidateResponse(valid=False, error=str(e))


@app.post("/search", response_model=SearchResponse)
async def similarity_search(req: SearchRequest):
    """
    Quantum similarity search pipeline:
    1. Query ChEMBL for structurally similar compounds (classical pre-filter)
    2. Compute RDKit descriptors for all candidates
    3. Run quantum kernel (Qiskit ZZFeatureMap) to re-rank
    4. Return both rankings for comparison
    """
    start = time.time()

    # Validate query
    try:
        smiles_to_mol(req.query_smiles)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {req.query_smiles}")

    # Step 1: Pull candidates from ChEMBL
    try:
        chembl_hits = await chembl_similarity_search(
            req.query_smiles,
            similarity_cutoff=req.chembl_cutoff,
            limit=req.chembl_limit,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"ChEMBL API error: {e.response.status_code}",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ChEMBL API unavailable: {str(e)}")

    if not chembl_hits:
        raise HTTPException(
            status_code=404,
            detail="No similar compounds found in ChEMBL. Try lowering the similarity cutoff.",
        )

    # Filter out compounds that fail RDKit parsing
    valid_hits = []
    for hit in chembl_hits:
        try:
            smiles_to_mol(hit["smiles"])
            valid_hits.append(hit)
        except Exception:
            continue

    if not valid_hits:
        raise HTTPException(status_code=404, detail="No valid compounds returned from ChEMBL.")

    lib_names = [h["name"] for h in valid_hits]
    lib_smiles = [h["smiles"] for h in valid_hits]
    lib_chembl_ids = [h["chembl_id"] for h in valid_hits]

    # Step 2: Compute descriptors
    all_smiles = [req.query_smiles] + lib_smiles
    try:
        descriptor_matrix = compute_descriptor_matrix(all_smiles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Descriptor computation failed: {str(e)}")

    descriptor_matrix = descriptor_matrix[:, : req.n_qubits]

    # Step 3: Normalize for quantum encoding
    normalized, scaler = normalize_features(descriptor_matrix)
    query_features = normalized[0:1]
    library_features = normalized[1:]

    # Step 4: Quantum kernel computation
    feature_map = create_zz_feature_map(
        req.n_qubits, reps=req.n_layers, entanglement="linear"
    )
    kernel = create_quantum_kernel(feature_map=feature_map)
    quantum_sims = compute_kernel_matrix(query_features, kernel, Y=library_features).flatten()

    # Step 5: Classical Tanimoto for comparison
    tanimoto_sims = []
    for smi in lib_smiles:
        try:
            tanimoto_sims.append(tanimoto_similarity(req.query_smiles, smi))
        except Exception:
            tanimoto_sims.append(0.0)

    # Step 6: Build ranked results
    desc_names = list(DESCRIPTOR_FUNCTIONS.keys())[: req.n_qubits]

    q_order = np.argsort(-quantum_sims)
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
                chembl_id=lib_chembl_ids[idx],
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
    n_compounds = len(valid_hits)
    all_q_ranks = [q_rank_map[i] for i in range(n_compounds)]
    all_t_ranks = [t_rank_map[i] for i in range(n_compounds)]
    rank_corr = float(np.corrcoef(all_q_ranks, all_t_ranks)[0, 1]) if n_compounds > 1 else 1.0
    top_n = min(req.top_k, n_compounds)
    disagreements = sum(1 for i in range(top_n) if q_order[i] != t_order[i])

    circuit_info = get_circuit_info(feature_map)
    circuit_info.pop("circuit_diagram", None)

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return SearchResponse(
        query_smiles=req.query_smiles,
        query_name=None,
        query_descriptors={
            k: round(float(v), 2)
            for k, v in compute_descriptors(req.query_smiles).items()
        },
        results=results,
        stats={
            "library_size": n_compounds,
            "chembl_cutoff": req.chembl_cutoff,
            "rank_correlation": round(rank_corr, 3),
            "disagreements": disagreements,
            "n_qubits": req.n_qubits,
            "n_layers": req.n_layers,
        },
        circuit_info=circuit_info,
        computation_time_ms=elapsed_ms,
    )
