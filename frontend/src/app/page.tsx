"use client";

import { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_QUERIES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" },
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Sorafenib", smiles: "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1" },
];

interface CompoundResult {
  rank: number;
  name: string;
  smiles: string;
  chembl_id: string | null;
  quantum_similarity: number;
  tanimoto_similarity: number;
  quantum_rank: number;
  tanimoto_rank: number;
  rank_changed: boolean;
  descriptors: Record<string, number>;
}

interface SearchResult {
  query_smiles: string;
  query_name: string | null;
  query_descriptors: Record<string, number>;
  results: CompoundResult[];
  stats: {
    library_size: number;
    chembl_cutoff: number;
    rank_correlation: number;
    disagreements: number;
    n_qubits: number;
    n_layers: number;
  };
  circuit_info: {
    num_qubits: number;
    num_parameters: number;
    depth: number;
  };
  computation_time_ms: number;
}

type LoadingStep = {
  label: string;
  status: "pending" | "active" | "done";
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [nQubits, setNQubits] = useState(8);
  const [nLayers, setNLayers] = useState(2);
  const [chemblCutoff, setChemblCutoff] = useState(70);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validation, setValidation] = useState<{
    valid: boolean;
    name?: string;
  } | null>(null);
  const [loadingSteps, setLoadingSteps] = useState<LoadingStep[]>([]);

  // Validate SMILES on input change (debounced)
  useEffect(() => {
    if (!query.trim()) {
      setValidation(null);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/validate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles: query.trim() }),
        });
        const data = await res.json();
        setValidation(data);
      } catch {
        // API might not be running yet
        setValidation(null);
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [query]);

  const runSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const steps: LoadingStep[] = [
      { label: "[1/5] Computing molecular descriptors...", status: "active" },
      { label: "[2/5] Normalizing features to [0, pi]...", status: "pending" },
      { label: "[3/5] Encoding onto quantum circuit...", status: "pending" },
      { label: "[4/5] Computing quantum kernel similarities...", status: "pending" },
      { label: "[5/5] Comparing with Tanimoto baseline...", status: "pending" },
    ];
    setLoadingSteps(steps);

    // Animate through steps
    const stepInterval = setInterval(() => {
      setLoadingSteps((prev) => {
        const activeIdx = prev.findIndex((s) => s.status === "active");
        if (activeIdx < prev.length - 1) {
          const next = [...prev];
          next[activeIdx] = { ...next[activeIdx], status: "done" };
          next[activeIdx + 1] = { ...next[activeIdx + 1], status: "active" };
          return next;
        }
        return prev;
      });
    }, 800);

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_smiles: query.trim(),
          n_qubits: nQubits,
          n_layers: nLayers,
          top_k: 20,
          chembl_cutoff: chemblCutoff,
          chembl_limit: 50,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `API error: ${res.status}`);
      }

      const data: SearchResult = await res.json();
      clearInterval(stepInterval);
      setLoadingSteps((prev) => prev.map((s) => ({ ...s, status: "done" as const })));

      // Small delay to show final step completing
      setTimeout(() => {
        setResult(data);
        setLoading(false);
      }, 400);
    } catch (err: any) {
      clearInterval(stepInterval);
      setError(err.message || "Failed to connect to QLeadFinder API");
      setLoading(false);
    }
  }, [query, nQubits, nLayers, chemblCutoff]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !loading) runSearch();
  };

  const maxQuantumSim = result
    ? Math.max(...result.results.map((r) => r.quantum_similarity))
    : 1;
  const maxTanimotoSim = result
    ? Math.max(...result.results.map((r) => r.tanimoto_similarity))
    : 1;

  return (
    <div className="page">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo-icon">Q</div>
          <div>
            <h1>QLeadFinder</h1>
            <div className="header-subtitle">
              Quantum-Enhanced Lead Discovery
            </div>
          </div>
        </div>
        <div className="header-badges">
          <span className="badge">IBM Qiskit</span>
          <span className="badge">RDKit</span>
          <span className="badge">ChEMBL 2.4M+</span>
          <span className="badge">
            {nQubits} Qubits
          </span>
        </div>
      </header>

      {/* Search */}
      <section className="search-section">
        <div className="search-card">
          <span className="search-label">Query Molecule (SMILES)</span>
          <div className="search-row">
            <div className="search-input-wrap">
              <input
                className={`search-input ${
                  validation
                    ? validation.valid
                      ? "valid"
                      : "invalid"
                    : ""
                }`}
                type="text"
                placeholder="Enter a SMILES string, e.g. CC(=O)Oc1ccccc1C(=O)O"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
              />
              {validation && (
                <div
                  className={`validation-msg ${
                    validation.valid ? "valid" : "invalid"
                  }`}
                >
                  {validation.valid
                    ? `Valid molecule${validation.name ? ` (${validation.name})` : ""}`
                    : "Invalid SMILES string"}
                </div>
              )}
            </div>
            <button
              className="search-btn"
              onClick={runSearch}
              disabled={loading || !query.trim()}
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </div>

          <div className="search-helpers">
            <span className="helper-label">Try:</span>
            {EXAMPLE_QUERIES.map((ex) => (
              <button
                key={ex.name}
                className="helper-chip"
                onClick={() => setQuery(ex.smiles)}
              >
                {ex.name}
              </button>
            ))}
          </div>

          <div className="settings-row">
            <div className="setting">
              <label>Qubits:</label>
              <select
                value={nQubits}
                onChange={(e) => setNQubits(Number(e.target.value))}
              >
                {[2, 3, 4, 5, 6, 7, 8].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <div className="setting">
              <label>Layers:</label>
              <select
                value={nLayers}
                onChange={(e) => setNLayers(Number(e.target.value))}
              >
                {[1, 2, 3, 4].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <div className="setting">
              <label>ChEMBL Cutoff:</label>
              <select
                value={chemblCutoff}
                onChange={(e) => setChemblCutoff(Number(e.target.value))}
              >
                {[40, 50, 60, 70, 80, 90].map((n) => (
                  <option key={n} value={n}>
                    {n}%
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="param-descriptions">
            <div className="param-item">
              <span className="param-name">Qubits</span>
              <span className="param-desc">
                Number of molecular properties encoded onto the quantum circuit. Each qubit represents one descriptor (e.g., molecular weight, lipophilicity, polar surface area). More qubits = richer comparison, but slower.
              </span>
            </div>
            <div className="param-item">
              <span className="param-name">Layers</span>
              <span className="param-desc">
                Depth of the quantum circuit. Each layer applies rotation and entanglement gates that capture interactions between molecular properties. More layers = more expressive similarity measure.
              </span>
            </div>
            <div className="param-item">
              <span className="param-name">ChEMBL Cutoff</span>
              <span className="param-desc">
                Minimum structural similarity (Tanimoto) to pre-filter candidates from ChEMBL{"'"} 2.4M+ compound database. Lower values cast a wider net; higher values return only close structural analogs.
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Loading State */}
      {loading && (
        <div className="loading-container">
          <div className="spinner" />
          <div className="loading-text">
            Running quantum similarity search...
          </div>
          <div className="loading-steps">
            {loadingSteps.map((step, i) => (
              <div key={i} className={`loading-step ${step.status}`}>
                {step.status === "done" ? "\u2713" : step.status === "active" ? "\u25B6" : "\u00A0\u00A0"}{" "}
                {step.label}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="error-card">
          {error}
          {error.includes("Failed to connect") && (
            <div style={{ marginTop: "0.5rem", fontSize: "0.8rem", opacity: 0.8 }}>
              Make sure the API is running: <code>bash run_api.sh</code>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          {/* Stats */}
          <div className="stats-bar">
            <div className="stat-card">
              <div className="stat-value">
                {result.stats.disagreements}/{result.results.length}
              </div>
              <div className="stat-label">Ranking Disagreements</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {result.stats.rank_correlation.toFixed(2)}
              </div>
              <div className="stat-label">Rank Correlation</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{result.stats.library_size}</div>
              <div className="stat-label">ChEMBL Hits (&gt;{result.stats.chembl_cutoff}%)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {result.computation_time_ms < 1000
                  ? `${result.computation_time_ms.toFixed(0)}ms`
                  : `${(result.computation_time_ms / 1000).toFixed(1)}s`}
              </div>
              <div className="stat-label">Computation Time</div>
            </div>
          </div>

          {/* Results Table */}
          <section className="results-section">
            <div className="results-header">
              <div className="results-title">
                Similar Compounds
                {result.query_name && (
                  <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                    {" "}to {result.query_name}
                  </span>
                )}
              </div>
              <div className="results-meta">
                {result.circuit_info.num_qubits}q / depth{" "}
                {result.circuit_info.depth}
              </div>
            </div>

            <div className="results-table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Compound</th>
                    <th>Quantum Sim</th>
                    <th>Tanimoto Sim</th>
                    <th>Rank Change</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r) => (
                    <tr key={r.rank}>
                      <td className="rank-num">{r.rank}</td>
                      <td>
                        <div className="compound-name">
                          {r.chembl_id ? (
                            <a
                              href={`https://www.ebi.ac.uk/chembl/compound_report_card/${r.chembl_id}/`}
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{ color: "var(--text-primary)", textDecoration: "none" }}
                            >
                              {r.name}{" "}
                              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                                {r.chembl_id}
                              </span>
                            </a>
                          ) : (
                            r.name
                          )}
                        </div>
                        <div className="compound-smiles">{r.smiles}</div>
                      </td>
                      <td>
                        <div className="sim-bar-wrap">
                          <span className="sim-value" style={{ color: "var(--green)" }}>
                            {r.quantum_similarity.toFixed(4)}
                          </span>
                          <div className="sim-bar">
                            <div
                              className="sim-bar-fill quantum"
                              style={{
                                width: `${(r.quantum_similarity / maxQuantumSim) * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="sim-bar-wrap">
                          <span className="sim-value" style={{ color: "var(--blue)" }}>
                            {r.tanimoto_similarity.toFixed(4)}
                          </span>
                          <div className="sim-bar">
                            <div
                              className="sim-bar-fill tanimoto"
                              style={{
                                width: `${(r.tanimoto_similarity / maxTanimotoSim) * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      </td>
                      <td>
                        {r.rank_changed ? (
                          <span className="rank-delta changed">
                            T#{r.tanimoto_rank} → Q#{r.quantum_rank}
                          </span>
                        ) : (
                          <span className="rank-delta same">&mdash;</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Query descriptors */}
          <section style={{ marginTop: "1.5rem" }}>
            <div
              className="results-header"
              style={{ marginBottom: "0.75rem" }}
            >
              <div className="results-title">Query Descriptors</div>
            </div>
            <div
              style={{
                display: "flex",
                gap: "0.5rem",
                flexWrap: "wrap",
              }}
            >
              {Object.entries(result.query_descriptors).map(([key, val]) => (
                <div
                  key={key}
                  style={{
                    background: "var(--bg-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "6px",
                    padding: "0.5rem 0.75rem",
                    fontSize: "0.8rem",
                  }}
                >
                  <span style={{ color: "var(--text-muted)" }}>{key}</span>{" "}
                  <span
                    className="mono"
                    style={{ color: "var(--green)", fontWeight: 600 }}
                  >
                    {val}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {/* Footer */}
      <footer className="footer">
        <div className="footer-divider" />
        <div className="footer-content">
          <div className="footer-built">
            <span className="footer-label">Built by</span>
            <a href="https://scriptome.ai" target="_blank" rel="noopener noreferrer" className="footer-logo-link">
              <img src="/scriptome_logo.png" alt="Scriptome.AI" className="footer-logo" />
            </a>
          </div>
          <div className="footer-tech">
            Powered by IBM Qiskit &middot; RDKit &middot; ChEMBL &middot; Next.js
          </div>
        </div>
      </footer>
    </div>
  );
}
