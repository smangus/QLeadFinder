#!/bin/bash
# Quick test: start the API server locally
# Run from the QLeadFinder root directory with venv activated
cd "$(dirname "$0")"
echo "Starting QLeadFinder API on http://localhost:8000"
echo "Docs at http://localhost:8000/docs"
echo ""
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
