#!/bin/bash
# Script to launch the Sepsis Streamlit Inference App
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$REPO_ROOT"
source .venv/bin/activate

echo "Starting Streamlit Inference App on http://localhost:8501..."
streamlit run 7_inference/app.py --server.port 8501 --server.address 0.0.0.0
