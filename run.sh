#!/bin/bash
# Tabular ML Lab — Run Script (macOS/Linux)
set -e

echo "🔬 Starting Tabular ML Lab..."

# Find virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Run ./setup.sh first to set up the environment."
    exit 1
fi

echo "🌐 Opening at http://localhost:8501"
streamlit run app.py --server.headless true
