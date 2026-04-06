#!/bin/bash
# Tabular ML Lab — Setup Script (macOS/Linux)
set -e

echo "🔬 Setting up Tabular ML Lab..."

# ── Try uv first (handles Python version automatically) ──────────
if command -v uv &>/dev/null; then
    echo "📦 Found uv — setting up with Python 3.12..."
    uv venv --python 3.12 .venv
    source .venv/bin/activate

    echo "📥 Installing dependencies..."
    uv pip install -r requirements.txt

    # Install optional packages that need Python <=3.12
    echo "📥 Installing optional packages (TDA, UMAP)..."
    uv pip install giotto-tda umap-learn 2>/dev/null || \
        echo "⚠️  Optional packages (giotto-tda, umap-learn) failed — TDA/UMAP features will be unavailable."

    mkdir -p .cache
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "To run:  source .venv/bin/activate && streamlit run app.py"
    echo "Or:      ./run.sh"

# ── Fallback to pip ───────────────────────────────────────────────
else
    echo "📦 uv not found — using pip (install uv for best experience: https://docs.astral.sh/uv)"
    echo ""

    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate

    pip install --upgrade pip
    echo "📥 Installing dependencies..."
    if pip install -r requirements.txt; then
        echo ""
        echo "✅ Setup complete!"
    else
        echo ""
        echo "⚠️  Some dependencies failed to install."
        echo "Core features still work. Optional packages (giotto-tda, umap-learn) require Python <=3.12."
        echo "Install uv (https://docs.astral.sh/uv) to automatically use the right Python version."
    fi

    mkdir -p .cache
    echo ""
    echo "To run:  source venv/bin/activate && streamlit run app.py"
    echo "Or:      ./run.sh"
fi

echo ""
echo "Optional: For AI-powered interpretation, install and run Ollama:"
echo "  https://ollama.ai → ollama serve → ollama pull qwen3.5:9b"
echo "  Or use OpenAI/Anthropic API keys in the app sidebar."
