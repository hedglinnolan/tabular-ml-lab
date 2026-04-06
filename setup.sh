#!/bin/bash
# Tabular ML Lab — Setup Script
set -e

echo "🔬 Setting up Tabular ML Lab..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
if pip install -r requirements.txt; then
    echo ""
    echo "✅ Setup complete!"
else
    echo ""
    echo "⚠️  Some dependencies failed to install."
    echo "The app may still work — optional packages (giotto-tda, umap-learn) require Python <=3.12."
    echo "Core features work on Python 3.10-3.13."
fi

# Create cache directory
mkdir -p .cache

echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Optional: For AI-powered interpretation, install and run Ollama:"
echo "  https://ollama.ai → ollama serve → ollama pull qwen3.5:9b"
echo "  Or use OpenAI/Anthropic API keys in the app sidebar."
