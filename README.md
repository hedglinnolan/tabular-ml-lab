# 🔬 Tabular Machine Learning Lab

**Publication-grade machine learning for tabular research data.**

A guided, interactive Streamlit application for researchers working with tabular data who need defensible methodology and publication-ready outputs. Built for nutrition scientists, epidemiologists, biostatisticians, and anyone who works with structured data and needs to publish papers.

## Features

### Guided Research Workflow
- **Upload & Audit** — Multi-file projects, intelligent merge builder, data quality assessment
- **EDA** — Comprehensive exploration with model-aware diagnostics
- **Feature Selection** — LASSO path, RFE-CV, stability selection, univariate screening with FDR correction
- **Preprocessing** — Per-model sklearn pipelines with intelligent defaults
- **Train & Compare** — Multiple model families with automatic baseline comparison
- **Explainability** — SHAP, permutation importance, calibration, decision curves
- **Report Export** — Methods sections, TRIPOD checklists, publication-quality figures

### Publication-Ready Outputs
- **Table 1** — Stratified descriptive statistics with p-values and SMD
- **Bootstrap 95% CIs** — BCa confidence intervals on all metrics
- **Calibration Analysis** — Reliability diagrams, Brier score, ECE
- **Decision Curve Analysis** — Net benefit curves for clinical utility
- **Subgroup Analysis** — Stratified metrics with forest plots
- **TRIPOD Checklist** — Auto-tracked compliance for prediction model reporting
- **Auto-Generated Methods Section** — Draft text based on actual workflow choices
- **CONSORT-Style Flow Diagrams** — Sample selection visualization
- **Journal-Quality Figures** — Export-ready plots with DPI control

### Intelligent Guidance
- Smart defaults with progressive disclosure (simple by default, advanced when needed)
- AI-powered interpretation (Ollama, OpenAI, or Anthropic)
- Contextual "why does this matter?" help throughout
- Reviewer concern flagging

### Models
- **Linear:** Ridge, Lasso, ElasticNet, Logistic Regression
- **Trees:** Random Forest, ExtraTrees, HistGradientBoosting
- **Distance:** KNN
- **Margin:** SVM (SVR/SVC)
- **Probabilistic:** Naive Bayes, LDA
- **Neural Networks:** PyTorch MLP with configurable architecture and loss functions
- **Automatic Baselines:** Mean/majority-class predictor + simple linear/logistic regression

## Quick Start

```bash
# Clone
git clone https://github.com/hedglinnolan/glucose-mlp-interactive.git
cd glucose-mlp-interactive

# Option A: Automated setup
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
streamlit run app.py

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependencies

## Optional: AI Interpretation

For AI-powered analysis interpretation:

**Ollama (free, local):**
```bash
# Install from ollama.ai
ollama serve
ollama pull llama3.2
```

**OpenAI or Anthropic:** Configure API keys in the app sidebar (🤖 LLM Settings).

## For Researchers

This tool is designed to help you go from raw data to a publication-ready manuscript. The workflow enforces methodological rigor:

1. Proper train/validation/test splits (no data leakage)
2. Bootstrap confidence intervals on all metrics
3. Automatic comparison against null/baseline models
4. Calibration analysis for clinical models
5. TRIPOD compliance tracking
6. Reproducibility manifest (seeds, versions, configurations)

## License

MIT
