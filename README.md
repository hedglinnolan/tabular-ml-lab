<p align="center">
  <h1 align="center">🔬 Tabular ML Lab</h1>
  <p align="center">
    <strong>From raw data to a manuscript-ready starting point. No coding required.</strong>
  </p>
  <p align="center">
    <a href="https://app.tabularml.dev">Live Demo</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#features">Features</a> ·
    <a href="https://github.com/hedglinnolan/tabular-ml-lab/issues">Report Bug</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/pytorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </p>
  <p align="center">
    <a href="https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker">
      <img src="https://img.shields.io/badge/🎓_University_Deployment-Docker_+_OIDC_Auth-0066cc?style=for-the-badge" alt="University Deployment">
    </a>
  </p>
</p>

---

> 🎓 **University IT Administrators:** Looking to deploy for your institution? Check out the [**university-docker branch**](https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker) for Docker deployment with KeyCloak OIDC authentication and institutional LLM integration. Complete setup guide included.

---

An interactive research workbench for scientists who work with tabular data and need to publish papers. Upload your CSV, and the app guides you through a complete, defensible ML workflow — from exploratory analysis to a compilable LaTeX manuscript draft with auto-generated methods, results, and structured discussion.

**Built for researchers, not ML engineers.** The app does the mechanical work of writing a prediction model paper. Your only edits are domain-specific context no tool can provide.

> 🌐 **Try it now:** [app.tabularml.dev](https://app.tabularml.dev) (Note: I am actively developing a new branch and I use this demo website to test out new features. Functionality may break once in a while.)

## Features

### 📋 10-Step Guided Workflow

| Step | Page | What it does |
|------|------|-------------|
| 1 | **Upload & Audit** | Load CSVs/Excel, merge multiple files, data quality checks |
| 2 | **EDA** | Distributions, correlations, Table 1, missing data analysis, interactive decision hub |
| 3 | **Feature Engineering** | PCA, polynomial features, log transforms, ratios, binning, TDA features |
| 4 | **Feature Selection** | LASSO path, RFE-CV, univariate, stability selection, consensus ranking |
| 5 | **Preprocess** | Per-model pipelines: imputation, scaling, encoding, outlier handling, power transforms |
| 6 | **Train & Compare** | 18 model families with bootstrap CIs, baseline comparison, optional Optuna optimization |
| 7 | **Explainability** | SHAP, permutation importance, PDP, calibration, decision curves, subgroup analysis |
| 8 | **Sensitivity Analysis** | Seed robustness, feature dropout — prove your results aren't fragile |
| 9 | **Statistical Validation** | Traditional stats for Table 1: t-tests, ANOVA, chi-square, custom hypothesis tests |
| 10 | **Report Export** | LaTeX manuscript, markdown report, TRIPOD checklist, methodology audit log |

### 📄 Publication-Ready Manuscript Generation

The export pipeline does the mechanical assembly work of writing a prediction model paper:

- **Compilable LaTeX manuscript** with structured abstract, methods, results, and discussion
- **Auto-generated methods section** reflecting your exact workflow decisions — preprocessing parameters, feature selection thresholds, model hyperparameters, split strategy
- **Results section** with width-contained performance tables, bootstrap CIs, explainability findings, sensitivity analysis
- **Structured discussion skeleton** with result-specific prompts referencing your actual best model, top features, and metrics
- **Table 1** with stratified descriptives, statistical tests, and footnoted custom tests
- **TRIPOD checklist** auto-tracked from your workflow
- **Commented figure references** matching your export filenames — uncomment after placing figures
- **Markdown report** with the same content for quick review

**What you still write:** Clinical context, study design rationale, interpretation of findings, comparison with prior work, domain-specific limitations.

**What the app writes for you:** Everything else — sample sizes, split ratios, model names, hyperparameters, metrics with CIs, preprocessing details, feature importance rankings, sensitivity results, software versions, methodological considerations.

### 🧠 18 Models, Zero Configuration

| Category | Models |
|----------|--------|
| **Linear** | Ridge, Lasso, ElasticNet, Logistic Regression, GLM, Huber |
| **Trees** | Random Forest, ExtraTrees, HistGradientBoosting |
| **Distance** | KNN (regression & classification) |
| **Margin** | SVM (SVR / SVC) |
| **Probabilistic** | Gaussian Naive Bayes, LDA |
| **Neural** | PyTorch MLP (configurable architecture and loss) |
| **Baselines** | Auto-generated mean/majority + simple linear/logistic |

Every model gets its own preprocessing pipeline. No data leakage. No shortcuts.

### 🤖 AI-Powered Interpretation (Optional)

Connect a local LLM or cloud API for plain-language analysis interpretation:

| Backend | Setup |
|---------|-------|
| **Ollama** (free, local) | `ollama serve && ollama pull llama3.2` |
| **OpenAI** | API key in sidebar |
| **Anthropic** | API key in sidebar |

---

## Quick Start

### Linux / macOS

```bash
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab

# Automated
chmod +x setup.sh && ./setup.sh
./run.sh

# Or manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell)

```powershell
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab

# Automated
.\setup.ps1
.\run.ps1

# Or manual
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Requirements

- Python 3.10+ (tested on 3.12)
- ~2GB disk for dependencies (PyTorch, scikit-learn, SHAP)
- GPU optional (only used by neural network models)

---

## For Researchers

This tool enforces methodological rigor so reviewers don't have to:

- ✅ Proper train/validation/test splits (no data leakage)
- ✅ Bootstrap confidence intervals on all reported metrics
- ✅ Automatic comparison against null and simple baselines
- ✅ Calibration analysis for clinical prediction models
- ✅ Sensitivity analysis to demonstrate robustness
- ✅ TRIPOD compliance tracking throughout the workflow
- ✅ Reproducibility manifest (seeds, versions, configurations)
- ✅ Methods section generated from your actual analysis choices with specific parameters
- ✅ LaTeX manuscript template populated with your results

**Your data stays private.** All processing happens in your browser session. Nothing is written to disk. No data is sent anywhere (unless you opt into cloud LLM interpretation).

---

## Project Structure

```
tabular-ml-lab/
├── app.py                    # Landing page and sidebar
├── pages/                    # 10 workflow pages
│   ├── 01_Upload_and_Audit.py
│   ├── 02_EDA.py
│   ├── 03_Feature_Engineering.py
│   ├── 04_Feature_Selection.py
│   ├── 05_Preprocess.py
│   ├── 06_Train_and_Compare.py
│   ├── 07_Explainability.py
│   ├── 08_Sensitivity_Analysis.py
│   ├── 09_Hypothesis_Testing.py
│   └── 10_Report_Export.py
├── ml/                       # Core ML modules
│   ├── model_registry.py     # 18 model definitions
│   ├── bootstrap.py          # BCa bootstrap CIs
│   ├── calibration.py        # Calibration metrics & plots
│   ├── dataset_profile.py    # Automated data profiling
│   ├── feature_selection.py  # LASSO, RFE, stability selection
│   ├── latex_report.py       # LaTeX manuscript generator
│   ├── publication.py        # Methods section generator
│   ├── sensitivity.py        # Seed & dropout robustness
│   ├── table_one.py          # Table 1 generator
│   └── ...
├── models/                   # Model implementations
├── utils/                    # Theme, session state, LLM UI
├── tests/                    # pytest suite
├── setup.sh / setup.ps1      # Cross-platform setup
├── run.sh / run.ps1          # Cross-platform run
└── requirements.txt
```

---

## 🎓 Institutional Deployment

**University IT Administrators:** Deploy Tabular ML Lab on your institutional infrastructure with Docker, KeyCloak OIDC authentication, and LLM integration.

👉 **[university-docker branch](https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker)**

**What's included:**
- 🐋 **Docker/Kubernetes configs** — Production-ready deployment
- 🔐 **KeyCloak OIDC SSO** — Standards-based authentication
- 🤖 **Institutional LLM** — Connect to your Ollama/vLLM infrastructure
- ⚡ **Compute profiles** — Optimize for your hardware
- 📚 **Complete docs** — UNIVERSITY_DEPLOYMENT.md, DOCKER_DEPLOYMENT.md, COMPUTE_PROFILES.md

**Perfect for:**
- 📖 Statistics courses — Students analyze data without coding
- 🔬 PhD research — Publication-ready outputs with TRIPOD checklists
- 🎯 Capstone projects — Guided ML workflow ensures quality
- 👨‍🏫 Faculty research — Bootstrap CIs, SHAP, calibration analysis

**Security:**
- ✅ On-premises deployment (no external APIs required)
- ✅ Session-isolated (no persistent data between users)
- ✅ Non-root containers
- ✅ Health check endpoints

See [UNIVERSITY_DEPLOYMENT.md](https://github.com/hedglinnolan/tabular-ml-lab/blob/university-docker/UNIVERSITY_DEPLOYMENT.md) for complete setup guide.

---

## Contributing

Issues and PRs welcome. If you use this in your research, please cite:

```
Hedglin, N. (2026). Tabular ML Lab [Computer software]. 
https://github.com/hedglinnolan/tabular-ml-lab
```

## License

MIT — use it however you want.
