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
    <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/pytorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </p>
  <p align="center">
    <a href="https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker">
      <img src="https://img.shields.io/badge/🎓_University_Deployment-Docker_+_AD_Auth-0066cc?style=for-the-badge" alt="University Deployment">
    </a>
  </p>
</p>

---

> 🎓 **University IT Administrators:** Looking to deploy for your institution? Check out the [**university-docker branch**](https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker) for Docker deployment with Active Directory authentication and institutional Ollama integration. Complete setup guide included.

---

An interactive research workbench for scientists who work with tabular data and need to publish papers. Upload your CSV, and the app guides you through a complete, defensible ML workflow — from exploratory analysis to a draft methods section and supporting manuscript materials.

**Built for researchers, not ML engineers.** Start with the core guided path, then use advanced options only when your study needs them.

> 🌐 **Try it now:** [app.tabularml.dev](https://app.tabularml.dev)  
> ⚠️ **Note:** The live demo currently runs the `feature/feature-engineering` branch (experimental features in testing). The stable release is on `main`.

### 🎉 What's New in v1.1 (Human-Centered Design Improvements)

- **💾 Session Save/Resume** — Pause your 45-minute workflow and continue later. Download `.pkl` files to save progress.
- **🔍 Diagnostic Assistant** — When models perform poorly (AUC < 0.65), get intelligent explanations: weak features, insufficient data, class imbalance, or high missing data.
- **📊 Explainability Prioritization** — Three-tier system (Essential/Recommended/Advanced) helps you focus on what reviewers actually care about.
- **📐 Statistical Validation** — Renamed "Hypothesis Testing" to emphasize publication value. Generate p-values for Table 1 and validate ML findings with traditional statistics.
- **🧬 Feature Engineering** — NEW optional step: create polynomial features, log transforms, ratios, binning, and topological features (TDA). Integrates seamlessly with feature selection.
- **🔗 Workflow Connective Tissue** — Every page now explains "Why This Step?" and "What Happens Next?" for better learning experience.

<!-- 
## Screenshots
TODO: Add screenshots of key pages
![Upload & Audit](docs/screenshots/upload.png)
![Train & Compare](docs/screenshots/train.png)
![Report Export](docs/screenshots/report.png)
-->

## Features

### 📋 10-Step Workflow with a Guided Default Path

| Step | Page | What it does |
|------|------|-------------|
| 1 | **Upload & Audit** | Load CSVs/Excel, merge multiple files, data quality checks |
| 2 | **EDA** | Distributions, correlations, Table 1, missing data analysis |
| 3 | **Feature Engineering** | 🆕 Advanced / optional: create polynomial, ratio, binning, or TDA features when baseline modeling needs help |
| 4 | **Feature Selection** | LASSO path, RFE-CV, stability selection, consensus ranking |
| 5 | **Preprocess** | Per-model pipelines: MICE imputation, scaling, encoding, outliers |
| 6 | **Train & Compare** | 18 model families with bootstrap CIs and baseline comparison |
| 7 | **Explainability** | SHAP, permutation importance, calibration, decision curves |
| 8 | **Sensitivity Analysis** | Seed robustness, feature dropout — prove your results aren't fragile |
| 9 | **Statistical Validation** | 🆕 Traditional stats for Table 1: t-tests, ANOVA, chi-square |
| 10 | **Report Export** | Auto-generated methods section, TRIPOD checklist, LaTeX tables |

### 📊 Manuscript-Ready Starting Materials

- **Table 1** with stratified descriptives, p-values, and SMD
- **Bootstrap 95% CIs** (BCa, 1000 resamples) on all metrics
- **Calibration analysis** — reliability diagrams, Brier score, ECE
- **Decision curve analysis** for clinical utility
- **Subgroup analysis** with forest plots
- **TRIPOD checklist** auto-tracked from your workflow
- **CONSORT-style flow diagrams**
- **Auto-generated methods section** reflecting your actual choices
- **Journal-quality figures** with DPI control

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
- ✅ Methods section text generated from your actual analysis choices

**Your data stays private.** All processing happens in your browser session. Nothing is written to disk. No data is sent anywhere (unless you opt into cloud LLM interpretation).

---

## Project Structure

```
tabular-ml-lab/
├── app.py                    # Landing page and sidebar
├── pages/                    # 9 workflow pages
│   ├── 01_Upload_and_Audit.py
│   ├── 02_EDA.py
│   ├── 03_Feature_Selection.py
│   ├── 04_Preprocess.py
│   ├── 05_Train_and_Compare.py
│   ├── 06_Explainability.py
│   ├── 07_Sensitivity_Analysis.py
│   ├── 08_Hypothesis_Testing.py
│   └── 09_Report_Export.py
├── ml/                       # Core ML modules
│   ├── model_registry.py     # 18 model definitions
│   ├── bootstrap.py          # BCa bootstrap CIs
│   ├── calibration.py        # Calibration metrics & plots
│   ├── feature_selection.py  # LASSO, RFE, stability selection
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

**University IT Administrators:** Deploy Tabular ML Lab on your institutional infrastructure with Docker, Active Directory authentication, and Ollama integration.

👉 **[university-docker branch](https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker)**

**What's included:**
- 🐋 **Docker/Kubernetes configs** - Production-ready deployment
- 🔐 **Active Directory SSO** - Reverse proxy authentication (nginx/Apache examples)
- 🤖 **Institutional Ollama** - Connect to your LLM infrastructure
- ⚡ **Compute profiles** - Optimize for your hardware (GTX 1080 Ti → multi-GPU clusters)
- 📚 **Complete docs** - UNIVERSITY_DEPLOYMENT.md, DOCKER_DEPLOYMENT.md, COMPUTE_PROFILES.md

**Quick start:**
```bash
git clone -b university-docker https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
cp .env.example .env
nano .env  # Configure OLLAMA_URL, COMPUTE_PROFILE, AUTH settings
docker build -t tabular-ml-lab .
docker run -d -p 8501:8501 --env-file .env tabular-ml-lab
```

**Perfect for:**
- 📖 Statistics courses - Students analyze data without coding
- 🔬 PhD research - Publication-ready outputs with TRIPOD checklists
- 🎯 Capstone projects - Guided ML workflow ensures quality
- 👨‍🏫 Faculty research - Bootstrap CIs, SHAP, calibration analysis

**Security features:**
- ✅ On-premises deployment (no external APIs)
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
