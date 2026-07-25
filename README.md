<p align="center">
  <h1 align="center">🔬 Tabular Machine Learning Lab</h1>
  <p align="center">
    <strong>From raw data to a manuscript-ready starting point. No coding required.</strong>
  </p>
  <p align="center">
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

## ⬇️ Download the App (no coding, no terminal)

Run Tabular ML Lab on your own computer like a normal desktop app — **your data
never leaves your machine.**

### ⬇️ [**Download the latest release (.zip)**](https://github.com/hedglinnolan/tabular-ml-lab/releases/latest)

*(alternate link, if the one above is blocked: [download the current code as a zip](https://github.com/hedglinnolan/tabular-ml-lab/archive/refs/heads/main.zip))*

**1.** Download the zip and unzip it anywhere you like (Desktop, Documents…).
The unzipped folder **is** the app — don't delete it after setup.

**2.** Open the folder and double-click the starter for your system:

| System | Double-click | First-time security prompt (once per computer) |
|--------|--------------|------------------------------------------------|
| **Windows** | `Start Tabular ML Lab.bat` | SmartScreen may appear → click **More info** → **Run anyway** |
| **Mac** | `Start Tabular ML Lab.command` | **Right-click → Open → Open** (plain double-click is blocked for downloaded files) |

**3.** The first launch sets everything up **once**: it downloads a private copy
of Python and the analysis libraries (roughly 600 MB, a few minutes — the window tells
you what's happening). Then the app opens in your browser automatically.

**Coming back later:** setup never repeats. On Windows, a **Tabular ML Lab**
icon is now on your Desktop and Start Menu; on Mac, a **Tabular ML Lab** app
appeared inside the folder (drag it to your Dock if you like). One click starts
the app in seconds — **no internet needed after the first launch.** The browser
tab is just the app's window: the icon starts the app, the tab shows it, and
closing the small console window quits it.

**Updating:** download the new zip and delete the old folder — the app never
stores your data inside its own folder, so nothing of yours is lost.

**Uninstalling:** delete the folder (plus the Desktop shortcut on Windows). That's everything.

### If the download is blocked

Some universities and hospitals block downloads at the network level, so the
file may never arrive — often with no explanation. These paths fail for
different reasons, so if one is blocked another usually works.

**1. Try the alternate download link above.** The release file and the
"current code" file are served from different addresses.

**2. Install from a terminal** (nothing is downloaded through your browser):

```bash
# macOS / Linux — paste into Terminal
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab && uv run streamlit run app.py
```

```powershell
# Windows — paste into PowerShell
irm https://astral.sh/uv/install.ps1 | iex
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab; uv run streamlit run app.py
```

**3. Ask IT to allow these addresses.** Sending them this exact list usually
resolves it in one message — all are standard developer infrastructure:

```
github.com
codeload.github.com          <- needed for zip downloads specifically
objects.githubusercontent.com
release-assets.githubusercontent.com
raw.githubusercontent.com
pypi.org
files.pythonhosted.org
astral.sh
```

**4. Ask a colleague to send you the folder.** The app is just files — a
zip on a USB stick or shared drive works exactly the same. Only the first
launch needs internet, to fetch Python and the libraries.

<details>
<summary>What the starter actually does (for the curious or cautious)</summary>

The starter uses <a href="https://github.com/astral-sh/uv">uv</a> (a widely
used open-source tool from Astral) to install a private Python 3.12 and the
libraries from <code>requirements.txt</code> into hidden folders
(<code>.tools/</code>, <code>.venv/</code>) inside the app folder — nothing is
installed system-wide, no admin rights needed. It then runs
<code>streamlit run app.py</code>, which serves the app only to your own
computer (localhost) and opens your browser. The security prompts appear
because the starters aren't code-signed — the source is fully inspectable in
this repository.
</details>

## Features

### 📋 10-Step Guided Workflow

| Step | Page | What it does |
|------|------|-------------|
| 1 | **Upload & Audit** | Load CSVs/Excel, merge multiple files, data quality checks |
| 2 | **EDA** | Distributions, correlations, Table 1, missing data analysis, interactive decision hub |
| 3 | **Feature Engineering** | PCA, polynomial features, log transforms, ratios, binning, TDA features |
| 4 | **Feature Selection** | LASSO path, RFE-CV, univariate, stability selection, consensus ranking |
| 5 | **Preprocess** | Per-model pipelines: imputation, scaling, encoding, outlier handling, power transforms |
| 6 | **Train & Compare** | 22 models (the neural network enables with one click) with bootstrap CIs, automatic baseline comparison, calibration analysis, optional Optuna optimization |
| 7 | **Explainability** | SHAP, permutation importance, PDP, external validation, subgroup analysis |
| 8 | **Sensitivity Analysis** | Seed robustness, feature dropout — prove your results aren't fragile |
| 9 | **Statistical Validation** | Traditional stats for Table 1: t-tests, ANOVA, chi-square, custom hypothesis tests |
| 10 | **Report Export** | LaTeX manuscript, markdown report, TRIPOD checklist, methodology audit log |

### 📄 Publication-Ready Manuscript Generation

**What you write:** Clinical context, study design rationale, interpretation of findings, comparison with prior work.

**What the app writes:** Sample sizes, split ratios, preprocessing parameters, model hyperparameters, metrics with bootstrap CIs, feature importance rankings, sensitivity results, software versions — a compilable LaTeX manuscript with methods, results, and a structured discussion skeleton populated from your actual analysis.

Also generates: Table 1 with stratified descriptives and statistical tests, TRIPOD checklist auto-tracked from your workflow, and a markdown report for quick review.

### 🧠 22 Models, Zero Configuration

| Category | Models |
|----------|--------|
| **Linear** | Ridge, Lasso, ElasticNet, Logistic Regression, GLM, Huber |
| **Trees** | Random Forest, ExtraTrees, HistGradientBoosting |
| **Boosting** | XGBoost, LightGBM (regression & classification) |
| **Distance** | KNN (regression & classification) |
| **Margin** | SVM (SVR / SVC) |
| **Probabilistic** | Gaussian Naive Bayes, LDA |
| **Neural** | PyTorch MLP (configurable architecture and loss) |
| **Baselines** | Auto-generated mean/majority + simple linear/logistic |

Every model gets its own preprocessing pipeline. No data leakage. No shortcuts.

### 🤖 AI-Powered Interpretation (Optional)

Connect a local LLM or cloud API for plain-language analysis interpretation. The app is model-agnostic — select any backend and model in the sidebar.

| Backend | Setup | Notes |
|---------|-------|-------|
| **Ollama** (free, local) | [Install Ollama](https://ollama.ai), then `ollama serve` | See model recommendations below |
| **OpenAI** | API key in sidebar | GPT-4o recommended |
| **Anthropic** | API key in sidebar | Claude Sonnet recommended |

**Ollama model selection** depends on your hardware:

| Available RAM | Recommended model | Pull command |
|---------------|-------------------|--------------|
| 8 GB | `qwen3.5:1.5b` | `ollama pull qwen3.5:1.5b` |
| 16 GB | `qwen3.5:9b` (app default) | `ollama pull qwen3.5:9b` |
| 32 GB+ / GPU | `qwen3.5:32b` or `llama3.1:70b` | `ollama pull qwen3.5:32b` |

Any Ollama-compatible model works — type the model name in the sidebar LLM Settings. Larger models produce better interpretations but require more memory.

---

## Quick Start

We recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) first — it automatically downloads the right Python version and installs all dependencies including optional packages (TDA, UMAP).

### Linux / macOS

```bash
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
chmod +x setup.sh && ./setup.sh
./run.sh
```

### Windows (PowerShell)

```powershell
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
.\setup.ps1
.\run.ps1
```

The app opens at **http://localhost:8501**.

> Without uv, the setup scripts fall back to `pip` with your system Python. Core features work on Python 3.10-3.13, but optional packages (giotto-tda, umap-learn) require 3.12 or earlier.

For preflight checks, troubleshooting, and a smoke test checklist, see [QUICKSTART.md](QUICKSTART.md).

### Requirements

- **With uv:** Any Python 3.10+ installed (uv handles the rest)
- **Without uv:** Python 3.10-3.12 recommended
- ~2GB disk for dependencies (PyTorch, scikit-learn, SHAP)
- GPU optional (only used by neural network models)

> **First launch:** The app loads ~60 packages including PyTorch and scikit-learn. Expect 15-30 seconds on a typical work laptop before the browser opens. Subsequent launches are faster.

---

## For Researchers

This tool enforces methodological rigor so reviewers don't have to:

- ✅ Test set locked away at upload — feature engineering and selection fit on training rows only (no leakage into held-out evaluation), with an explicit, watermarked exploratory mode if you want full-data screening
- ✅ Bootstrap confidence intervals on all reported metrics
- ✅ Automatic comparison against null and simple baselines
- ✅ Calibration analysis for clinical prediction models
- ✅ Sensitivity analysis to demonstrate robustness
- ✅ TRIPOD compliance tracking throughout the workflow
- ✅ Reproducibility manifest (seeds, versions, configurations)
- ✅ Methods section generated from your actual analysis choices with specific parameters
- ✅ LaTeX manuscript template populated with your results

**Your data stays private.** When you run the app locally, all processing happens on your own machine and nothing is written to disk — projects, datasets, and analysis state live in memory for the duration of your browser session (saved sessions are files you download yourself). No data leaves your machine unless you opt into a cloud LLM backend, in which case column names, summary statistics, and small data excerpts are sent to that provider for interpretation. If you use the hosted demo instead of a local install, your uploads are processed on the demo server for your session.

---

## Project Structure

```
tabular-ml-lab/
├── app.py                    # Landing page and sidebar
├── pages/                    # 11 workflow pages
│   ├── 01_Upload_and_Audit.py
│   ├── 02_EDA.py
│   ├── 03_Feature_Engineering.py
│   ├── 04_Feature_Selection.py
│   ├── 05_Preprocess.py
│   ├── 06_Train_and_Compare.py
│   ├── 07_Explainability.py
│   ├── 08_Sensitivity_Analysis.py
│   ├── 09_Hypothesis_Testing.py
│   ├── 10_Report_Export.py
│   └── 11_Theory_Reference.py
├── ml/                       # Core ML modules
│   ├── model_registry.py     # 22 model definitions
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
