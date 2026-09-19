<p align="center">
  <h1 align="center">🔬 Tabular Machine Learning Lab</h1>
  <p align="center">
    <strong>From raw data to a manuscript-ready starting point. No coding required.</strong>
  </p>
  <p align="center">
    <a href="#what-it-does">What it does</a> ·
    <a href="#download">Download</a> ·
    <a href="#quick-start">Install from a terminal</a> ·
    <a href="#privacy">Privacy</a> ·
    <a href="#server">Run on a server</a> ·
    <a href="https://github.com/hedglinnolan/tabular-ml-lab/issues">Report Bug</a>
  </p>
  <p align="center">
    <a href="https://github.com/hedglinnolan/tabular-ml-lab/actions/workflows/ci.yml"><img src="https://github.com/hedglinnolan/tabular-ml-lab/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
    <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </p>
</p>

---

> **Two ways to run this.** On **your own laptop**, download it and double-click — nothing to configure, no accounts, no admin rights, and your data never leaves your machine. If your lab would rather run one shared copy on **a university server**, that works too: the app has no accounts or login of its own, and your institution's single sign-on goes in front of it. See [Running it on a university server](#server) below.

---

An interactive research workbench for scientists who work with tabular data and need to publish papers. Upload your table — CSV, Excel, Parquet, TSV or JSON — and the app guides you through a complete, defensible ML workflow — from exploratory analysis to a LaTeX manuscript draft with auto-generated methods, results, and structured discussion, ready to paste into Overleaf or compile with a full TeX installation.

**Built for researchers, not ML engineers.** The app does the mechanical work of writing a prediction model paper — the numbers, the parameters, the tables. What's left to you is the domain context no tool can provide: the introduction and objectives, the study-design rationale, the interpretation, and the literature. It is free and open source (MIT), and it runs on your own computer.

<a name="what-it-does"></a>
## ✨ What it does

### 📋 The guided workflow: 7 core steps, 3 optional deep-dives

| Step | Page | What it does |
|------|------|-------------|
| 1 | **Upload & Audit** | Load CSV, Excel, Parquet, TSV or JSON; join or stack several files into one table; transpose wide assay exports; full data-quality audit |
| 2 | **EDA** | Distributions, correlations, Table 1, missing-data analysis, and a findings panel that counts blockers, warnings and opportunities, with classical diagnostics (VIF, residual normality, influence) one click away |
| 3 | **Feature Engineering** *(optional)* | PCA, polynomial features, log transforms, ratios, binning, UMAP, missingness indicators, TDA\* |
| 4 | **Feature Selection** | LASSO path, RFE-CV, univariate, stability selection, consensus ranking |
| 5 | **Preprocess** | Per-model pipelines: imputation, scaling, encoding, outlier handling, power transforms |
| 6 | **Train & Compare** | Up to 13 models for a regression target or 12 for classification (22 in the library), with bootstrap CIs, automatic baseline comparison, calibration analysis, optional Optuna tuning, and a time estimate measured on your own machine before you click Train |
| 7 | **Explain & Validate** | SHAP, permutation importance, PDP, Bland–Altman agreement, external validation against a second file, subgroup checks, and cohort runs — the same study fitted separately on each group and exported side by side |
| 8 | **Sensitivity Analysis** *(optional)* | Seed robustness, feature dropout — prove your results aren't fragile |
| 9 | **Statistical Validation** *(optional)* | Traditional stats for Table 1: t-test / Mann–Whitney, ANOVA / Kruskal–Wallis, chi-square / Fisher, correlation, normality and paired tests — with FDR correction, and results that flow straight into Table 1 |
| 10 | **Report Export** | LaTeX manuscript (compiled to PDF too, if pdflatex is installed), markdown report with a decision audit trail, TRIPOD checklist, and one ZIP holding your models, predictions, figures and a reproducibility manifest |

*\* TDA and UMAP come from two extra packages. Every install path adds them on a best-effort basis (Python 3.12 on Windows, Mac and Linux x86-64) and the Feature Engineering page tells you if they're missing.*

The shortest defensible path is the 7 core steps. The three optional branches
are folded into an "Advanced workflow" group in the sidebar's default "Quick
workflow" mode, and open whenever you want them.

**📖 Theory Reference.** An eleventh page that is not a workflow step: eight cited
chapters with interactive demos, from data quality to TRIPOD. The app's inline
advice links straight to the relevant chapter and names the section, so "why does
skew matter?" is one click from the number that raised the question.

**How much data?** A typical clinical table — a few thousand rows and up to a
few hundred columns — is comfortable. Wider tables work, with every cap named as
it fires. Details under [How much data can it handle?](#how-much-data-can-it-handle).

### 📄 Publication-Ready Manuscript Generation

**What you write:** Clinical context, study design rationale, interpretation of findings, comparison with prior work.

**What the app writes:** Sample sizes, split ratios, preprocessing parameters, model hyperparameters, metrics with bootstrap confidence intervals, feature importance rankings, sensitivity results, and a reproducibility manifest of exact software versions and seeds — a compilable LaTeX manuscript with methods, results, and a structured discussion skeleton populated from your actual analysis.

Also generates: Table 1 with stratified descriptives and statistical tests, a 22-item TRIPOD checklist part-filled from your workflow (the items the app cannot infer stay blank for you to complete), and a markdown report for quick review.

<a name="for-researchers"></a>
### 🔬 For Researchers

This tool enforces methodological rigor so reviewers don't have to:

- ✅ Test set locked away on the upload page the moment you pick a target (15% by default, decided once) — feature engineering and selection fit on training rows only (no leakage into held-out evaluation), with an explicit, watermarked exploratory mode if you want full-data screening
- ✅ BCa bootstrap confidence intervals (1,000 resamples) on the headline metrics — one click on Step 6, and the intervals print beside the point estimates in the manuscript's results table
- ✅ Automatic comparison against null and simple baselines
- ✅ Calibration analysis for clinical prediction models
- ✅ Sensitivity analysis to demonstrate robustness
- ✅ TRIPOD reporting evidence collected as you work, and a part-filled checklist at export
- ✅ Reproducibility manifest (seeds, versions, configurations)
- ✅ Methods section generated from your actual analysis choices with specific parameters
- ✅ LaTeX manuscript template populated with your results
- ✅ Pre-export validation — downloads stay disabled until the draft's numbers agree with the analysis they describe, unless you explicitly override
- ✅ ~2,600 automated tests run on every change, including a real LaTeX round-trip

### 🚫 What this is not

- **Not survival analysis.** There are no Cox or other time-to-event models; an outcome here is a number or a class.
- **Not a differential-expression or batch-correction pipeline.** It runs downstream of tools like limma or DESeq2, on the features you bring, and does not integrate several omics layers.
- **Not causal inference.** Feature importance is predictive, and the generated Discussion says so.
- **Not a hosted service.** Nothing runs anywhere except where you install it.

### 🧠 16 Algorithms, 22 Model Variants, Sensible Defaults

| Category | Models |
|----------|--------|
| **Linear** | Ridge, Lasso, ElasticNet, Logistic Regression, GLM (plain OLS / logistic), Huber |
| **Trees** | Random Forest, ExtraTrees |
| **Boosting** | HistGradientBoosting, XGBoost, LightGBM (regression & classification) |
| **Distance** | KNN (regression & classification) |
| **Margin** | SVM (SVR / SVC) |
| **Probabilistic** | Gaussian Naive Bayes, LDA |
| **Neural** | PyTorch MLP (architecture and loss recommended from your data, or set your own) |
| **Baselines** | Auto-generated mean/majority + simple linear/logistic |

Six of these ship as separate regression and classification variants, which is why 16 algorithms make 22 registry entries; the picker only ever offers you the ones that fit your target.

Each model can get its own preprocessing pipeline — different imputation, scaling and encoding for a Ridge than for a Random Forest — configured on the Preprocess step. Pipelines are fitted on the training rows only; validation and test data are transformed, never fitted. A model you add on Train & Compare without preparing it on Preprocess borrows whichever pipeline was built first, and the app names the owner on the model card and again before the run starts.

<a name="privacy"></a>
## 🔒 Privacy and data handling

**On your own computer.** All processing happens on your machine and nothing of yours is saved to disk — projects, datasets, and analysis state live in memory for the duration of your browser session, and saved sessions are files you download yourself. The only place your content ever touches the filesystem is a scratch folder the app uses to compile your manuscript to PDF, and it deletes that again immediately. Nothing leaves your machine unless you opt into a cloud AI backend (below).

**On a shared server.** The same rules apply with the server in place of your laptop: uploads are processed there, held in the server's memory for the life of your session, and gone when the session ends or the container restarts. Each person's session is their own; nothing is written to the server's disk between sessions. [Running it on a university server](#server) has the deployment-side picture.

**With the optional AI interpretation.** If you turn on a cloud backend, column names, summary statistics and the result tables you're looking at — not your raw rows — are sent to that provider for interpretation. The local Ollama option sends nothing off your machine, and the feature is off unless a backend is set up.

<a name="download"></a>
## ⬇️ Download the App (no coding, no terminal)

The download and the double-click starters launch the **Classic** app.

Run Tabular ML Lab on your own computer like a normal desktop app — **your data
never leaves your machine.**

### ⬇️ [**Download the latest release (.zip)**](https://github.com/hedglinnolan/tabular-ml-lab/releases/latest)

*(alternate link, if the one above is blocked: [download the current code as a zip](https://github.com/hedglinnolan/tabular-ml-lab/archive/refs/heads/main.zip))*

**1.** Download the zip and unzip it anywhere you like. A plain local folder is
best — Desktop and Documents are often synced to OneDrive or iCloud, which slows
setup down and syncs tens of thousands of library files. The unzipped folder
**is** the app — don't delete it after setup.

**2.** Open the folder and double-click the starter for your system:

| System | Double-click | First-time security prompt (once per computer) |
|--------|--------------|------------------------------------------------|
| **Windows** | `Start Tabular ML Lab.bat` | SmartScreen may appear → click **More info** → **Run anyway** |
| **Mac** | `Start Tabular ML Lab.command` | **Right-click → Open → Open** (plain double-click is blocked for downloaded files) |

**3.** The first launch sets everything up **once** — no admin rights needed, and
nothing is installed system-wide: it downloads a private copy of Python and the
analysis libraries — about 400 MB on Windows, 350 MB on a Mac, a few minutes on a
normal connection — and unpacks them to roughly **1.5 GB on disk**. The window
tells you what's happening. Then the app opens in your browser automatically.

**Coming back later:** setup never repeats. On Windows, a **Tabular ML Lab**
icon is now on your Desktop and Start Menu; on Mac, a **Tabular ML Lab** app
appeared inside the folder (drag it to your Dock if you like). One click starts
the app in seconds, with **no internet needed after the first launch** (only the
app's fonts come from the web, and they fall back to your system font offline).
The browser tab is just the app's window: the icon starts the app and the tab
shows it. On Windows, the app runs in a small console window minimized to your
taskbar — closing that window quits it.

<details>
<summary><b>Quitting on a Mac, updating, and uninstalling</b></summary>

**Quitting on a Mac:** the app icon runs it in the background — to quit it, open
Activity Monitor and stop the `python3.12` process running Streamlit, or start the
app from `Start Tabular ML Lab.command` instead if you want a window you can
close.

**Updating:** download the new zip and delete the old folder — the app never
stores your data inside its own folder, so nothing of yours is lost. The new
folder reinstalls its libraries on first launch: a few minutes and about 1.4 GB
written again, though uv serves most of it from what it already downloaded
rather than fetching it a second time.

**Uninstalling:** delete the folder, plus the two **Tabular ML Lab** shortcuts on
Windows (Desktop and Start Menu). Setup also left a private Python and a download
cache in your user profile — on Windows, `%APPDATA%\uv` and `%LOCALAPPDATA%\uv`
(the same `uv` folders live in your home directory on Mac). Deleting those
reclaims up to another 1-2 GB, most of it the shared download cache; leave them
if you use other Python tools built on uv.

</details>

<details>
<summary><b>If your university or hospital blocks the download</b></summary>

Some universities and hospitals block downloads at the network level, so the
file may never arrive — often with no explanation. These paths fail for
different reasons, so if one is blocked another usually works.

**1. Try the alternate download link above.** The release file and the
"current code" file are served from different addresses.

**2. Ask a colleague to send you the folder.** The app is just files — a zip on a
USB stick or shared drive works exactly the same. Ask them to send the zip they
downloaded, not their installed copy (delete `.venv` and `.tools` first if in
doubt). Only the first launch needs internet, to fetch Python and the libraries.

**3. Ask IT to allow these addresses.** Sending them this exact list usually
resolves it in one message — all are standard developer infrastructure:

```
github.com
codeload.github.com          <- needed for zip downloads specifically
objects.githubusercontent.com
release-assets.githubusercontent.com
astral.sh
releases.astral.sh           <- astral.sh redirects here to serve the installer
pypi.org
files.pythonhosted.org
fonts.googleapis.com         <- cosmetic only; the app runs fine without it
fonts.gstatic.com            <- cosmetic only
```

**4. Install from a terminal** (nothing goes through your browser). This is the
same install as [Quick Start](#quick-start) below — it needs Git, and it prints
the address to open instead of opening your browser for you.

</details>

<details>
<summary>What the starter actually does (for the curious or cautious)</summary>

The starter uses <a href="https://github.com/astral-sh/uv">uv</a> (a widely
used open-source tool from Astral) to install a private Python 3.12 and the
libraries from <code>requirements.txt</code>, then tries the optional extras in
<code>requirements-optional.txt</code>. The libraries go into
<code>.venv/</code> beside the app (with uv itself in <code>.tools/</code>); the
private Python and uv's download cache go into your own user folder — on
Windows, <code>%APPDATA%\uv</code> and <code>%LOCALAPPDATA%\uv</code>. Nothing is
installed system-wide and no admin rights are needed. It then runs
<code>streamlit run app.py</code>, which serves the app only to your own
computer (localhost) and opens your browser. The security prompts appear
because the starters aren't code-signed — the source is fully inspectable in
this repository.
</details>

<a name="quick-start"></a>
## Quick Start (from a terminal)

We recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) first — it downloads its own Python 3.12, so you don't need Python installed at all, and the setup script then also adds the two optional extras (TDA, UMAP). If they fail to build, the script says so and everything else still works.

### Linux / macOS

```bash
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
chmod +x setup.sh run.sh && ./setup.sh
./run.sh
```

### Windows (PowerShell)

```powershell
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
.\setup.ps1
.\run.ps1
```

> If PowerShell refuses with *running scripts is disabled on this system*, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first (it applies to that window only), or use the double-click starter instead.

Then open **http://localhost:8501** in your browser. The scripts print that address but don't open it for you — the terminal stays busy while the app runs, and `Ctrl+C` there stops it.

> Without uv, the setup scripts fall back to `pip` with your system Python, and the optional extras (TDA, UMAP) are skipped — they can only be added safely on the uv path, because one of them would otherwise downgrade scikit-learn for the whole app.

For troubleshooting, a dependency self-check (`scripts/smoke_check.py`), and a walkthrough you can use to confirm the install works, see [QUICKSTART.md](QUICKSTART.md).

### Requirements

- **With uv (recommended):** no Python needed — uv downloads its own Python 3.12. You do need Git for the `git clone` step.
- **Without uv:** Python 3.12 — the version CI, the Dockerfile and the setup scripts all pin (3.13 works locally but isn't covered by CI) — plus Git
- 4 GB RAM minimum for one person — this is also what decides how large a file the app will accept. Sizing a shared server for a lab or a class instead? See [Running it on a university server](#server).
- About 1.5 GB of disk once installed on Windows or macOS (PyTorch alone is roughly a third of it), and several GB on Linux, where the PyPI PyTorch wheel also pulls in CUDA libraries
- GPU optional (only used by neural network models)

> **First launch:** The app loads about 25 libraries — scikit-learn, pandas, Streamlit and their dependencies. Expect 15-30 seconds on a typical work laptop before the page is ready. PyTorch, the largest one, isn't loaded on the landing page; it comes in when you open Step 6, Train & Compare. Subsequent launches are faster.

### How much data can it handle?

Short version: **a typical clinical table — a few thousand rows and up to a few
hundred columns — is comfortable.** What strains the app is the column count, not
the file size. Reading a big CSV is cheap — a 194 MB RNA-seq matrix (60,000 rows ×
1,000 columns) parses in about 4.3 seconds at roughly 1.3 GB peak. (Excel is the
exception: an `.xlsx` reads roughly fifty times slower per cell, so the app counts
the sheet and quotes you the wait *before* it starts parsing.) Ingest is not the
bottleneck. What decides whether your study is comfortable here is how wide the
table is:

- **Up to ~500 columns** — comfortable, at any reasonable row count. This is the shape the guided workflow was built and tuned for.
- **~500-2,000 columns** — workable, but the exploratory pages slow down and start working on a declared subset: above 1,000 numeric columns the correlation screen keeps the 1,000 highest-variance of them and says how many pairs that covers; VIF is withheld above 200 numeric features; model-agnostic SHAP quotes a price up to 1,000 features and is declined above that (tree and linear models are explained normally either way). Nothing is dropped silently — every cap that fires names itself on the page, and is carried into the manuscript's Discussion as a limitation.
- **Above ~2,000 columns** — the app warns you and lets you continue. Nothing hangs: the correlation screen and permutation importance both wait to be asked, and when you ask, the screen works from the 1,000 highest-variance columns and says exactly how many of the possible pairs that covered. What a very wide table costs you is completeness, not the run — so subsetting to the features you actually intend to model still buys a sharper analysis.

<details>
<summary><b>If you work with omics, and how memory is budgeted</b></summary>

**If you work with omics:** a targeted panel of a few hundred to a couple of
thousand features is the sweet spot. (If your matrix arrives genes-as-rows, the
upload page transposes it for you.) A full 20,000-gene matrix will usually load
(a 500 × 20,000 matrix is only a few hundred megabytes; one that is also tall is
refused on memory) and the run does finish, but the wide analyses report on a
declared subset rather than on every gene: above 5,000 columns the correlation
screen waits for you to ask for it, then states how many columns it kept and that
it kept the highest-variance ones; VIF is unavailable; and SHAP falls back to the
tree and linear explainers. Treat that as a screening pass, not the whole study,
and cut to your candidate set before you model.

**On memory.** A frame costs roughly 6.6-7.2× the size of its CSV by the time the
app is holding it, so plan on several GB of free RAM per GB of CSV. Nothing is
gated on file size except one loose backstop — a file over 1,500 MB is refused
without being read, because reading it is what would exhaust memory. Below that,
uploads are admitted by measuring the *parsed frame* against the memory your
machine can actually give, because the same matrix written at two different
decimal precisions differs several-fold on disk and not at all in RAM. If a file
won't fit, the app tells you at upload instead of half-way through the workflow —
and again when you join or stack files, because two files that each fit can link
into one table that does not.

Running this on a shared or containerized server? The deployment-side knobs — upload
ceiling, proxy body size, container memory — are in
[UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md#hardware).

</details>

---

## 🤖 Optional: AI interpretation

**The AI never writes your paper — and it is off unless a backend is set up.** The manuscript, the methods and results text, Table 1 and the model coaching are all generated deterministically from your own numbers. The optional LLM only comments on a result you are already looking at; those comments never enter the LaTeX manuscript, and they reach the markdown report only if you tick **Include LLM interpretations in report** on Report Export (off by default).

Connect a local LLM or a cloud API for plain-language interpretation of what you're looking at. Three backends are supported — Ollama, OpenAI (or any OpenAI-compatible endpoint) and Anthropic — and you pick the backend and type the model name in the sidebar under **🤖 LLM Settings**. On a shared server the administrator can set the default backend, the Ollama address and the API keys in `.env`; keys set there stay on the server and are never shown or saved. A **🧠 Deep Analysis** button then sits under the plots and tables on EDA, Feature Selection, Train & Compare, Explain & Validate, Sensitivity Analysis and Statistical Validation — click it and you get a short, specific read on that one result, and you can ask a follow-up question in the same panel.

| Backend | Setup | Notes |
|---------|-------|-------|
| **Ollama** (free, local) | [Install Ollama](https://ollama.ai), then pull a model | If Ollama isn't already running, the app tries to start it for you; if that doesn't come up within a few seconds, run `ollama serve` yourself. The address is `localhost:11434` on your own machine; on a server it is whatever the administrator set, and you can change it in the sidebar. |
| **OpenAI** | API key in sidebar, or set on the server | Defaults to `gpt-4o-mini`; type any other model name over it |
| **Anthropic** | API key in sidebar, or set on the server | Defaults to `claude-sonnet-5`; type any other model name over it |

**Ollama model size** tracks your RAM — `qwen3.5:1.5b` at 8 GB, `qwen3.5:9b` at 16 GB (the app default), `qwen3.5:32b` at 32 GB or with a GPU (`ollama pull qwen3.5:9b`, and so on). Any Ollama-compatible model works; type its name in the sidebar. Larger models produce better interpretations but require more memory, and the first answer can take a minute or two while the model loads.

---

<a name="server"></a>
## 🎓 Running it on a university server (Docker)

Everything above describes running the app on one person's computer. You can also
run it once, on a server, so a whole lab or class shares it — one container, one
shared memory budget, no per-user setup and nothing for students to install:

```bash
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
cp .env.example .env
docker compose up --build
```

That's the whole deployment on a closed network. Putting it behind your
institution's login is more work, and all of it happens outside the app: swap the
`ports:` mapping in `docker-compose.yml` for `expose: ["8501"]` so only the proxy
can reach the container, configure the proxy to do the authenticating, and set the
proxy values whose defaults otherwise make the app look broken — request body
size, read timeout, WebSocket upgrade, and sticky sessions if you run more than
one replica. [UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md) walks through
all of it, with worked examples for nginx, Apache and OAuth2 Proxy. The
`Dockerfile` and `docker-compose.yml` live here on `main` alongside the app, so
there is no separate branch to keep in sync.

**About logins.** The app has none of its own — as shipped, `docker compose up`
publishes the app on the host's port 8501, so anyone who can reach that machine on
your network can use it. That is fine on a lab network or VPN, where the network
is the access control. To open it wider, put the single sign-on your institution
already runs (Shibboleth, CAS, Azure AD, Keycloak) in front of it and let that
decide who gets through.

**Size your RAM here.** Everyone on that server shares one container and one
memory budget — `APP_MEMORY_LIMIT` in `.env`, 4 GB as shipped — and uploads are
admitted against what is left of it. Rule of thumb from the admission gate: a
numeric frame is budgeted at about 32 bytes per cell while it is held (8 bytes
with a 4× safety margin), so a 10,000 × 500 table wants roughly 160 MB of headroom
and a 100,000 × 2,000 table roughly 6.4 GB. Add one such headroom per person
likely to be mid-analysis at the same time, on top of the few hundred megabytes
the app itself occupies. 4 GB suits a single user or a small lab on
spreadsheet-sized data; 16 GB is a reasonable start for a class or for omics
panels. The full arithmetic is in
[UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md#hardware).

**Updating, and what a restart costs.** Updating is `git pull` and
`docker compose up --build`. Every restart ends every session in progress —
uploads, fitted models and audit trails live only in the container's memory — so
do it when nobody is mid-analysis, and pin a class to a release tag so a pull
cannot change behavior mid-semester. Nothing is backed up because nothing is
stored: people keep their work by downloading a saved session.

**Also worth knowing:**
- The container runs as a non-root user and exposes a health check endpoint.
- Nothing is kept on disk between sessions — each person's uploads, models and
  audit trail live in the server's memory for the life of their session.
- An optional Ollama sidecar (`docker compose --profile ollama up`, then
  `docker exec -it tabular-ml-lab-ollama-1 ollama pull qwen3.5:9b`) runs the AI
  interpretation features on your own hardware, so no data leaves your network.
  The app reaches it through `OLLAMA_BASE_URL`, which the compose file already
  sets.

---

## 🚧 Roadmap: TurboTab (preview)

This repository ships one analysis engine behind two front doors. **Classic** is
the Streamlit workbench described above — stable, and the one to use for
analyses you intend to publish. **TurboTab** is a faster single-page interface
being built on the same engine: upload, automatic diagnosis, ranked findings you
accept or reject, training, explainability and a manuscript draft, as one page.
It is under active development — screens, behavior, and saved state may change
without notice, and it is not yet covered by the guarantees this README makes
about Classic.

<details>
<summary><b>Running TurboTab</b> (terminal required)</summary>

TurboTab has no entry in the desktop starter yet — you start it from a terminal.
Nothing below applies to the Classic app; skip this if you are here to run an
analysis. From a clone of this repository:

```bash
# macOS / Linux
python3 -m venv venv && venv/bin/pip install -r requirements.txt
make turbotab
# or, equivalently:
venv/bin/python scripts/serve_turbotab.py --port 8777
```

```powershell
# Windows (PowerShell) — there is no `make` here, so call the script directly
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\python scripts/serve_turbotab.py --port 8777
```

Then open http://127.0.0.1:8777. `make turbotab` runs `./venv/bin/python`, so it
needs the `venv/` above; if you already have the double-click starter's `.venv/`,
that interpreter works too — it is built from the same `requirements.txt` — so
point the last command at it instead. The launcher checks its own interpreter
before serving and refuses with an explanation rather than starting a server that
would fail mid-workflow — if it refuses, it tells you exactly which package and
which interpreter to fix. `make turbotab-check` runs that environment check on
its own.

</details>

---

## Project Structure

<details>
<summary><b>Repository layout</b></summary>

```
tabular-ml-lab/
├── Start Tabular ML Lab.bat / .command   # Double-click starters (see launcher/)
├── app.py                    # Landing page and sidebar
├── pages/                    # The 10 workflow steps, plus a theory reference
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
├── data_processor.py         # Ingest and cleaning core
├── ml/                       # Core ML modules
│   ├── model_registry.py     # 16 algorithms, 22 model variants
│   ├── bootstrap.py          # BCa bootstrap CIs
│   ├── calibration.py        # Calibration metrics & plots
│   ├── dataset_profile.py    # Automated data profiling
│   ├── feature_selection.py  # LASSO, RFE, stability selection
│   ├── latex_report.py       # LaTeX manuscript generator
│   ├── publication.py        # Methods section generator
│   ├── regime.py             # Compute budgets for wide tables, and their disclosure
│   ├── sensitivity.py        # Seed & dropout robustness
│   ├── table_one.py          # Table 1 generator
│   └── ...
├── models/                   # Model implementations
├── utils/                    # 35 modules: cohort branches, upload admission, the sealed
│                             #   test set, provenance, sessions, theme, LLM UI
├── turbotab/                 # The second front door (see the roadmap above)
├── launcher/                 # What the double-click starters run
├── scripts/                  # Smoke check, TurboTab launcher
├── tests/                    # ~2,600 tests (unit, Streamlit integration, LaTeX round-trip)
├── docs/                     # Design notes, audits, TurboTab specs
├── Dockerfile / docker-compose.yml   # University-server deployment
├── setup.sh / setup.ps1      # Cross-platform setup
├── run.sh / run.ps1          # Cross-platform run
├── requirements.txt          # The app; requirements-optional.txt holds the TDA/UMAP extras
└── CITATION.cff
```

</details>

[ARCHITECTURE.md](ARCHITECTURE.md) describes the Streamlit app's layers, page flow
and session-state keys — it predates `turbotab/` and the cohort-branch work, so
read it as the engine's design notes rather than a current file listing.

---

## Contributing

Issues and PRs welcome. `make ci` runs the same test tiers as GitHub Actions
before you push (it expects a `venv/` built with `pip install -r requirements.txt -r requirements-dev.txt`),
and [ARCHITECTURE.md](ARCHITECTURE.md) has the engine's design notes.

## Citing this tool

If you use this in your research, please cite (GitHub's "Cite this repository"
button offers the same reference in BibTeX and APA, from `CITATION.cff`):

```
Hedglin, N. (2026). Tabular ML Lab (Version 1.0.0) [Computer software].
https://github.com/hedglinnolan/tabular-ml-lab
```

## License

MIT — use it however you want.
