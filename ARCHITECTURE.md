# ARCHITECTURE.md — Product Architecture & Commander's Intent

**Read this before touching any code. This is the design philosophy of the Tabular Machine Learning Lab.**

*Last updated: 2026-04-04*

---

## What This App Is

A research workbench that guides scientists through a complete, defensible ML workflow on tabular data — from raw CSV to compilable LaTeX manuscript. The user provides domain expertise; the app provides methodological rigor.

**Stack:** Streamlit (~22K lines Python), Ollama for LLM-powered interpretation, SQLite for project/dataset persistence.

**URL:** https://app.tabularml.dev (deployed via Cloudflare Tunnel → Streamlit on port 8501)

**CI/CD:** Push to `main` → GitHub Actions self-hosted runner on clawserver → `git pull` + `pip install` + `systemctl restart tabular-ml-lab`

---

## The Three Layers

Every feature, every page, every interaction exists within three layers that must stay synchronized:

### Layer 1: Working Data (lowest level)

What is actually happening to the data. Feature engineering, imputation, scaling, encoding, splitting, training, evaluation.

**Key property:** This layer **forks per model.** When a user selects Ridge + Random Forest + MLP, three separate preprocessing pipelines exist simultaneously. Each model gets a bespoke version of the data shaped to its assumptions. This per-model pipeline architecture is the app's core differentiator — most tools assume one universal pipeline.

**Implemented via:** `preprocessing_pipelines_by_model`, model registry capabilities, per-model config in Preprocess page.

### Layer 2: Insight (middle level)

Based on the data profile, is the user making the right choices? This layer observes, recommends, tracks decisions, and narrates them for publication.

**Lifecycle:** Detect → Recommend → Act → Narrate

**Key property:** Insights are **model-aware.** "BMI is skewed" matters for Ridge and MLP but not for Random Forest. Coaching contextualizes recommendations via `model_scope` on each insight. Resolution details capture per-model actions.

**Implemented via:** `InsightLedger` (unified system in `utils/insight_ledger.py`). Single source of truth for observations, coaching, methodology tracking, TRIPOD auto-completion, and manuscript narrative generation.

**Four views of one ledger:**
1. **Page coaching** — filtered: "3 items need attention on this page" (rendered by `utils/coaching_ui.py`)
2. **Progress summary** — counts: "12 detected, 8 resolved, 4 open"
3. **Methodology audit** — chronological record of every decision
4. **Publication narrative** — manuscript-ready prose grouped by workflow phase

### Layer 3: Presentation (highest level)

Are users guided through the steps in an easily digestible fashion? This layer determines what the user sees, when, and how.

**Key property:** Consistent companion experience. The coaching component (`coaching_ui.py`) renders in the same position, with the same interaction pattern and visual language on every page. Severity-based icons (🚨 blocker, ⚠️ warning, ℹ️ info, 💡 opportunity) and consistent styling across all pages.

---

## The Workflow Pages (11 pages)

| # | Page | Purpose |
|---|------|---------|
| 01 | Upload & Audit | Project-based data management with intelligent merging, target/feature selection, task type detection |
| 02 | EDA | Exploratory analysis with auto-generated insights that feed the coaching layer |
| 03 | Feature Engineering | Optional transforms, interaction terms, domain-driven feature creation |
| 04 | Feature Selection | LASSO path, RFE-CV, univariate screening, stability selection |
| 05 | Preprocess | Per-model pipeline builder — imputation, scaling, encoding, power transforms |
| 06 | Train & Compare | Train models, evaluate, compare metrics, show diagnostics |
| 07 | Explainability | Permutation importance, SHAP, partial dependence, external validation, subgroup analysis |
| 08 | Sensitivity Analysis | Robustness checks and perturbation analysis |
| 09 | Hypothesis Testing | Traditional statistical tests to validate ML findings, Table 1 generation |
| 10 | Report Export | Generate downloadable manuscript bundle (LaTeX + figures + tables) |
| 11 | Theory Reference | Interactive statistical foundations — browsable reference with live demos |

---

## The OODA Loop Model

Each model the user selects spins up a new decision loop:

```
                    ┌→ Orient(Ridge) → Decide(Ridge) → Act(Ridge)
Observe (EDA) ──→  ├→ Orient(RF)    → Decide(RF)    → Act(RF)
                    └→ Orient(MLP)   → Decide(MLP)   → Act(MLP)
```

- **Observe:** EDA reveals facts about the data (skew, missing, correlations, sufficiency)
- **Orient:** Coaching contextualizes those facts for each selected model family via `model_scope`
- **Decide:** User configures preprocessing, feature selection, hyperparameters per model
- **Act:** Training, evaluation, explainability, sensitivity analysis

---

## Provenance Architecture (Dual System)

The app has two complementary provenance systems that together produce a complete manuscript:

### InsightLedger (`utils/insight_ledger.py`) — Coaching lifecycle
Handles observe → recommend → resolve. EDA detects "BMI is skewed," coaching recommends a transform, the user acts, and the resolution is recorded. Each insight carries:
- `model_scope`: which model families it applies to (empty = all)
- `theory_anchor`: links to an interactive demo on the Theory Reference page
- Severity levels: blocker, warning, info, opportunity

Drives the coaching UI and contributes the *why* to the manuscript.

### WorkflowProvenance (`utils/workflow_provenance.py`) — Pipeline record
Captures what actually happened at each workflow stage — structured, typed, incrementally built. Each page writes its section via `record_*()` methods as the user acts. Structured as a chain of typed dataclasses:

- `UploadProvenance` — target, task type, features, cleaning actions
- `EDAProvenance` — distributional findings, correlation structure
- `FeatureEngineeringProvenance` — transforms applied
- `FeatureSelectionProvenance` — methods used, features retained
- `PreprocessProvenance` — per-model pipeline configurations
- `TrainingProvenance` — models trained, metrics, CV strategy
- `ExplainabilityProvenance` — importance rankings, SHAP results

Consumers read the whole structure via `get_methods_context()`. Contributes the *what* to the manuscript.

### NarrativeEngine (`ml/narrative_engine.py`) — Manuscript generation
**This is the key integration layer.** Single entry point that reads from both WorkflowProvenance + InsightLedger and produces a complete, internally consistent manuscript draft.

```
WorkflowProvenance (what happened) + InsightLedger (what was considered)
    ↓
NarrativeEngine.generate() → ManuscriptDraft
    ↓
ManuscriptDraft.to_markdown() / .to_latex()
```

Replaces the old approach where Report Export assembled prose from 100+ scattered session_state reads.

### ManuscriptValidator (`ml/manuscript_validator.py`) — Pre-export QA
Runs consistency checks before export: model name agreement, metric consistency, population counts, section completeness. Catches errors before they reach the LaTeX output.

---

## Theory Reference System

The Theory Reference is a pedagogical layer that bridges coaching insights to interactive demonstrations:

### Theory Anchors (`utils/theory_anchors.py`)
Maps concept keys (e.g., `"skewness"`, `"collinearity"`) to:
- Plain-language "why this matters" summary (shown inline on coaching cards)
- Common misconception to surface
- "What to look for" prompt (trains the user's eye on their own results)
- Chapter/section pointer to the full Theory Reference page

### Theory Demos (`utils/theory_demos.py`)
Portable interactive Plotly demos that can render on any page via widget key namespacing. Examples: skewness distribution explorer, calibration curve demo, bias-variance tradeoff visualization.

### Theory Reference Page (`pages/11_Theory_Reference.py`)
Browsable statistical reference organized by chapter. Each chapter covers the mathematical assumptions, interactive visualizations, and practical implications for model selection.

---

## Model Architecture

### Model Registry (`ml/model_registry.py`)
Central registry of available models with capability declarations (handles missing, handles categorical, needs scaling, etc.). Per-model preprocessing defaults flow from these capabilities.

### Model Wrappers (`models/`)
Consistent interface for all model families:
- `glm.py` — OLS Linear Regression / Logistic Regression
- `huber_glm.py` — Huber-robust GLM
- `rf.py` — Random Forest
- `nn_whuber.py` — Neural Network with weighted Huber/BCE/CE loss
- `registry_wrappers.py` — Adapters for registry models to training infrastructure

---

## Data Persistence

### Project System (`utils/dataset_db.py`)
SQLite-backed project and dataset management. Supports:
- Project-based organization (multiple datasets per project)
- Dataset merging with intelligent column reconciliation
- Working table management

### Session Management (`utils/session_manager.py`)
Save/load full analysis sessions (pickle-based). Users can download and resume their workflow, preserving all state including ledger entries, provenance records, and trained models.

---

## Vertical Cuts

Whenever a new feature or ML model is added, it cuts across all three layers:

| What changes | Layer 1 (Data) | Layer 2 (Insight) | Layer 3 (Presentation) |
|---|---|---|---|
| New model added | Register capabilities, preprocessing defaults | Know when to recommend it, what it needs, what to warn about | Show in selector, render results |
| New transform added | Apply to data, track column lineage | Recommend when appropriate, resolve related insights | Expose controls, show before/after |
| New analysis added | Compute statistics | Generate insights from results | Render visualizations, coaching annotations |
| New theory concept | N/A | Add theory_anchor to relevant insights | Add demo to theory_demos.py, section to Theory Reference page |

---

## Design Principles

1. **The process brings clarity.** The product isn't the point — the understanding that emerges from building it is. Don't shortcut the methodology for speed.

2. **Model-aware, not model-specific.** Coach at the family level (linear, tree, neural), not the individual model level. The user selects Ridge — they should understand it as a regularized linear model, not just "Ridge."

3. **Reviewer-focused tone.** Coaching says "a reviewer would flag this" rather than "you should fix this." The app represents methodological standards, not personal opinion.

4. **Synchronicity across layers.** When data changes, insights update, and presentation reflects. When the user acts, the resolution propagates everywhere. No stale state, no orphaned recommendations.

5. **Per-model pipelines are first-class.** Every part of the app should respect that different models need different data. This is not a complication to manage — it's the core value proposition.

6. **Actions are a proxy for intent.** We can't read minds, but we can track what the user did and reconstruct their reasoning chain. The methodology log and resolution trail serve this purpose.

7. **The manuscript is the ultimate output.** Everything flows toward a defensible, publishable paper. If a feature doesn't eventually contribute to the manuscript narrative, question whether it belongs.

8. **Theory is pedagogically integrated.** Coaching doesn't just tell you *what* to do — it links to interactive demonstrations of *why* via theory anchors. The app teaches as it guides.

---

## File Map

| Concern | Primary files |
|---|---|
| **Data pipeline** | `ml/preprocessing.py`, `ml/preprocess_operators.py`, `ml/pipeline.py`, `ml/training.py` |
| **Model registry** | `ml/model_registry.py`, `models/*.py` (wrappers) |
| **Insight layer** | `utils/insight_ledger.py` (single source of truth) |
| **Coaching UI** | `utils/coaching_ui.py` (renders ledger insights consistently) |
| **Theory system** | `utils/theory_anchors.py` (inline bridges), `utils/theory_demos.py` (interactive demos), `pages/11_Theory_Reference.py` |
| **Provenance** | `utils/workflow_provenance.py` (structured pipeline record) |
| **Manuscript generation** | `ml/narrative_engine.py` (NarrativeEngine → ManuscriptDraft) |
| **Manuscript validation** | `ml/manuscript_validator.py` (pre-export consistency checks) |
| **Report export** | `ml/publication.py`, `ml/latex_report.py`, `pages/10_Report_Export.py` |
| **Table 1** | `ml/table_one.py` (publication-ready descriptive statistics) |
| **Dataset profile** | `ml/dataset_profile.py` (pure computation, feeds insight producers) |
| **EDA intelligence** | `ml/eda_recommender.py`, `ml/eda_actions.py` |
| **Feature selection** | `ml/feature_selection.py`, `ml/feature_steps.py` |
| **Evaluation** | `ml/eval.py`, `ml/calibration.py`, `ml/bootstrap.py`, `ml/sensitivity.py` |
| **Regime detection** | `ml/regime.py` (adaptive layout based on dataset shape) |
| **Data persistence** | `utils/dataset_db.py` (SQLite projects), `utils/session_manager.py` (save/load) |
| **Navigation** | `utils/storyline.py` (breadcrumbs), `utils/theme.py` (CSS/styling) |
| **Utilities** | `utils/reconcile.py`, `utils/column_utils.py`, `utils/widget_helpers.py`, `utils/seed.py` |
| **Tests** | `tests/` (unit + integration + workflow tests) |

---

## Earmarked Future Features

### Target Variable Transformation (Preprocessing)
The app supports power transforms (log1p, Yeo-Johnson) for *features* but has no UI for transforming the *target variable*. For regression tasks with skewed targets, this is a gap — the coach flags the problem but the user has no in-app way to fix it. The Preprocess page should offer target transforms with clear guidance and automatic back-transformation of predictions. Must integrate with the per-model pipeline.

---

*This document is the commander's intent for the Tabular Machine Learning Lab. Any agent working on this codebase should read it before making changes. If a proposed change conflicts with these principles, raise it — don't silently violate them.*
