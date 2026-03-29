# ARCHITECTURE.md — Product Architecture & Commander's Intent

**Read this before touching any code. This is the design philosophy of the Tabular ML Lab.**

---

## What This App Is

A research workbench that guides scientists through a complete, defensible ML workflow on tabular data — from raw CSV to compilable LaTeX manuscript. The user provides domain expertise; the app provides methodological rigor.

## The Three Layers

Every feature, every page, every interaction exists within three layers that must stay synchronized:

### Layer 1: Working Data (lowest level)

What is actually happening to the data. Feature engineering, imputation, scaling, encoding, splitting, training, evaluation.

**Key property:** This layer **forks per model.** When a user selects Ridge + Random Forest + MLP, three separate preprocessing pipelines exist simultaneously. Each model gets a bespoke version of the data shaped to its assumptions. This per-model pipeline architecture is the app's core differentiator — most tools assume one universal pipeline.

**Implemented via:** `preprocessing_pipelines_by_model`, model registry capabilities, per-model config in Preprocess page.

### Layer 2: Insight (middle level)

Based on the data profile, is the user making the right choices? This layer observes, recommends, tracks decisions, and narrates them for publication.

**Lifecycle:** Detect → Recommend → Act → Narrate

**Key property:** Insights must be **model-aware.** "BMI is skewed" matters for Ridge and MLP but not for Random Forest. Coaching should contextualize recommendations to the models the user has selected. Resolution details should capture per-model actions.

**Implemented via:** `InsightLedger` (unified system). Single source of truth for observations, coaching, methodology tracking, TRIPOD auto-completion, and manuscript narrative generation.

**Four views of one ledger:**
1. **Page coaching** — filtered: "3 items need attention on this page"
2. **Progress summary** — counts: "12 detected, 8 resolved, 4 open"
3. **Methodology audit** — chronological record of every decision
4. **Publication narrative** — manuscript-ready prose grouped by workflow phase

### Layer 3: Presentation (highest level)

Are users guided through the steps in an easily digestible fashion? This layer determines what the user sees, when, and how.

**Key property:** Consistent companion experience. The insight layer should feel like the same advisor on every page — same position, same interaction pattern, same visual language. When the user moves from EDA to Feature Engineering to Preprocess, the coaching component should be familiar and predictable.

---

## The OODA Loop Model

Each model the user selects spins up a new decision loop:

```
                    ┌→ Orient(Ridge) → Decide(Ridge) → Act(Ridge)
Observe (EDA) ──→  ├→ Orient(RF)    → Decide(RF)    → Act(RF)
                    └→ Orient(MLP)   → Decide(MLP)   → Act(MLP)
```

- **Observe:** EDA reveals facts about the data (skew, missing, correlations, sufficiency)
- **Orient:** Coaching contextualizes those facts for each selected model family
- **Decide:** User configures preprocessing, feature selection, hyperparameters per model
- **Act:** Training, evaluation, explainability, sensitivity analysis

We cannot know exactly what the user is thinking. But we can:
1. **Inform** on a per-model basis (presented cleanly at Layer 3)
2. **Track actions taken** as a proxy for their decision-making
3. **Generate narrative** that reconstructs the reasoning chain for publication

---

## Vertical Cuts

Whenever a new feature or ML model is added, it cuts across all three layers:

| What changes | Layer 1 (Data) | Layer 2 (Insight) | Layer 3 (Presentation) |
|---|---|---|---|
| New model added | Register capabilities, preprocessing defaults | Know when to recommend it, what it needs, what to warn about | Show in selector, render results |
| New transform added | Apply to data, track column lineage | Recommend when appropriate, resolve related insights | Expose controls, show before/after |
| New analysis added | Compute statistics | Generate insights from results | Render visualizations, coaching annotations |

If a new feature does not actively change data, it almost always **reads status** at every layer (e.g., auto-generated methodology entries that passively record what happened).

---

## The Per-Model Complication

This is the hardest architectural problem in the app.

Layer 1 already handles per-model pipelines. But Layers 2 and 3 still largely think in terms of one universal dataset. The gap:

- **Current:** "BMI is skewed → apply log transform" (blanket recommendation)
- **Target:** "BMI is skewed → applies to your Ridge and MLP pipelines. RF handles this natively."

- **Current:** Resolution records "Applied log1p to BMI" (one entry)
- **Target:** Resolution records "log1p applied in Ridge/MLP pipeline; raw features retained for RF"

- **Current:** Manuscript says "Features were log-transformed prior to modeling"
- **Target:** Manuscript says "Log transforms were applied to skewed features for regularized linear models; tree-based models operated on untransformed features"

Insight `model_scope` is the mechanism to close this gap. Each insight should know which model families it's relevant to, and coaching/resolution/narrative should respect that scope.

---

## Design Principles

1. **The process brings clarity.** The product isn't the point — the understanding that emerges from building it is. Don't shortcut the methodology for speed.

2. **Model-aware, not model-specific.** Coach at the family level (linear, tree, neural), not the individual model level. The user selects Ridge — they should understand it as a regularized linear model, not just "Ridge."

3. **Reviewer-focused tone.** Coaching says "a reviewer would flag this" rather than "you should fix this." The app represents methodological standards, not personal opinion.

4. **Synchronicity across layers.** When data changes, insights update, and presentation reflects. When the user acts, the resolution propagates everywhere. No stale state, no orphaned recommendations.

5. **Per-model pipelines are first-class.** Every part of the app should respect that different models need different data. This is not a complication to manage — it's the core value proposition.

6. **Actions are a proxy for intent.** We can't read minds, but we can track what the user did and reconstruct their reasoning chain. The methodology log and resolution trail serve this purpose.

7. **The manuscript is the ultimate output.** Everything flows toward a defensible, publishable paper. If a feature doesn't eventually contribute to the manuscript narrative, question whether it belongs.

---

## File Map

| Concern | Primary files |
|---|---|
| Data pipeline | `ml/preprocessing.py`, `ml/training.py`, model registry |
| Insight layer | `utils/insight_ledger.py` (single source of truth) |
| Coaching producers | EDA auto-insights (`pages/02_EDA.py`), `log_methodology()` bridge |
| Per-model config | `pages/05_Preprocess.py`, `preprocessing_pipelines_by_model` |
| Manuscript generation | `ml/publication.py`, `ml/latex_report.py`, `ledger.to_manuscript_narrative()` |
| Navigation/presentation | `utils/storyline.py` (breadcrumbs), `utils/theme.py` |
| Dataset profile | `ml/dataset_profile.py` (pure computation, feeds insight producers) |
| Regime detection | `ml/regime.py` (adaptive layout based on dataset shape) |
| Workflow provenance | `utils/workflow_provenance.py` (structured pipeline record) |

---

## Provenance Architecture

The app has two complementary provenance systems:

### InsightLedger (coaching lifecycle)
Handles observe → recommend → resolve. EDA detects "BMI is skewed," coaching recommends a transform, the user acts, and the resolution is recorded. This drives the coaching UI and narrative generation.

### WorkflowProvenance (pipeline record)
Captures what actually happened at each workflow stage — structured, typed, incrementally built. Each page writes its section via `record_*()` methods as the user acts. Consumers (NarrativeEngine, TRIPOD checker, consistency validator, Report Export) read the whole structure via `get_methods_context()`.

**Why two systems?** They serve different purposes. The InsightLedger tracks *recommendations and decisions* (coaching). WorkflowProvenance tracks *configurations and actions* (provenance). A coaching insight says "BMI is skewed → we recommend log transform for linear models." The provenance record says "Ridge pipeline: Yeo-Johnson power transform applied; RF pipeline: no transform." Both are needed for a complete manuscript — the InsightLedger explains *why*, the provenance records *what*.

**Migration path:** Report Export currently reads from ~100 scattered session_state keys. As the provenance layer matures, these reads migrate to `get_methods_context()`. The old reads remain as fallbacks during transition.

---

## Earmarked Future Features

### Target Variable Transformation (Preprocessing)
The app currently supports power transforms (log1p, Yeo-Johnson) for *features* in Preprocessing, but has no UI for transforming the *target variable*. For regression tasks with skewed targets, this is a gap — the coach flags the problem but the user has no in-app way to fix it. The Preprocess page should offer target transforms (log1p, Yeo-Johnson, Box-Cox) with clear guidance: "Your target is right-skewed. Transforming it can improve linear model performance. Predictions will be automatically back-transformed." Must integrate with the per-model pipeline — tree models may not need it.

### Theory Reference Guide
The app applies statistical theory across every page (why collinearity hurts linear models, why skew affects distance-based methods, why class imbalance misleads accuracy) but never shows its work. Two reference layers are needed:
1. **Model theory** — the mathematical assumptions underpinning each model family (linearity, independence, normality of residuals for linear; feature independence for Naive Bayes; smoothness for distance-based; etc.). Should be accessible from the model selector and anywhere models are compared.
2. **Coaching rationale** — why the coach recommends what it recommends. Each insight type should link to a brief explanation: "We flag collinearity for linear models because correlated predictors inflate coefficient variance (VIF > 10 ⟹ unstable estimates). See: James et al., ISLR §3.3.3." This makes the app pedagogically useful and lets reviewers verify the tool's reasoning.

Both should be structured as a browsable reference, not inline tooltips — something a student or reviewer can consult independently.

---

*This document is the commander's intent for the Tabular ML Lab. Any agent working on this codebase should read it before making changes. If a proposed change conflicts with these principles, raise it — don't silently violate them.*
