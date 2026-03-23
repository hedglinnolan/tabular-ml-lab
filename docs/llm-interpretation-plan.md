# LLM Interpretation Improvement Plan

**Branch:** `feature/llm-interpretation-improvements`
**Created:** 2026-03-22
**Status:** Planning

---

## Problem Statement

The LLM interpretation feature exists but underdelivers. Three independent problems:

1. **Context is thin** — the context builder (`build_llm_context`) has rich optional parameters for dataset profile, EDA insights, preprocessing config, and model family, but *no call site passes them*. The LLM gets a stats summary string and feature names — not enough to give specific advice.

2. **Model is underpowered** — `llama3.1:8b` on a GTX 1080 Ti (11GB VRAM) gives textbook-level paraphrasing, not analytical interpretation. Better models now fit in the same VRAM budget.

3. **Coverage is incomplete** — only 3 of 7 analysis pages offer LLM interpretation. Missing pages include ones where interpretation would be most valuable (Sensitivity Analysis, Hypothesis Testing).

---

## Workstream 1: Context Enrichment (Logical Layer)

### Current State
`build_llm_context()` accepts these params that are **never passed** at any call site:

| Parameter | What it provides | Where it lives in session_state |
|---|---|---|
| `dataset_profile` | n_rows, n_features, sufficiency, p/n ratio, missing count | `st.session_state.dataset_profile` |
| `eda_insights` | Accumulated findings from EDA runs | `st.session_state.eda_results` (findings lists) |
| `preprocessing_config` | Scaling, imputation, outlier treatment, encoding choices | `st.session_state.preprocessing_pipelines_by_model` or per-model config |
| `model_family` | linear/tree/neural/distance/margin/prob | Derivable from model name via `MODEL_TO_FAMILY` in `insight_ledger.py` |
| `data_domain_hint` | clinical/nutrition/epidemiology | Auto-inferred if feature_names passed (already works) |

### Plan

**1a. Build a `gather_session_context()` helper** in `utils/llm_ui.py` that pulls all available context from `st.session_state` in one call. Returns a dict matching `build_llm_context` kwargs. Every call site uses this instead of manually assembling params.

```python
def gather_session_context() -> dict:
    """Pull all available analysis context from session state."""
    import streamlit as st
    ctx = {}
    # dataset profile
    if "dataset_profile" in st.session_state:
        ctx["dataset_profile"] = st.session_state.dataset_profile
    # accumulated EDA insights
    eda_results = st.session_state.get("eda_results", {})
    findings = []
    for action_id, result in eda_results.items():
        findings.extend(result.get("findings", [])[:3])
    if findings:
        ctx["eda_insights"] = findings[:12]
    # preprocessing config (from first model's pipeline as representative)
    pipelines = st.session_state.get("preprocessing_pipelines_by_model", {})
    if pipelines:
        first_model = next(iter(pipelines))
        ctx["preprocessing_config"] = pipelines[first_model].get("config", {})
    # ledger unresolved items as context
    ledger = st.session_state.get("insight_ledger")
    if ledger:
        unresolved = ledger.get_unresolved()
        if unresolved:
            ctx["eda_insights"] = ctx.get("eda_insights", []) + [
                f"[UNRESOLVED] {ins.description}" for ins in unresolved[:5]
            ]
    return ctx
```

**1b. Update all existing call sites** to use `gather_session_context()`:
- `pages/02_EDA.py` — `_run_and_show()` (1 location)
- `pages/06_Train_and_Compare.py` — learning curves, pred vs actual, residuals, confusion matrix (4 locations)
- `pages/07_Explainability.py` — permutation importance, SHAP, Bland-Altman (3 locations)

**1c. Pass `model_family`** at Train and Explainability call sites using the existing `MODEL_TO_FAMILY` mapping from `insight_ledger.py`.

**1d. Add insight ledger context** — resolved insights tell the LLM what the user already addressed (e.g., "skew was detected and log-transformed"). Unresolved insights tell it what's still open. This is the single most valuable context addition — it turns the LLM from a one-shot interpreter into a session-aware advisor.

### Estimated effort: ~2 hours

---

## Workstream 2: Model Upgrade

### Current State
- **Hardware:** GTX 1080 Ti (11,264 MB VRAM), i5-8600K, 16GB RAM
- **Current model:** `llama3.1:8b` (4.9GB, Q4_K_M) — adequate for summarization, poor at analytical reasoning
- **VRAM budget:** ~10.5GB usable (need ~700MB for Ollama overhead + nomic-embed-text stays loaded for memory search)

### Model Candidates

| Model | Params | VRAM (Q4_K_M) | MATH Score | GPQA Diamond | Notes |
|---|---|---|---|---|---|
| **Qwen3.5-9B** | 9B | ~6.5GB | — | **81.7** | Beats GPT-OSS-120B on reasoning. Best efficiency/capability ratio in class. |
| **Phi-4** | 14B | ~8.5GB | **80.4** | — | Microsoft. Best MATH per GB. Tight fit but feasible. |
| **Gemma 3 12B QAT** | 12B | ~9.4GB | — | — | Google's quantization-aware training. Very tight on 1080 Ti. |
| **DeepSeek-R1 14B** | 14B | ~8.8GB | — | — | Explicit chain-of-thought. Good for step-by-step statistical reasoning. |
| llama3.1:8b (current) | 8B | 4.9GB | 68.0 | — | Baseline. Adequate but not competitive. |

### Recommendation

**Primary: Qwen3.5-9B** — fits comfortably (6.5GB leaves room for KV cache), dramatically better reasoning benchmarks, and the model is specifically praised for analytical tasks. The GPQA Diamond score (81.7) is a science/reasoning benchmark, which maps directly to our use case.

**Secondary test: Phi-4 at Q4_K_M** — if Qwen3.5-9B disappoints on statistical interpretation specifically, Phi-4's MATH score suggests it might handle numerical reasoning better. Tighter VRAM fit (~8.5GB) but feasible.

### Plan

**2a. Pull and test Qwen3.5-9B:**
```bash
ollama pull qwen3.5:9b
```

**2b. Run a head-to-head comparison** — same context (SHAP interpretation from a real analysis run), same prompt, compare llama3.1:8b vs Qwen3.5-9B output quality. Grade on: specificity, actionability, statistical accuracy, conciseness.

**2c. Update default model** in `llm_ui.py` and `llm_local.py` (or deprecate `llm_local.py` entirely — it's likely dead code).

**2d. Update system prompt** — with a better model, we can ask for more structured output. Current prompt says "3-5 key points." With Qwen3.5-9B we could structure as: (1) What the results show, (2) What's concerning, (3) What to do next — matching the app's advisor persona.

### Estimated effort: ~1 hour (mostly waiting for model download + testing)

---

## Workstream 3: Presentation Layer Coverage

### Current Coverage

| Page | LLM Interpretation | Analyses Covered |
|---|---|---|
| EDA | ✅ | All 7 EDA actions via `_run_and_show()` |
| Feature Engineering | ❌ | — |
| Feature Selection | ❌ | — |
| Preprocess | ❌ | — |
| Train & Compare | ✅ | Learning curves, pred vs actual, residuals, confusion matrix |
| Explainability | ✅ | Permutation importance, SHAP, Bland-Altman |
| Sensitivity Analysis | ❌ | — |
| Hypothesis Testing | ❌ | — |
| Report Export | ⚠️ | Can include cached LLM results, no new interpretation |

### Where interpretation adds the most value (priority order)

1. **Sensitivity Analysis** (HIGH) — Seed sensitivity, bootstrap stability, feature dropout. Users often don't know how to interpret stability metrics. "Is 2% variance in AUROC across seeds concerning?" — that's exactly what an LLM advisor should answer.

2. **Hypothesis Testing** (HIGH) — Statistical test results (t-tests, chi-squared, ANOVA, correlation). Users frequently misinterpret p-values, effect sizes, and multiple comparison corrections. High value, clear context to pass.

3. **Feature Selection** (MEDIUM) — Interpretation of which features were selected and why (LASSO path, RFE curve, univariate scores). Less critical because the coaching layer already advises here.

4. **Preprocess** (LOW) — Pipeline configuration is more about decisions than interpretation. The coaching layer already covers this well.

5. **Feature Engineering** (LOW) — Transform impact is shown via before/after distributions. LLM interpretation is marginal here.

### Plan

**3a. Add LLM interpretation to Sensitivity Analysis page** — after seed sensitivity results, bootstrap confidence intervals, and feature dropout tables. Build context from the stability metrics + model performance + dataset size.

**3b. Add LLM interpretation to Hypothesis Testing page** — after each test result (t-test, chi-squared, ANOVA, correlation matrix). Build context from test statistics + p-values + effect sizes + sample size + multiple comparison status.

**3c. Standardize the UX pattern** — currently the button reads "🔬 Interpret with AI" with an optional context expander above it. This works but could be tighter:
- Move user context text area *inside* the interpretation result area (show after first interpretation, for follow-up questions)
- Add a "📋 Add to Report" button on each interpretation that flags it for inclusion in Report Export
- Consider auto-collapsing the interpretation into a styled callout box (matching the coaching UI pattern)

### Estimated effort: ~3-4 hours

---

## Execution Order

1. **Context enrichment first** (Workstream 1) — improves quality for all existing call sites immediately
2. **Model upgrade second** (Workstream 2) — amplifies the context improvements
3. **Coverage expansion third** (Workstream 3) — extend to new pages with the improved context + model

Total estimated: ~6-7 hours across sessions.

---

## Cleanup

- **Deprecate `ml/llm_local.py`** — verify it's not imported anywhere, then remove. `utils/llm_ui.py` is the single LLM module.
- **Reconcile default models** — `llm_local.py` says `qwen2.5:7b`, `llm_ui.py` says `llama3.1:8b`. After model upgrade, there should be one default in one place.
- **Update ARCHITECTURE.md** — add LLM interpretation as a documented concern in the file map.
