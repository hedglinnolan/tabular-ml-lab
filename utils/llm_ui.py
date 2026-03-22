"""
Reusable UI for LLM-powered interpretation of analysis results.

Supports: Ollama (default), OpenAI API, Anthropic API.
Sends rich context with a system prompt that demands actionable, specific interpretation.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

_MAX_TABLE_ROWS = 20
_MAX_TABLE_CHARS = 2000

# ============================================================================
# Default model configuration — single source of truth
# ============================================================================

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# ============================================================================
# Domain detection
# ============================================================================

_DOMAIN_KEYWORDS = {
    "clinical": {"glucose", "bmi", "age", "weight", "height", "bp", "hdl", "ldl",
                  "hb", "waist", "hip", "cholesterol", "hba1c", "insulin",
                  "creatinine", "hemoglobin", "albumin", "triglycerides",
                  "systolic", "diastolic", "egfr", "sbp", "dbp"},
    "nutrition": {"calories", "carbs", "protein", "fat", "fiber", "sodium",
                  "potassium", "calcium", "iron", "vitamin", "nutrient",
                  "dietary", "intake", "servings", "kcal", "macronutrient"},
    "epidemiology": {"incidence", "prevalence", "odds_ratio", "hazard",
                     "exposure", "cohort", "case_control", "risk_factor",
                     "mortality", "morbidity", "survival"},
}


def _infer_domain_hint(feature_names: Optional[List[str]] = None) -> str:
    """Infer domain from feature/column names."""
    if not feature_names:
        return ""
    names_lower = " ".join(str(x).lower() for x in feature_names)
    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for k in keywords if k in names_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else ""


# ============================================================================
# System prompt for interpretation
# ============================================================================

INTERPRETATION_SYSTEM_PROMPT = """You interpret statistical analysis results in a tabular ML research workbench (Streamlit app). The researcher sees a specific plot or table and clicks "Interpret with AI" for expert analysis.

# ANALYTICAL CHECKLIST — address each section in order

## VALIDITY
- Are assumptions of this method met given the data context?
- Do known issues (collinearity, missingness, skew, small n) affect THIS result?
- Is the sample size adequate for this analysis?

## INTERPRETATION
- What does this specific result mean for this dataset?
- Distinguish statistical significance from practical significance
- Reference actual values from the results — add insight the numbers alone don't show

## CONCERNS
- What would a peer reviewer flag?
- Connect to unresolved issues from prior analysis steps if relevant

## NEXT STEP
- One concrete action the researcher can take in this tool

# DOMAIN CONSTRAINTS — follow these as hard rules
- VIF > 5 between features means SHAP and permutation importance rankings are unreliable for those features
- R² > 0.95 on tabular data is suspicious — flag possible data leakage
- Effect sizes matter more than p-values for clinical/practical relevance
- n < 30 per group weakens parametric test assumptions
- 48%+ missing data in a feature means imputed values dominate — flag reduced reliability
- Residual patterns in linear models suggest assumption violations; in tree models they indicate systematic prediction gaps (different diagnosis)

# OUTPUT RULES
- Maximum 5 points total across all sections
- If the researcher asked a question, answer it first
- Do NOT define methods (no "SHAP values measure the marginal contribution...")
- Do NOT explain formulas or general model properties
- Do NOT summarize the background context
- Do NOT restate numbers the researcher can already see
- Do NOT start with "Great question" or "Certainly" or "Based on the analysis"
"""

# Analysis-type-specific hints injected into the system prompt
# These add targeted sub-questions to the VALIDITY section
ANALYSIS_TYPE_HINTS: Dict[str, str] = {
    "learning_curves": (
        "- Is there evidence of overfitting (train loss much lower than val loss) or underfitting (both high)?\n"
        "- Has the model converged, or would more epochs help?"
    ),
    "pred_vs_actual": (
        "- Are errors evenly distributed, or concentrated in specific prediction ranges?\n"
        "- Does the scatter suggest heteroscedasticity?"
    ),
    "residuals": (
        "- Do residual patterns suggest violated assumptions for this model family?\n"
        "- Is there systematic under/over-prediction in specific ranges?"
    ),
    "confusion_matrix": (
        "- Does class imbalance or threshold choice affect this result?\n"
        "- Which misclassification type is more costly in this domain?"
    ),
    "permutation_importance": (
        "- Does collinearity between features affect the reliability of these importance rankings?\n"
        "- Are the top features consistent with domain expectations, or do they suggest data leakage?"
    ),
    "SHAP": (
        "- Does collinearity between features affect the reliability of these SHAP rankings?\n"
        "- Do SHAP interaction effects suggest feature dependencies the model is exploiting?"
    ),
    "bland_altman": (
        "- Is the performance difference practically meaningful or within noise?\n"
        "- Does the limits of agreement width suggest clinical/practical interchangeability?"
    ),
    "roc_curve": (
        "- Is this AUC sufficient for the clinical/practical context?\n"
        "- Does the curve shape suggest the model performs differently at different thresholds?"
    ),
    "pr_curve": (
        "- Does the PR curve reveal class imbalance issues that ROC masks?\n"
        "- Is precision maintained at practically useful recall levels?"
    ),
    "correlation": (
        "- Is the correlation coefficient practically meaningful, not just statistically significant?\n"
        "- Could confounders or non-linear relationships affect this result?"
    ),
    "two_group_comparison": (
        "- Is the effect size meaningful regardless of p-value?\n"
        "- Were assumptions (normality, equal variance) checked before this test?"
    ),
    "multi_group_comparison": (
        "- If significant, which specific group differences drive the result?\n"
        "- Is the effect size (eta-squared, omega-squared) practically meaningful?"
    ),
    "chi_squared": (
        "- Are expected cell counts adequate (>5) for chi-squared validity?\n"
        "- Does Cramér's V indicate a practically meaningful association?"
    ),
    "seed_sensitivity": (
        "- How much variance is attributable to random initialization vs genuine model instability?\n"
        "- Is the coefficient of variation acceptable for the intended application?"
    ),
    "bootstrap_ci": (
        "- Is the confidence interval width acceptable for practical decision-making?\n"
        "- Does the interval cross any clinically or practically meaningful thresholds?"
    ),
    "feature_dropout": (
        "- Which features cause the largest performance drops when removed?\n"
        "- Does removing a feature improve performance (suggesting noise or collinearity)?"
    ),
    "feature_selection": (
        "- Why did selection methods agree or disagree on specific features?\n"
        "- Are there domain-relevant features that were excluded that shouldn't have been?"
    ),
}


def _build_system_prompt(plot_type: str) -> str:
    """Compose the full system prompt with analysis-type-specific hints.

    The base analytical framework is static. Type-specific hints are injected
    into the VALIDITY section to guide the model's reasoning for this
    particular kind of analysis.
    """
    base = INTERPRETATION_SYSTEM_PROMPT
    hints = ANALYSIS_TYPE_HINTS.get(plot_type, "")
    if hints:
        base += f"\n# ANALYSIS-SPECIFIC CHECKS for {plot_type}\n{hints}\n"
    return base


# ============================================================================
# Session context gatherer
# ============================================================================

def gather_session_context() -> Dict[str, Any]:
    """Pull all available analysis context from Streamlit session state.

    Returns a dict of kwargs suitable for passing to build_llm_context().
    This is background context — the broad project state that helps the LLM
    give specific interpretations of focal results.
    """
    import streamlit as st

    ctx: Dict[str, Any] = {}

    # Dataset profile
    profile = st.session_state.get("dataset_profile")
    if profile and isinstance(profile, dict):
        ctx["dataset_profile"] = profile

    # Task type and sample info from data_config
    data_config = st.session_state.get("data_config")
    if data_config:
        if hasattr(data_config, "task_type") and data_config.task_type:
            ctx["task_type"] = data_config.task_type
        if hasattr(data_config, "feature_cols") and data_config.feature_cols:
            ctx["feature_names"] = list(data_config.feature_cols)

    # Sample size from working dataframe
    df = st.session_state.get("working_df")
    if df is not None and hasattr(df, "__len__"):
        ctx["sample_size"] = len(df)

    # Accumulated EDA findings
    eda_results = st.session_state.get("eda_results", {})
    findings: List[str] = []
    if isinstance(eda_results, dict):
        for _action_id, result in eda_results.items():
            if isinstance(result, dict):
                for f in result.get("findings", [])[:3]:
                    if isinstance(f, str) and f not in findings:
                        findings.append(f)
    if findings:
        ctx["eda_insights"] = findings[:12]

    # Insight ledger — unresolved and resolved items
    ledger = st.session_state.get("insight_ledger")
    if ledger and hasattr(ledger, "get_unresolved"):
        unresolved = ledger.get_unresolved()
        if unresolved:
            unresolved_strs = []
            for ins in unresolved[:6]:
                desc = getattr(ins, "finding", "") or getattr(ins, "description", "")
                sev = getattr(ins, "severity", "")
                if desc:
                    prefix = f"[{sev.upper()}] " if sev else ""
                    unresolved_strs.append(f"{prefix}{desc}")
            if unresolved_strs:
                existing = ctx.get("eda_insights", [])
                ctx["eda_insights"] = existing + [f"[OPEN ISSUE] {s}" for s in unresolved_strs]

        resolved = ledger.get_resolved()
        if resolved:
            resolved_strs = []
            for ins in resolved[:6]:
                finding = getattr(ins, "finding", "")
                action = getattr(ins, "resolved_by", "")
                if finding and action:
                    resolved_strs.append(f"{finding} → {action}")
            if resolved_strs:
                existing = ctx.get("eda_insights", [])
                ctx["eda_insights"] = existing + [f"[RESOLVED] {s}" for s in resolved_strs]

    # Preprocessing config — extract from first model's pipeline as representative
    pipelines = st.session_state.get("preprocessing_pipelines_by_model", {})
    if isinstance(pipelines, dict) and pipelines:
        first_key = next(iter(pipelines))
        pipeline_data = pipelines[first_key]
        if isinstance(pipeline_data, dict):
            config = pipeline_data.get("config", pipeline_data)
            if isinstance(config, dict):
                ctx["preprocessing_config"] = config

    return ctx


# ============================================================================
# Rich context builder
# ============================================================================

def build_eda_full_results_context(result: Dict[str, Any], action_id: str) -> str:
    """Build full EDA-results context: all findings, stats, per-figure/table data."""
    import pandas as pd

    parts: List[str] = []
    findings = result.get("findings", [])
    if findings:
        parts.append("Findings: " + "; ".join(findings))

    stats = result.get("stats", {})
    if stats:
        lines: List[str] = []
        if "correlation_tests" in stats:
            for t in stats["correlation_tests"]:
                if len(t) >= 4:
                    feat, r, p, name = t[0], t[1], t[2], t[3]
                    pv = f", p={p:.4f}" if p is not None and p == p else ""
                    lines.append(f"  {feat}: r={r:.3f}{pv} ({name})")
                elif len(t) >= 2:
                    lines.append(f"  {t[0]}: {t[1]}")
        elif "feature_correlations" in stats:
            for t in stats["feature_correlations"]:
                if len(t) >= 2:
                    lines.append(f"  {t[0]}: |r|={t[1]:.3f}")
        if "vif" in stats:
            vifs = stats["vif"]
            if isinstance(vifs, list):
                bits = []
                for c, v in vifs[:15]:
                    if isinstance(v, (int, float)) and v == v:
                        bits.append(f"{c}={v:.1f}")
                if bits:
                    lines.append("  VIF: " + "; ".join(bits))
        for k in ("shapiro_p", "shapiro_stat", "max_leverage", "max_cooks",
                   "n_high_leverage", "n_high_cooks"):
            if k in stats and stats[k] is not None:
                v = stats[k]
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if "residual_vs_predicted_corr" in stats and stats["residual_vs_predicted_corr"] is not None:
            lines.append(f"  residual_vs_predicted_corr: {stats['residual_vs_predicted_corr']:.4f}")
        if lines:
            parts.append("Stats: " + " | ".join(lines))

    figures = result.get("figures", [])
    corr_tests = stats.get("correlation_tests", []) if isinstance(stats, dict) else []
    corr_feats = [x[0] for x in corr_tests] if corr_tests else []

    for idx, (fig_type, fig_data) in enumerate(figures):
        n = idx + 1
        if fig_type == "table":
            if hasattr(fig_data, "head") and hasattr(fig_data, "to_csv"):
                df = fig_data
                try:
                    head = df.head(_MAX_TABLE_ROWS)
                    raw = head.to_markdown(index=False)
                except Exception:
                    try:
                        raw = head.to_csv(index=False)
                    except Exception:
                        raw = head.to_string()
                if len(raw) > _MAX_TABLE_CHARS:
                    raw = raw[:_MAX_TABLE_CHARS] + "\n..."
                parts.append(f"Table {n}:\n{raw}")
            else:
                parts.append(f"Table {n}: [tabular data]")
        else:
            desc = ""
            if fig_type == "plotly" and hasattr(fig_data, "layout") and hasattr(fig_data.layout, "title"):
                t = getattr(fig_data.layout.title, "text", None)
                if t:
                    desc = t
            if not desc and idx < len(corr_feats):
                feat = corr_feats[idx]
                for t in (corr_tests or []):
                    if len(t) >= 4 and t[0] == feat:
                        desc = f"{feat}: r={t[1]:.3f}, p={t[2]:.4f} ({t[3]})"
                        break
            if not desc:
                desc = "scatter" if "scatter" in str(action_id).lower() or "linearity" in action_id else "plot"
            parts.append(f"Figure {n}: {desc}.")

    return "\n\n".join(parts) if parts else ""


def build_llm_context(
    plot_type: str,
    stats_summary: str,
    model_name: Optional[str] = None,
    where: Optional[str] = None,
    existing: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    task_type: Optional[str] = None,
    data_domain_hint: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    eda_insights: Optional[List[str]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    model_family: Optional[str] = None,
) -> str:
    """Build structured context for LLM interpretation.

    Separates FOCAL context (the specific result to interpret) from BACKGROUND
    context (project state that helps the LLM be specific). The LLM should
    interpret the focal result, using background to ground its advice.
    """
    parts = []

    # ── FOCAL ANALYSIS (what the user is looking at right now) ──
    parts.append("# FOCAL ANALYSIS — interpret THIS result")
    parts.append(f"Analysis type: {plot_type}")
    if where:
        parts.append(f"Location: {where}")
    parts.append(f"Results:\n{stats_summary}")

    if model_name:
        model_desc = model_name
        if model_family:
            model_desc += f" ({model_family} family)"
        parts.append(f"Model: {model_desc}")

    if metrics:
        kv = "; ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in list(metrics.items())[:8])
        parts.append(f"Performance metrics: {kv}")

    if existing:
        parts.append(f"Automated summary (do NOT repeat — add what it misses): {existing}")

    # ── BACKGROUND (project state for grounding — do not summarize) ──
    bg_parts = []

    # Dataset basics
    context_bits = []
    if task_type:
        context_bits.append(f"Task: {task_type}")
    if sample_size is not None:
        context_bits.append(f"n={sample_size:,}")
    domain = data_domain_hint or (_infer_domain_hint(feature_names) if feature_names else "")
    if domain:
        context_bits.append(f"Domain: {domain}")
    if context_bits:
        bg_parts.append(" | ".join(context_bits))

    # Dataset profile
    if dataset_profile:
        profile_parts = []
        for key in ("n_rows", "n_features", "n_numeric", "n_categorical",
                     "data_sufficiency", "p_n_ratio", "n_features_with_missing"):
            val = dataset_profile.get(key)
            if val is not None:
                profile_parts.append(f"{key}={val}")
        if profile_parts:
            bg_parts.append("Dataset: " + ", ".join(profile_parts))

    # Features
    if feature_names:
        if len(feature_names) <= 12:
            bg_parts.append(f"Features ({len(feature_names)}): {', '.join(str(x) for x in feature_names)}")
        else:
            feats = ", ".join(str(x) for x in feature_names[:10])
            bg_parts.append(f"Features ({len(feature_names)}): {feats}, … (+{len(feature_names)-10} more)")

    # Preprocessing
    if preprocessing_config:
        prep_parts = []
        for key in ("numeric_scaling", "numeric_imputation", "numeric_outlier_treatment",
                     "categorical_encoding", "use_pca", "use_kmeans_features"):
            val = preprocessing_config.get(key)
            if val is not None and val != "none" and val is not False:
                prep_parts.append(f"{key}={val}")
        if prep_parts:
            bg_parts.append("Preprocessing: " + ", ".join(prep_parts))

    # Prior findings + insight ledger state
    if eda_insights:
        bg_parts.append("Prior findings & decisions:\n" + "\n".join(f"  - {i}" for i in eda_insights[:15]))

    if bg_parts:
        parts.append("\n# BACKGROUND — use to ground your interpretation, do NOT summarize")
        parts.extend(bg_parts)

    return "\n".join(parts)


# ============================================================================
# LLM Backend abstraction
# ============================================================================

def _get_llm_backend(session_state: Optional[Dict] = None) -> str:
    """Get configured LLM backend from session state or default."""
    if session_state:
        return session_state.get("llm_backend", "ollama")
    return "ollama"


def _call_llm(
    context: str,
    system_prompt: str = INTERPRETATION_SYSTEM_PROMPT,
    backend: str = "ollama",
    model: str = "",
    api_key: str = "",
    ollama_url: str = "http://localhost:11434",
) -> Optional[str]:
    """Call LLM backend with context and system prompt.

    Supports: ollama, openai, anthropic.
    Returns the response text or None on error.
    """
    if backend == "ollama":
        return _call_ollama(context, system_prompt, model or DEFAULT_OLLAMA_MODEL, ollama_url)
    elif backend == "openai":
        return _call_openai(context, system_prompt, model or DEFAULT_OPENAI_MODEL, api_key)
    elif backend == "anthropic":
        return _call_anthropic(context, system_prompt, model or DEFAULT_ANTHROPIC_MODEL, api_key)
    else:
        logger.warning(f"Unknown LLM backend: {backend}")
        return None


def _call_ollama(context: str, system_prompt: str, model: str, url: str) -> Optional[str]:
    """Call Ollama API using the chat endpoint.

    Strategy for thinking models (Qwen3.5):
    1. First try WITH thinking enabled (num_predict=3000, 75s timeout)
       - Thinking produces better quality but is slower
       - If content is returned, use it
       - If only thinking is returned (budget exhausted), extract from thinking
    2. If that times out, retry WITHOUT thinking (num_predict=2048, 30s timeout)
       - Still good quality thanks to structured system prompt
       - Fast fallback ensures the user always gets a response
    """
    import requests
    try:
        # Ensure running
        try:
            requests.get(f"{url}/api/tags", timeout=2)
        except Exception:
            import subprocess
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(3)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        # Attempt 1: thinking enabled, moderate budget
        try:
            resp = requests.post(
                f"{url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 3000},
                },
                timeout=75,
            )
            if resp.ok:
                data = resp.json()
                message = data.get("message", {})
                content = (message.get("content") or "").strip()
                if content:
                    return content
                # Thinking but no content — extract from thinking tail
                thinking = (message.get("thinking") or "").strip()
                if thinking:
                    logger.info(f"Ollama thinking-only response ({len(thinking)} chars) — extracting")
                    fallback = thinking[-1000:].strip()
                    last_break = fallback.rfind("\n\n")
                    if last_break > 100:
                        fallback = fallback[last_break:].strip()
                    if fallback:
                        return fallback
        except requests.exceptions.Timeout:
            logger.info("Ollama thinking attempt timed out at 75s — falling back to no-think mode")

        # Attempt 2: thinking disabled, fast fallback
        resp = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.3, "num_predict": 2048},
            },
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            content = (data.get("message", {}).get("content") or "").strip()
            return content or None
        else:
            logger.warning(f"Ollama error: {resp.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _call_openai(context: str, system_prompt: str, model: str, api_key: str) -> Optional[str]:
    """Call OpenAI API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        return None


def _call_anthropic(context: str, system_prompt: str, model: str, api_key: str) -> Optional[str]:
    """Call Anthropic API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": context}],
            max_tokens=800,
            temperature=0.3,
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Anthropic call failed: {e}")
        return None


# ============================================================================
# Streamlit UI components
# ============================================================================

def _apply_pending_llm_widget_restore():
    """Apply deferred LLM widget state before sidebar widgets are instantiated."""
    import streamlit as st

    pending = st.session_state.pop("_pending_widget_state_restore", None)
    if not pending:
        return

    llm_keys = {
        "llm_backend",
        "ollama_model",
        "openai_api_key",
        "openai_model",
        "anthropic_api_key",
        "anthropic_model",
    }
    for key, value in pending.items():
        if key in llm_keys:
            st.session_state[key] = value
        else:
            # Preserve any unrelated deferred widget state for the next owner.
            remaining = st.session_state.get("_pending_widget_state_restore", {})
            remaining[key] = value
            st.session_state["_pending_widget_state_restore"] = remaining


def render_llm_settings_sidebar():
    """Render LLM configuration in sidebar."""
    import streamlit as st

    _apply_pending_llm_widget_restore()

    with st.sidebar.expander("🤖 LLM Settings", expanded=False):
        backend = st.selectbox(
            "LLM Backend",
            ["ollama", "openai", "anthropic"],
            index=["ollama", "openai", "anthropic"].index(
                st.session_state.get("llm_backend", "ollama")
            ),
            key="llm_backend",
            help="Choose which LLM to use for interpretation. Ollama runs locally (free), OpenAI and Anthropic require API keys.",
        )

        if backend == "ollama":
            st.text_input(
                "Ollama model",
                value=st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL),
                key="ollama_model",
                help="Model name (e.g., qwen3.5:9b, llama3.1:8b, gemma2)",
            )
            st.caption("Ollama is running locally on this server — no API key needed.")
        elif backend == "openai":
            st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("openai_api_key", ""),
                key="openai_api_key",
                type="password",
            )
            st.text_input(
                "Model",
                value=st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL),
                key="openai_model",
            )
        elif backend == "anthropic":
            st.text_input(
                "Anthropic API Key",
                value=st.session_state.get("anthropic_api_key", ""),
                key="anthropic_api_key",
                type="password",
            )
            st.text_input(
                "Model",
                value=st.session_state.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL),
                key="anthropic_model",
            )


def _run_llm_call(context: str, plot_type: str, sk: str) -> None:
    """Execute LLM call and store result in session state."""
    import streamlit as st

    backend = st.session_state.get("llm_backend", "ollama")
    model = ""
    api_key = ""
    ollama_url = "http://localhost:11434"

    if backend == "ollama":
        model = st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL)
    elif backend == "openai":
        model = st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL)
        api_key = st.session_state.get("openai_api_key", "")
        if not api_key:
            st.session_state[sk] = "__no_key__"
            return
    elif backend == "anthropic":
        model = st.session_state.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL)
        api_key = st.session_state.get("anthropic_api_key", "")
        if not api_key:
            st.session_state[sk] = "__no_key__"
            return

    with st.spinner(f"🧠 Analyzing... ({model}, up to ~60s)"):
        sys_prompt = _build_system_prompt(plot_type) if plot_type else INTERPRETATION_SYSTEM_PROMPT
        result = _call_llm(
            context, sys_prompt,
            backend=backend, model=model, api_key=api_key, ollama_url=ollama_url,
        )

    if result:
        st.session_state[sk] = result
    else:
        st.session_state[sk] = "__error__"
        logger.error(f"LLM call returned None: backend={backend}, model={model}")


def render_interpretation_with_llm_button(
    context: str,
    key: str,
    result_session_key: Optional[str] = None,
    plot_type: str = "",
) -> None:
    """Render LLM deep analysis button and styled result callout.

    UX flow:
    1. Button: "🧠 Deep Analysis" — clean, no pre-click clutter
    2. On click: LLM call, result appears in styled callout box
    3. Inside callout: follow-up text area + re-analyze button
       (only shown after first result, not before)
    """
    import streamlit as st

    sk = result_session_key or f"llm_result_{key}"
    user_ctx_key = f"{key}_user_context"
    reanalyze_key = f"{key}_reanalyze"

    # Show button only if no result yet
    res = st.session_state.get(sk)
    if not res:
        if st.button("🧠 Deep Analysis", key=key, help="Get expert-level AI interpretation of these results"):
            _run_llm_call(context, plot_type, sk)
            res = st.session_state.get(sk)

    # Display result in styled callout
    if res == "__no_key__":
        backend = st.session_state.get("llm_backend", "ollama")
        st.warning(f"Please configure your {backend.title()} API key in the sidebar (🤖 LLM Settings).")
    elif res == "__unavailable__":
        st.caption(
            "To use this feature: (1) Install Ollama from [ollama.ai](https://ollama.ai). "
            "(2) Run `ollama serve` in a terminal. "
            "(3) Pull a model: `ollama pull qwen3.5:9b`."
        )
    elif res == "__error__":
        st.warning(
            f"Could not get interpretation. Check sidebar LLM Settings. "
            f"Current: backend={st.session_state.get('llm_backend', 'ollama')}, "
            f"model={st.session_state.get('ollama_model', DEFAULT_OLLAMA_MODEL)}. "
            f"Verify Ollama is running: `curl http://localhost:11434/api/tags`"
        )
        # Allow retry
        if st.button("🔄 Retry", key=f"{key}_retry"):
            st.session_state.pop(sk, None)
            _run_llm_call(context, plot_type, sk)
            res = st.session_state.get(sk)
    elif res:
        # Styled callout box with indigo left border
        st.markdown(
            f'<div style="border-left: 3px solid #6366f1; padding: 12px 16px; '
            f'background: rgba(99, 102, 241, 0.04); border-radius: 4px; '
            f'margin: 8px 0 12px 0;">'
            f'<strong style="color: #6366f1;">🧠 AI Analysis</strong>'
            f'<div style="margin-top: 8px;">{res}</div></div>',
            unsafe_allow_html=True,
        )

        # Follow-up area (inside the result context, not before it)
        with st.expander("💬 Ask a follow-up", expanded=False):
            st.text_area(
                "Follow-up question",
                key=user_ctx_key,
                placeholder="E.g., 'Is this good enough for a JAMA submission?' or 'What about the outliers in BMI?'",
                label_visibility="collapsed",
            )
            if st.button("🔄 Re-analyze", key=reanalyze_key):
                ctx = context or ""
                user_txt = (st.session_state.get(user_ctx_key) or "").strip()
                if user_txt:
                    ctx += f"\n\nResearcher's follow-up question: {user_txt}"
                st.session_state.pop(sk, None)
                _run_llm_call(ctx, plot_type, sk)
