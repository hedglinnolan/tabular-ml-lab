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

INTERPRETATION_SYSTEM_PROMPT = """You are a senior biostatistician and data scientist reviewing analysis results for a research publication. Your role is to provide interpretation that a scientist can use directly.

RULES:
1. EXPLAIN what the results MEAN for this specific dataset — don't just restate numbers
2. Give ACTIONABLE recommendations (what should the researcher do next?)
3. Flag concerns a PEER REVIEWER would raise
4. Be SPECIFIC — reference the actual values, features, and context provided
5. If you see red flags (e.g., data leakage, overfitting, insufficient sample size), say so clearly
6. Suggest what to CHECK or INVESTIGATE further
7. Use clear, professional language suitable for a methods/results section discussion
8. Keep it concise — 3-5 key points, not a wall of text

DO NOT:
- Simply paraphrase or restate the statistical output
- Give generic textbook explanations
- Be vague ("the results look good") — be precise about WHY
- Ignore potential problems to be polite"""


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
    """Build rich context for the LLM with all available information.

    Now includes dataset profile, accumulated EDA insights, preprocessing choices,
    and model family context for much more specific interpretation.
    """
    parts = []

    # Header
    parts.append(f"## Analysis: {plot_type}")
    parts.append(f"Statistical results: {stats_summary}")

    # Dataset context
    context_parts = []
    if task_type:
        context_parts.append(f"Task: {task_type}")
    if sample_size is not None:
        context_parts.append(f"Sample size: n={sample_size:,}")
    domain = data_domain_hint or (_infer_domain_hint(feature_names) if feature_names else "")
    if domain:
        context_parts.append(f"Domain: {domain}")
    if context_parts:
        parts.append("Context: " + " | ".join(context_parts))

    # Dataset profile (if available)
    if dataset_profile:
        profile_parts = []
        for key in ("n_rows", "n_features", "n_numeric", "n_categorical",
                     "data_sufficiency", "p_n_ratio", "n_features_with_missing"):
            val = dataset_profile.get(key)
            if val is not None:
                profile_parts.append(f"{key}={val}")
        if profile_parts:
            parts.append("Dataset profile: " + ", ".join(profile_parts))

    # Model info
    if model_name:
        parts.append(f"Model: {model_name}" + (f" ({model_family})" if model_family else ""))
    if metrics:
        kv = "; ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in list(metrics.items())[:8])
        parts.append(f"Performance metrics: {kv}")

    # Features
    if feature_names:
        if len(feature_names) <= 12:
            parts.append(f"Features ({len(feature_names)}): {', '.join(str(x) for x in feature_names)}")
        else:
            feats = ", ".join(str(x) for x in feature_names[:10])
            parts.append(f"Features ({len(feature_names)}): {feats}, … (+{len(feature_names)-10} more)")

    # Preprocessing
    if preprocessing_config:
        prep_parts = []
        for key in ("numeric_scaling", "numeric_imputation", "numeric_outlier_treatment",
                     "categorical_encoding", "use_pca", "use_kmeans_features"):
            val = preprocessing_config.get(key)
            if val is not None and val != "none" and val is not False:
                prep_parts.append(f"{key}={val}")
        if prep_parts:
            parts.append("Preprocessing: " + ", ".join(prep_parts))

    # Accumulated EDA insights
    if eda_insights:
        parts.append("Prior EDA findings:\n" + "\n".join(f"  - {i}" for i in eda_insights[:8]))

    # Existing interpretation (as background only)
    if existing:
        parts.append(f"Existing automated summary (use as background, do NOT paraphrase): {existing}")

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
        return _call_ollama(context, system_prompt, model or "llama3.1:8b", ollama_url)
    elif backend == "openai":
        return _call_openai(context, system_prompt, model or "gpt-4o-mini", api_key)
    elif backend == "anthropic":
        return _call_anthropic(context, system_prompt, model or "claude-sonnet-4-20250514", api_key)
    else:
        logger.warning(f"Unknown LLM backend: {backend}")
        return None


def _call_ollama(context: str, system_prompt: str, model: str, url: str) -> Optional[str]:
    """Call Ollama API."""
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

        resp = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": context,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 800},
            },
            timeout=60,
        )
        if resp.ok:
            return resp.json().get("response", "").strip()
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

def render_llm_settings_sidebar():
    """Render LLM configuration in sidebar."""
    import streamlit as st

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
                value=st.session_state.get("ollama_model", "llama3.1:8b"),
                key="ollama_model",
                help="Model name (e.g., llama3.1:8b, mistral, gemma2)",
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
                value=st.session_state.get("openai_model", "gpt-4o-mini"),
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
                value=st.session_state.get("anthropic_model", "claude-sonnet-4-20250514"),
                key="anthropic_model",
            )


def render_interpretation_with_llm_button(
    context: str,
    key: str,
    result_session_key: Optional[str] = None,
) -> None:
    """Render LLM interpretation button with rich context.

    Now uses the configured backend and enriched system prompt.
    """
    import streamlit as st

    sk = result_session_key or f"llm_result_{key}"
    user_ctx_key = f"{key}_user_context"

    # Get backend config
    backend = st.session_state.get("llm_backend", "ollama")

    with st.expander("💬 Add context for the AI (optional)", expanded=False):
        st.caption(
            "Tell the AI what to focus on — e.g., clinical implications, "
            "concerns about sample size, specific features of interest."
        )
        st.text_area(
            "Your context",
            key=user_ctx_key,
            placeholder="E.g., 'Focus on whether these results are strong enough for a JAMA submission' or 'I'm worried about the outliers in BMI'",
            label_visibility="collapsed",
        )

    if st.button("🔬 Interpret with AI", key=key, help="Get expert-level interpretation of these results"):
        ctx = context or ""
        user_txt = (st.session_state.get(user_ctx_key) or "").strip()
        if user_txt:
            ctx += f"\n\nResearcher's specific question/focus: {user_txt}"

        # Gather additional context from session state
        eda_insights = []
        insights_list = st.session_state.get("insights", [])
        if isinstance(insights_list, list):
            for ins in insights_list[:10]:
                if isinstance(ins, dict) and "finding" in ins:
                    eda_insights.append(ins["finding"])
        if eda_insights:
            ctx += "\n\nPrior findings from this analysis session:\n" + "\n".join(f"- {i}" for i in eda_insights)

        # Call LLM
        model = ""
        api_key = ""
        ollama_url = "http://localhost:11434"

        if backend == "ollama":
            model = st.session_state.get("ollama_model", "llama3.1:8b")
        elif backend == "openai":
            model = st.session_state.get("openai_model", "gpt-4o-mini")
            api_key = st.session_state.get("openai_api_key", "")
            if not api_key:
                st.session_state[sk] = "__no_key__"
        elif backend == "anthropic":
            model = st.session_state.get("anthropic_model", "claude-sonnet-4-20250514")
            api_key = st.session_state.get("anthropic_api_key", "")
            if not api_key:
                st.session_state[sk] = "__no_key__"

        if st.session_state.get(sk) != "__no_key__":
            with st.spinner(f"Getting interpretation from {backend} ({model})..."):
                result = _call_llm(
                    ctx, INTERPRETATION_SYSTEM_PROMPT,
                    backend=backend, model=model, api_key=api_key, ollama_url=ollama_url,
                )

            if result:
                st.session_state[sk] = result
            else:
                st.session_state[sk] = "__error__"
                logger.error(f"LLM call returned None: backend={backend}, model={model}")
            # No st.rerun() — result is in session state and displays below
            # Avoids resetting tab/expander position on the page

    # Display result
    res = st.session_state.get(sk)
    if res == "__no_key__":
        st.warning(f"Please configure your {backend.title()} API key in the sidebar (🤖 LLM Settings).")
    elif res == "__unavailable__":
        st.caption(
            "To use this feature: (1) Install Ollama from [ollama.ai](https://ollama.ai). "
            "(2) Run `ollama serve` in a terminal. "
            "(3) Pull a model: `ollama pull llama3.2`."
        )
    elif res == "__error__":
        st.warning(
            f"Could not get interpretation. Check sidebar LLM Settings. "
            f"Current: backend={st.session_state.get('llm_backend', 'ollama')}, "
            f"model={st.session_state.get('ollama_model', 'llama3.1:8b')}. "
            f"Verify Ollama is running: `curl http://localhost:11434/api/tags`"
        )
    elif res:
        st.markdown(f"**🔬 AI Interpretation:**\n\n{res}")
