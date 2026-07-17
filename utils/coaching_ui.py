"""
Reusable coaching UI component — the insight companion that lives on every page.

Renders ledger insights in a consistent, model-aware format.
Call render_page_coaching(page_id) at the top of any page.
"""
from typing import Optional, List
import streamlit as st
from utils.insight_ledger import (
    get_ledger, InsightLedger, Insight,
    MODEL_TO_FAMILY, FAMILY_DISPLAY_NAMES, models_to_families,
    SEVERITY_ORDER,
)
from utils.theory_anchors import infer_theory_anchor, render_theory_link


SEVERITY_ICONS = {
    "blocker": "🚨",
    "warning": "⚠️",
    "info": "ℹ️",
    "opportunity": "💡",
}

SEVERITY_STYLES = {
    "blocker": "error",
    "warning": "warning",
    "info": "info",
    "opportunity": "success",
}


def _get_selected_models() -> List[str]:
    """Get the user's currently selected models from session state.

    The model-selection checkboxes (train_model_<key>, shared by the
    Preprocess and Train pages) are the source of truth; an explicit
    'selected_models' key acts as an override for programmatic use. Nothing
    wrote 'selected_models' before, which left model-aware coaching —
    grouping and per-family scoping — permanently inactive.
    """
    explicit = st.session_state.get("selected_models")
    if explicit:
        return list(explicit)
    from_checkboxes = [k.replace("train_model_", "")
                       for k, v in st.session_state.items()
                       if k.startswith("train_model_") and v is True]
    if from_checkboxes:
        return from_checkboxes
    # train_model_* keys become widget-bound on the Train page, so Streamlit
    # garbage-collects them when the user navigates elsewhere. Fall back to
    # the durable record of models whose pipelines were built on Preprocess.
    built = st.session_state.get("preprocess_built_model_keys")
    return list(built) if built else []


def render_page_coaching(
    page_id: str,
    show_resolved: bool = True,
    show_model_grouping: bool = True,
    compact: bool = False,
) -> None:
    """Render the coaching companion for a page.

    This is the standard entry point — call it near the top of each page.
    It handles:
    - Model-aware grouping when models are selected
    - Fallback to flat list when no models selected yet
    - Resolved insights (collapsed)
    - Blocker banners

    Args:
        page_id: The page identifier (e.g., "05_Preprocess")
        show_resolved: Whether to show resolved insights
        show_model_grouping: Whether to group by model family
        compact: If True, use minimal rendering (no expander wrapper)
    """
    ledger = get_ledger()
    if len(ledger) == 0:
        return  # No insights yet — nothing to show

    unresolved = ledger.get_unresolved(page=page_id)
    resolved = ledger.get_resolved(page=page_id) if show_resolved else []
    selected_models = _get_selected_models()

    if not unresolved and not resolved:
        return  # Nothing relevant to this page

    # Blocker banner — always visible, never collapsed
    blockers = [i for i in unresolved if i.severity == "blocker"]
    if blockers:
        for b in blockers:
            st.error(f"🚨 **Blocker:** {b.finding}")
            if b.recommended_action:
                st.caption(f"→ {b.recommended_action}")

    # Non-blocker insights
    non_blockers = [i for i in unresolved if i.severity != "blocker"]

    if not non_blockers and not resolved:
        return

    # Build the label
    n_open = len(non_blockers)
    n_resolved = len(resolved)
    label_parts = []
    if n_open:
        label_parts.append(f"{n_open} open")
    if n_resolved:
        label_parts.append(f"{n_resolved} resolved")
    label = f"📋 Coaching ({', '.join(label_parts)})"

    if compact:
        _render_insights_body(non_blockers, resolved, selected_models, show_model_grouping, page_id=page_id)
    else:
        with st.expander(label, expanded=bool(non_blockers)):
            _render_insights_body(non_blockers, resolved, selected_models, show_model_grouping, page_id=page_id)


def _render_insights_body(
    unresolved: List[Insight],
    resolved: List[Insight],
    selected_models: List[str],
    show_model_grouping: bool,
    page_id: str = "page",
) -> None:
    """Render the insights content — called inside or outside an expander."""
    # Track which theory demos have been shown to avoid duplicates
    shown_demos: set = set()

    if unresolved:
        if show_model_grouping and selected_models:
            _render_model_grouped(unresolved, selected_models, page_id=page_id, shown_demos=shown_demos)
        else:
            _render_flat(unresolved, page_id=page_id, shown_demos=shown_demos)

    if resolved:
        st.markdown("---")
        st.caption("**Resolved:**")
        for ins in resolved[:5]:
            st.caption(f"✅ ~~{ins.finding}~~ → {ins.resolved_by}")
        if len(resolved) > 5:
            st.caption(f"... and {len(resolved) - 5} more")


_SEVERITY_RANK = {"blocker": 0, "warning": 1, "info": 2, "opportunity": 3}


def _render_model_grouped(insights: List[Insight], selected_models: List[str], page_id: str = "page", shown_demos: set = None) -> None:
    """Render insights grouped by model family.

    Each insight renders exactly ONCE: universal insights first (they affect
    every model), then one card per scoped insight under its first matched
    family — the full applicability list is already shown by the insight's
    italic scope chip. Rendering the same card once per matched family
    produced two or three identical cards for a single skewness finding.
    Clean families collapse into a single trailing reassurance line.
    """
    families = models_to_families(selected_models)
    universal = []
    by_family = {}

    for ins in insights:
        if not ins.model_scope:
            universal.append(ins)
        else:
            matched_families = [f for f in families if f in ins.model_scope]
            if matched_families:
                display = FAMILY_DISPLAY_NAMES.get(matched_families[0], matched_families[0])
                by_family.setdefault(display, []).append(ins)
            else:
                # Insight has model_scope but none of the user's models match
                # This means it's irrelevant — skip it
                pass

    # Universal insights first — class imbalance / missing-data warnings must
    # not render below per-family info notes.
    if universal:
        universal = sorted(universal, key=lambda i: _SEVERITY_RANK.get(i.severity, 9))
        st.markdown(f"**All Models** ({len(universal)} item{'s' if len(universal) > 1 else ''})")
        for ins in universal:
            _render_single_insight(ins, page_context=page_id, shown_demos=shown_demos)

    # Family sections, most severe contents first
    def _section_rank(items):
        return min((_SEVERITY_RANK.get(i.severity, 9) for i in items), default=9)

    for family_display, items in sorted(by_family.items(), key=lambda kv: _section_rank(kv[1])):
        items = sorted(items, key=lambda i: _SEVERITY_RANK.get(i.severity, 9))
        st.markdown(f"**{family_display}** ({len(items)} item{'s' if len(items) > 1 else ''})")
        for ins in items:
            _render_single_insight(ins, page_context=page_id, shown_demos=shown_demos)

    # Collapse clean families into one line. "Clean" = no insight's scope
    # includes the family at all (an insight filed under its first matched
    # family still counts for its other matched families via its scope chip).
    matched_any = set()
    for ins in insights:
        if ins.model_scope:
            matched_any.update(f for f in families if f in ins.model_scope)
    clean = [FAMILY_DISPLAY_NAMES.get(f, f) for f in families if f not in matched_any]
    if clean:
        st.markdown(f"✅ No family-specific issues for {', '.join(clean)}")


def _render_single_insight(ins: Insight, page_context: str = "coaching", shown_demos: set = None) -> None:
    """Render one insight with its theory link and inline demo if available.

    Args:
        shown_demos: Set of anchor keys already shown on this page.
            If the anchor was already shown, skip the demo (avoid duplicates).
    """
    if shown_demos is None:
        shown_demos = set()

    icon = SEVERITY_ICONS.get(ins.severity, "ℹ️")
    scope_hint = ""
    if ins.model_scope:
        scope_names = [FAMILY_DISPLAY_NAMES.get(f, f) for f in ins.model_scope]
        scope_hint = f" _{', '.join(scope_names)}_"
    st.markdown(f"  {icon} {ins.finding}{scope_hint}")
    if ins.recommended_action:
        st.caption(f"    → {ins.recommended_action}")

    # Theory link — only show demo on first occurrence of each anchor
    anchor_key = infer_theory_anchor(ins)
    if anchor_key:
        if anchor_key not in shown_demos:
            shown_demos.add(anchor_key)
            ctx = f"{page_context}_{anchor_key}"
            render_theory_link(anchor_key, compact=False, page_context=ctx)
        # For subsequent insights with the same anchor, no demo — just a quiet note
        # (the user already has the demo above)


def _render_flat(insights: List[Insight], page_id: str = "page", shown_demos: set = None) -> None:
    """Render insights as a flat list (no models selected yet)."""
    for ins in insights:
        _render_single_insight(ins, page_context=page_id, shown_demos=shown_demos)


def render_coaching_summary_badge(page_id: str) -> None:
    """Render a compact badge showing insight count for sidebar use.

    e.g., "⚠️ 3 items" or "✅ All resolved"
    """
    ledger = get_ledger()
    unresolved = ledger.get_unresolved(page=page_id)

    if not unresolved:
        if ledger.get_resolved(page=page_id):
            st.caption("✅ All insights resolved")
    else:
        blockers = sum(1 for i in unresolved if i.severity == "blocker")
        warnings = sum(1 for i in unresolved if i.severity == "warning")
        if blockers:
            st.caption(f"🚨 {blockers} blocker{'s' if blockers > 1 else ''}, {len(unresolved)} total")
        elif warnings:
            st.caption(f"⚠️ {len(unresolved)} insight{'s' if len(unresolved) > 1 else ''} to review")
        else:
            st.caption(f"ℹ️ {len(unresolved)} note{'s' if len(unresolved) > 1 else ''}")
