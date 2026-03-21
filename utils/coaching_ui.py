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
    """Get the user's currently selected models from session state."""
    return st.session_state.get("selected_models", [])


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
        _render_insights_body(non_blockers, resolved, selected_models, show_model_grouping)
    else:
        with st.expander(label, expanded=bool(non_blockers)):
            _render_insights_body(non_blockers, resolved, selected_models, show_model_grouping)


def _render_insights_body(
    unresolved: List[Insight],
    resolved: List[Insight],
    selected_models: List[str],
    show_model_grouping: bool,
) -> None:
    """Render the insights content — called inside or outside an expander."""
    if unresolved:
        if show_model_grouping and selected_models:
            _render_model_grouped(unresolved, selected_models)
        else:
            _render_flat(unresolved)

    if resolved:
        st.markdown("---")
        st.caption("**Resolved:**")
        for ins in resolved[:5]:
            st.caption(f"✅ ~~{ins.finding}~~ → {ins.resolved_by}")
        if len(resolved) > 5:
            st.caption(f"... and {len(resolved) - 5} more")


def _render_model_grouped(insights: List[Insight], selected_models: List[str]) -> None:
    """Render insights grouped by model family."""
    families = models_to_families(selected_models)
    universal = []
    by_family = {}

    for ins in insights:
        if not ins.model_scope:
            universal.append(ins)
        else:
            matched_families = [f for f in families if f in ins.model_scope]
            if matched_families:
                for f in matched_families:
                    display = FAMILY_DISPLAY_NAMES.get(f, f)
                    by_family.setdefault(display, []).append(ins)
            else:
                # Insight has model_scope but none of the user's models match
                # This means it's irrelevant — skip it
                pass

    # Report clean families
    all_family_displays = [FAMILY_DISPLAY_NAMES.get(f, f) for f in families]
    dirty_families = set(by_family.keys())

    # Render per-family insights
    for family_display in all_family_displays:
        items = by_family.get(family_display, [])
        if items:
            st.markdown(f"**{family_display}** ({len(items)} item{'s' if len(items) > 1 else ''})")
            for ins in items:
                icon = SEVERITY_ICONS.get(ins.severity, "ℹ️")
                st.markdown(f"  {icon} {ins.finding}")
                if ins.recommended_action:
                    st.caption(f"    → {ins.recommended_action}")
        else:
            st.markdown(f"**{family_display}** — ✅ no issues")

    # Universal insights (affect all models)
    if universal:
        st.markdown(f"**All Models** ({len(universal)} item{'s' if len(universal) > 1 else ''})")
        for ins in universal:
            icon = SEVERITY_ICONS.get(ins.severity, "ℹ️")
            st.markdown(f"  {icon} {ins.finding}")
            if ins.recommended_action:
                st.caption(f"    → {ins.recommended_action}")


def _render_flat(insights: List[Insight]) -> None:
    """Render insights as a flat list (no models selected yet)."""
    for ins in insights:
        icon = SEVERITY_ICONS.get(ins.severity, "ℹ️")
        # Add model scope hint if available
        scope_hint = ""
        if ins.model_scope:
            scope_names = [FAMILY_DISPLAY_NAMES.get(f, f) for f in ins.model_scope]
            scope_hint = f" _{', '.join(scope_names)}_"
        st.markdown(f"{icon} {ins.finding}{scope_hint}")
        if ins.recommended_action:
            st.caption(f"  → {ins.recommended_action}")


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
