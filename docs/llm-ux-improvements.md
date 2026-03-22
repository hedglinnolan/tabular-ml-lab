# LLM Interpretation UX Improvements

**Branch:** `feature/llm-interpretation-improvements`
**Created:** 2026-03-22

---

## Current Problems

### 1. Two competing "Interpretation" labels
The automated narrative renders as `**Interpretation:** ...` and the LLM result renders as `**🔬 AI Interpretation:** ...`. Both are bold markdown paragraphs with no visual distinction. The relationship between them (automated = quick summary, AI = expert deep dive) isn't communicated.

### 2. Expander creates visual noise before the button
The `💬 Add context for the AI (optional)` expander appears ABOVE the button, before the user has even decided to click. This is 12+ expanders cluttering the page across model diagnostics. Most users won't type custom context on first use.

### 3. Button breaks visual flow
Natural flow: plot → interpretation → next plot. The LLM button + expander sit between the interpretation and the next section, creating a speed bump that must be scrolled past.

### 4. No visual hierarchy between automated and AI interpretation
Both are plain markdown. The AI interpretation carries more weight (it's context-aware, connects to background findings) but looks identical to the automated summary.

### 5. Button text is redundant
"Interpret with AI" when an interpretation already exists above it. Doesn't convey what the AI adds beyond the automated narrative.

### 6. Classification ROC/PR curves have no interpretation button
These are displayed for classification models but get no LLM interpretation, despite being among the most commonly misunderstood diagnostic plots.

---

## Proposed Changes

### A. Rename and restyle the automated narrative
**Current:** `**Interpretation:** The residual distribution shows...`
**Proposed:** `**Summary:** The residual distribution shows...`

This is a quick, deterministic summary. Call it what it is. Reserve "interpretation" for the LLM's deeper analysis.

### B. Restructure the LLM interaction as a collapsible callout BELOW the summary
**Current flow:**
```
[Plot]
[**Interpretation:** automated text]
[Expander: add context]
[Button: Interpret with AI]
[Result: AI Interpretation text]
```

**Proposed flow:**
```
[Plot]
[**Summary:** automated text]
[Button: 🧠 Deep Analysis]
  → when clicked, result appears in a styled callout box:
  ┌─────────────────────────────────────────┐
  │ 🧠 AI Analysis                          │
  │                                         │
  │ [interpretation text]                   │
  │                                         │
  │ ┌─ 💬 Ask a follow-up (optional) ─────┐│
  │ │ [text area]                          ││
  │ │ [🔄 Re-analyze]                      ││
  │ └─────────────────────────────────────┘│
  └─────────────────────────────────────────┘
```

**Key changes:**
- Context text area moves INSIDE the result area (shown only after first click)
- Result renders in a visually distinct container (colored border, subtle background)
- Follow-up question capability replaces the pre-click expander
- Button text changes to "🧠 Deep Analysis" (clearer value prop)

### C. Visual styling for AI results
Use a styled container that's visually distinct from the automated summary:
```python
st.markdown(
    f'<div style="border-left: 3px solid #6366f1; padding: 12px 16px; '
    f'background: rgba(99, 102, 241, 0.04); border-radius: 4px; '
    f'margin: 8px 0;">'
    f'<strong>🧠 AI Analysis</strong><br/><br/>{result}</div>',
    unsafe_allow_html=True,
)
```

### D. Consolidate per-model diagnostics
On the Train page, a user looking at HistGB results sees:
- Pred vs Actual → Summary + LLM button
- Residuals → Summary + LLM button  
- (Confusion matrix if classification → Summary + LLM button)

That's 2-3 LLM buttons per model. Consider a SINGLE "🧠 Analyze all diagnostics" button per model that sends ALL diagnostic results in one context package. The LLM can then cross-reference (e.g., "the residual skew explains the pred vs actual scatter pattern"). This reduces button clutter while improving interpretation quality.

**Trade-off:** Loses the per-plot specificity. Could offer both: individual buttons on each plot + a combined button at the end of the model's diagnostic section.

### E. Add interpretation to missing analysis types

**ROC/PR curves (classification):** Add button after the ROC and PR curve plots. Analysis type hint: "Is this AUC sufficient for the clinical/practical context? Does the PR curve reveal issues that ROC masks (class imbalance)?"

**Feature Selection (04):** Add button after the consensus feature table. Analysis type hint: "Why did these methods agree/disagree on specific features? Are there domain-relevant features that were excluded?"

---

## Implementation Priority

1. **Restyle AI result display** (callout box) — highest visual impact, least code change
2. **Move context expander inside result** — reduces pre-click clutter
3. **Rename automated narrative to "Summary"** — clarifies hierarchy
4. **Rename button to "🧠 Deep Analysis"** — clearer value prop
5. **Add ROC/PR interpretation** — new coverage for classification
6. **Consider combined per-model analysis button** — discuss with Nolan first
