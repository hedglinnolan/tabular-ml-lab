# Tabular ML Lab — Product Review (2026-03-11)

**Reviewer:** Claude (CPO hat) + Nolan (confirmed alignment)
**Status:** Capture for iterative work over coming weeks

---

## The Thesis (Strong)

"Clinical/biomedical researchers who need ML in their papers shouldn't have to write Python. Give them a guided workflow that produces publication-ready materials."

**Niche:** The gap between SPSS and PyCaret. The researcher who currently emails a data science collaborator because they can't do ML themselves. Nobody else generates a methods section + TRIPOD checklist + Table 1 from the same interface.

**Competitors and why we're different:**
- **PyCaret** — low-code but still code. No publication output.
- **JADBio** — closest (AutoML for biomarker discovery, no-code). But SaaS, proprietary, expensive, omics-focused.
- **AutoML platforms** (H2O, Vertex, Azure ML) — enterprise. No publication output. Overkill for 500-row clinical datasets.
- **TabPFN** — foundation model, Python-only. Research, not publication.
- **SPSS / JMP** — traditional stats. No ML workflow. No SHAP. No TRIPOD.

---

## What's Working

1. **10-step workflow mirrors how a reviewer reads a methods section.** Pedagogical AND productive.
2. **Methodology logging + generation is a real moat.** No competitor does this.
3. **Guardrails prevent retraction-worthy mistakes.** Data leakage warnings, group splitting, double-transform detection.
4. **Explainability suite is comprehensive.** SHAP + permutation + PDP + calibration + subgroup + decision curves.

---

## What Needs Work (Priority Order)

### P0: Simplify Upload & Audit for first contact
- Page 1 is 1700 lines. Project management, multi-dataset merging, column reconciliation.
- Promise: "30-60 minutes, no coding." Reality: feels like configuring an enterprise ETL tool.
- **Fix:** "Simple Mode" — drop CSV, pick target column, go. Advanced features behind an expander. First-time UX under 2 minutes to reach EDA.

### P0: Make Export genuinely publication-ready
- Methods section has `[PLACEHOLDER: Add specific software versions]`
- TRIPOD checklist items say "Review and update"
- **Fix:** Fill placeholders automatically. Add software versions. Generate complete TRIPOD checklist. Methods section should require zero editing for parts the app controls. Domain-specific context is the only thing the user should add.

### P1: Quick Path vs Full Path
- 10 pages is too many for a researcher who just wants results
- Linear navigation implies ALL steps required when they're not
- FE + Feature Selection could be one page with tabs
- Sensitivity + StatVal are often skipped
- **Fix:** "Quick Path" (Upload → EDA → Preprocess → Train → Export) and "Full Path" (all 10). Sidebar dims optional pages. Progress celebration at milestones.

### P1: No onboarding / guided tour
- No contextual "why" — only "what"
- No first-time walkthrough
- No progress celebration ("You've completed EDA! Here's what you've learned")
- `render_guidance()` blocks are static, don't adapt to user state
- **Fix:** Adaptive contextual help. First-run onboarding overlay. Milestone celebrations.

### P2: EDA page underpowered for its importance
- EDA is where researchers spend the most time understanding data
- Compared to Sweetviz, DataPrep, ydata-profiling — visual density is low
- **Missing:** Interactive scatter plots, missing data heatmaps, outlier visualization, distribution overlays
- These exist in the codebase but feel secondary to automated analysis buttons

### P2: AI features feel bolted on
- Optional LLM integration (Ollama/OpenAI/Anthropic) is mentioned but feels like afterthought
- **Opportunity:** Use LLM to interpret results in plain English — "Your model shows Age and BMI are the strongest predictors, with a non-linear threshold effect around BMI 30"
- In 2026, a research tool without meaningful AI integration feels dated

### P3: No collaboration or sharing
- Single-user Streamlit session
- No way to share results with co-author, export link, or have supervisor review
- For academic research, collaboration is essential

---

## Strategic Questions (Open)

1. **Who deploys this?** University research computing departments. Have we talked to even one? Product-market fit isn't "is the app good" — it's "will research computing departments install this for their faculty?"

2. **Retention loop?** Researcher uses this once per paper (2-3/year). Very low frequency. Community opportunity: shared preprocessing recipes, published methodology templates, domain model benchmarks.

3. **Quality bar for "publication-ready"?** We say outputs are drafts. That's legally correct but commercially weak. Value prop should be: "your only edits are domain-specific context that no tool can provide."

---

## Bottom Line

The app has a real value proposition that no competitor matches. The engineering is solid. The statistical methodology is defensible.

But it's currently designed for people who already know ML and want a convenient wrapper. The actual target audience — clinical researchers who *don't* know ML — would struggle with cognitive load, linear rigidity, and incomplete export.

**The hardest engineering is done. The hardest product work hasn't started.**

Gap between "technically correct tool" and "product researchers love" = UX, onboarding, and polish.

---

## Action Items (for coming weeks)
- [ ] P0: Simple Mode for Upload & Audit
- [ ] P0: Auto-fill Export placeholders + complete TRIPOD
- [ ] P1: Quick Path / Full Path toggle with sidebar dimming
- [ ] P1: First-run onboarding experience
- [ ] P2: Enrich EDA with interactive visuals
- [ ] P2: Meaningful LLM integration for result interpretation
- [ ] P3: Explore sharing/collaboration features
- [ ] Talk to at least 1 research computing department about deployment
