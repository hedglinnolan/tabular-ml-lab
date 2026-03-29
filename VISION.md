# VISION.md — Competitive Position & Differentiation Strategy

*Last updated: 2026-03-29*

---

## The Landscape

Every tool in this space optimizes for the same thing: **model performance.** Upload data → get the best model → deploy it. The race is to find the highest AUC with the least code.

| Tool | Focus | Output | Provenance | Publication Support |
|------|-------|--------|------------|-------------------|
| **AutoGluon** | Ensemble stacking, multimodal | Model + leaderboard | None | None |
| **PyCaret** | Low-code rapid prototyping | Model + comparison plots | None | None |
| **H2O AutoML** | Scalable enterprise AutoML | Model + basic report | Minimal | None |
| **FLAML** | Cost-efficient hyperparameter search | Model + config | None | None |
| **Auto-sklearn** | Bayesian meta-learning | Model + metrics | None | None |
| **MLJAR** | AutoML with markdown reports | Model + per-model README.md | **Partial** — SHAP, feature importance, training logs | Markdown reports, but no methods section, no TRIPOD |
| **TabPFN 2.5** | Foundation model, zero-shot | Predictions (no training) | None (black box by design) | None |
| **AutoML-Med** (2025) | Medical tabular pipeline optimization | Optimized pipeline | Minimal | None |

### What MLJAR Gets Right (and Where It Stops)

MLJAR (mljar-supervised) is the closest competitor in spirit. It generates markdown documentation for every trained model — feature importance, SHAP plots, training curves, hyperparameter logs. It's "not a black box" by design.

But MLJAR's reports are **engineering artifacts**, not **scientific manuscripts**. They tell you *what happened* but not *why it was the right thing to do*. There's no:
- Methodology justification (why Ridge vs RF, why log transform, why these features)
- Per-model preprocessing documentation (what was done differently for each model family)
- Reporting guideline compliance (TRIPOD+AI, CONSORT flow)
- Reviewer-oriented framing ("a reviewer would flag this")

---

## Where Nobody Plays

There is a **complete vacuum** between "I trained a model" and "I published a defensible paper about it."

Researchers currently bridge this gap manually:
1. Train models in PyCaret/AutoGluon/notebooks
2. Copy-paste metrics into a Word document
3. Write the methods section from memory (hoping they remember what they actually did)
4. Manually check TRIPOD+AI compliance (27 items, most missed)
5. Get reviewer comments that their methods section doesn't match their results
6. Scramble to reconstruct what actually happened

**Nobody automates this.** The publication pipeline is the unsolved problem.

### The TRIPOD+AI Opportunity

TRIPOD+AI (BMJ, April 2024) is a 27-item checklist that is rapidly becoming mandatory for ML prediction model studies in medical and health journals. SciSpace offers an LLM agent that checks *existing manuscripts* against the checklist. But:

- No tool generates TRIPOD-compliant content *from the workflow itself*
- No tool checks compliance *during* the analysis (before you've written anything)
- No tool maps specific workflow actions to specific TRIPOD items in real time

This is like having a building code inspector who only checks the finished building, never the blueprints or construction.

---

## The Tabular ML Lab's Position

**We are not an AutoML tool.** We don't compete with AutoGluon on model performance. We don't compete with TabPFN on zero-shot prediction. We don't compete with PyCaret on speed-to-prototype.

**We are the tool that makes the paper possible.**

The output isn't a model — it's a **defensible, publishable manuscript** with full provenance tracking from raw data to final results. The model is a byproduct of the methodology. The methodology is the product.

### Core Differentiators

**1. Manuscript-First Architecture**
Every feature exists because it contributes to a publishable paper. The methods section isn't generated after the fact — it's assembled from structured provenance captured at every decision point. The InsightLedger tracks observations → recommendations → actions → narrative in a single lifecycle.

**2. Per-Model Pipeline Provenance**
When Ridge gets log-transformed features and Random Forest gets raw features, the methods section says so. No other tool tracks preprocessing divergence at the model-family level and surfaces it in publication prose. This is the core differentiator from MLJAR (which documents per-model performance but not per-model preprocessing rationale).

**3. TRIPOD+AI Compliance by Construction**
Rather than checking compliance after the paper is written, the app maps workflow actions to TRIPOD+AI checklist items in real time. The user sees "14/27 TRIPOD items addressed" as they work, with coaching on what's missing. By the time they export, the manuscript is already compliant.

**4. Reviewer-Oriented Coaching**
The coaching layer doesn't say "you should scale your features." It says "a reviewer would note that your linear models received unscaled features, which inflates the contribution of high-variance predictors." This framing teaches methodology while guiding decisions. No other tool does this.

**5. The User Provides Domain Expertise, The App Provides Rigor**
AutoML tools remove the human. We keep the human in the loop but make their decisions auditable. The researcher decides to use Yeo-Johnson — the app records why it was recommended, when it was applied, which models it affected, and how to describe it in the methods section.

---

## What We Must Build to Own This Position

### Immediate (Weeks)
- **#53: Wire model configs to methods section** — The single most embarrassing gap. The methods section doesn't list which models were trained.
- **#51: Complete MODEL_TO_FAMILY mapping** — Coaching is blind to 7 models.
- **#49: Unify methodology_log and InsightLedger** — Can't claim provenance tracking with two disconnected systems.

### Near-term (1-2 Months)
- **#50: Per-model pipeline provenance** — The flagship differentiator doesn't work yet. Methods says blanket "preprocessing was applied" for all models.
- **TRIPOD+AI real-time tracker** — Map each workflow action to TRIPOD items. Show compliance progress. Generate a compliance report alongside the manuscript.
- **#48: External validation workflow** — TRIPOD requires it. Currently no way to upload a holdout dataset.

### Medium-term (3-6 Months)
- **Tabular foundation model integration** — Add TabPFN 2.5 as a model option alongside traditional ML. Position it as "compare your Ridge/RF/MLP against a foundation model baseline." This is novel — no other interactive tool offers this comparison.
- **Collaborative manuscript editing** — Export to Overleaf/Google Docs with provenance annotations. Let co-authors edit the narrative while keeping the provenance chain intact.
- **Multi-dataset support** — TRIPOD+AI requires development and validation on separate datasets. The app should support this workflow natively.

---

## Who This Is For

**Primary:** Researchers in health sciences, social sciences, and applied domains who use tabular data and need to publish their ML work in peer-reviewed journals. They know their domain, they know enough statistics to be dangerous, and they struggle with the gap between "model that works" and "paper that's accepted."

**Secondary:** Graduate students learning ML methodology. The coaching layer is pedagogical by design — it teaches *why* decisions matter, not just *what* buttons to click.

**Not for:** ML engineers who need production deployment. Data scientists who just want the best AUC. Kaggle competitors. AutoML power users.

---

## The Tagline

**"From CSV to manuscript. Every decision documented."**

Or, more precisely: *The research workbench that closes the gap between training a model and publishing a paper about it.*

---

## Competitive Moat

The moat isn't the ML algorithms (everyone has those). The moat is:
1. **The provenance architecture** — InsightLedger + per-model pipeline tracking is hard to bolt onto an existing AutoML tool
2. **TRIPOD+AI mapping** — requires deep domain expertise to map workflow actions to reporting items; this is editorial knowledge, not engineering
3. **Reviewer-oriented coaching corpus** — the accumulated knowledge of "what a reviewer would flag" is a curated asset that improves over time
4. **The three-layer architecture** (Data → Insight → Presentation) — designed from the ground up for manuscript generation, not retrofitted

Anyone can add a "generate report" button to PyCaret. Nobody can retroactively instrument their pipeline with per-decision provenance tracking.

---

---

## Roadmap — GitHub Milestones

### M1: Manuscript Foundation (by May 2026)
*Fix the provenance pipeline so the methods section accurately reflects what happened.*

| # | Issue | Type | Effort |
|---|-------|------|--------|
| 51 | 7 models missing from MODEL_TO_FAMILY mapping | Bug fix | 30 min |
| 55 | Deprecation time bombs (use_container_width, utcnow) | Housekeeping | 1 hr |
| 53 | Methods section doesn't list trained models or configs | **Critical gap** | 3-4 hrs |
| 49 | Unify methodology_log and InsightLedger | **Architecture** | 8-10 hrs |
| 56 | Insight lifecycle gaps (orphaned insights, training provenance) | Architecture | 4-6 hrs |

**Execution order:** 51 → 55 → 53 → 49 → 56. Each unblocks the next. By the end, the methods section is trustworthy.

### M2: Per-Model Provenance (by June 2026)
*The flagship differentiator: different models get different preprocessing, and the manuscript says so.*

| # | Issue | Type | Effort |
|---|-------|------|--------|
| 50 | Per-model pipeline provenance tracking | **Core differentiator** | 8-10 hrs |
| 46 | Model-scoped coaching (per-family recommendations) | Differentiator | 6-8 hrs |
| 52 | EDA recommendations → InsightLedger integration | Architecture | 3-4 hrs |
| 60 | Reviewer-oriented coaching corpus with citations | Differentiator | 6-8 hrs |
| 61 | End-to-end NarrativeEngine | **Architecture** | 12-16 hrs |

**Execution order:** 50 → 52 → 46 → 61 → 60. #50 is the proof-of-concept. #61 is the capstone that replaces the stitched-together methods generation.

### M3: TRIPOD+AI Compliance (by August 2026)
*Real-time compliance tracking during the workflow. No other tool does this.*

| # | Issue | Type | Effort |
|---|-------|------|--------|
| 57 | TRIPOD+AI real-time compliance tracker | **Core differentiator** | 10-12 hrs |
| 59 | Manuscript consistency validator | Safety net | 6-8 hrs |
| 54 | Consistency validation integration | Architecture | 4-6 hrs |
| 48 | External validation workflow | TRIPOD requirement | 8-10 hrs |

**Execution order:** 54 → 59 → 57 → 48. The validator (#54/#59) is the safety net; TRIPOD tracker (#57) builds on it; external validation (#48) is the final TRIPOD requirement.

### M4: Frontier Models (by October 2026)
*Position the app at the methodological frontier.*

| # | Issue | Type | Effort |
|---|-------|------|--------|
| 58 | TabPFN 2.5 foundation model integration | Differentiator | 6-8 hrs |

TabPFN is the perfect test of per-model provenance: it needs NO preprocessing while Ridge needs the full pipeline. If M2 is done correctly, TabPFN's methods section entry writes itself.

---

## Dependency Graph

```
M1: Foundation
  #51 (family mapping) ──────────────────────────────────────→ done
  #55 (deprecations) ────────────────────────────────────────→ done
  #53 (models in methods) ───────────────────────────────────→ done
  #49 (unify tracking) ─────→ #56 (lifecycle gaps) ─────────→ M1 complete
                                    │
M2: Per-Model Provenance            ↓
  #50 (pipeline provenance) → #52 (EDA→ledger) → #46 (scoped coaching)
                                                        │
                                                        ↓
                                    #61 (NarrativeEngine) → #60 (coaching corpus)
                                            │                       │
M3: TRIPOD+AI                               ↓                       ↓
  #54 (consistency checks) → #59 (validator) → #57 (TRIPOD tracker) → M3 complete
                                                        │
                                                        ↓
                                              #48 (external validation)

M4: Frontier
  #58 (TabPFN 2.5) ← requires M2 complete for provenance to work
```

---

*This document should be revisited quarterly. The competitive landscape moves fast — TabPFN went from research curiosity to Nature publication to 2.5 release in 18 months. But the publication gap persists because it's a harder problem than performance optimization. That's our advantage.*
