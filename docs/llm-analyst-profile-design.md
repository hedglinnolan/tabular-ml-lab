# Design: Data Analyst Profile for Local LLM Interpretation

**Branch:** `feature/llm-interpretation-improvements`
**Created:** 2026-03-22
**Status:** Design (pre-implementation)

---

## Research Synthesis

This design draws on six bodies of research. The key finding is that what works for large models (70B+, API-hosted) often fails or backfires for small models (≤14B, local). Our 9B Qwen3.5 needs a different strategy than what most "prompt engineering guides" recommend.

### 1. Persona Prompting: Mostly Doesn't Work for Small Models

**Source:** Zheng et al. (2026), "When 'A Helpful Assistant' Is Not Really Helpful" — 162 personas × 2,410 questions × 4 LLM families.

**Finding:** Adding personas to system prompts does NOT improve model performance on objective tasks. The effect of each persona is "largely random." Gender-neutral, in-domain, work-related roles do marginally better, but with small effect sizes.

**Source:** Araujo et al. (2025, EMNLP), "Principled Personas" — 9 LLMs × 27 tasks.

**Finding:** Three desiderata for persona prompting: (1) expertise advantage, (2) robustness to irrelevant attributes, (3) fidelity to persona attributes. Results: expert personas usually show positive or non-significant changes, BUT irrelevant persona attributes (random names, descriptions) can **degrade performance by up to 30 percentage points**. Mitigation strategies (explicit instructions, two-step refinement) only work for models ≥70B. Smaller models are "largely insensitive to such constraints."

**Implication for us:** Our current "senior biostatistician" persona is at best neutral, at worst wasting tokens and introducing noise. For a 9B model, **task structure matters more than identity framing**. Don't tell the model *who* it is — tell it *what to do*.

### 2. Hybrid Prompting Wins for Statistical Reasoning

**Source:** Frontiers in AI (2025), "Prompt engineering for accurate statistical reasoning with LLMs in medical research" — GPT-4.1 and Claude 3.7 Sonnet, validated on Claude 4 and o3/o4 mini.

**Finding:** Four prompting strategies compared on statistical tasks:
- **Zero-shot:** Sufficient for descriptive tasks. Fails for inferential tasks (skips assumption checking).
- **Explicit instruction:** Better — forces stepwise analysis. But rigid.
- **Chain-of-thought:** Reveals reasoning but can hallucinate intermediate steps.
- **Hybrid (winner):** Combines explicit instructions + reasoning scaffolds + format constraints. Scored highest on all four evaluation criteria:
  1. **Assumption checking** — does the model verify statistical assumptions?
  2. **Test/method selection** — is the chosen interpretation appropriate?
  3. **Output completeness** — are all relevant aspects addressed?
  4. **Interpretive quality** — is the explanation specific and actionable?

**Implication for us:** Our system prompt should be a **hybrid prompt** — explicit analytical steps embedded in the instruction, not a personality description. The four criteria above should be our evaluation rubric.

### 3. Contextual Layering Over Flat Prompts

**Source:** Enterprise implementations (Reddit r/ChatGPTPromptGenius, 2025); Anthropic Claude 4 prompting docs; InsightLens (PacificVis 2025).

**Pattern:** Instead of one-shot prompts, build context hierarchies:
- **Layer 1:** Organizational/domain knowledge (static)
- **Layer 2:** Session/project context (dynamic, accumulated)
- **Layer 3:** Specific task/query (focal)

**InsightLens** specifically studied data analysts using LLMs and found the #1 pain point was **contextual forgetting** — the LLM losing track of what was previously discovered. Their solution: structured insight management with multi-level navigation.

**Implication for us:** We've already built the layering (BACKGROUND vs FOCAL in workstream 1). But the prompt itself should be structured in layers too — the system prompt defines the *analytical framework*, the context provides the *project state*, and the user message provides the *focal question*.

### 4. Few-Shot Examples: Skip Them

**Source:** Frontiers paper (2025) explicitly excluded few-shot prompting: "its reliance on curated examples introduces variability and complicates standardization." The Few-shot Dilemma (arxiv 2025.09) found LLM performance peaks with 2-3 examples then **declines** with more.

**Implication for us:** Don't embed example interpretations in the system prompt. Every example eats tokens that could be context, and for a 9B model with limited context window, that's a real cost. The hybrid structure (explicit steps) achieves what few-shot aims for without the overhead.

### 5. Thinking Models Need Different Handling

**Source:** Qwen team blog; Unsloth docs; r/LocalLLaMA.

**Key insight:** Qwen3.5 has built-in chain-of-thought via its thinking mode. Adding "think step by step" to the prompt is **redundant and wasteful** — the model already thinks before responding. Instead:
- Let thinking mode handle the reasoning
- Use the system prompt to structure **what** to reason about, not **how** to reason
- The system prompt should be a checklist of analytical concerns, not a reasoning scaffold

**Implication for us:** With Qwen3.5's native thinking, our prompt should be a structured **analytical checklist** — the model will internally reason through each point before composing its response.

### 6. LLM-as-Data-Analyst: The Semantic Gap

**Source:** Tang et al. (2025), "LLM/Agent-as-Data-Analyst: A Survey" — comprehensive review of the field.

**Design goal that matters for us:** Semantic-aware design — the LLM should understand what the data *means* (clinical significance, practical relevance) not just what the numbers *are* (statistical significance).

**Implication for us:** The prompt should explicitly distinguish between statistical interpretation ("R² = 0.78") and practical interpretation ("does this model predict well enough for clinical use?"). This is the gap between a textbook answer and an expert answer.

---

## Design: The Analyst Profile

Based on the research, the "profile" is **not a persona** — it's a **structured analytical framework** that tells Qwen3.5 exactly what to evaluate and how to structure its output. The model's thinking mode handles the reasoning; the prompt handles the structure.

### Architecture: Three-Layer Prompt Composition

```
┌─────────────────────────────────────────────┐
│ SYSTEM PROMPT (static)                       │
│ = Analytical framework + output structure    │
│   What to evaluate, how to structure output  │
│   No personality, no persona, no filler      │
├─────────────────────────────────────────────┤
│ CONTEXT: BACKGROUND (dynamic, per-session)   │
│ = gather_session_context() output            │
│   Dataset profile, prior findings, ledger    │
│   Preprocessing, model family                │
├─────────────────────────────────────────────┤
│ CONTEXT: FOCAL (dynamic, per-button-click)   │
│ = The specific analysis result               │
│   Plot type, stats summary, metrics          │
│   User's question (if any)                   │
└─────────────────────────────────────────────┘
```

### System Prompt: The Analytical Framework

Derived from the Frontiers paper's four evaluation criteria, adapted for our app's context:

```
You interpret statistical analysis results for a tabular ML research workbench.

ANALYTICAL CHECKLIST — evaluate each before responding:

1. VALIDITY CHECK
   - Are the statistical assumptions of this method met given the data?
   - Does the sample size support this analysis?
   - Are there known data issues (collinearity, missingness, skew) that affect this specific result?

2. RESULT INTERPRETATION
   - What does this result tell the researcher about their data/model?
   - Distinguish statistical significance from practical significance
   - Reference specific values — do not restate what is already displayed

3. CONCERNS
   - What would a peer reviewer flag about this specific result?
   - Are there methodological limitations the researcher should disclose?
   - Connect to known issues from prior analysis steps if relevant

4. NEXT STEP
   - One concrete, actionable recommendation
   - Should reference something the researcher can do in this tool

OUTPUT RULES:
- 3-5 focused points, not a wall of text
- Never summarize the background context
- Never give textbook definitions of methods
- If the researcher asked a specific question, answer it directly first
```

### What This Changes From Current

| Aspect | Current | New |
|---|---|---|
| Identity framing | "You are a senior biostatistician and data scientist" | None — task structure only |
| Analytical guidance | "Flag concerns a peer reviewer would raise" (vague) | Four-point checklist with specific sub-questions |
| Output structure | "3-5 key points" | Implicit structure via checklist (validity → interpretation → concerns → next step) |
| Redundant CoT | "EXPLAIN what results MEAN" (redundant with Qwen3.5 thinking) | Let thinking mode handle reasoning; prompt handles *what* to reason about |
| Token budget | Spent on persona description + generic rules | Spent on analytical sub-questions that directly improve output |
| Background handling | "Use to ground your interpretation" | "Are there known data issues that affect this specific result?" (active reference, not passive instruction) |

### Analysis-Type Variants

The analytical checklist should subtly adapt based on what's being interpreted. Not a different prompt per analysis — a set of **type-specific sub-questions** injected into the checklist:

**For model diagnostics** (residuals, pred vs actual, learning curves):
- "Is there evidence of overfitting or underfitting?"
- "Do residual patterns suggest violated assumptions for this model family?"

**For feature importance** (SHAP, permutation importance):
- "Does collinearity between features affect the reliability of these importance rankings?"
- "Are the top features consistent with domain expectations, or do they suggest data leakage?"

**For model comparison** (Bland-Altman, confusion matrix):
- "Is the performance difference practically meaningful or within noise?"
- "Does class imbalance or threshold choice affect this comparison?"

**For hypothesis testing** (t-test, ANOVA, chi-squared — future workstream 3):
- "Were assumptions checked before this test was selected?"
- "Is the effect size meaningful, regardless of p-value?"

**For sensitivity analysis** (seed sensitivity, bootstrap — future workstream 3):
- "How much variance is attributable to random initialization vs. genuine model instability?"
- "Is the confidence interval width acceptable for the intended application?"

### Implementation Plan

1. **Create `ANALYST_SYSTEM_PROMPT` in `llm_ui.py`** — the static analytical framework (replaces `INTERPRETATION_SYSTEM_PROMPT`)

2. **Create `ANALYSIS_TYPE_HINTS` dict** — maps analysis types to 2-3 type-specific sub-questions

3. **Compose at call time:** system prompt = framework + type-specific hints (injected into the VALIDITY CHECK section)

4. **No changes to `build_llm_context()` or `gather_session_context()`** — the workstream 1 context pipeline is correct; this only changes the system prompt layer

5. **Test:** Same head-to-head comparison methodology from workstream 2 — same SHAP context, old prompt vs new prompt, grade on the four Frontiers criteria

### What We're NOT Doing (And Why)

- **No persona/role framing** — research shows it doesn't help ≤14B models and can hurt
- **No few-shot examples** — eats context tokens, introduces variability, and Qwen3.5's thinking mode achieves the same guided reasoning
- **No explicit "think step by step"** — redundant with Qwen3.5's native thinking; would waste thinking tokens on meta-reasoning about reasoning
- **No fine-tuning** — premature; prompt structure is cheaper and reversible. Fine-tuning is the nuclear option if this doesn't work.
- **No Ollama Modelfile persona** — same as the persona point; task structure > identity framing

---

## Evaluation Rubric

Borrowed directly from the Frontiers paper, adapted for our interpretive (not computational) use case:

| Criterion | Score 1 (Poor) | Score 3 (Adequate) | Score 5 (Excellent) |
|---|---|---|---|
| **Validity awareness** | Ignores assumptions/limitations | Mentions limitations generically | Connects specific data issues to this result |
| **Interpretive specificity** | Generic ("results look good") | References values but surface-level | Explains mechanism (why, not just what) |
| **Reviewer anticipation** | No concerns raised | Flags obvious issues | Identifies subtle issues a reviewer would catch |
| **Actionability** | No next steps or vague ("collect more data") | Reasonable suggestion | Specific action doable in this tool |

Minimum acceptable: average ≥ 3.0 across criteria. Target: ≥ 4.0.

---

## References

1. Zheng et al. (2026). "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models." arXiv:2311.10054v3.
2. Araujo et al. (2025). "Principled Personas: Defining and Measuring the Intended Effects of Persona Prompting on Task Performance." EMNLP 2025. arXiv:2508.19764.
3. Frontiers in AI (2025). "Prompt engineering for accurate statistical reasoning with large language models in medical research." doi:10.3389/frai.2025.1658316.
4. InsightLens (2025). "Augmenting LLM-Powered Data Analysis with Interactive Insight Management and Navigation." arXiv:2404.01644v2.
5. Tang et al. (2025). "LLM/Agent-as-Data-Analyst: A Survey." arXiv:2509.23988v3.
6. Wharton GenAI Labs (2025). "The Decreasing Value of Chain of Thought in Prompting." — CoT gains are marginal for reasoning models.
