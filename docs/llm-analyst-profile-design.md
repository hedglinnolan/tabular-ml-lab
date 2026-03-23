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

## Agentic System Prompt Patterns (from Production Systems)

Cross-referencing the academic research with patterns from 30+ production agent prompts (Cursor, Claude Code, Manus, v0, same.new, Bolt.new, ChatGPT, Cline, Augment, Windsurf — sourced from awesome-ai-system-prompts and system-prompts-and-models-of-ai-tools repos):

### Patterns That Confirm the Research

**1. Structure > Identity (confirming Zheng/Araujo)**

The highest-performing agent prompts spend almost zero tokens on personality and almost all tokens on operational structure. Manus's prompt is a pure operational blueprint: `<agent_loop>`, `<planner_module>`, `<knowledge_module>` — not a single sentence about who Manus "is" as a person. Claude Code's sub-agent creation spec says: "Be specific rather than generic — avoid vague instructions. Include concrete examples when they would clarify behavior." The pattern is universal: **task structure beats identity framing**.

The exception (Claude's "enjoys helping humans" persona) comes from Anthropic for a 200B+ parameter model. At 9B, this is dead weight.

**2. XML/Markdown Section Tags (confirming hybrid prompting)**

Every high-performing prompt uses structural tags to separate concerns:
- Manus: `<system_capability>`, `<agent_loop>`, `<todo_rules>`
- same.new: `<tool_calling>`, `<making_code_changes>`
- Bolt.new: `<artifact_instructions>`, `<system_constraints>`

This maps directly to the Frontiers paper's hybrid prompting finding — structure the prompt as explicit sections, not flowing prose. For our small model, this is especially important: Qwen3.5 at 9B will parse `# VALIDITY CHECK` as a distinct section better than it will parse a paragraph that implies a validity check.

**3. Environment Context Injection (confirming contextual layering)**

Cline injects `SYSTEM INFORMATION` (OS, shell, working directory). Manus injects `<system_capability>` (available tools, sandbox specs). Same.new injects OS version and IDE context. Every production agent tells the model *exactly what environment it's operating in*.

For our app: this means the system prompt should tell Qwen3.5 that it's embedded in a Streamlit-based tabular ML workbench, what analysis pages exist, what the user can do next. Not as abstract knowledge — as operational context.

**4. One-Step-at-a-Time (confirming thinking model approach)**

Manus's `<agent_loop>` enforces "Choose only ONE tool call per iteration." Bolt.new says "Think HOLISTICALLY and COMPREHENSIVELY BEFORE creating an artifact." Same.new and Cline both mandate "Wait for execution results before proceeding."

For our non-agentic (single-shot interpretation) use case, the analogue is: **address one analysis result at a time**, don't try to synthesize the whole session. This confirms our FOCAL vs BACKGROUND architecture — the model should deeply interpret one thing, not shallowly scan everything.

### Novel Patterns Worth Adopting

**5. Anti-Patterns as Instructions (from Claude Code, v0)**

The most effective prompts don't just say what to do — they explicitly list what NOT to do, with specificity:
- Claude Code: "DO NOT ADD ANY COMMENTS unless asked"
- v0: "MUST NOT apologize or provide an explanation" when refusing
- same.new: "NEVER refer to tool names when speaking to the USER"

Our current "DO NOT" list is good but too generic ("Don't give textbook explanations"). Production prompts are surgical: "Do not define what SHAP values are. Do not explain the formula for R². Do not list general advantages of tree models."

**6. Domain Constraint Anchoring (from v0)**

v0's prompt embeds deep domain constraints: "ALWAYS uses icons from lucide-react", "uses Tailwind CSS for styling", "ONLY uses the AI SDK via 'ai' and '@ai-sdk'." These aren't preferences — they're hard constraints that eliminate a class of wrong answers.

For our app: we should embed statistical domain constraints:
- "Effect sizes matter more than p-values for clinical relevance"
- "n < 30 per group invalidates t-test normality assumptions"
- "VIF > 5 means SHAP/permutation importance rankings are unreliable"
- "R² > 0.95 on tabular data is suspicious — check for data leakage"

These are the kind of bright-line rules that a 9B model can follow mechanically even when it can't reason through the statistics deeply.

**7. Quality Assurance Checkpoints (from Claude Code agent architect)**

Claude Code's agent creation prompt includes: "Build in quality assurance and self-correction mechanisms." The best agents include verification steps — not just "do the task" but "verify the output meets criteria."

For our interpretation prompt: add a brief self-check at the end: "Before responding, verify: (1) Did you reference specific values from the focal analysis? (2) Did you connect to at least one background finding? (3) Is your next step actionable within this tool?"

**8. Conciseness as a Hard Rule (from Claude, ChatGPT)**

Claude's prompt: "provides the shortest answer it can... avoiding tangential information." ChatGPT 4o: "ULTRA IMPORTANT: Do NOT be verbose." This is critical for our use case — a 9B model given freedom to write will produce fluff. Hard token budget or point count is more reliable than "be concise."

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

Derived from the Frontiers paper's four evaluation criteria, production agent patterns, and domain constraint anchoring:

```
You interpret statistical analysis results in a tabular ML research workbench (Streamlit app). The researcher sees a specific plot or table and clicks "Interpret with AI" for expert analysis.

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

# SELF-CHECK (internal, before responding)
- Did I reference specific values from the focal analysis?
- Did I connect to at least one background finding?
- Is my next step something the researcher can do in this tool?
```

### What This Changes From Current

| Aspect | Current | New | Source |
|---|---|---|---|
| Identity framing | "You are a senior biostatistician and data scientist" | None — task structure only | Zheng 2026, Araujo 2025 |
| Analytical guidance | "Flag concerns a peer reviewer would raise" (vague) | Four-section checklist with specific sub-questions | Frontiers 2025 |
| Output structure | "3-5 key points" | Ordered sections (validity → interpretation → concerns → next step) | Manus agent_loop pattern |
| Anti-patterns | "Don't give generic explanations" (vague) | Specific bans: "Do NOT define methods", "Do NOT explain formulas" | v0, Claude Code |
| Domain constraints | None | Hard rules: VIF>5, R²>0.95, effect sizes>p-values | v0 domain anchoring |
| Self-verification | None | Three-point self-check before responding | Claude Code architect |
| Conciseness | "Keep it concise" | "Maximum 5 points total" + banned opening phrases | Claude, ChatGPT |
| Environment context | None | "tabular ML research workbench (Streamlit app)" | Cline, Manus, same.new |
| Redundant CoT | "EXPLAIN what results MEAN" | Let thinking mode handle reasoning; prompt handles *what* to reason about | Qwen3 docs, Wharton CoT paper |
| Background handling | "Use to ground your interpretation" | "Do known issues affect THIS result?" (active query, not passive instruction) | Frontiers hybrid prompting |

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

### Academic Research
1. Zheng et al. (2026). "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models." arXiv:2311.10054v3.
2. Araujo et al. (2025). "Principled Personas: Defining and Measuring the Intended Effects of Persona Prompting on Task Performance." EMNLP 2025. arXiv:2508.19764.
3. Frontiers in AI (2025). "Prompt engineering for accurate statistical reasoning with large language models in medical research." doi:10.3389/frai.2025.1658316.
4. InsightLens (2025). "Augmenting LLM-Powered Data Analysis with Interactive Insight Management and Navigation." arXiv:2404.01644v2.
5. Tang et al. (2025). "LLM/Agent-as-Data-Analyst: A Survey." arXiv:2509.23988v3.
6. Wharton GenAI Labs (2025). "The Decreasing Value of Chain of Thought in Prompting." — CoT gains are marginal for reasoning models.

### Production Agent Prompts (from open-source repos)
7. awesome-ai-system-prompts (dontriskit/awesome-ai-system-prompts) — 8-principle analysis of 30+ production agent prompts. Patterns analyzed: Manus (explicit agent loop, modular tags), v0 (domain constraint anchoring, anti-pattern specificity), same.new (XML structure, tool etiquette), Claude Code (sub-agent architect, quality assurance checkpoints), Bolt.new (holistic pre-thinking), Cline (environment injection), ChatGPT 4.5 (conciseness as hard rule, inline schemas).
8. system-prompts-and-models-of-ai-tools (x1xhlol) — 131K-star collection, 30,000+ lines of raw prompt content. Used for cross-referencing patterns across Cursor, Windsurf, Devin AI, Augment Code, and others.
9. claude-code-system-prompts (Piebald-AI) — Claude Code's agent creation architect prompt: "Be specific rather than generic. Include concrete examples. Balance comprehensiveness with clarity. Build in quality assurance and self-correction mechanisms."
10. BrightCoding (2026). "System Prompts for AI Agents: The Complete Guide." — Synthesis of 8 core principles from 20+ production agents, with performance metrics (Manus: 78% autonomous completion, same.new: 89% satisfaction, v0: 94% component generation success).
