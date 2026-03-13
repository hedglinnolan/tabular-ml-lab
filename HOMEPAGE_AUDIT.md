# Homepage Audit: Product Design & Trust Evaluation

**Date:** 2026-03-10  
**Page:** `app.py` (Application Landing Page)  
**Goal:** Establish trust and confidence with new users

---

## Critical Issues Found

### ❌ Issue 1: Workflow Contradiction (HIGH)

**Problem:** Step 9 appears both IN the workflow AND as "also available"

```
Step 9: "Hypothesis Testing — Statistical tests without ML"
Also available: "Hypothesis Testing — Statistical tests without ML"
```

**User confusion:** "Is this optional or required?"

**Fix:** Remove from "Also available" section - it's part of the main workflow.

---

### ❌ Issue 2: Naming Inconsistency (CRITICAL)

**Problem:** Page 9 was renamed to "Statistical Validation" but homepage still says "Hypothesis Testing"

**Current:**
- Step 9: "Hypothesis Testing — Statistical tests without ML"

**Actual page name:**
- `pages/09_Hypothesis_Testing.py` but title is "Statistical Validation"

**Fix:** Update to "Statistical Validation" for consistency.

---

### ❌ Issue 3: Missing Target Audience (HIGH)

**Problem:** Homepage doesn't say WHO this is for

**Questions new users have:**
- "Is this for me?"
- "Do I need to know Python?"
- "Do I need to be a data scientist?"

**Fix:** Add clear "Who This Is For" section

---

### ❌ Issue 4: Overclaiming (TRUST ISSUE)

**Current claims:**
- "From raw data to publication-ready results" ← Implies no work needed
- "Reviewer Proof" ← Nothing is reviewer-proof
- "Built for researchers, by researchers" ← Generic claim

**Reality:**
- Users still need to interpret results
- Reviewers may still have questions
- Need domain expertise to use correctly

**Fix:** Be more honest about what the app does vs. what the user must do

---

### ❌ Issue 5: No Context for "Why This?" (HIGH)

**Missing:**
- Why use this instead of R/Python notebooks?
- What problem does this solve?
- What's the alternative (manual workflow)?

**Fix:** Add comparison / value proposition that's specific, not generic

---

### ❌ Issue 6: Vague AI Mention (MEDIUM)

**Current:**
- "AI Interpretation: Ollama / OpenAI / Anthropic"

**User confusion:**
- "What does AI actually do here?"
- "Is this required?"
- "Do I need an API key?"

**Fix:** Be specific about what LLM assistance provides (optional insights, not required)

---

### ❌ Issue 7: No Time Expectation (MEDIUM)

**Missing:**
- "How long will this take?"
- "Can I save and resume?"

**User needs to know:**
- Session save/resume exists
- Typical workflow is 30-60 min (varies by data size)

---

### ❌ Issue 8: No Use Case Examples (MEDIUM)

**Missing:**
- What kind of data works well?
- What's a concrete example?
- What fields is this used in?

**Fix:** Add 1-2 concrete examples (e.g., "Clinical trial outcomes prediction" or "Biomarker discovery")

---

## What's Working Well ✅

1. **Clear workflow steps** - 10-step numbered list is scannable
2. **Visual hierarchy** - Hero → Benefits → Steps → CTA
3. **Progressive disclosure** - Capabilities collapsed by default
4. **Honest capabilities list** - Shows what's actually included
5. **Call-to-action** - Clear next step (Upload & Audit)

---

## Recommended Redesign Structure

```
1. Hero
   - Value prop: specific, honest
   - Target audience: clear

2. Problem/Solution (NEW)
   - What problem does this solve?
   - Who is this for?
   - What makes it different?

3. How It Works
   - 10-step workflow (existing)

4. When to Use This (NEW)
   - ✅ Good for: Clinical trials, biomarker studies, small-to-medium tabular datasets
   - ⚠️ Not designed for: Time series, image data, NLP, production deployment

5. Getting Started (existing "Quick start" but clearer)

6. Capabilities (existing, but move AI explanation here)

7. FAQ (NEW - Optional)
   - "Do I need to know Python?" → No
   - "How long does this take?" → 30-60 min
   - "Can I save my work?" → Yes
   - "Is my data private?" → Yes, session-only

8. Footer (existing)
```

---

## Key Principles to Follow

1. **Don't lie** - If it's not ready for publication without review, say so
2. **Be specific** - "Clinical researchers" not "researchers"
3. **Set expectations** - "Generates draft methods section" not "publication-ready"
4. **Explain tradeoffs** - "Easy to use but less flexible than custom code"
5. **Build trust** - Acknowledge limitations upfront

---

## Next Steps

Should I:
1. Rewrite the homepage following these principles?
2. Just fix the critical issues (naming, contradiction)?
3. Show you a draft for review first?

