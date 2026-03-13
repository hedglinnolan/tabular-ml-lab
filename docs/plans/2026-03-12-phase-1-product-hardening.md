# Phase 1 Product Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve product trust and default-path clarity on the `feature/feature-engineering` branch without changing core analysis behavior or breaking the happy path.

**Architecture:** This batch is intentionally limited to low-risk product hardening. We will modify shared workflow rendering and user-facing copy, avoiding changes to ML/data-flow internals. The highest-risk shared change is sidebar workflow logic in `utils/theme.py`, so verification will focus on import safety, syntax, and preserving existing session-state keys.

**Tech Stack:** Streamlit, Python, markdown copy, shared UI helpers in `utils/theme.py`

---

## Scope Guardrails

**In scope:**
- Sidebar progress truthfulness
- Default-path vs advanced-path framing in shared UI
- Home page wording that reduces cognitive load
- Export wording calibrated to actual product maturity
- README wording updates that match current app reality

**Out of scope for this batch:**
- Upload & Audit structural simplification
- Quick Path / Full Path stateful mode
- Feature Engineering algorithm changes
- Session serialization redesign
- Export generator logic changes
- Data-flow / preprocessing / model-training internals

---

### Task 1: Capture baseline and identify exact text/logic targets

**Files:**
- Inspect: `utils/theme.py`
- Inspect: `app.py`
- Inspect: `pages/03_Feature_Engineering.py`
- Inspect: `pages/10_Report_Export.py`
- Inspect: `README.md`

**Step 1: Re-read targeted sections and note exact strings/branches to change**
- Sidebar checklist item labels and completion logic in `utils/theme.py`
- Hero + workflow + result copy in `app.py`
- Experimental/optional framing in `pages/03_Feature_Engineering.py`
- Promise language in `pages/10_Report_Export.py`
- Workflow/version messaging in `README.md`

**Step 2: Record baseline verification commands**
Run:
```bash
python -m py_compile app.py utils/theme.py pages/03_Feature_Engineering.py pages/10_Report_Export.py
```
Expected: no output

**Step 3: Commit baseline only if needed**
No commit required if no files changed.

---

### Task 2: Fix sidebar workflow truthfulness and default-path framing

**Files:**
- Modify: `utils/theme.py`

**Step 1: Write the failing expectation as a checklist in the plan**
Expected behavior after implementation:
- Statistical Validation should be completable when its results exist
- Sidebar should distinguish core/default steps from advanced/optional steps
- Feature Engineering should remain visible but clearly optional

**Step 2: Implement minimal shared logic changes**
Change `render_sidebar_workflow()` to:
- compute `stat_validation_run` from a real session-state signal rather than hardcoded `False`
- rename/checklist labels to better reflect default path and optionality
- visually label Feature Engineering and Sensitivity Analysis as optional/advanced without hiding them

**Implementation notes:**
- Reuse existing state keys where possible
- If no durable statistical-validation key exists, use a conservative derived signal from session state and avoid inventing broad new state behavior in this batch
- Do not break current `current_page` matching

**Step 3: Run syntax verification**
Run:
```bash
python -m py_compile utils/theme.py
```
Expected: no output

**Step 4: Run targeted grep/readback**
Run:
```bash
grep -n "Statistical Validation\|Feature Engineering\|optional\|advanced" utils/theme.py
```
Expected: updated labels/logic visible

**Step 5: Commit**
```bash
git add utils/theme.py
git commit -m "fix: improve workflow truthfulness and optional-step framing"
```

---

### Task 3: Simplify home page into a calmer default-path narrative

**Files:**
- Modify: `app.py`

**Step 1: Update hero and top-level promise language**
Adjust copy so the app reads as:
- a guided default path for first-run researchers
- with advanced options available when needed
- “draft/manuscript-ready starting point” rather than overclaiming final publication readiness

**Step 2: Update workflow explanation**
Revise workflow section to:
- identify the core path explicitly
- describe Feature Engineering / Sensitivity / Statistical Validation as optional extensions
- reduce the sense that all 10 steps are always mandatory

**Step 3: Update results/output framing**
Keep strong value proposition, but calibrate language around export and user responsibility.

**Step 4: Run syntax verification**
Run:
```bash
python -m py_compile app.py
```
Expected: no output

**Step 5: Spot-check changed strings**
Run:
```bash
grep -n "guided\|optional\|advanced\|draft\|starting point" app.py
```
Expected: updated copy present

**Step 6: Commit**
```bash
git add app.py
git commit -m "refactor: clarify default path on home page"
```

---

### Task 4: Reposition Feature Engineering as advanced/optional, not implied default

**Files:**
- Modify: `pages/03_Feature_Engineering.py`

**Step 1: Update the page banner and intro copy**
Revise copy to emphasize:
- this page is optional
- most first-pass users should skip it initially
- it is best used after baseline modeling or with clear domain rationale

**Step 2: Strengthen skip-path language**
Make the skip behavior and recommendation clearer in copy only, without altering data flow.

**Step 3: Keep existing warnings/guardrails intact**
Do not modify reset/save/downstream invalidation behavior.

**Step 4: Run syntax verification**
Run:
```bash
python -m py_compile pages/03_Feature_Engineering.py
```
Expected: no output

**Step 5: Spot-check changed strings**
Run:
```bash
grep -n "optional\|advanced\|skip\|baseline" pages/03_Feature_Engineering.py
```
Expected: revised positioning language visible

**Step 6: Commit**
```bash
git add pages/03_Feature_Engineering.py
git commit -m "docs: reposition feature engineering as advanced optional step"
```

---

### Task 5: Calibrate export promise language without touching export logic

**Files:**
- Modify: `pages/10_Report_Export.py`

**Step 1: Revise page intro copy**
Adjust wording from “publication-ready” toward:
- manuscript-ready starting point
- draft materials generated from workflow choices
- user review still required for study-specific context

**Step 2: Preserve the strong value proposition**
Do not make the feature sound weak; make it sound honest.

**Step 3: Run syntax verification**
Run:
```bash
python -m py_compile pages/10_Report_Export.py
```
Expected: no output

**Step 4: Spot-check changed strings**
Run:
```bash
grep -n "publication-ready\|draft\|manuscript\|review" pages/10_Report_Export.py
```
Expected: calibrated wording visible

**Step 5: Commit**
```bash
git add pages/10_Report_Export.py
git commit -m "docs: calibrate export promise language"
```

---

### Task 6: Align README to current product reality

**Files:**
- Modify: `README.md`

**Step 1: Update branch/version messaging**
Ensure README accurately reflects:
- feature-engineering branch as experimental/testing branch
- 10-step workflow
- optional/advanced nature of certain steps
- export as draft/manuscript-starting materials rather than zero-edit perfection

**Step 2: Avoid broad doc rewrites**
Keep changes narrowly scoped to currently misleading sections.

**Step 3: Run quick grep verification**
Run:
```bash
grep -n "9 workflow\|10-step\|feature/feature-engineering\|draft\|optional" README.md
```
Expected: outdated references removed or corrected

**Step 4: Commit**
```bash
git add README.md
git commit -m "docs: align readme with current workflow and product posture"
```

---

### Task 7: End-to-end verification for Batch 1

**Files:**
- Verify: `app.py`
- Verify: `utils/theme.py`
- Verify: `pages/03_Feature_Engineering.py`
- Verify: `pages/10_Report_Export.py`
- Verify: `README.md`

**Step 1: Run compile checks together**
Run:
```bash
python -m py_compile app.py utils/theme.py pages/03_Feature_Engineering.py pages/10_Report_Export.py
```
Expected: no output

**Step 2: If test suite exists and is affordable, run targeted smoke tests**
Run one of:
```bash
pytest -q
```
or, if suite is too heavy,
```bash
pytest -q -k "session or workflow or export"
```
Expected: pass, or clearly documented unrelated pre-existing failures

**Step 3: Run git diff review**
Run:
```bash
git diff -- app.py utils/theme.py pages/03_Feature_Engineering.py pages/10_Report_Export.py README.md
```
Expected: copy/shared-logic-only changes; no ML/data-pipeline changes

**Step 4: Final commit**
```bash
git add app.py utils/theme.py pages/03_Feature_Engineering.py pages/10_Report_Export.py README.md
git commit -m "feat: harden product trust and clarify default workflow"
```

---

## Execution Notes for Delegation

If delegating Task 2-6 to a coding subagent:
- Restrict the subagent to this repo only: `/home/claw/.openclaw/workspace/glucose-mlp-interactive`
- Tell it explicitly: **no changes to ML/data-flow logic**
- Require it to stop after edits and before final claim so main agent can review diffs and run verification
- Main agent must inspect changed strings and shared logic in `utils/theme.py` before reporting success

## Success Criteria

Batch 1 is successful if:
- Sidebar no longer lies about Statistical Validation completion
- Default path feels clearer in shared UI and top-level copy
- Feature Engineering remains available but is clearly optional/advanced
- Export language is more honest without sounding weak
- README no longer obviously contradicts the app
- No syntax/test regressions introduced
