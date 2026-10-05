# How this analysis was produced, and how far to trust it

Provenance for everything in this folder, so the next person can tell measurement from inference
and knows where to re-check.

---

## What was measured deterministically

Scripted over the working tree — no model judgment involved. Reproduce with the snippets below.

| Fact | Value | Method |
|---|---|---|
| Streamlit coupling per module | 41 refs across `ml/` (3 files); 0 in `models/` | `grep -rn "import streamlit\|st.session_state"` |
| Module sizes | `ml/` 17,735 · `models/` 1,000 · `utils/` 11,442 · `pages/`+`app.py` 19,835 | `wc -l` |
| `visualizations.py` portability | 12 `return fig`; **0** `st.pyplot`/`st.plotly_chart` | grep |
| `st.session_state` key surface | **163 distinct keys** | AST + regex over all `.py` |
| Import graph / fan-in | `session_state` 16 importers (10 pages), `insight_ledger` 14/7, `workflow_provenance` 12/10, `theme` 12/11, `test_lockbox` 12/10 | `ast` walk, `data/import-graph.json` |
| Import cycles | **3**, all through the state layer | DFS over the graph |
| Test coverage | **34 of 70** core modules imported by any test; all of `models/` uncovered | AST over `tests/` |
| Caching decorators | 18, incl. 4 in `ml/macro_shape.py` | grep |
| Exception swallowing | 91 `except Exception` in core; ~24 return a clean default | grep + context |

```bash
# session_state key surface
grep -rhoE "st\.session_state(\.[A-Za-z_]\w*|\[['\"][^'\"]+['\"]\])" --include=*.py . | sort -u | wc -l

# coupling census
grep -rln "import streamlit\|st\.session_state" --include=*.py ml/ models/ utils/ *.py

# is visualizations.py portable?
grep -c "return fig" visualizations.py; grep -c "st\.pyplot\|st\.plotly_chart" visualizations.py
```

## What was verified by hand

Fifteen findings — everything in `TRANSITION_PLAN.md` §01 and §02 — were re-read directly in
source against `origin/main` @ `24c3446`, after PR #145. These are **Tier 0** in the ledger. Where
PR #145 changed the picture, the disposition says so (`PARTIAL` rather than `OPEN`).

## What came from agents

Two workflow runs over the repository:

| Run | Agents | Outcome |
|---|---|---|
| Inventory (first attempt) | 9 | **2 completed** — data ingestion, profiling/EDA. Seven died on a session usage limit. |
| Transition analysis | 10 | **10 completed.** Six domain maps, three cross-cuts (stage contracts, silent-failure hunt, test strategy), one completeness critic. ~2M tokens, 573 tool calls. |

Combined: **663 functions catalogued, 76 invariants, 231 critical/high findings, 110 landmines.**
Deduplicated to **385 ledger rows**.

Raw output is preserved at `data/raw-analysis.json` (~820 KB). The ledger is a summary; go to the
raw file when you need a function's full transition note.

## How far to trust it

**Trust directly:** the deterministic measurements, and Tier 0.

**Verify before acting:** all 370 Tier 1 rows. They were produced by agents reading the repo at
commit `fbe422a` — *before* PR #145, which changed `utils/test_lockbox.py` by +312 lines, added
`utils/replay.py` (405 lines), and landed 14 new test files. Some Tier 1 findings are certainly
stale. They are marked `UNVERIFIED` for that reason, and **re-verifying them is the first loop
iteration**, not a formality.

Agents were instructed to quote real signatures and never invent a function, and every claim
carries a `file:line` evidence field — but agent output is evidence to check, not fact. Two claims
I spot-checked did require correction in framing (the NN adapter's non-training `fit()` is
deliberate, not accidental; the "permanently stale" section was recoverable, just undiscoverably).

## Known gaps

- **The first inventory run's six lost domains** were re-covered by the second run, but at a
  different prompt framing. Ingestion and EDA therefore have deeper function-level semantics than
  the rest.
- **`pages/11_Theory_Reference.py` (5,162 lines)** was classified as static content rather than
  logic on density evidence, not a full read. Confirm before dropping it from the engineering backlog.
- **No runtime verification of the existing app.** Everything is static analysis. The three live
  bugs were confirmed by reading code, not by reproducing them in a browser. Reproducing
  LIVE-001 (two datasets in one session, compare PCA) is worth doing before fixing.
- **The prototypes are not implementations.** Synthetic data and fake math throughout, by design.
  They prove interaction and design, nothing about the engine.

## Prototype verification

`prototypes/interview-feed.html` was driven end-to-end in headless Chromium — 60+ assertions
across three suites covering both upload branches, the stale cascade, retrain-after-draft, and
manuscript composition. Zero console errors; no horizontal overflow at 1280px or 390px; both
themes checked.

That process caught four real defects that static review missed: a strict-mode error that would
have broken the entire script, a shared job timer where a re-answer and a recompute silently
cancelled each other, an after-table marking unchanged values as changed, and a recompute firing
on an intermediate sub-answer before the new row count existed.

**The lesson worth carrying into the rebuild:** these were all state-machine bugs invisible to
reading and obvious to clicking. TurboTab's own frontend will need the same treatment.
