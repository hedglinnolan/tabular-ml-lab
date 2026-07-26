# Design brief — Tabular ML Lab: a scroll-driven architecture explainer

**For:** a Claude design (Fable 5) session **with access to this repository.**
**Deliverable:** one self-contained, scroll-animated HTML file that explains, to a
technically-literate but non-engineer audience (research collaborators,
reviewers, conference attendees), **(a) how the modeling process works** and
**(b) how the app's four architectural planes interact.** Animations play as the
reader scrolls; nothing autoplays off-screen.

This document is the *instruction set*. Follow the content precisely — it is
grounded in the real code, and you can verify every claim by opening the files
named below. Treat the visual/motion direction as a strong, opinionated
starting point and elevate it with your own craft; treat the **architecture
facts and the honesty theme as non-negotiable.**

---

## 0. Read these first (ground truth — do not paraphrase from memory)

Open and skim these before designing. Every label, arrow, and number in the
piece must trace to them. If something here disagrees with the code, **the code
wins** — and flag it.

| Plane / concept | File(s) | Key symbols |
|---|---|---|
| **Data plane** | `utils/session_state.py` | `get_data()`, `set_data()`, `reset_downstream_results()`; state keys: `raw_data → df_engineered → selected_features → X_train/X_val/X_test → trained_models → model_results` |
| **Test-set lockbox** | `utils/test_lockbox.py` | `ensure_lockbox()`, `train_row_mask()`, `is_exploratory()` |
| **Advisory plane** | `utils/insight_ledger.py` | `class InsightLedger`, `class Insight`; `upsert / resolve / acknowledge / rollback_resolutions / prune_auto_generated` |
| **Record plane** | `utils/workflow_provenance.py` | `class WorkflowProvenance`; section slots `eda, feature_engineering, feature_selection, split, preprocessing, training, explainability, sensitivity, coach`; `record_*()` methods |
| **Narrative plane** | `ml/narrative_engine.py` | `class NarrativeEngine`, `class ManuscriptDraft` (evidence map, `[AUTHOR REQUIRED]` scaffolds, ownership preamble) |
| **Modeling coach** | `ml/model_coach.py`, `ml/coach_probe.py` | `select_top_picks()`, `model_viability()`, `run_probe()`, `run_post_training_diagnostics()` |
| **The pages (pipeline stages)** | `pages/01…10*.py` | the ten workflow steps, in order |
| **Prior architecture notes** | `CODE_REVIEW.md` (esp. the "Design principles" and the 2026-07 sections) | the invariants the visualization should embody |

---

## 1. The single idea the piece must land

> **One dataset flows through ten steps; four planes watch that flow and turn it
> into a manuscript you can defend. The planes never let the story outrun the
> evidence.**

Two axes carry this:

- **Horizontal axis — the modeling pipeline** (§3): the dataset's linear journey,
  Upload → … → Manuscript.
- **Vertical axis — the four planes** (§2): parallel lanes that each stage writes
  to and reads from. The planes are the app's spine; the pages are just where the
  user stands.

The payoff is the **honesty machinery** (§4): the lockbox, the invalidation
cascade, and provenance→manuscript. If a viewer leaves remembering only one
thing, it should be *"the tool physically cannot claim what it didn't record,
and it sealed the test set before anyone looked."*

---

## 2. The four planes (the vertical axis) — define each on screen

Render these as **four horizontal lanes**, top to bottom, each with its own accent
(see §6). Each lane has a fixed identity the reader learns once and then tracks:

1. **Data plane — "the material."** `utils/session_state.py`.
   The dataframe and everything derived from it: raw table → engineered features →
   the train/val/test split → fitted models → metrics. This is the only plane that
   holds the actual numbers. Voice: *concrete, physical.*
   Reads/writes: every stage. Governed by `set_data()` and `get_data()`'s
   precedence (`df_engineered > filtered_data > raw_data`).

2. **Advisory plane — "the coach."** `utils/insight_ledger.py`.
   A ledger of findings: EDA insights, the modeling coach's shortlist and
   evidence-probe verdicts, preprocessing guards, post-training diagnostics.
   Findings are *upserted* by producers and *resolved / acknowledged* when the
   user acts, or *pruned* when their producer re-runs. Voice: *observing, advising —
   never deciding for you.* It watches the data plane and speaks.

3. **Record plane — "the notary."** `utils/workflow_provenance.py`.
   An append-only record of what was actually *done* at each step — the split
   strategy and seed, the preprocessing recipe, which models trained, the coach
   shortlist that was followed. One typed slot per stage (`eda`, `split`,
   `training`, …). Voice: *dispassionate, factual.* It does not advise; it testifies.

4. **Narrative plane — "the author's desk."** `ml/narrative_engine.py`.
   Compiles the record plane into a `ManuscriptDraft`: Methods and Results written
   from provenance, an **evidence map** (each sentence → its source), an
   **ownership preamble**, and **`[AUTHOR REQUIRED]`** blanks wherever a claim needs
   human judgment. Voice: *careful, publishable, refuses to overclaim.* It reads the
   Record plane only — never invents.

**The crucial relationship to show:** Advisory *advises* the human; Record
*witnesses* the human's actual choices; Narrative *publishes only what Record
witnessed.* Advice that was never acted on does not reach the manuscript. Make this
visible: an advisory finding that is dismissed should visibly *not* travel to the
Record or Narrative lanes.

---

## 3. The modeling pipeline (the horizontal axis) — ten stages

A left-to-right (on desktop) spine of nodes. As the reader scrolls, a **playhead**
advances node by node; each node, when it becomes active, fires typed "packets"
down into the lanes it writes to. Keep node copy terse; the lane pulses do the
explaining.

| # | Stage (page) | Data plane writes | Advisory | Record | Narrative |
|---|---|---|---|---|---|
| 1 | **Upload & Audit** (01) | `raw_data`; **lockbox drawn** — a 15% test slice sealed | audit warnings | `record_upload` | — |
| 2 | **EDA** (02) | — (reads train rows only) | auto-insights (skew, missingness, correlations) | `record_eda_analysis` | — |
| 3 | **Feature Engineering** (03) | `df_engineered` | — | `record_feature_engineering` | — |
| 4 | **Feature Selection** (04) | `selected_features` | consensus notes | `record_feature_selection` | — |
| 5 | **Preprocess + Coach** (05) | preprocessing pipelines | **coach shortlist + evidence-probe verdict** | `record_preprocessing`, `record_coach` | — |
| 6 | **Train & Compare** (06) | `X_train/val/test` (from lockbox), `trained_models`, CV | post-training diagnostics (prefer-simpler, CI overlap, heteroscedasticity) | `record_split`, `record_training` | — |
| 7 | **Explainability** (07) | importances, SHAP | (drift/robustness notes) | `record_explainability` | — |
| 8 | **Sensitivity** (08) | seed / dropout robustness | — | `record_sensitivity` | — |
| 9 | **Hypothesis Testing** (09) | group tests, effect sizes | — | `record_statistical_test` | — |
| 10 | **Report Export** (10) | — | — | reads all slots | **`NarrativeEngine` compiles `ManuscriptDraft`** |

By stage 10 the reader should *see* the Narrative lane assemble itself from the
accumulated Record lane, blanks and all.

---

## 4. The three honesty disciplines (the emotional core — give each its own beat)

These are the moments that make the piece more than a flowchart. Each deserves a
dedicated scroll scene where the pipeline pauses and the mechanism animates.

- **A. The lockbox (`utils/test_lockbox.py`).** At stage 1, a slice of the Data
  lane (~15%) is drawn into a **sealed vault** and stays visibly sealed through
  stages 2–5. Every target-aware step (EDA, FE fits, feature selection, the coach
  probe) operates on `train_row_mask()` rows only — show these stages reaching
  *around* the vault. The vault opens **exactly once**, at stage 6 (Train &
  Compare), to score the final model. Tagline: *"The test set is sealed before
  anyone looks at the data."*

- **B. The invalidation cascade (`reset_downstream_results()`).** Show what happens
  when the user changes something upstream (new data, a re-drawn lockbox, an applied
  feature selection): a **"clear" wave** ripples downstream across *all four lanes*,
  wiping stale results — pipelines, splits, models, insights, provenance sections.
  Tagline: *"Absent is better than false."* (This is the principle the 2026-07
  `CODE_REVIEW.md` section formalizes.) Ideal as a scroll-triggered "what if I
  change my mind?" interlude.

- **C. Provenance → manuscript.** At the end, draw the literal wires from Record
  slots to manuscript sentences, and show an **`[AUTHOR REQUIRED]`** blank where the
  chain has no evidence to draw on. Tagline: *"It writes what happened, and leaves a
  labeled blank where only you can decide."*

---

## 5. Scroll narrative — scene-by-scene structure

Design as a vertical sequence of full-height (or tall) scenes. Each scene: a
**pinned/sticky visual** that animates as its text scrolls past, or a
scroll-progress-driven reveal. Suggested sequence (you may merge/rename, but keep
the arc):

0. **Hero.** The one idea (§1) as a headline. A faint, still diagram of the spine +
   four lanes previews the whole system. A scroll cue.
1. **"Meet the four planes."** Introduce the lanes one at a time as the reader
   scrolls — each lane draws in with its accent, name, one-line role, and its
   source file as a small monospace tag. End with all four visible, empty, waiting.
2. **Stage-by-stage build (the heart).** The playhead walks the pipeline. For each
   stage: the node lights, a one-line description appears, and packets fly into the
   lanes it writes to (per §3's table). The lanes visibly *accumulate*. Pace this so
   ~2–3 stages share the viewport's scroll budget; don't make it 10 identical beats —
   cluster (Ingest 1–2 · Shape 3–5 · Model 6 · Interrogate 7–9 · Publish 10).
3. **Discipline A — the lockbox.** Pipeline pauses; the vault seals at stage 1 and
   the reader scrolls the train-only stages reaching around it; the vault opens at
   stage 6. (§4A)
4. **Discipline B — the cascade.** A "you changed your mind" moment: a clear wave
   sweeps all four lanes. (§4B)
5. **Discipline C — the manuscript.** Record slots wire into manuscript sentences;
   an `[AUTHOR REQUIRED]` blank glows where evidence is absent. (§4C)
6. **The whole system, at rest.** All four lanes full, the spine complete, the
   manuscript emitted. A closing line restating §1. Optional: a compact legend
   mapping each lane to its file, so a curious viewer can go read the code.

---

## 6. Visual system (ground it in the app's own identity)

The app already has an identity — **honor it, don't reinvent it.**

- **Brand & palette.** Indigo is the app's signal color: `#667EEA` / `#7C8CF0`
  (light indigo) → `#4034A8` (deep violet). Neutrals should be indigo-biased, not
  pure gray. The app icon ("Distill") is a white Erlenmeyer flask holding a punched
  data table on an indigo squircle — the piece's motif language is **lab + tabular
  cells**, never finance/charts.
- **Four lane accents.** Give each plane a distinct, harmonious hue that still
  reads as one family. Suggestion (tune for contrast in both themes): Data =
  indigo `#7C8CF0`; Advisory = amber/gold (it *advises*, warm); Record = teal/slate
  (factual, cool); Narrative = a deeper violet/plum (the destination). Keep semantic
  colors (a "clear/invalidate" red-orange, a "sealed" state) separate from the four
  accents.
- **Typography.** Pair a characterful display face for scene headlines with a clean
  humanist sans for body, and a **monospace** for code symbols, file names, and
  state-key labels (`df_engineered`, `record_split`, `[AUTHOR REQUIRED]`) — the
  monospace tie-in to "this is real code" is doing real work. Inline faces as
  `@font-face` data URIs (the Artifact CSP blocks font CDNs); if you can't, use a
  strong system stack deliberately, not a silent fallback.
- **Theme-aware.** Support light and dark via `prefers-color-scheme` **and** a
  `data-theme` override; give both the same care. A dim conference room likely
  means dark — make dark the confident default look.
- **Motif consistency.** Data as small rounded **cells** (echoing the icon's table);
  the lockbox as a sealed cell/vault; packets as small tokens moving along wires.
  Keep it lab-instrument precise, not playful.

---

## 7. Motion & technical constraints

- **Self-contained, single HTML file.** All CSS/JS inline; no external requests
  (CSP blocks them). No animation libraries — hand-roll with `IntersectionObserver`
  for enter triggers and a scroll-progress calc for scrubbed sequences. SVG or
  Canvas for the spine/lanes/packets; prefer Canvas for many moving tokens.
- **Scroll-driven, not autoplay.** Progress is tied to scroll position; scrolling
  up reverses/re-arms. Nothing important animates while off-screen.
- **Animate only `transform` and `opacity`** for 60fps; avoid layout-thrashing
  properties. Throttle scroll work with `requestAnimationFrame`.
- **`prefers-reduced-motion`:** provide a graceful static/step-through version —
  every scene must still communicate with motion disabled (packets appear in place;
  the vault shows a sealed vs. open state without the travel).
- **Responsive.** On narrow screens the four lanes may stack or the spine may become
  vertical; content must never require horizontal page scroll. Wide diagrams live in
  their own `overflow-x:auto` container.
- **Accessibility.** Real headings, sufficient contrast in both themes, visible
  focus states, `aria` labels on the diagram; the piece should be readable as a
  document even if JS fails.
- **Performance budget.** Target a light file; lazy-init heavy Canvas scenes when
  near viewport.

---

## 8. Accuracy guardrails (the part that makes it trustworthy)

- **Do not invent stages, planes, methods, or numbers.** Ten stages, four planes.
  If you show an example metric, mark it clearly as illustrative — or pull the real
  seed-42 NHANES figures only if the user provides them; otherwise keep numbers
  schematic.
- **Every code label must be real.** `df_engineered`, `train_row_mask`,
  `record_training`, `InsightLedger.prune_auto_generated`, `[AUTHOR REQUIRED]` — all
  exist; open the files to copy exact names.
- **The honesty theme is the soul, not decoration.** The lockbox, the cascade, and
  provenance→manuscript are why this app exists. If a design trade-off threatens one
  of those three beats, protect the beat.
- **The four planes advise/witness/publish in that order of authority.** Never draw
  the Advisory plane writing directly into the Narrative plane — advice reaches the
  manuscript only by being acted on and thereby recorded.

---

## 9. Acceptance checklist

- [ ] A first-time viewer can name the four planes and say what each does.
- [ ] The ten-stage pipeline is correct and in order; each stage's lane-writes match §3.
- [ ] The lockbox seals at stage 1, is respected through 2–5, opens once at stage 6.
- [ ] The invalidation cascade visibly clears **all four** lanes.
- [ ] The manuscript assembles from Record slots and shows at least one `[AUTHOR REQUIRED]` blank.
- [ ] Every code/state label on screen exists in the repo.
- [ ] Works scrolled up and down; degrades cleanly under `prefers-reduced-motion`; no horizontal page scroll; light + dark both polished.
- [ ] Single self-contained HTML, no external requests, 60fps on a laptop.

---

*Authored from the live architecture of Tabular ML Lab. The downstream design
session has repo access — when in doubt, open the file and read it.*
