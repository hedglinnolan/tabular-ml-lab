# Nolan's drive rubric

Distilled 2026-09-27 from the record of his past drives, so milestones can be reviewed through his
eyes when he is not available to drive. The orchestrator runs §5 on every milestone.

**Provenance.** **[N]** = words the source attributes to Nolan (the product owner, "his words", a
quoted user). **[obs]** = an agent's or tester's observation, not his words. `F:ID` = a row in
`docs/turbotab/archive/data/findings.json`; bare `GUIDED-1xx` ids are narrated in
`docs/turbotab/archive/DRIVE_LOG_NHANES.md`. Most of his own at-the-screen reactions come from his real-NHANES
drive of 2026-08-04 and the product-owner rows `F:DRIVE-001…010`. He calls himself *"the product
design guy"* and does not read code (`docs/turbotab/archive/prompts/PM_TRANSITION.md` §02): he judges only what is on
screen.

## 1 · What he is trying to feel

- **Domain tradecraft, made beautiful and digestible — not just correct.** [N] *"The whole point of
  this app is to be a beautiful expression of how to conduct informed math modeling in the
  researcher's specific domain, with real tradecraft and results to back up their decisions, all
  presented to the user in a dynamic, easily digestible manner. In addition to being correct, the
  engine must surface and it must be beautiful."* — `PRODUCT_VISION.md` §06b
- **More capable, never narrowed; the app speaks frankly instead of deciding for him.** [N] *"Guided
  should be easier to understand and more dynamic, not less capable."* · *"The shape of the data
  changes the model shelf, but never to the extent that a user has no option to select a bad model.
  We do our best to fit based on their selection, but the app surfaces the concerns outright."* —
  `PRODUCT_VISION.md` §04b
- **The app knows his field from the first screen, and that shapes what he is offered.** [N] *"add
  domain-informed content starting at step 10 of the app that cascades down into the routing and what
  options are presented to the user"* — later corrected to **step 0** (`RETROSPECTIVE.md` §01, §01b).
- **His answers actually change the analysis.** [N] *"We straightforwardly do ask prediction v
  inference, I am just not sure we really wire it to any concrete changes in the engine yet."* —
  `F:GUIDED-231`
- **He can see his data richly, in his domain's own pictures.** [N] On interactive, exportable
  domain EDA figures: *"this is the money maker"* (`F:DRIVE-009`). *"Missingness pattern analysis to me
  would mean show me the co-missingness pattern visually."* (GUIDED-168)
- **He can understand how each model reasons about nutrition; usefulness beats novelty.** [N] *"SHAP
  in particular I have high hopes for being the vehicle by which a nutrition modeler can understand
  the inductive bias each model brings to the table in solving their problem."* · *"Just because a
  CRAN package exists in the world with a specific plot doesn't mean that would not still be a useful
  feature to ship in our app."* — `PRODUCT_VISION.md` §06c

Standing positions the record paraphrases: the steps are not the product, the connective tissue
between them is (`F:GUIDED-178`); hard questions stay hard and the answer is to invest in teaching
(`F:GUIDED-160`); a diagnosis never ships without its lever (`RETROSPECTIVE.md` §01b).

## 2 · What reliably frustrates him (most recurring first)

1. **He presses something and nothing visible happens** — he cannot tell "not wired" from "refused
   silently". [N] *"It appears the app does nothing when I click those buttons."* (a correct 409 that
   never reached him, GUIDED-167) · *"Clicking 'set these entries to missing' does nothing."*
   (GUIDED-165) · *"…when I went to click the very bottom card, it just dismissed itself."*
   (GUIDED-156). Also GUIDED-161/162, `F:DRIVE-004`.
2. **The app says something that is not true right now** — a transcript line saying entries "were
   set to missing" over unchanged data (GUIDED-165); 8 of 9 columns called "written as text" when they
   were not (GUIDED-158); [obs] "No model has been fitted yet" after fitting (`F:DRIVE-056/060`); green
   "complete" over failed runs (`F:DRIVE-065`); one study reported with three different Ns
   (`F:DRIVE-031/045/046`).
3. **A problem flagged with no way to fix it** — no row list for impossible values, no way to mark a
   column unclean (`F:DRIVE-007`); a missingness panel that showed a change without letting him make
   it (`F:DRIVE-008`); three instincts, one route offered (GUIDED-166). [obs] *"Being warned four
   times and empowered zero times is worse than it sounds"* (`F:DRIVE-057`).
4. **He loses his place** — [N] *"Every step is lit up except train. And there is no option to
   train."* (GUIDED-159); auto-scroll past the card he was reading (`F:DRIVE-006`); [obs] reflow under
   the cursor causing wrong state changes, twice to the target (`F:DRIVE-054`); a rail that highlights
   a step without going there (`F:DRIVE-047`).
5. **Walls** — ten look-alike cards as "accidental infinite scroll"; his fix was one card with
   horizontal paging (`F:DRIVE-005`). [N] *"Even when hovering over the blurbs… I am not sure I would
   understand the difference."* (135-word disclosure, GUIDED-160). [obs] critical and trivial cards
   looking the same; 150–250-word tooltips over controls; red never used where numbers could not be
   trusted (`F:DRIVE-058`).
6. **Labels that don't say what will happen** — "earmark it", "decide at Explore", "mark for
   manuscript" (`F:DRIVE-003`); a title promising one analysis and delivering another (GUIDED-168).
7. **The app forgets what was settled or contradicts what it knows** — [N] *"Didn't we just settle
   how to handle missingness as informative or not?"* (GUIDED-166); [obs] naming `patient_id` then
   recording no person column (`F:DRIVE-057`); `record_id` accepted as target, refused as predictor
   (`F:DRIVE-062`).
8. **Machinery showing through** — `{n_bins}` in a methods sentence, Python `None` in the checklist
   (GUIDED-175/179); [obs] `[object Object]`, raw `**` markdown, dependency names and page paths
   (`F:DRIVE-038/059`).
9. **The app choosing for him quietly** — [N] median fill on `meds_hbp` is *"possibly a bad idea"*
   (GUIDED-163); [obs] default imputation disclosed in small print after the fit (`F:DRIVE-061`); the
   event level never asked for a 0/1 target (`F:DRIVE-032`).

## 3 · What earned praise (the bar)

His own endorsements: the spirit of the before/after missingness panel (`F:DRIVE-008`); bulk repair
as one worked example → pick the set → apply once (`F:DRIVE-002`); a pager for same-kind facts
(`F:DRIVE-005`); his working table in a right-hand panel beside the decisions (`F:GUIDED-174`).

Observed by others: the two-press preview — before/after table, "CELLS CHANGED 0 → 140", "Nothing has
happened yet" — "the best screen in the app"; arithmetic that respects the reader ("1 row moves a rate
by 0.125" read like "a statistician who respects you"); absences that give a reason ("a model grading
its own homework"); domain catches (glucose in two units, censored assay values, a decimal comma, grain
contradicting the data); checks that can actually fail; a fold row that says what it hides ("7 more —
3 warnings, 4 cautions"). Run 1 called the Explore layer "genuinely beautiful".

## 4 · Domain expectations he brings

- **Blanks have meaning** — `meds_hbp` (71% blank) means "not asked"; median fill would put every
  unknown on BP medication (GUIDED-163). The missingness decision must hold across steps (GUIDED-166).
- **Impossible values** — the full row list to fix at source, a way to mark a column unclean, and three
  routes: set to missing, exclude rows, mark the column corrupted (`F:DRIVE-007`, GUIDED-166).
- **Reproducible recodes** — which of `female`/`male` became 1 (GUIDED-157); `imputed_*` columns are
  `bool`, not text (GUIDED-158).
- **Identifiers aren't data** — `SEQN` in the nutrient picker returned a SETTLED EAR claim (GUIDED-170,
  critical); energy has no EAR either (`F:DRIVE-034`).
- **Subgroups two ways** — separate models per subgroup, or one pooled model with post-hoc subgroup
  analysis (`F:GUIDED-106`); right after the target he expects to choose features and whether to slice
  (`F:DRIVE-010`).
- **A methods section a nutrition reviewer accepts** — energy-adjustment model, misreporting rule and
  cut-off, usual-intake handling, complex survey design, DRI edition (`F:GUIDED-123`); inference-family
  models on the shelf (`F:GUIDED-105`).
- **Figures from his field** — per-feature distributions for nutrition, relative abundance for
  metabolomics, 2-D PCA with k-means; interactive and exportable (`F:DRIVE-009`).
- **Held-out discipline (his ruling)** — anything answering "is this data corrupted?" may see every
  row; anything informing a modeling choice may not see the sealed rows (`F:GUIDED-096`).
- **Silent on his file, not raised by him** (adjudicator's predictions, confirmed): no survey weights
  or design variables (`WTMEC2YR`, `WTDRD1`, `SDMVSTRA`, `SDMVPSU`); nine pooled cycles
  (`cycle_begin_year`) never remarked on; `imputed_*` flags not linked to their base columns.
- **Which N** [obs] — 21,849 uploaded, 6,297 with an outcome, 5,352 trained (Runs 2–5, Drive 8).

## 5 · Review checklist — every screen, every milestone

1. Does every press produce a visible acknowledgment at the control that quotes what was recorded — and
   does a refusal look different from a dead control?
2. Is every receipt, header, count and summary true at the moment it is read, including after a fit, an
   undo or a changed answer?
3. Could a reader reproduce each recorded decision from its sentence alone (which level became 1, which
   rows changed, which method ran)?
4. Does every study-level number name its denominator, and do all screens agree on it?
5. Does every warning arrive with the control that acts on it, in the same place?
6. When the app already holds an answer, does it use it instead of asking again or claiming ignorance?
7. Does the page stay still unless he pressed a navigation control — nothing reflowing between paint
   and click?
8. Can he tell where he is, what comes next and what is reachable — and does pressing a step go there?
9. Are pushed cards few and ranked, the critical one looking critical, same-kind facts paged into one
   card, colors matching the stakes?
10. Does every control's label state its effect (what, where, undoable?) and every title deliver the
    analysis it names?
11. Is the in-card explanation two or three sentences, deeper teaching on his own columns in the panel,
    and no tooltip covering a control?
12. Is the page free of internals — `None`, `{placeholders}`, `[object Object]`, raw markdown, paths,
    dependency names?
13. Does the app avoid choosing silently and never remove an option, while stating its concerns
    outright?
14. Does each answer (lens, purpose, grain…) visibly change something downstream — or say plainly that
    it changed nothing?
15. Can he see his data: the working table beside the decisions, before/after previews on real rows,
    domain figures he can interact with and export?
16. On NHANES: is `SEQN` an identifier everywhere, and does the app speak to survey design, pooled
    cycles and the `imputed_*` flags?
17. Do nutrition claims refuse where the science refuses, with evidence badges he could defend to a
    reviewer?
18. **Milestone:** can he take his real NHANES file from upload to a fitted model and a complete,
    self-consistent draft with no dead end — and does the draft say what the screens showed?
