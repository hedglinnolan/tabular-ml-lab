# Handoff

**State (2026-10-01): M1 is done and tagged `next-m1`.** The production app runs the full NHANES
journey in a real browser: the Record with its card-based decisions, the pipeline banner, and the
stage. The stage has the transform player (the flip plays each method's real storyboard), previews
of every option on the user's own data, finding evidence, savable figures in journal style with
provenance captions, and the Results (models against a baseline, coefficients, and substitution
curves with a refit uncertainty band). Checks: 444 Python tests, 100 frontend tests, and the
m1-journey Playwright spec against a real server. Run records: `m1/w1-result.json` and
`m1/w2-result.json`. Screens: `m1/screens/`.

**Next: M2** (BLUEPRINT §9). It covers the opening sequence for all five lenses (orientation,
grain and repeats, eligibility, the seal), findings with preview-before-apply and deferral, and the
wide omics data path, benchmarked. Start with the "M2-first" and "[M2]" items in INBOX.md (the energy
card's word budget first). Then triage the rest of that file, merge `main` in, write M2_CONTRACT.md,
and launch.

---

## History: the pause of 2026-09-27 (mid-M1)

Paused at Nolan's request (weekly usage limit). Everything is committed and pushed. Start here.

## Where it stands

- **M0 is done** and tagged `next-m0`.
- **M1 part 1 is done and merged** on `turbotab-next`: validators, the server-side interview
  Router, column roles, the participant flow, the sealed split, the generic consequence diff, and
  previews for every M1 question. Also three model families, in-fold pipelines, fit, substitution
  curves, publishable decision sentences, teaching for all 11 questions, pack proposals, and
  findings that name their lever. Workers now spawn on demand.
- **Checks:** 395 Python tests and 27 frontend tests pass.
- **The live NHANES journey** runs end to end through the API in about 10 s. Fit takes about 2.5 s,
  and previews take 1–30 ms. Details are in `m1/w1-result.json` → `server.measurements`.
- **M1 part 2 has not started.** That's the frontend: the Record and the pipeline panel rebuilt on
  the chosen design, then integration, two reviewers, and fixes.

## The design decision — mostly made (see BLUEPRINT §11.1)

Nolan ruled from the screenshots: `stage` is the base; no per-option preview thumbnails; `stage`'s lineage diagram is canonical; prototype a **pipeline banner** (the whole pipeline as a compact strip at the top, everything below is the working window). **The slider then became a transform player** (BLUEPRINT §11.1): one two-state *now ⇄ with this choice* flip that plays the method's own labeled storyboard as it flips (forward, or in reverse on the way back; brisk; interruptible) and drives every view; step dots pause on any step; switching options morphs directly between results without replaying the storyboard; saves capture any labeled state. Backend work this adds: consequence views gain optional `steps`, and the energy-adjustment builder supplies residual/density/partition storyboards. **The design is fully ruled — build it.**

### Background: the three prototypes

Three prototypes answer the wall-of-text problem (BLUEPRINT §11) on the same real NHANES fixture.
Branches are on origin:

| Branch | Angle | Visible words (S1 preview) |
|---|---|---|
| `explore/stage` | Hovering or focusing an option morphs the pipeline panel into its consequence: a primary before/after view, plus lineage and distribution | 269 |
| `explore/inline` | Every option card carries a sparkline preview; the focused card enlarges in a fixed stage under the strip; a pin compares two options | 181 |
| `explore/scrub` | One before→after transformation per option, scrubbed; table cells roll like an odometer; sidenotes | 243 |

Screenshots, frame strips, word counts and each designer's rationale and weaknesses are in each
branch under `docs/turbotab-next/m1/explore/<angle>/`, and in `m1/w1-result.json` → `explore`.

**Nolan has seen `stage/s1-density-light.png` and said:** *"I really like the screenshot you sent me.
It's a really good and promising design style."* Make `stage` the base. Graft in what the others do
better, if it survives a side-by-side look:

- inline's compare pin and its lower word count;
- scrub's column-identity morph in the table (`fat_total` → `fat_total_adj` → `fat_total_per_kcal`)
  and its sidenotes.

Also fix the weaknesses the designers named:

- Stage's first-preview transition snaps.
- Histogram bins morph across different units. Crossfade when the unit changes, and morph only
  within the same unit.
- Hover-to-preview needs a touch equivalent.

## Found by the live journey — carry these into M1 part 2

1. **Complete cases drop 86% of NHANES.** `meds_hbp` and `meds_chol` are blank for 18,405 rows, and
   a blank there means "not asked" (DRIVE_RUBRIC §4). The missing-values question must say so
   through its preview, and should offer leaving those columns out, before the user records
   "complete cases".
2. **Nested nutrients distort substitution.** The fat_total → carb curve holds `fat_sat`, `fat_mon`
   and `fat_poly` fixed while `fat_total` moves. The design warns about it, but roles should mark
   nested components (sugar ⊂ carb, fat subtypes ⊂ fat_total), and substitution pairs should
   respect them.
3. **Boosted trees score below the mean baseline** (CV R² −0.04) with no concern attached. The shelf
   and results should say when a family underperforms the baseline.
4. **The linear substitution band has zero width.** Resampling rows through one fitted linear model
   cannot vary its average effect. The band needs a refit bootstrap (PRODUCT_VISION §06c, mark 2).

The other 36 notes from this run are in `INBOX.md` under "From M1 workflow 1", untriaged.

## A framing Nolan added at the pause

*Modeling decision provenance* — his name for a reproducibility problem in nutrition research:
readers can't reconstruct which choices turned raw data into a model's inputs. He noted the lineage
diagram illustrates it well. It is now North star item 4 in the BLUEPRINT. It raises the lineage
and the row flow from teaching aids to deliverables: M1 part 2 should make them publication-quality;
M5's export carries them as figures plus a replayable provenance record.

## Next session, in order

1. Look at the three prototypes side by side (their screenshots and frames). Rule on the design and
   record the ruling in BLUEPRINT §11.
2. Merge the chosen prototype's reusable components and `explore/fixtures` into `turbotab-next`.
3. Launch M1 workflow 2 on that design and the four points above: the frontend Record (questions
   from the Router, previews, teaching drawer, findings with levers), the pipeline panel (Rows flow,
   Columns lineage, Results with metrics and substitution curves), integration, two reviewers
   (function, and taste against DRIVE_RUBRIC), and a fixer.
4. Drive it with Playwright on the real NHANES export, triage the inbox, merge `main` in, and tag
   `next-m1`.
