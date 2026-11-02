# PP1: the first engine batch after the pilot

**TL;DR.** Nolan, 2026-11-01 18:40: "Go through the list! You are empowered to keep working." The scoping audit
(read-only, `durable/pp1/scope_report.json`) found every item's premise true or partly true. Eight packages build
now, four at a time, on `pkg/*` branches off 977ebbfa: the plasma fixes he earmarked (QC repairs after the split, the
QC correction's hidden blanks, findings that read the QC rows, imputation pooled per feature, the refusal moved before
the lock), the imputation width rule, the stepwise sensitivity analysis on aliased designs, two dead-end exits, the
lock's own record, and two CI pins. Nothing lands in the main checkout until the pilot's glucose runs end.

## Packages (wave script wf_3675cbf5-37b; briefs in durable/pp1/<name>_brief.json)
| package | scope items | base | notes |
|---|---|---|---|
| qc-rows | 2, 4(1) | 977ebbfa | guard reads the draw, not the split's existence; NaN as null |
| stepwise-alias | 7 | 977ebbfa | R's aliasing, QR inverse, df n - rank, failed copy refused |
| imputation-cap | 3 | 977ebbfa | rows-per-parameter rule from the engine's own constants; refused before the lock |
| findings-rows | 5, 4(2) | 977ebbfa | findings and left-censoring judged on participants; settled finding keeps its repairs |
| decision-words | 8, 9 | 977ebbfa | role_unconfirmed's three exits; imputation_levels only where rows repeat |
| lock-withdrawal-record | 10 | 977ebbfa | recorded_by turbotab with its note; assembly maps the old ones to 'app' |
| ci-pins | 11 | 977ebbfa | REML pin on what float64 certifies; reach's timeout says why |
| featurewise-mi | 1 | qc-rows's head | per-feature censored MI, Rubin + Barnard-Rubin, BH; check before the lock; single fill leaves the inference exits |

Later (PP1b): curve-over-plans (scope item 6) on the integrated branch, after featurewise-mi and
lock-withdrawal-record. Then the integrator on feat/pp1-int (the shared stages/__init__.py version lines, generated
files), ci/pp1-int.

## Rulings (dated)
- 2026-11-01, the orchestrator as methods owner, within Nolan's "empowered": every call adds a capability or makes a
  number right; none removes one. The single censoring-aware fill leaves the feature-wise exits under inference only
  because the pooled route replaces it (it shrinks the event difference 29% at 60% censored, coverage 0.865).
- The imputation width uses the engine's existing rows-per-parameter constants, not a new rule.
- An aliased term in the Wald backward elimination is set aside untested, named as R's lm names it (NA).
- Scoring: the pilot's score keeps the pre-registered rule for run_19 (refused); a secondary, marked not
  pre-registered, counts a feature with no q as not discovered.
- Machine: four packages at a time; the quiet throttle runs 21:57 to 08:58 (durable/night_pp1.sh, PID in
  night_pp1.pid); a glucose analyst run never starts while PP1's workflow is live.
- 2026-11-02, the orchestrator: imputation-cap's width rule refuses a model only where it is wider than 80 columns
  AND short of the rows-per-parameter rule, so nothing the base drew is refused (amending the brief's ruling).
- 2026-11-02: the left-censoring reading sets aside every column with two or fewer distinct values on the
  participants, as the omics_scale reading does (completing findings-rows's "as the omics_scale finding does").
- 2026-11-02: the answer's leavable set reaches the fit, so the fit and the answer refuse in the same words with the
  same ways forward; the exposure is never offered to leave out.

## Status (2026-11-02 ~11:15)
- On feat/pp1-int (the orchestrator's merges): qc-rows, lock-withdrawal-record, ci-pins (approved);
  stepwise-alias, decision-words, imputation-cap, findings-rows (each at its repair cap, its findings left carried
  into the follow-up); the follow-up's pp1f-findings and pp1f-sweep (approved) and pp1f-impute (at its cap: one
  minor to the backlog, its stage-version minor in 15efc6e1). decisions.py imports smcfcs after row_floor.
- featurewise-mi (pkg/featurewise-mi f55a6309, 9.10M fresh, past the 6M tripwire, two blockers: the Tobit fit not
  certified at its maximum on sparse designs; an OverflowError at imputation.py:364) waits for Nolan: one bounded
  honest-refusal repair, or park it.
- Fresh spend: the wave 22.91M (46 agents), round 3 2.12M, the follow-up 1.99M.
- Next: the integrator (generated files, fast tier, ci/pp1-int, CI read); PP1b curve-over-plans after Nolan's
  featurewise-mi ruling; nothing to main until the glucose runs and their replays end.
