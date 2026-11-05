# Handoff

The current state only. When a new state replaces this one, the replaced body goes, verbatim and
under a dated heading, to the top of `HANDOFF_HISTORY.md`, which holds every earlier state and the
rulings recorded in them. Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live
in the documents they amend.

## TL;DR (2026-11-04)

- **The swarm pilot is closed** (Nolan, 2026-11-04: "Close the pilot now; report glucose apart").
  - Forty analyst runs on five datasets cost about 9.1M fresh tokens.
  - Haiku and Sonnet both passed the accuracy bar on the walking fixture.
  - The stance question is deferred (2 decisions that matter on walking, short of the 3 required).
  - Model prices are filled in `research/swarm/prereg.json` (df8baed8). The referee audit ran: its
    planted bad reasons were not caught, so no production analyst is named yet.
  - Glucose is reported apart. The report is the orchestrator's `durable/pilot/PILOT_REPORT.md`.
- **An external vision audit (2026-11-04)** found the engine on course and the experience drifting.
  Nolan's ruling on it: "Adhere to the wisdom of the audit": first make one Estimate analysis work
  end to end in the app, then the voice pass, then the return thread, then export.
- **The post-pilot order:**
  - batch 1: Estimate end to end;
  - batch 2: the voice pass;
  - batch 3: the return thread;
  - batch 4: C7a placement, "Write my own", Word and LaTeX.
- **Landed here** (local, not pushed): the post-pilot packages through integration pass 7
  (feat/pp1-int at 00e4364e, CI green on all 9 jobs), fast-forwarded.

## State on 2026-11-04, checked against git

- **turbotab-next** (local; pushed only with Nolan's clearance) is feat/pp1-int at 00e4364e,
  fast-forwarded, plus this file's commit. It holds df8baed8 (the prereg prices) through a907e2b9.
- **Integration pass 5** (b6085792) landed earlier the same day. **Pass 6** (feat/pp1-int at
  925b8b03) adds sixteen packages:
  - PP2's engine follow-ups: decision-exits, imputation-draws, imputation-agreement,
    curve-hold-live, feature-table-applies, lens-contradiction, mi-words, survey-weight-words,
    family-effect-words, why-audit;
  - PP3's front end, from Nolan's Q1 to Q13 rulings: first-look, reason-box, writeup (read-only),
    act-on-hold;
  - the front-end packages turn-exits and small-fields.
  - Pass 6's CI failed one Linux float tolerance (test_imputation_draws.py), which pass 7 fixed.
- **Pass 7** (00e4364e) adds batch 1's first two packages, b1-writeup-opens and b1-straight-line,
  that tolerance fix and one test literal. ci/pp1-int carries its CI: run 41943389984, 9 of 9
  green.
  - Stage versions now: fit 44, sensitivity 28, calibration 24, secondary 18, scales 15, effects
    19, modification 10, mattered 6, findings 26, proposals 26, forms 2.
- **Batch 1, approved and not yet integrated:**
  - b1-decide-cap (bbaf816c): the Decide cap per stage;
  - b1-results-check (e8fcacad): Results draws the straight line's check.
- **Building:** b1-decision-exits (no refused press is a dead end). Pass 8 takes all three.
- **Next:**
  - b1-estimate-journey: one real-server Estimate journey ending in the download, a heavy run, in
    daytime;
  - Nolan's own walk of one Estimate analysis (his ruling 7);
  - then pp3-retire-classic may land.
- **Parked at cap:**
  - b1-ask-less's readings (5e2d0a6e): missing-value codes outside Classic's list still read as
    amounts, and transformed names outside a block list get a unit;
  - pp3-unit-card: Q6 waits on an engine design for reopen bookkeeping;
  - pp2-server-reads;
  - pp2-findings-omics.
- **The INBOX** holds each pass's carried items, by package, from line 1165 (pass 7's block last).
- **Local and private** (Nolan, 2026-10-17: "Let's not make those public"):
  - `.claude/settings.local.json` and `.claude/hooks/` run the Never-rules guard and the analyst
    hook.
  - None of these is ever committed.

## Open for Nolan

- **A corrected referee pass**, as a declared deviation, before any production analyst is named.
  It would ask whether a reason's premises match what the analyst was shown, with fresh plants, at
  about 1M tokens.
- **His 10-verdict blind check** of the referee's judgments.
- **His walkthrough of run_41.**
- **The common-sense audit after batch 1.**
- **The voice pass plan** (batch 2): four questions, about 21M fresh.
- **His review of the design choices made without him** (Q13: the orchestrator's
  `durable/pp3/design_choices.json`).

## Heavy runs to schedule with Nolan

- **The b1-estimate-journey run:** daytime, after batch 1 integrates.
- **The T1 rerun at 600 datasets and the reference-journey recapture,** still waiting for a time.
