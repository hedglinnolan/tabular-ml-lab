# Handoff

The current state only. When a new state replaces this one, the replaced body goes, verbatim and
under a dated heading, to the top of `HANDOFF_HISTORY.md`, which holds every earlier state and the
rulings recorded in them. Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live
in the documents they amend.

## TL;DR (2026-10-19)

- **What:** the build is on the swarm path. Nolan ruled on 2026-10-17 that the swarm's pieces come
  before the drawn screens of W2–W4. On 2026-10-18 he kept the full path and said: "Do everything
  right up until we are ready to start the swarm and I will have a discussion with you about it."
- **Landed** locally on turbotab-next (8a3e0fff, on GitHub as `ci/full-int-3b`; turbotab-next
  itself is held until CI is green):
  - W1 part 2: the quest shell, the Predict cards and Predict Results, fixed through two
    walkthroughs;
  - the swarm's engine pieces;
  - the claims gates;
  - the Sullivan 2021 replication;
  - the swarm's driver, record, checks and intake, with a dry run whose 18 checks all agree;
  - the false sentences an analyst reads (R1-F's first batch) and RM's referee and audit fixes.
- **The pre-pilot report** is `plans/PILOT_READINESS.md`: four of five gates met; RB, the live
  lockdown check and one shell race remain; the pilot after Nolan's usage resets (Thursday 1 pm).
- **Spend:** 96.0M fresh subagent tokens from 2026-10-17 00:30 to 2026-10-19, against the ~80M
  reference (raised from 50M on 2026-10-18; "if we blow past 80M that's fine"); 126.5M since
  Thursday, about 91% of Nolan's week.

## State on 2026-10-19, checked against git

- **turbotab-next** is 8a3e0fff locally. It holds:
  - 12345c40, which merged feat/w1p2-x (W1 part 2 with the swarm engine pieces), feat/r1-gates,
    feat/sullivan-r and feat/sullivan-a;
  - feat/int-3 on top: feat/land-fix-1 (the path fuzzer's I11 fix, the NHANES m2 journey, CI pinned
    to ubuntu-24.04), feat/rm-core (RM's referee and audit fixes, without R1-M), feat/rf-1b (R1-F's
    first batch), feat/drv-int and feat/drv-fix (the driver, record, checks, intake, analyst type
    and hook, the dry run), and three fixes made at integration.
- **CI on 8a3e0fff** (`ci/full-int-3b`): every job green but the real-server journey
  `quest-nhanes-predict`, where the page's quest bar shows "Who's in" reopened while GET /quest does
  not (passes on this machine; passed on 12345c40). **turbotab-next is not pushed until it is fixed.**
- **Environments:**
  - `venv-next/` in the main checkout is TurboTab v2's environment (CI's pins); `scripts/py` runs
    it from any worktree.
  - R 4.6.1 with `survey` and `haven` is on the machine. No CI job installs R.
- **Local and private** (Nolan, 2026-10-17: "Let's not make those public"):
  - `.claude/settings.local.json` runs `.claude/hooks/guard.py` (five Never rules) and, since
    2026-10-19, `research/swarm/analyst_hook.py` for every tool (an analyst runs only the driver).
  - None of these settings is ever committed.

## Done since 2026-10-16

Each line gives the package's fresh subagent tokens, measured from its transcripts.

- **W1 part 2** (plans/W1_PART2_PLAN.md):
  - Stage A, B and C built the shell, the Predict cards, Predict Results, the held-out baseline and
    its verdict, and the methods sections.
  - The orchestrator's walkthrough of NHANES glucose under Predict found 5 blockers, 7 majors and
    13 minors (notes/walkthrough/2026-10-17-nhanes-glucose-predict.md).
  - Fix rounds W1-R to W1-X took them, about 18M in all.
  - The re-walk on 712747ff found findings 1–17 and 19–25 fixed, and 10 more (N1–N10), which W1-W
    and W1-X fixed.
- **Rulings made on the way, recorded in the code and its tests:**
  - **R-g:** roles proposed below high confidence with the same role, confidence and reason are
    one group, confirmed in one press. A block that mixes groups is refused.
  - **Under Predict the roles card names no preparation step** after four rounds of false step
    notes. It says only "N columns hold a predictor role, whichever group holds them."
  - **N9:** a noticing whose deciding question is already answered is proposed as a limitation.
  - **The methods** may fold alike confirmations into one sentence, and the entry lists its
    records.
- **The swarm's engine pieces** (plans/SWARM_ENGINE_PLAN.md, about 6M with integration):
  - GET /fits, a record of every completed fit with its paired movement under Predict;
  - GET /results;
  - GET /mattered, the specification curve after the lock, checked against R's `survey` package;
  - analyst mode: re-seal refused, a claims render log in every mode, a preview log, and the
    decision's `note`;
  - each file's provenance and SHA-256.
- **The claims gates** (plans/R1_PLAN.md R1-G, 1.46M):
  - every claim's origin;
  - meta-test 7: a swarm claim needs two referees, neither its author nor a fork, plus a basis
    audit;
  - meta-test 8: every grade change is a history line;
  - research/registry's decoy queue, which reproduces the backtest's own gate result.
- **The Sullivan 2021 replication** (plans/SULLIVAN_PLAN.md,
  notes/replication/sullivan2021.md, about 1.3M):
  - Through the engine, Table 4 matches the paper at printed precision, with exact n.
  - A blind R-only reference agrees with the engine to about 1e-11 on equal rows. Every row
    difference is traced to a named decision.
  - The paper's stated M4 (Model 2 plus medication) does not reproduce its printed values. The
    Table 4 model plus medication reproduces all of them; the report gives both readings.
  - It found 14 engine gaps (INBOX).

## Next, in order

The goal (Nolan, 2026-10-18): everything up to the swarm's start, then a discussion with him
before any pilot run. His usage resets Thursday at 1 pm; nothing optional runs before it (his
words of 2026-10-18: a buffer for other tasks this week). Each step's status ends with one line of
cost against the ~80M reference, which is not a cap.

1. **The discussion** of `plans/PILOT_READINESS.md` (§4's four questions). **[Nolan]**
2. **Nolan restarts Claude Code** (`claude --resume`) so the analyst agent type loads; then the
   live lockdown check and the analyst's model id (Haiku 5.5, not 4.5).
3. **After the reset:** RB (its brief: the orchestrator's `rb_args.json`, 25 claims at most), and
   the `quest-nhanes-predict` shell race; then CI green and turbotab-next pushed.
4. **The measured pair** (one Haiku, one Opus analyst on the Predict fixture), then the pilot,
   daytime and unthrottled, as Nolan schedules it.

## Open for Nolan

- **His own walk** of NHANES glucose under Predict on the landed product, when he chooses. It is
  no longer a gate for the swarm path (his go-ahead of 2026-10-18).
- **The pilot's analysts are Haiku 5.5** (his ruling of 2026-10-18): three per dataset, with one
  Opus ceiling run per dataset; Sonnet runs only if Haiku fails SWARM_PLAN §7, and after he is
  told (SWARM_PLAN §6).
- **The schedule** for the pilot's runs, which are heavy runs on his machine.

## Nolan's context

- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit.
  - The orchestrator's re-walk on 2026-10-18 gave a held-out MSE of 1,131 for ridge on 21
    predictors (20% sealed), with a cross-validated R² of 0.156 (0.141 to 0.171).
  - The light walk had given R² 0.155 to 0.18. He never passed about 0.25, and that only after
    trimming glucose's tails before upload (INBOX, "From Nolan's question on predicting glucose").
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **No preview release of v2** (2026-10-17): "I'm all in on the analytic engine's work."

## Heavy runs to schedule with Nolan

- **The T1 rerun at 600 datasets:** about 60 minutes at 4 jobs, approved on 2026-10-11, and still
  waiting for a time.
- **The reference-journey recapture:**
  - including `src/mocks/fixtures/m3-nhanes-prediction.json`, which still carries the design's
    estimand under Predict;
  - and the dietary-prediction capture;
  - then the materiality calibration on it.
- **The swarm pilot's runs** (SWARM_PLAN §6), after the discussion; daytime and unthrottled
  (`plans/PILOT_READINESS.md` §3: the throttle made long commands miss the prompt cache).
