# Handoff

The current state only. When a new state replaces this one, the replaced body goes, verbatim and
under a dated heading, to the top of `HANDOFF_HISTORY.md`, which holds every earlier state and the
rulings recorded in them. Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live
in the documents they amend.

## TL;DR (2026-10-19)

- **What:** the build is on the swarm path. Nolan ruled on 2026-10-17 that the swarm's pieces come
  before the drawn screens of W2–W4. On 2026-10-18 he kept the full path and said: "Do everything
  right up until we are ready to start the swarm and I will have a discussion with you about it."
- **Landed** on turbotab-next and pushed (2026-10-19):
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

- **turbotab-next** holds:
  - 12345c40, which merged feat/w1p2-x (W1 part 2 with the swarm engine pieces), feat/r1-gates,
    feat/sullivan-r and feat/sullivan-a;
  - feat/int-3 on top: feat/land-fix-1 (the path fuzzer's I11 fix, the NHANES m2 journey, CI pinned
    to ubuntu-24.04), feat/rm-core (RM's referee and audit fixes, without R1-M), feat/rf-1b (R1-F's
    first batch), feat/drv-int and feat/drv-fix (the driver, record, checks, intake, analyst type
    and hook, the dry run), and three fixes made at integration.
- **CI:** full tier green on db92976e (all 13 jobs; `m1-journey` timed out once waiting for the
  old screen's exclusions card and passed on a rerun of the same commit, its second flake after
  e7fd917f's). Earlier: green on 5901298a.
- **Windows, 2026-10-20** (Nolan's first start on a Windows laptop; three fixes, each with its test):
  - the launcher installs with uv's copy mode on Windows: a OneDrive-managed user folder refused
    uv's hard links (os error 396);
  - the decision log and the fit record no longer call `os.pread`, which Windows lacks (the first
    answer raised AttributeError); CI's Windows job runs both checks;
  - every file the engine moves into place goes through `core/atomic.py`, which on Windows retries
    while another process holds the target open: stage jobs failed, and a stage's `latest` pointer
    stayed on the previous result.
  - `probe/windows` (never merged) walks the two quest journeys on a Windows runner and annotates
    every server traceback. On db92976e's code both journeys pass, and the only traceback is
    asyncio's harmless connection reset (INBOX).
- **Since 8a3e0fff, on turbotab-next:** the analyst model set to `claude-haiku-5-5`; the shell
  race fix; SWARM_PLAN version 2 (after the independent audit in `notes/swarm-audit/`), under
  review by a Fable xhigh reader before anything in its §10 is built.
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

The goal (Nolan, 2026-10-18): everything up to the swarm's start, then the pilot. The plan is
`plans/SWARM_PLAN.md` version 5.1, reviewed by five independent readers (`notes/swarm-audit/`); the
fifth's verdict was BUILD (2026-10-19). A sixth (2026-10-20, after Nolan's first walk) found the
driver answers no declaration; package 1 now adds them, and a crash sweep runs before the freeze. Each step's status ends with one line of cost against the
reference, which is not a cap.

1. **After Nolan's usage resets (Thursday 1 pm),** the build in SWARM_PLAN §10, packages 1–7
   (38–61M fresh, about 42–67M with a buffer), with its order and its notes for the builders.
2. **The pilot in the same week** (SWARM_PLAN §6–§7), on the thin buffer Nolan accepted: daytime,
   unthrottled, within 40% of the week, the Estimate fixture first.
3. **The live lockdown check and the analyst's model id** (package 7) need a fresh session on the
   home machine, since agent types load when a session starts.

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
