# Ready to swarm? The pre-pilot report (DRIVER_PLAN §4)

Written 2026-10-19 by the orchestrator for Nolan, under his go-ahead of 2026-10-18: "Do everything
right up until we are ready to start the swarm and I will have a discussion with you about it."
This is that discussion's paper.

**Superseded in part the same day:** after the discussion and an independent audit
(`notes/swarm-audit/2026-10-19-plan-audit.md`), SWARM_PLAN version 2 redesigns the pilot (two
kinds of claim, mechanical arms, replay scoring, a pre-registered decision rule) and adds a build
before it (its §10). Its gates, cost and order replace §1–§5 below; this report stays as the
record of the state on the morning of 2026-10-19.

## TL;DR

- **Almost.** Four of SWARM_PLAN §9's five gates are met. The fifth is your word on the schedule.
  Two smaller items remain before the first run:
  - **RB**, the claims your NHANES glucose walkthrough shows, which the first gate names;
  - **the live lockdown check**, which needs Claude Code restarted so the analyst agent type
    loads.
- **The driver works end to end.** A scripted analyst ran the Predict fixture, the Estimate
  fixture and the null decoy through the driver to their records. All 18 mechanical checks gave
  the result written down before the runs.
- **Proposed timing:** after your usage resets on Thursday at 1 pm.
  - RB and one small shell fix go first.
  - Then one measured pair of runs (a Haiku and an Opus analyst on one fixture), so the full
    pilot is sized from real numbers.
  - Then the pilot itself, in daytime and unthrottled.
- **Cost so far:**
  - 126.5M fresh subagent tokens since Thursday, which by your 82% figure is about 91% of the
    week;
  - 96.0M of that since Saturday 00:30, against the ~80M reference;
  - about 11M of it was waste: last night's integration under the quiet-hours throttle
    (lesson below).

## 1. The gates (SWARM_PLAN §9)

| Gate | Status | Evidence |
|---|---|---|
| R0 landed, and the walkthrough's claims registered (R1) | **Half.** R0 landed 2026-10-16. RG's gates, R1-F's first batch and RM's referee and audit fixes have landed locally. RB, the walkthrough's claims, has not run. | RB's brief is ready (25 claims at most, refusals and defaults first). It runs after the reset. |
| W2–W4's needs in §8 built | Met | SWARM_ENGINE_PLAN's pieces: GET /fits (it now marks the locked fit), GET /results, GET /mattered (checked against R's `survey`), analyst mode, provenance. |
| The Sullivan 2021 replication done by hand | Met | Table 4 at printed precision with exact n; an R-only reference agrees with the engine to about 1e-11 (`notes/replication/sullivan2021.md`). |
| The stage-3 audit's decoy machinery | Met | `research/registry`'s decoy queue reproduces the backtest's gate result; `research/swarm/decoys.py` makes the pilot's null decoy. |
| Your word on the schedule | **Open** | §4 below. |

DRIVER_PLAN's own stopping point (D6) asks for the dry run and the restriction check:
- **The dry run:** 18 of 18 checks agree with their construction on turbotab-next 8a3e0fff
  (unthrottled):

  | Run | Driver commands | Time |
  |---|---|---|
  | Predict fixture, exploring | 60 | 201 s |
  | Estimate fixture | 52 | 60 s |
  | Null decoy | 55 | 288 s |

  The single-pass check flags the two single-pass runs and passes the exploring one, as built;
  the decoy yields no finding.
- **The restriction check, half done:**
  - The analyst hook is wired locally in `.claude/settings.local.json`, never committed, and the
    driver confirms it.
  - Fed a simulated analyst's calls, it refuses Edit and every command but the driver's, allows
    the driver's `ask`, and leaves every other agent alone.
  - The live check, one analyst agent asked to edit, write, fetch and run outside the driver,
    needs a session that loaded `analyst.md` at its start. A nested session with permissions
    bypassed was refused by Claude Code's safety classifier, rightly.
  - **Your step:** restart Claude Code with `claude --resume`.
  - The same check shows which model `model: haiku` resolves to, Haiku 5.5 or 4.5; this
    environment lists 4.5 as the newest Haiku.

## 2. What the pilot is

SWARM_PLAN §6, unchanged except where noted:
- **Six datasets:**
  - Bang 2009 (Predict, its own 1999–2006 intake);
  - Sullivan 2021 (Estimate);
  - your NHANES fasting glucose (Predict);
  - a Predict fixture (`clinical_readmission_predict`, declared C 0.7419);
  - an Estimate fixture (`survey_support_estimate`, a null effect of age);
  - a null decoy (`outpatient_admission_predict`).

  Each has a manifest (URL, release date, license, SHA-256) and a question card.
- **Analysts:**
  - Haiku 5.5 at high effort, three per dataset: 18 runs;
  - Opus 5.5 at high effort, one per dataset as the ceiling: 6 runs.

  Sonnet runs only if Haiku fails §7, and only after you are told.
- **The referee audit:** Opus referees on a sample of each run's attempts (`research/swarm/referee.py`).
- **Scored on:** §7's six acceptance lines, all checked mechanically from the records:
  - seal integrity;
  - logging;
  - exploration, with the single-pass rate at most 1 in 5;
  - correctness;
  - robustness, with at least 9 runs in 10 reaching the export alone;
  - spend.

## 3. Cost, measured where it can be

Nothing has measured an analyst run yet: the dry run's analyst is a script and spends no tokens.
So the first step after the reset is a **measured pair**: one Haiku and one Opus analyst on the
Predict fixture, run one after the other. Their fresh tokens, fits and wall time size the rest.

**A wide guess until then, to be replaced by that measurement:**
- a Haiku run, about 0.3–1M fresh;
- an Opus run, about 1–3M;
- the referees, about 3–6M in all.

That puts the whole pilot near 13–40M fresh, or roughly 10–30% of a week at this week's rate.
That range is why the measured pair comes first. If the projection threatens the week's usage the
build needs, you are told before the rest runs (§7.6).

**The lesson from last night:**
- **What happened:** under the quiet-hours throttle, test commands took 5–9 minutes. That is past
  a subagent's roughly 5-minute prompt-cache life, so every next turn re-read the agent's whole
  context as fresh input. The integration package cost about four times its normal spend
  (8.9M build, 2.6M repair).
- **For the pilot:** an analyst's fit can run long. Run it in daytime, unthrottled, with each
  analyst's commands kept short. SWARM_PLAN §6's "under the quiet-hours throttle" should be
  reversed. That is your call, since it means fan noise in the daytime.

## 4. Questions for you

1. **The schedule.** Thursday after 1 pm for RB, the shell fix and the measured pair; Friday for
   the pilot proper? Or another day?
2. **Daytime, unthrottled runs.** Three analysts per dataset at once is six workers: a heavy run,
   audible. Is daytime acceptable, and which hours?
3. **The share of a week the pilot may use.** I suggest up to about 40% of a fresh week, with the
   measured pair deciding whether it fits.
4. **The analyst model.** If `model: haiku` turns out to be 4.5, may I set the full Haiku 5.5
   model id in `analyst.md`, or would you rather check it first?

## 5. Set-up before the first run (the orchestrator's list)

- Restart Claude Code (your step), then the live lockdown check and the model id.
- Run RB.
- Fix the real-server journey `quest-nhanes-predict`. On CI's runner it sees the page's quest
  bar show "Who's in" as reopened while `GET /quest` does not, after last night's goal-change
  naming fix. It passes on this machine. The pilot does not use the shell, but CI must be green
  before turbotab-next is pushed. turbotab-next is held locally at 8a3e0fff; the same commit is
  on GitHub as `ci/full-int-3b`.
- **A permission rule** for the pilot session, so an analyst's driver commands are not prompted:
  `Bash(scripts/py -m research.swarm.driver:*)` in `.claude/settings.local.json`, local only.
- **The pilot's launcher:**
  - it writes each known answer as `{feature, family?, measure, value}` or `{null: true}`;
  - `start` names each run itself (`run_NN` plus a secret part), so no label tells an analyst it
    is on a decoy;
  - the fit budget is set from the first runs' measured fit times (§6).

## 6. Known limitations the pilot will meet

These are filed in INBOX. None changes a check's verdict, but the pilot's report will name each
where it shows.

- **The Estimate fixture's known answer is judged by refusal.** With multiple imputation ranked
  first, the effects stage refuses to pool the marginal risks over the imputed copies, so the
  check passes on a refusal, not a number. Pooling marginal risks by Rubin's rules is an engine
  package.
- **Table 2's caption** names the declared marginal measure over rows that hold conditional odds
  ratios (`export/tables.py`).
- **Interval edge cases:**
  - a held-out AUC at perfect separation gets an interval of zero width;
  - the cross-validated AUC interval counts a fold with constant scores as having no variance.
  - A constant score's held-out interval is now refused, which the decoy meets.
- **Deferred:**
  - the export's list of the claims that acted (R1-M) is deferred for a redesign scoped to the
    stage runs the export reports;
  - R1-F's later batches (RF-2, RF-3: the backtest's defects outside teaching and cards) wait.
- **Stale captures:** some frontend captures and mocks still hold sentences changed this week;
  their recapture is a heavy run to schedule with you.
