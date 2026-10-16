# Onboarding audit: the base instructions an agent gets when it joins

A second, independent revision followed (`REVISION2.md`). Where the two differ about the files as
they now stand, `REVISION2.md` is current; this audit's findings and evidence stand as recorded.

An independent audit (Fable 5.1, 2026-10-15) of what a Claude agent is told when it first joins
TurboTab, as the orchestrator or as an executor, and the revision that followed on
`docs/agent-onboarding`. Nolan asked for appropriate guidance with clarity, consistency and
conciseness. Evidence is cited as `file:line`, or as a file in the read-only bundle
(`scratchpad/onboarding_audit/`): `preambles/`, the 21 shared "COMMON" blocks the orchestrator
hand-wrote into its workflow scripts; `memory_copy/`, the orchestrator's memory; and
`orchestrator_held_rules.md`. Nolan's rulings are kept as ruled; §4 lists the places where two
rulings seemed to meet.

## 1 · Inventory: where instructions came from, by role

| Role | Source | How it reaches the agent | Size |
|---|---|---|---|
| Orchestrator | `MEMORY.md` index + 19 memory files | loaded into every orchestrator session, no one else | ~200 KB; `turbotab-next-progress.md` alone 118 KB |
| Orchestrator | `docs/turbotab-next/HANDOFF.md` | read at session start | 434 lines, of which about 30 were the current state |
| Orchestrator | BLUEPRINT §0, §8, §10; the DoD; `plans/*.md` §7 rulings | read when planning | — |
| Orchestrator | the workflow-authoring skill | loaded when writing a script | — |
| Every executor | the script's COMMON preamble plus a package brief | pasted into the prompt, rewritten per wave | 21 preambles, 52.5 KB, about 13K tokens in all; 1–4.8 KB each |
| Every executor | `turbotab/frontend/CLAUDE.md` | injected automatically only when a file under `turbotab/frontend/` is edited | 100 lines |
| Every executor | **no root CLAUDE.md** | — | the `.gitignore` ignored `CLAUDE.md` and `.claude/` (`.gitignore:38`, `:74`) |
| Builder, repairer | the plan row, the "Shared files" list, the §7 rulings, the specs the prompt names | prompt | — |
| Verifier, reviewer | the builder's report, the journal, FOUNDATION, DRIVE_RUBRIC | prompt | — |
| Integrator | merge order, the verifiers' problems, "Landing protocol" (HANDOFF 2026-10-10) | prompt; the protocol lived only in HANDOFF and memory | — |
| Auditor | one preamble (`preambles/turbotab-ground-up-audit-*`) and the backtest prompts (`notes/claims-backtest/scripts/prompt_*.md`) | prompt | — |
| Human contributor | README, CONTRIBUTING, `turbotab/README.md` | the repo | Classic-oriented |

Facts the orchestrator held that no repo file recorded (`orchestrator_held_rules.md`): the email
rule, the model floor, the decoy rule, the check-runs API, the venv facts, the node_modules
symlink, the worktree-from-main fact, the report of three doubtful claims.

## 2 · Findings

### Consistency

- **C1 · Which interpreter.** Early prompts named `.venv`; the backtest's basis-audit prompt of
  2026-10-14 still does (`notes/claims-backtest/scripts/prompt_basis.md:42`: `.venv/bin/python -m
  pytest`). `.venv` has no pytest (verified: `ModuleNotFoundError`). CONTRIBUTING.md describes uv
  building `.venv/` while the Makefile hard-codes `venv/`. The hook resolves `TURBOTAB_PYTHON`,
  then `./venv`, then `turbotab/.venv`, then the main worktree's (`.githooks/lib.sh:19-40`); a
  linked worktree has none of its own. **Fix:** the root CLAUDE.md names the one interpreter and
  says why, once.
- **C2 · Which test command.** Four forms across the preambles: `-n 4 --timeout=900` (resume-fixes),
  `-n 2 --timeout=900 | tail -15` (every wave), `-m "not slow" -q -n 2 -p no:cacheprovider`
  (s1-fixes), and CI's `-m "not slow" -q -n 2 -rfE` (`.github/workflows/v2.yml:71`). **Fix:** one
  verified line in CLAUDE.md, the union of CI's selection and the preambles' guards.
- **C3 · How long a local run may be.** "~5 minutes" (ground-up audit), "a few minutes" (C6a phase
  2), "about 10 minutes" (s1-fixes, held rules); BLUEPRINT §8 and `turbotab/README.md:21` called
  the whole core and server suite a "fast check" under a minute, when CI's full tier takes 50 to 58
  minutes (`v2-full.yml:33`). **Fix:** ten minutes for any one run; heavy means more than a few
  minutes at full CPU; both stale sentences corrected.
- **C4 · Classic's tests.** `preambles/turbotab-resume-fixes-*`: "Classic (ml/, pages/, utils/,
  tests/) is out of scope: do not modify it." Nolan, 2026-10-08: "Classic tests may be edited, but
  not Classic app code" (HANDOFF_HISTORY, 2026-10-10, "Rulings since 2026-10-08"). **Fix:** the
  ruling, in CLAUDE.md.
- **C5 · pip install.** Seven preambles and the held rules forbid it; `plans/WAVE_C6A_PLAN.md:375`
  says "Builders run `pip install -c turbotab/server/constraints.txt`". The venv is still off the
  pins (verified: scikit-learn 1.9.0 vs 1.9.1, xgboost 3.3.0 vs 3.4.1, shap 0.52.0 vs 0.53.0,
  numpy 2.4.6 vs 2.5.3, fastapi 0.140.13 vs 0.143.0, pydantic 2.13.4 vs 2.14.0). **Fix:** the rule
  and the drift are stated in CLAUDE.md; the plan line should be struck and the venv aligned by the
  orchestrator at a scheduled moment (§6).
- **C6 · Attribution.** Every preamble ends with `Co-Authored-By: Claude Opus 5.5`, including the
  prompts that run Sonnet builders and Fable auditors; the living-methods preamble carries a
  different session id from the same script set. The harness also hands each agent its own
  attribution lines in a system reminder. **Fix:** the template carries the two lines as a field
  with the model name as a placeholder, and §6 asks the orchestrator whether the field is needed
  at all.
- **C7 · Push authority.** "never push" (s1), "don't push at all" (groundwork), "pushing
  ci/<name> is allowed" (waves), "pushing ci/wave-e1p-int is allowed" (repair). Not a
  contradiction, a per-wave grant. **Fix:** CLAUDE.md: never push unless the prompt names the
  branch; the integrator definition names `ci/<name>`.

### Clarity

- **L1 · The base branch.** The harness creates worktrees from `main`, about 593 commits behind.
  Nine preambles say so in nine wordings, inside a longer bullet ("Your worktree may start from
  main: FIRST run `git checkout -B <branch> turbotab-next`"). **Fix:** CLAUDE.md states it; the
  builder definition makes the check step 1 and gives the command.
- **L2 · HANDOFF.md mixed state, history and rules**, newest block on top with dated blocks
  below, and said "Two rulings are pending" (2026-10-10 block) after FOUNDATION §0 had superseded
  them (`plans/WALKABLE_PLAN.md:336` asked HANDOFF to say so). **Fix:** §5.
- **L3 · CONTRIBUTING.md** describes Classic's setup (uv, `.venv`, Streamlit tiers,
  `OMP_NUM_THREADS=1`); a v2 agent reading it on day one gets the wrong environment. **Fix:**
  CLAUDE.md states the v2 environment; CONTRIBUTING is left for human Classic contributors.
- **L4 · Plans in session scratch.** `preambles/turbotab-w1-walk-it-part1-*`: "THE PLAN:
  /private/tmp/.../scratchpad/plans/WALKABLE_PLAN.md (if absent, your brief below is complete)".
  The plan was later committed to `docs/turbotab-next/plans/`. **Fix:** the template points at
  repo paths.
- **L5 · Binding rules restated differently each wave.** "No estimate before the lock" appears in
  eight preambles in five wordings (before the plan lock; before the lock or Fit; under Estimate and
  Describe no estimate stage before the track's plan lock; nothing served before Fit under Predict;
  no estimate and no score before Fit). **Fix:** the frontend CLAUDE.md points at FOUNDATION §5
  rule 6; the template says to name a rule, not paraphrase it.

### Conciseness

- **N1 · Restatements.** Over the 21 preambles: the attribution lines 19 times, the commit-message
  style 19, the interpreter path 17, the hook and American spelling 17, "never bypass" 16, "never
  push" 14, the pkill rule 14, `-n 2` targeted tests 14, "Classic must not change" 13, the
  independent-reference rule 11, "do not run the full suite" 11, "tests first" 8, the node_modules
  symlink 8, the worktree-from-main fix 8, "no estimate before the lock" 8, the heavy-run rule 8, the pip
  rule 7, `--reuse` 6, context discipline 6. About 13K tokens of preamble were written by hand, and
  most of each one restates the previous one with drift.
- **N2 · HANDOFF.md** was 434 lines and 30 KB; the current state was about 30 lines.
- **N3 · The orchestrator's memory** is about 200 KB per session; the progress log is 118 KB of
  entries whose facts are in `git log` and the plans (`MEMORY_RECOMMENDATIONS.md`).
- **N4 · Context cost of the revision.** The root CLAUDE.md is 91 lines, 832 words (about 1,100
  tokens), paid once per agent per turn. Each agent definition is 24–29 lines and reaches only the
  agents of that role. ORCHESTRATOR.md (98 lines) is read, not injected.

### Fitness

- **F1 · The frontend CLAUDE.md contradicted Nolan's calm ruling.** It prescribed three voices
  (serif, sans, mono chips) and "every hue is a claim" from DESIGN_LANGUAGE §02–§03; Nolan, 2026-10-05,
  found every design too busy and FOUNDATION §2 rules one type family and plain text, no chips.
  `plans/WALKABLE_PLAN.md:186` had flagged the conflict and left it to W1-E. **Fix:** the file now
  points at FOUNDATION as binding and says the M1–M3 components keep their style until rebuilt
  (match them when fixing; build new screens to FOUNDATION). `reference/README.md`'s row for
  DESIGN_LANGUAGE now says §02–§04 are amended.
- **F2 · A root CLAUDE.md could not be committed.** `.gitignore:74` ignored `CLAUDE.md` ("not for
  public repo", from the Classic era) and `:38` ignored `.claude/`. The frontend one is tracked
  despite the rule. **Fix:** `CLAUDE.md` is no longer ignored; `.claude/*` is, except
  `.claude/agents/`. Whether these files may be public is a question for Nolan (§6).
- **F3 · Rules in every agent's context that one role needs.** The model floor, the decoy rule,
  the check-runs API, resume mechanics, the quiet-hours throttle: orchestrator or integrator
  matters. **Fix:** ORCHESTRATOR.md and `integrator.md`; none in CLAUDE.md. The sealed decoys'
  location appears in no agent-facing file.
- **F4 · The verifier contract.** Verifiers returned "confirmed" beside real defects and the repair
  did not run (memory, 2026-10-09; `orchestrator_held_rules.md`). The rule existed only in memory.
  **Fix:** `verifier.md` says a verdict beside a listed defect is a contradiction and the repair
  runs on any non-empty problem list; ORCHESTRATOR.md §3 item 4.
- **F5 · Severity scales** existed only in the ground-up audit preamble. **Fix:** `auditor.md`
  (critical, major, minor) and `verifier.md` (blocker, major, minor, note).

### Gaps (needed repeatedly, told ad hoc or never)

- **G1** `node_modules` is absent in worktrees (8 preambles; a reviewer found it the hard way,
  `preambles/s1-fixups-*`). Now CLAUDE.md.
- **G2** `TURBOTAB_PYTHON` for the hook in a worktree (2 preambles, both from 2026-10-15). Now CLAUDE.md.
- **G3** The email rule (no repo file; an agent sent it to Unpaywall on 2026-10-15). Now CLAUDE.md.
- **G4** "Report your three most doubtful claims" (HANDOFF and the registry plan only; 8 of 9
  suspect picks in the backtest were defects). Now CLAUDE.md's Report and every agent definition.
- **G5** "A change to a family's defaults runs every reproducibility pin" (HANDOFF only). Now CLAUDE.md.
- **G6** The venv is off CI's pins (nowhere). Now CLAUDE.md.
- **G7** Guards are tested both ways; a number is right or refused (held rules only). Now CLAUDE.md.
- **G8** Where the current state is, where rules are, where rulings are, for a new orchestrator
  session (memory only). Now CLAUDE.md "Where truth lives" and ORCHESTRATOR.md §4.

### Stale

- **S1** BLUEPRINT §8's "both should stay under ~1 minute" and `turbotab/README.md`'s fast-check
  sentence (C3). Corrected.
- **S2** HANDOFF's pending rulings (L2). Corrected.
- **S3** `turbotab/frontend/CLAUDE.md`'s layout lacked `src/components/views` and `src/explore/`
  (`INBOX.md:153` had noted the gap). Added.
- **S4** `plans/WAVE_C6A_PLAN.md:375` tells builders to pip install (C5). Left for the
  orchestrator: plans are the orchestrator's record.
- **S5** Memory: the heavy-runs memory cites `AGENT_ONBOARD.md §06` (archived);
  `turbotab-rebuild-decisions.md` says the legacy `api.py` "keep[s] running" (retired 2026-10-05)
  and "all 367 noticings ship" (re-ruled 2026-10-09); `ajcn-paper-sprint-state.md` is from August
  and its `v1.0-paper` tag does not exist; the `MEMORY.md` line for the progress log describes an
  entry of 2026-10-03. All in `MEMORY_RECOMMENDATIONS.md`.

## 3 · The revision, and why each piece

**Root `CLAUDE.md`** (91 lines). Every line is paid for by every agent on every turn, so it holds
only what every agent can break: what the project is (two sentences); where truth lives (so an
agent reads the ruling, not a paraphrase); the environment (C1, G1, G2, G6); the test rules (C2, C3,
G5, G7); the generated files and their commands (6 preambles); the machine and the person (the
heavy runs, pkill, the email, no proprietary tools); branches and commits (L1, C4, C7, the hook); and
the report shape (G4). Nothing in it is a ruling about the product; those are pointed at.

**`.claude/agents/{builder,verifier,integrator,auditor}.md`.** The workflow API's `agent()` takes
`agentType`, resolved from the same registry as the Agent tool, and the harness injects CLAUDE.md
into subagents (the workflow-authoring reference). So the role rules that 21 preambles restated
(tests first and failing on base; the verifier's problem list and severities; the integrator's
regeneration, fast-tier run and CI read; the auditor's read-only evidence rules) can be stated
once each, versioned and spell-checked. The call still sets the model, so the Opus/Sonnet policy
stays the orchestrator's. **Not yet verified in a live workflow:** no workflow was run for this
audit. The orchestrator should confirm with one Agent call naming `agentType: 'builder'` and asking
it to quote its step 1 (§6). Until then, the same text can be pasted as a preamble section.

**`docs/turbotab-next/ORCHESTRATOR.md`** (98 lines): what Nolan rules on and what the orchestrator
owns; his standing rules with dates and his words; how a wave runs; upkeep of HANDOFF, INBOX and
memory. Everything in it was previously in memory, in HANDOFF's "Landing protocol" and "Subagent
policy", or in no file.

**`turbotab/frontend/CLAUDE.md`**: F1 and S3; the American-spelling and legacy-suite rules were
removed because the root carries them; the openapi command uses the one interpreter.

**`PREAMBLE_TEMPLATE.md`**: what still changes per wave and so still belongs in the prompt.

**Also corrected**, because an agent reads them on day one: BLUEPRINT §8's fast-check bullet,
`turbotab/README.md`'s fast-check sentence, `reference/README.md`'s DESIGN_LANGUAGE row.

**Not done:** CONTRIBUTING.md (public, Classic's); the plans (the orchestrator's record); any
design or methods document beyond a citation.

## 4 · Nolan's rulings: kept as ruled; where two seemed to meet

- 2026-09-27 "workflows are fine any time for now until I say otherwise" and the quiet-hours
  requests of 2026-10-04 and 2026-10-05: not a conflict; the first anticipates the second.
  ORCHESTRATOR.md §2 states both and the throttled answer.
- 2026-10-06 "decide or dismiss each before the lock" and 2026-10-09 the triage sweep: re-ruled;
  the DoD records both. Not restated here.
- 2026-10-07 "three scope-cut groups stay in v2" and 2026-10-08 C6c displaced to v2.x: his
  displacement rule, recorded in the DoD.
- 2026-10-08 "all 367 noticings ship" and 2026-10-09 the reference-journey rule: re-ruled in the DoD.
- The resume-fixes preamble's "Classic tests out of scope" against his 2026-10-08 ruling (C4) is
  orchestrator drift, not two rulings.
- "No proprietary tools" (2026-10-10, recorded only in the progress memory beside the conformal
  proposer) is now in CLAUDE.md as a product rule; its exact scope is asked in §6.

## 5 · HANDOFF.md: the structure applied

`HANDOFF.md` now holds the current state only: State, Landed since, Known red, Next in order,
Leftovers that still hold, Open for Nolan, Nolan's context, Heavy runs to schedule. Each new state
replaces the body; the replaced block is appended, dated and verbatim, to `HANDOFF_HISTORY.md`.

Every piece of content survives. Lines 35–434 of the old file are in `HANDOFF_HISTORY.md` verbatim
(checked line by line by the split script; the saved original top block is in the bundle), under
three added dated headings; the 2026-10-11 leftovers and Nolan's context, which the top block said
"still hold", are in the current file; "Process rules learned on 2026-10-15" are now rules in
CLAUDE.md (the reproducibility pins) and ORCHESTRATOR.md §3 (fix the defect, file the feature; the
pins); "Landing protocol" and "Subagent policy" are ORCHESTRATOR.md §2 and §3. The pending rulings
are marked superseded under Open for Nolan, as WALKABLE_PLAN §6 asked.

Fourteen citations that named a historical HANDOFF section were repointed to `HANDOFF_HISTORY.md`:
`V2_DEFINITION_OF_DONE.md:28`, `calm/FOUNDATION.md:36` and `:605`, `RECIPES_AND_TUNING.md:39`,
`:1111` and `:1682`, `crosswalk/CROSSWALK.md:3`, `:122`, `:175`, `:1467` and `:2041`,
`crosswalk/SIZING.md:180`, `M1_CONTRACT.md:198`, and the docstring of
`turbotab/core/tests/test_stability_selection.py:1`.
`CROSSWALK.md:68` ("HANDOFF.md records no ruling") and `WALKABLE_PLAN.md:336` (the instruction now
carried out) are unchanged. The audit of 2026-10-09 is a record and was not edited.

## 6 · Questions

For Nolan:
1. May a root `CLAUDE.md` and `.claude/agents/` be public? The `.gitignore` said "not for public
   repo" in 2025; `turbotab/frontend/CLAUDE.md` has been public since 2026-10-05, and
   `release.yml` would ship both in the Classic release zip.
2. "No proprietary tools" (2026-10-10): does it cover dependencies and services in the product
   only, or also the tools agents use while building?
3. Is the `v1.0-paper` tag on `main` still wanted for the AJCN availability statement?

For the orchestrator:
1. Confirm that a workflow `agent()` call with `agentType: 'builder'` loads `.claude/agents/builder.md`
   from the main checkout once this branch lands, and that the call's `model` wins over the
   definition (none sets a model). If it does not, paste the definitions as preamble sections.
2. Confirm whether workflow agents receive the harness's attribution reminder; if they do, drop
   the attribution field from the template and let the harness supply it.
3. Strike `plans/WAVE_C6A_PLAN.md:375` and align the venv with `constraints.txt` at a scheduled
   moment, then delete the drift sentence from CLAUDE.md.
4. Apply `MEMORY_RECOMMENDATIONS.md`.
5. W1-E owns `turbotab/frontend/CLAUDE.md`; merge this branch before W1-E starts, or brief W1-E's
   builder on the new file.

## 7 · Verification

Run in the worktree on 2026-10-15 with `OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2`, about 15 s in all:
- `python -m turbotab.core.reference.methods --check`: "METHODS_REFERENCE.md is current".
- `python -m turbotab.core.reference.packet --all --reuse --check`: silent, exit 0.
- `python -m turbotab.server.openapi --write`: no diff against the committed `openapi.json`.
- `python -m pytest tests/test_american_spelling.py -q -p no:cacheprovider` (all three tests, the
  enumeration control included): 3 passed, with the new untracked files in its population.
- `python -m pytest turbotab/core/tests/test_reference.py -m "not slow" -q -n 2 -p no:cacheprovider --timeout=900`:
  70 passed in 2 s, the exact form CLAUDE.md gives.
- `npm run check` was not run: no frontend code changed, only `turbotab/frontend/CLAUDE.md`.
- The pre-commit hook ran on every commit with `TURBOTAB_PYTHON` set; no `--no-verify`.
