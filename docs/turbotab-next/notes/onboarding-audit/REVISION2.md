# Onboarding guidance: the second revision

A second, independent revision of the agent onboarding guidance (2026-10-15), made after the first
audit and revision (`AUDIT.md`, commits f891b5be and eebe7f20). Four cold-start probes (a builder, a
verifier, an orchestrator and an auditor) each tried to work from the first revision alone and
recorded what they had to guess, what conflicted, what was missing, what was noise for their role,
and what they would have done wrong. Nolan asked for this pass "just so we are being deliberate and
thorough", with "an agent-friendly structure and guidance provided, with the right strategic
direction and pacing".

Each change below names its source: **P-B**, **P-V**, **P-O** and **P-A** are the builder, verifier,
orchestrator and auditor probes; **A** is a first-audit finding (`AUDIT.md` §2 id); **S** is the
strategy and pacing Nolan and the orchestrator set (now `ORCHESTRATOR.md` §1–§2).

## 1 · The structure

| File | Who reads it | What it holds |
|---|---|---|
| `CLAUDE.md` (root) | every agent, every turn | what to do first, where your role's rules are, what never to do, the trust bar, the environment and test commands, commits |
| `.claude/agents/<role>.md` | the agent of that role | what its prompt must name, its order of work, its checks, its report |
| `docs/turbotab-next/ORCHESTRATOR.md` | the orchestrator, at session start | start of a session, direction, pace, who decides what, how a wave runs, upkeep |
| `docs/turbotab-next/HANDOFF.md` | the orchestrator | the state checked against git, the order of work with each step's purpose, where to wait for Nolan |
| `notes/onboarding-audit/PREAMBLE_TEMPLATE.md` | the orchestrator, writing prompts | the fields that change per task, matched to what each role file requires |
| `scripts/py` | every agent | the main checkout's venv Python, from any worktree |

The first screen of `CLAUDE.md` now answers the three questions a cold agent has. **What first:**
"First" items 1–4: the orchestrator goes to ORCHESTRATOR.md, every other agent to its role file,
whose first lines list what its prompt must name, and an agent missing any of it returns at once
instead of inventing it. **Which file:** item 2. **Never:** the next section, one line per rule.

## 2 · What changed, and why

### Root `CLAUDE.md` (91 → 73 lines)

- **Points to the role files** and says the orchestrator's exceptions win (P-B: "CLAUDE.md never
  mentions .claude/agents/"; P-O: "Never edit or commit in the main checkout" had no exception for
  the one agent that lands there).
- **A missing prompt field means return, not guess** (P-B invented a branch name and owned-files
  list; P-V found only a look-alike branch, `feat/mc2b4-omics-switches`, and a verdict on it would
  have judged a different change; P-A had no list of five citations). The required fields now live in each role file, so an
  auditor is not told to stop for lack of a branch it never has.
- **The premise is a hypothesis** (P-B would have trusted INBOX:396's diagnosis of the wrong
  threshold; P-V would have used RECIPES F4's stale "what the code does today"; P-A's task paired
  `P_GLU` with the wrong cycle). One rule for all roles; each role file says what to do with it.
- **Scope in one line** (S: "fix the defect, file the feature"; P-B would have fixed the adjacent
  half of INBOX:396).
- **One definition of a heavy run:** over about ten minutes, more than two workers, a sweep, T1,
  T12, a recapture, or a whole suite, the integrator's shards excepted (P-V: "a few minutes of full
  CPU" and "under about ten minutes" let a six-minute sweep be both allowed and forbidden; P-O:
  CLAUDE.md forbade the fast tier the integrator must run). `--timeout` is now 600, inside the
  ten-minute cap, not 900 (P-B).
- **A named legacy or acceptance file may run** when it reaches the code; only whole suites are
  out (P-B: the card's only tests are legacy files, which "never run the legacy suites" seemed to
  forbid).
- **A test named on purpose drops `-m "not slow"`, and its count is checked** (P-V: a slow-marked
  new test is silently deselected and the run reads green).
- **Finding the tests that reach your code** has a command, verified: `git grep -l <name> --
  'turbotab/*test_*.py'` catches core, server, acceptance and legacy tests (P-B, P-V: "no map from
  code to tests"). The first draft of this revision used `turbotab/**/test_*.py`, which misses the
  legacy files; caught by running it.
- **The trust bar as four imperatives** (S): test first from known truth; guards both ways; a
  number right or refused, a caveat being no fix; sentences generated from the value the code
  uses.
- **Never create a tag** (P-O: memory says to tag `v1.0-paper`, and `release.yml` publishes on any
  `v*` tag).
- **Installs:** a browser and R packages outside a scratch library are named (the frontend file
  said `npx playwright install chromium`; the verifier needs R references).
- **A scratch worktree under your scratchpad is yours** (P-V: no way to run a test on the base
  without touching the builder's tree; see `verifier.md`).
- **Attribution** comes from the agent's harness reminder, and only failing that from the prompt
  (orchestrator's instruction; A C6). No model name is written anywhere.
- **Moved out, because not every agent can break them:** "Where truth lives" (P-B, P-A: noise;
  it sent P-B to HANDOFF for a priority that HANDOFF does not list) to ORCHESTRATOR.md; the
  generated-file commands to `builder.md` and `integrator.md`; the generic report shape, which
  matched no role but the builder (P-V, P-A), to the role files; "match the idiom" and the slow
  marker to `builder.md`; Rscript to `verifier.md`.
- **`TURBOTAB_PYTHON` is no longer a step.** The hook already finds the main checkout's venv from
  a linked worktree (`.githooks/lib.sh` `resolve_python`, run here with the variable unset); the
  variable is needed only if the hook prints GATES CANNOT RUN.

### `scripts/py` (new)

Four lines: `exec` the venv Python of the first entry of `git worktree list`. It replaces the
absolute home path in every command (the public-repo rule), works from the main checkout and from
`.worktrees/` and `.claude/worktrees/` alike, where no one relative path does, and fails loudly if
run from anywhere but a tree's root, which is also where it must run so that `turbotab` imports
from the right tree (P-V's wrong-tree trap).

### `.claude/agents/`

- **Each begins with what its prompt must name** (all four probes).
- **No model policy** in the definitions (P-B and P-V were Sonnet agents told "Opus for guards"
  with nothing they could do about it). The tier rule is the orchestrator's, in ORCHESTRATOR.md §4.
- **`builder.md`:** a premise step; existing tests that pin the behavior are found and run first
  (P-B would have deleted the two-column rule and reopened the one-clinical-column hole that
  `test_omics.py` guards); "you own the files and their tests" replaces "plus new test files"
  (P-B: the natural home for the test was the existing file); update every place that states a
  changed rule, docstrings and generated sentences included (P-B; S: text written apart from
  behavior); the stored-reference pin by node id, with the re-pin rule (S: switch off the one
  change and show every other number holds); `--check` is the builder's and regeneration the
  integrator's (P-B); the narrower-edge stop (S, B-281).
- **`verifier.md`:** a setup check before judging (P-V; A F4: eight verifiers once reviewed an
  empty diff, and the check lived only in ORCHESTRATOR.md); read-only in the builder's tree; the
  base-commit run through a detached scratch worktree, verified end to end here; regressions by
  name and the pin; defined severities, with notes kept out of the problem list so a note does not
  trigger a repair (P-V); "there is no approve, but" (S); design reviews also ask whether the
  screen hands the researcher's question back (S, design from meaning).
- **`integrator.md`:** the regeneration commands, corrected: `turbotab.core.reference.methods`
  writes with no flag and has no `--write`, which the first draft of this revision had wrong; the
  four shards one at a time; the pin by node id.
- **`auditor.md`:** web etiquette, one request a second, raw responses saved (P-A); a repo tool
  that writes a tracked file runs on a copy (P-A: `citations_check` rewrites `citations.json`, line
  420); a stated rule for any sample; documented exceptions are not new findings; a severity for
  bibliographic errors. **The `tools:` line is removed:** it omitted WebFetch for a role that
  fetches (P-A), and a tool allowlist may also withhold tools the harness relies on (StructuredOutput
  for a schema, ToolSearch for deferred tools). Read-only is a rule in the text, as it was in effect
  before, since Bash can write.

### `ORCHESTRATOR.md`

P-O's verdict was that the guidance "carries mechanics well … but only thinly carries the
project's strategic direction and pace", which lived in memory only.
- **§0 Start of a session** (P-O: no protocol for a bare "continue"): check HANDOFF against git,
  reply briefly, continue.
- **§1 Direction:** understanding not AutoML, art into science, design from meaning, calm over
  complete, why TurboTab wins, provenance, and the trust bar with the backtest's four failure kinds
  and "abundance in the corpus, authority in the registry" (S).
- **§2 Pace:** walkable sooner and pull not push; velocity over ceremony; usage is finite, with the
  backtest's cost as the scale to warn at; **default caps of two repair rounds and two
  critique-and-revise rounds** (P-O: "no round cap is given"; S: rounds capped); size work to the
  evidence; one definition of heavy runs and the quiet-hours throttle described without a session
  path; **the autonomy rule's boundary** (P-O: "launch the next phase without waiting" against
  usage pauses).
- **§3 Who decides what:** what Nolan rules on now includes what is published; how to show him
  work (P-O: "where Nolan receives reviews"); the methods and DoD delegations; his standing rules.
- **§4 How a wave runs:** `agentType` and the workflow-authoring skill named (P-O); the model tier
  moved here; landing in the main checkout stated as the orchestrator's exception; never a `v*`
  tag; "approve, but" with a silent wrong number is a blocker (S).
- **§5 Upkeep:** worktree cleanup (P-O: about 110 stale worktrees); memory keeps feedback that
  shapes the orchestrator's behavior.

### `HANDOFF.md`

- **The state is checked against git** (P-O, P-V: it said `turbotab-next` was the stage-1 merge;
  it is e6e58809, feat/s1-int is not on it, and fix/s1-inference is not in feat/s1-int).
- **The B-281 count is six rounds,** as the commits and ORCHESTRATOR.md say, not four (P-O).
- **The main checkout's untracked `notes/replication/`** is named, with the one file that differs
  (P-O: the merge would be refused).
- **The goal and each step's purpose are stated, and the waits for Nolan are marked [Nolan]**
  (P-O; S: walkable sooner, stopping rules set up front). Step 1 carries B-281's stop: a defect in
  the refusal is fixed, a new edge of partial reporting goes to INBOX.
- **Step 2 carries the first audit's orchestrator to-dos** (P-O: they were only in AUDIT.md §6),
  plus one check this revision adds: whether an agent in a nested worktree sees CLAUDE.md twice.
- **The calibration intercept is back among the leftovers as a methods ruling** (P-O: it was under
  Open for Nolan, against ORCHESTRATOR.md's "never hand a methods question back"; in the history
  it was a W1 part 1 leftover).
- **The public question states both options** (task instruction).
- The T1 rerun's 2026-10-11 approval is marked as still needing a time (P-O).

### Elsewhere

- `PREAMBLE_TEMPLATE.md`: BRANCH, PREMISE (with how its number was measured: P-B could not
  reproduce "11 of 600"), BUILDER'S REPORT, WEB and a conditional ATTRIBUTION; the integrator's
  fields; a "who needs it" column matched to the role files.
- `turbotab/frontend/CLAUDE.md`: `scripts/py`; no browser install.
- `V2_DEFINITION_OF_DONE.md` §6: one line saying the road is the plan of 2026-10-07 and HANDOFF holds
  the current order (P-O: two roadmaps, the older not marked).
- `HANDOFF_HISTORY.md` and ORCHESTRATOR.md agree that history is newest first.
- Personal detail: "beside his bed" is gone from CLAUDE.md, ORCHESTRATOR.md, BLUEPRINT §8 and two
  lines of MODEL_FAMILY_CONTRACT; the guidance says heavy runs are scheduled with the owner.
  AUDIT.md's three mentions are reworded and it points here.

## 3 · Rejected or reversed from the first revision

1. Absolute home paths in commands (replaced by `scripts/py`).
2. `TURBOTAB_PYTHON` as a required commit step (the hook does not need it).
3. "Where truth lives" in the per-turn file (moved to the orchestrator's files and the prompt's READ).
4. One report shape in CLAUDE.md (each role's own, in its file).
5. The model policy in agent descriptions (the agent cannot act on it).
6. "Plus new test files" (a test belongs beside the file it tests).
7. The auditor's `tools:` allowlist (see above).
8. HANDOFF's statement that stage 1 had merged, and its count of four B-281 rounds.
9. The calibration intercept under Open for Nolan.
10. Two thresholds for a heavy run, and a per-test timeout above the run cap.
11. The always-on attribution field with a model placeholder in the template.

Kept as the first revision made them: the split into root, role files, orchestrator's guide and a
current-state handoff with its history; the verifier contract; the frontend file's move to
FOUNDATION; the fast-check corrections in BLUEPRINT §8 and `turbotab/README.md`; the fourteen
repointed citations; the email rule; the decoy rule kept out of agent-facing files.

## 4 · Context cost

The root `CLAUDE.md` is 73 lines and about 820 words, against 91 lines and 832 words. The words
fell less than the lines because four rules every probe needed were added (go to your role file,
return when a field is missing, the premise is a hypothesis, the scope line), and the two
exceptions that resolved conflicts (the orchestrator's, the scratch worktree). Everything that one
role alone can break moved to that role's file, which only that role loads.

## 5 · Verification

Run in this worktree with `OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2`, about a minute in all; nothing
heavy, nothing installed, nothing left running.
- `scripts/py -c "import sys, pytest, xdist, pytest_timeout, turbotab; …"`: `sys.prefix` is the
  main checkout's venv, and `turbotab` imports from this worktree.
- The targeted command exactly as CLAUDE.md gives it, on `turbotab/core/tests/test_reference.py`:
  70 passed in 2 s.
- `scripts/py -m turbotab.core.reference.methods --check`: current. `scripts/py -m
  turbotab.core.reference.packet --all --reuse --check`: exit 0.
- `scripts/py -m turbotab.server.openapi --write`, then `npm run gen:api` with `node_modules`
  symlinked from the main checkout: neither file changed; the link was removed.
- The stored-reference pin's node id: `--collect-only` collects 1 test.
- `git grep -l _classify -- 'turbotab/*test_*.py'`: `test_omics.py` and two legacy files; the
  `turbotab/**/test_*.py` form finds only the first.
- The verifier's base-commit procedure, end to end: `git worktree add --detach` in the scratchpad,
  a test copied in and run there (70 passed, `turbotab` imported from the scratch tree),
  `git worktree remove --force`, and the worktree list shows it gone.
- `git merge-base --is-ancestor 54fe6a95 HEAD`: holds.
- The hook's `resolve_python` with `TURBOTAB_PYTHON` unset, in this linked worktree: the main
  checkout's `venv/bin/python`.
- `tests/test_american_spelling.py`: 3 passed. The pre-commit hook ran on each commit.
- Not run: `npm run check` (no frontend code changed), the check-runs API (no web requests in this
  task), the fast-tier shards (integrator's, about fifteen minutes).

## 6 · Questions

For Nolan:
1. **May `CLAUDE.md` and `.claude/agents/` be public?** A: merge them as they are, versioned and
   checked with the code; they hold rules and repo paths, no personal detail. B: keep them
   untracked in the main checkout, where the orchestrator and its workflow agents still read them;
   they are then unversioned and absent from a fresh clone, and CLAUDE.md's pointer to the role
   files must name the main checkout's copy. `turbotab/frontend/CLAUDE.md` has been public since
   2026-10-05 either way.
2. **Where should the orchestrator stop and wait?** The default now: a short status after each
   step of the order, waiting for you only at the public question, the walkthrough and each heavy
   run, and a warning before any step the size of the backtest. Does your 2026-10-11 "pause after
   wave 1 part 2" still stand beyond that?
3. **Is the `v1.0-paper` tag still wanted?** (From the first audit.)
4. **"No proprietary tools":** the product only, or also agents' build tools? (From the first audit.)
5. **About 27 older committed documents** (audit notes, research notes, prompts) contain absolute
   home paths. Scrub them, or leave the record as written?

For the orchestrator:
1. Before merging fix/s1-inference, confirm from the stage-1 workflow's journal that a fresh
   reviewer approved B-281's fallback round.
2. Run step 2's one-call check: `agentType` loads the role file; the agent sees one copy of
   CLAUDE.md, not its worktree's and the main checkout's; and whether its system reminder carries
   attribution lines.
3. Keep or change the default caps (two repair rounds, two critique rounds) per plan.
4. Apply `MEMORY_RECOMMENDATIONS.md` only after the merge, with this revision's amendments at its
   top: the email memory shrinks instead of going; the throttle's full session path is kept; the
   2026-10-11 pause is marked lifted.
5. The throttle script lives in one session's directory. Keep it there with its full path in
   memory, or commit it (it holds nothing personal)?
6. Which copy of `notes/replication/estimate_candidates.md` is newer, the main checkout's or
   feat/s1-int's?
7. If a tracked `scripts/py` is unwelcome in a docs branch, put the interpreter's path in each
   prompt instead and revert CLAUDE.md's commands.

## 7 · Doubts

1. `scripts/py` assumes the first entry of `git worktree list` is the checkout with the venv and
   that git can read its metadata in every agent's sandbox. Both hold here; an agent on a base
   that predates this branch lacks the script until step 1 of `builder.md` moves it.
2. Whether a nested worktree's agent loads CLAUDE.md twice. The workflow reference says subagents
   get the orchestrator's CLAUDE.md files; it does not say whether the worktree's own copy is also
   loaded. Step 2 checks it.
3. The round caps, the checkpoint rule and the autonomy boundary are inferred from the strategy and
   the incidents, not ruled by Nolan; and removing the auditor's `tools:` line rests on how the
   harness treats an allowlist, which I could not test from here.
