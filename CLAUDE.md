# TurboTab: rules for every agent

Claude agents build TurboTab v2 (`turbotab/`) under one orchestrating session. TurboTab teaches a
researcher to own each modeling decision; it is not AutoML. This file reaches every agent on every
turn, so it holds only the rules any agent can break.

## First
1. The orchestrating session, the one the owner talks to, reads `docs/turbotab-next/ORCHESTRATOR.md`;
   its exceptions to the rules below (landing in the main checkout, pushing `turbotab-next`) win.
2. Every other agent reads its role's file, `.claude/agents/<role>.md` (builder, verifier,
   integrator or auditor; with no role named, your prompt and this file are your rules). Its first
   lines list what your prompt must name. If any is missing, do not invent it (a branch name, a
   worktree, a scope): return at once and say what is missing.
3. The task's premise (an INBOX line, a fix list, a spec's account of today's code) is a
   hypothesis. Check it in the code first. Where it is wrong, the fix follows the code, not the
   diagnosis; a task whose premise fails may not need doing, so say so before you build.
4. Do the task and stop at the stopping rule. A feature or a neighboring defect goes in your
   report, not into the change.

## Never
- Install or upgrade anything in the shared environment (`pip`, `npm install`, a browser, an R
  package outside a scratch library). If something is missing, report it.
- Start a heavy run: a command over about ten minutes, more than two workers, a sweep, T1, T12, a
  journey recapture, or a whole suite (core, server, acceptance, the legacy `tests/` and
  `turbotab/test_*.py`; the integrator's fast-tier shards excepted). Heavy runs are scheduled
  with the owner, so report the need instead.
- `pkill`, `killall` or any pattern kill. Kill only the PIDs you started (`$!`).
- Put the owner's email address, or any personal identifier, in a request to any service: send no
  email parameter, or `research@example.org`.
- Edit, commit, check out, stash or reset outside the worktree and branch your prompt names,
  including the main checkout and every other agent's worktree. A detached scratch worktree you
  add under your scratchpad, and remove, is yours.
- Push, or create a tag, unless your prompt names the exact branch. A `v*` tag publishes a release.
- `git commit --no-verify`.
- Hand-edit a generated file: `turbotab/server/openapi.json`, `turbotab/frontend/src/api/generated.ts`,
  `docs/turbotab-next/reference/METHODS_REFERENCE.md`, `docs/turbotab-next/review-packets/*.md`.
- Edit Classic's app code (`ml/`, `pages/`, `utils/`, `app.py`); Classic's `tests/` may change.
- Add a proprietary tool or service to the product.

## The trust bar
- Test first, from known truth: a new behavior's test fails on the base commit, and its expected
  value comes from construction, an independent library (R, statsmodels, scikit-learn, numpy) or
  a derivation, never from the engine's own output or constant.
- Test a guard both ways: it fires where it should and stays silent where it should.
- A number is right or it is refused. A caveat beside a number the fit cannot certify is no fix.
- Where float64 or a solver can tip a case either way, the test pins that claim (right, or refused),
  not which refusal fires: that differs between platforms (CI's Linux runner, twice in 2026-10).
- A sentence the user reads says only what the code does. A number in it is interpolated from the
  value the code uses, never retyped beside the constant that holds it.

## Environment and tests
- Python is `scripts/py`: the main checkout's `venv-next`, pinned as CI is (a worktree has no venv;
  `venv/` is Classic's; `.venv` lacks pytest). Run from your worktree's root so `turbotab` imports
  from your tree, and prefix every command with `OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2`.
- A targeted run: `OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py -m pytest <paths> -m "not slow" -q -n 2 -p no:cacheprovider --timeout=600 2>&1 | tail -15`
- A test you name on purpose: drop `-m "not slow"` and check that the summary counts it, not as
  deselected. Find the tests that reach your code with `git grep -l <name> -- 'turbotab/*test_*.py'`;
  run an acceptance or legacy file only by name, and only when it reaches your code.
- For frontend work in a worktree, symlink the main checkout's `turbotab/frontend/node_modules`.
  Leave nothing behind: no server, background process, scratch worktree or symlink you made.
- Read what your prompt names, by section or line range; grep first, keep output short. For what
  the code does, the code is the truth.

## Commits and the report
- A commit message is one or two plain sentences saying what is now true, in the voice of
  `git log`, then a blank line and the attribution lines from your harness's system reminder
  (only if it gives none, those your prompt supplies).
- The pre-commit hook checks that every `.py` file parses and that `.py` and `.md` prose is
  American English. If it prints GATES CANNOT RUN, set `TURBOTAB_PYTHON` to the main checkout's
  `venv-next/bin/python`.
- Return your report as data in your final response, in the shape your prompt's schema or your
  role's file gives, with your three most doubtful claims and what you did not run or could not
  check. A short true report beats a green one; a result you did not read is not reported.
