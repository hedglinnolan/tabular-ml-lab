# TurboTab

Tabular ML Lab is a research workbench for nutrition researchers: upload a table, make every
modeling decision with its consequence shown, and leave with a defensible result and a methods
section. TurboTab v2, in `turbotab/`, is the rebuild; Classic, the Streamlit app, is `ml/`,
`pages/`, `utils/` and `app.py`. Many Claude agents build v2 under one orchestrating Claude for the
product owner, Nolan Hedglin, who judges only what is on screen.

## Where truth lives

- `docs/turbotab-next/BLUEPRINT.md` is the contract; §0 holds rulings that are not re-litigated.
- `docs/turbotab-next/V2_DEFINITION_OF_DONE.md` is the finish line. New ideas go to `INBOX.md`
  there, one line each, never into the work in flight.
- `docs/turbotab-next/HANDOFF.md` is the current state and the ordered next steps.
- Presentation: `docs/turbotab-next/calm/FOUNDATION.md` (it amends DESIGN_LANGUAGE §02–§04 and
  BLUEPRINT §11). Methods: `MODELING_SEQUENCE.md`, `RECIPES_AND_TUNING.md`,
  `MODEL_FAMILY_CONTRACT.md`, `SURFACING_POLICY.md`. Plans: `docs/turbotab-next/plans/`.
- The orchestrator's own guide is `docs/turbotab-next/ORCHESTRATOR.md`.
- Your prompt names your task, branch, worktree, owned files, stopping rule and report shape.

## Environment

- Python is the main checkout's `venv/bin/python`: `/Users/nhedglin/tabular-ml-lab/venv/bin/python`
  (3.13, with pytest, xdist and pytest-timeout). A linked worktree has no venv of its own, and
  `.venv` has no pytest. Run from your worktree's root so `turbotab` imports from your tree.
- Set `OMP_NUM_THREADS=2` and `TURBOTAB_WORKERS=2` for every run.
- Never `pip install`, `npm install` or upgrade anything: other agents are testing in the same
  environment. A worktree lacks `turbotab/frontend/node_modules`; symlink the main checkout's for
  the duration and remove the link before you finish. If a package is missing, stop and report it.
- The venv is not exactly on `turbotab/server/constraints.txt` (CI is); if a pinned number passes
  here and fails on CI, suspect versions first.
- Rscript is available for independent references. Install R packages only into a scratch library.

## Tests

- Targeted runs only:
  `OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 /Users/nhedglin/tabular-ml-lab/venv/bin/python -m pytest <paths> -m "not slow" -q -n 2 -p no:cacheprovider --timeout=900 2>&1 | tail -15`
- Keep any one local run under about ten minutes. Never run the whole core and server suite, the
  whole acceptance suite (`turbotab/core/tests/acceptance`) or the legacy suites (`tests/`,
  `turbotab/test_*.py`) locally: CI's full tier (`.github/workflows/v2-full.yml`) runs them, and the
  fast tier (`v2.yml`) is `-m "not slow"`. Mark a test over 35 s `slow`. Run acceptance tests by
  name, choosing those that reach the code you touched; a change to a family's defaults also runs
  every reproducibility pin.
- A new behavior starts with a test that fails on the base commit, with its truth from
  construction, an independent library (R, statsmodels, scikit-learn, numpy) or a derivation, never
  from the engine's own output. A guard is tested both ways: it fires where it should and stays
  silent where it should. A reported number is right or refused.
- Frontend: `npm run check` in `turbotab/frontend`, output through `tail`.

## Generated files

Never hand-edit these; regenerate them, and the integrator regenerates them after every merge:
`turbotab/server/openapi.json` and `turbotab/frontend/src/api/generated.ts`
(`/Users/nhedglin/tabular-ml-lab/venv/bin/python -m turbotab.server.openapi --write`, then `npm run gen:api`);
`docs/turbotab-next/reference/METHODS_REFERENCE.md`
(`/Users/nhedglin/tabular-ml-lab/venv/bin/python -m turbotab.core.reference.methods --check` says when it is stale);
`docs/turbotab-next/review-packets/*.md`
(`/Users/nhedglin/tabular-ml-lab/venv/bin/python -m turbotab.core.reference.packet --all --reuse --check`; always `--reuse`, never rerun the journey fits).

## The machine and the person

- The machine is beside Nolan's bed. No heavy run (T1, T12, a journey recapture, anything over a
  few minutes of full CPU) without the orchestrator scheduling it with him.
- Never `pkill`, `killall` or any pattern kill. Record the PIDs you start (`$!`), kill only those,
  and leave no server or background process running when you finish.
- Nolan's email address, which you may see in context, identifies him only. Never put it, or any
  personal identifier, in a request to any service: use no email parameter, or a placeholder such
  as `research@example.org`.
- No proprietary tools or services in the product.

## Branches and commits

- Work only in the worktree and on the branch your prompt names. Never edit or commit in the main
  checkout, never switch to a branch another worktree holds, and never push unless your prompt
  says which branch (`ci/<name>` branches run CI on GitHub's runners; `turbotab-next` and `main`
  are the orchestrator's).
- Classic's app code is not edited; Classic's `tests/` and the README may be.
- Edit only the files your package owns, plus new test files. A change you need in another file
  goes in your report as an open issue, not into the file.
- Commit with `TURBOTAB_PYTHON=/Users/nhedglin/tabular-ml-lab/venv/bin/python` so the pre-commit
  hook can run; never `--no-verify`. The hook's gates: every `.py` parses, and `.py` and `.md`
  prose is American English (`tests/test_american_spelling.py`).
- A commit message is one or two plain sentences saying what is now true, in the voice of
  `git log`, then a blank line, then the two attribution lines your prompt gives.
- Match the surrounding code's idiom, naming and comment density.

## Report

Return data, not a message: branch and commits; files changed; the tests you ran and their counts;
every number that moved and its cause; the changes you needed in files you do not own; and your
three most doubtful claims.
