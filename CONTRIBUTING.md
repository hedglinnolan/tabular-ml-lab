# Contributing

Tabular ML Lab is used by researchers who mostly do not write code, and it
emits manuscripts, so a bug report from a non-developer and a pull request from
a developer are both contributions. This page covers both.

## Reporting a bug

Open an issue with the **Bug report** template; it asks for the four things
that make a report actionable:

1. **Your operating system** (Windows, macOS, Linux) and, if you know it, the
   version.
2. **How you started the app**: the double-click starter from the release zip,
   the terminal install (`setup.sh` / `setup.ps1` then `run.sh` / `run.ps1`),
   or the Docker container on a server.
3. **Which version.** The release you downloaded, or the `version:` line in
   `CITATION.cff` in your copy of the folder.
4. **What you clicked, what happened, and what you expected.** A screenshot
   helps; please do not attach data with real participants in it.

Questions that are not bugs ("can it do X?", "which model should I pick?")
belong in [GitHub Discussions](https://github.com/hedglinnolan/tabular-ml-lab/discussions),
which is enabled and read. Security problems go through
[SECURITY.md](SECURITY.md), not a public issue.

## Setting up to develop

The app is a Streamlit workbench. The recommended setup is the one the README
describes under Quick Start, which uses [uv](https://docs.astral.sh/uv/) to
download its own Python 3.12:

```bash
# macOS / Linux
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
chmod +x setup.sh run.sh && ./setup.sh
./run.sh
```

```powershell
# Windows (PowerShell)
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
.\setup.ps1
.\run.ps1
```

Two environments can exist side by side, and it is worth knowing which one
you are in. The uv path and the double-click starter both build **`.venv/`**.
The `Makefile` hard-codes **`venv/`** (`./venv/bin/python`), which is the pip
fallback's environment, so `make ci` and the other `make` targets only work
after:

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt   # Windows: venv\Scripts\pip
```

`requirements-dev.txt` is not optional for the test suite: without matplotlib
and plotly from it, 26 integration tests fail in a way that looks like real
breakage.

## Running the tests the way CI does

CI (`.github/workflows/ci.yml`) runs on Ubuntu with Python 3.12 in three
tiers. Run them the same way locally, from the repository root, with your
environment's Python:

```bash
# Tier 1: unit and workflow tests. OMP_NUM_THREADS=1 matters: sklearn's and
# torch's OpenMP runtimes share the process and can segfault it otherwise.
OMP_NUM_THREADS=1 python -m pytest tests/ --ignore=tests/integration \
  --ignore=tests/test_suite_is_order_independent.py \
  --ignore=tests/test_a_fixed_row_names_a_test_that_actually_runs.py \
  --ignore=tests/test_nn_modernization.py \
  -q --timeout=120

# Tier 1b: the one file that runs torch, in a fresh interpreter.
OMP_NUM_THREADS=1 python -m pytest tests/test_nn_modernization.py -q --timeout=120

# Tier 2: Streamlit AppTest integration and the LaTeX round-trip.
python -m pytest tests/integration -q --timeout=120
```

The two excluded meta-tests re-run the whole suite in subprocesses; they are a
local or nightly gate, not something to run on every change. On a multi-core
machine, `-n 4 --dist loadfile` (pytest-xdist, in `requirements-dev.txt`)
makes Tier 1 several times faster; use `--dist loadfile`, because one workflow
test file shares state across its class. The PDF round-trip in Tier 2 needs a
TeX installation with `adjustbox` and `microtype`; without one it fails
locally and passes in CI, which installs the full texlive set.

## Gates a pull request must pass

Besides the tiers above, a few tests check the repository rather than the
code, and they fail for reasons that are easy to miss:

- **American English** (`tests/test_american_spelling.py`) scans every tracked
  `.py` and `.md` file. "analyze", "modeling", "behavior"; and note that
  "analyses" is the noun, "analyzes" the verb
  (`tests/test_analyses_is_the_noun.py`).
- **Documented commands exist** (`tests/test_a_documented_command_still_works.py`)
  parses every fenced shell block in `README.md` and the TurboTab docs: a
  `make` target it names must exist in the `Makefile`, and a `scripts/` path
  must exist on disk.
- **The deployment layer** (`tests/test_deployment_layer.py`) checks that the
  Dockerfile, compose file and deployment guide agree, and that no document
  claims the app authenticates anyone (it does not; single sign-on goes in
  front of it).
- **CI also runs CodeQL.** A substring check on a URL-looking string will be
  flagged; match the mechanism you mean instead.

## Conventions

- **Commit titles are sentences** that say what changed and why it was wrong
  before: "The constant column is actually dropped", not "fix bug". The body
  carries the evidence.
- **One pull request per concern.** Five small PRs merge; one large one waits.
- **Tests carry their reasoning.** A test's docstring says what defect it
  pins and what would have to be true for the test to be wrong. A test that
  asserts an absence needs a positive control, or its silence means nothing.
- **Never `.clear()` a session-state container in place.** Cohort branches
  hold those objects by reference, so clearing one empties an archived branch
  through it; always assign a fresh container. The invariants around this are
  in `docs/COHORT_BRANCHES_MVP.md`, section 12.
- **A value that reads as a measurement must be one.** No `999.0` for an
  undefined statistic, no `nan` that prints as a score, no caveat about a
  comparison that was never made. The app writes Methods sections; every
  number in it is a claim.
- **Every cap that changes an analysis says so** on the page and in the
  manuscript. Capping a computation means auditing whatever writes prose
  about its result.

## Where the design is written down

[ARCHITECTURE.md](ARCHITECTURE.md) describes the Streamlit app's layers, page
flow and session-state keys. It predates `turbotab/` (the second front door
onto the same engine) and the cohort-branch work, so read it as the engine's
design notes rather than a current map; `docs/` holds the later design records
and audits. The README's "Project Structure" section is the current file layout.
