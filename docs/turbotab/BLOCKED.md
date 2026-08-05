# Blocked

What an unsupervised loop could not do, written down instead of worked around.
The rule this file exists for is in `LOOP.md` §05: *if you are blocked or
something looks structurally wrong, stop and write what you found here rather
than guessing.*

Each entry names the command, what it did, what it should have done, and — the
part that matters — **what was NOT done to make it pass.**

---

## §11 item 12's three software defaults could not be verified, and two of the three are outside the gate

**Found:** L50/F2 — the metabolomics hedges, 2026-08-04.
**File:** `docs/turbotab/research/METABOLOMICS_PACK.md` §11 item 12, and §02
items 4 and 6.
**Instrument:** `docs/turbotab/tools/evidence.py check`, and its own extractor
run directly over `docs/turbotab/research/*.md`.

§11 item 12 is a hard stop: *"any claim about a specific software default —
MetaboAnalyst's IQR filter, `pmp`'s blank fold change, structToolbox's D-ratio.
These change between versions. **[verify-at-build]** and, better, read them from
the user's installed version rather than hard-coding."*

**None of the three could be read here, and the honest form of that is a
refusal rather than a number.** All three belong to R/Bioconductor packages;
none is installed and there is no R runtime on this machine (`which R` and
`which Rscript` both find nothing, `pip list` finds no `pmp` and no
`structToolbox`). §11's own preferred remedy — read it from the user's
installed version — has nothing to read. So `packs.software_default()` **always
refuses**, with the badge, and with what to do instead: take the value from the
release you actually ran and state it in your methods with the version. There
is no branch in that function that returns a number, which is structural rather
than careful — a function that could return one is a place a later loop puts a
constant.

**The gap in the gate, and it is the part worth carrying.** `evidence.py`'s
check 6 holds a number out of the code only when the marker **names** it —
`_VERIFY` captures the text after the colon and `_NUMBER` reads digits out of
that capture, so a bare `[verify-at-build]` contributes nothing. Run over all
five research files, the entire held-out set is **one pair**:
`{('METABOLOMICS_PACK.md', '50')}`, from §02's
`[verify-at-build: 50% and the SD-vs-MAD default]`.

`METABOLOMICS_PACK.md` carries exactly **two** markers in its content — §02
line 154, named; §11 line 922, bare — plus the legend at line 11 that defines
what the marker means. So of item 12's three software defaults, only
structToolbox's D-ratio criterion is protected by the gate, **because §02
happens to name its number**. MetaboAnalyst's IQR-filter setting and `pmp`'s
blank fold change are marked only by §11's bare marker, and a later loop could
hard-code either one and the gate would print a green tick.

**What was NOT done to make this pass.**

- **No value was looked up and shipped.** Egress from this environment is open
  — which is itself a correction to the standing premise, since the research
  files were built under a blocked proxy and that is no longer the environment.
  A web-sourced number would still be a hard-coded constant about a version
  this app has not read, which is exactly what item 12 forbids. Reading it
  would not have honored the marker; it would have converted a refusal into a
  claim.
- **No number was quoted "for context".** A refusal that names the value it is
  refusing to supply has supplied it.
- **`METABOLOMICS_PACK.md` was not edited** to turn §11's bare marker into a
  named one. That is editing the evidence to fit the instrument, and the
  numbers would have had to come from somewhere this loop cannot reach.
- **`evidence.py` was not changed** to treat a bare `[verify-at-build]` as
  holding out the numbers in its own paragraph. That is a gate moving in the
  same loop as the work that pressured it, which `AGENT_ONBOARD.md` §08.2
  forbids by name: after the fact it is indistinguishable from relaxing — or
  tightening — a gate under pressure. It should land on a passing run, with the
  reasoning recorded before it is load-bearing.

**The consequence to carry.** The hedge register ships all thirteen §11 items
and none of them carries a tool-specific constant. The gate green tick means
*the one number a marker named is out of the code*; it does not mean *no
software default is hard-coded anywhere*, and the difference is two of the three
numbers item 12 is about.

---

## `make test` still runs zero tests, now for `TEST-038`'s reason

**Found:** L28 setup, 2026-07-30.
**Rows:** `TEST-039` (this row's own cause, resolved), `TEST-038` (the blocker
that remains).

The L28 prompt's setup step said to install the dev requirements so `make test`
runs, and to write here if it still could not.

**What was done.**

```
venv/bin/python -m pip install -r requirements-dev.txt
```

installed `pytest-timeout 2.4.0` (with `kaleido`, `choreographer`, `orjson`,
`platformdirs`, `simplejson`, `logistro`). `TEST-039`'s diagnosis was exactly
right: `PYTEST_OPTS := --timeout=60 -q` at `Makefile:18` was passing an option
no installed plugin recognized, every target exited **4** having run zero tests,
and installing the declared-but-absent plugin fixed that.

**What it does now.**

```
$ make test
ERROR tests/test_nn_modernization.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 warning, 1 error in 1.94s
make: *** [test] Error 2
```

`tests/test_nn_modernization.py` imports `torch` at module scope, `torch` is not
installed, and **a missing import at collection aborts the entire run** — so the
target still runs zero tests. The exit code moved from 4 to 2 and the count did
not move at all.

That is `TEST-038`, filed at L13 and still open: `models/nn_whuber.py:5` imports
`torch` unguarded while `utils/seed.py:7-10` wraps the identical import in
`try/except ImportError` with a comment saying it is optional. One of the two is
right about whether `torch` is optional and they cannot both be.

**What was NOT done, and why.**

- **`torch` was not installed.** It is ~1.1 GB, and `TEST-038`'s whole argument
  is that it should not be mandatory to collect the test suite. Installing it
  would make the symptom disappear while confirming the claim the finding
  disputes.
- **The Makefile was not edited** to add `--ignore=tests/test_nn_modernization.py`.
  That is the working-around this file exists to prevent: it would make the
  documented command green while leaving a module nobody can collect, and the
  next person would inherit a `make test` that silently skips a file.
- **`test_nn_modernization.py` was not guarded.** It is the right fix and it
  belongs to `TEST-038`, whose `act` field already specifies it — guard the
  import the way `utils/seed.py` does and raise from the wrapper's constructor,
  so the model advertises itself as unavailable instead of taking the module
  down. Doing it here would close another loop's finding inside this one, in a
  file the prompt did not scope.

**The consequence to carry.** Every test count this project reports still comes
from a hand-rolled invocation:

```
venv/bin/python -m pytest tests/ turbotab/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py -q
venv/bin/python -m pytest tests/integration -q
```

Those numbers are real, and they are evidence about a command written down
nowhere. `LOOP.md` §05 says adjudicating a loop now includes running `make test`
once; it will keep failing, at collection, until `TEST-038` is closed.

**And the structural gap behind both rows.** Nothing checks that `venv/`
satisfies `requirements-dev.txt`. `pytest-timeout` was declared at
`requirements-dev.txt:25` and absent from the interpreter the `Makefile` names,
and no gate could see the difference. This is the third decay of the same claim
in the same file — `LOOP.md` §05 records the first two — and the lesson keeps
failing to catch it because the lesson is prose.
