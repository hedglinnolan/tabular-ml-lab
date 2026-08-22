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

---

## L66's three pending files allocate three ledger ids twice, and the contract that produced them has no allocator

**Found:** L66 orchestration, 2026-08-22, after all three agents finished.
**Files:** `docs/turbotab/tools/pending/L66-A1.py` on `TurboTab-L66-A1`, and
`docs/turbotab/tools/pending/L66-A2.py` on `TurboTab-L66-A2`.
**Instrument:** reading both files after both agents reported.

`L66` §00.6.2 has each agent write its intended `ledger.py` invocations to a
pending file instead of touching `findings.json`, and §00.6.3 has the
adjudicator apply the three files in agent order. **Nothing in that contract
allocates ids.** Three agents ran concurrently, each read the same ledger, each
found the same next free id, and two of them took it.

**Six distinct findings are competing for three ids:**

| id | Agent 1 filed | Agent 2 filed |
|---|---|---|
| `MISC-034` | high, `FIXED` — a container written by `innerHTML` in one branch and `ownChild` appends in another keeps what the assign branch left, forever | medium, `OPEN` — the 49 STATE+CONTRACT rows called "not closeable" are two different things, and the half called a boundary catalogue holds two criticals and thirteen highs |
| `TEST-114` | high, `FIXED` — the page harness could not be asked what the page repaints | medium, `OPEN` — 143 of 1,005 rows carry an id whose prefix contradicts their `area` field |
| `TEST-115` | medium, `OPEN` — the deck-region guard's matcher, two prongs, 0-of-0 today | high, `OPEN` — guards skipping worktrees by `'.worktrees' not in str(path)` are vacuous when the repository is itself checked out under `.worktrees/` |

`IMPORT-268`–`271` from Agent 3 do not collide.

**A second defect in the same file, independent of the collision.** Agent 1's
pending file expresses its invocations as **comments** naming Python variables
rather than as argv lists, and its three `add` lines carry no `--id`. `add`
requires `--id`. So `L66-A1.py` cannot be applied mechanically even after the
ids are settled; Agents 2 and 3 wrote argv lists that can. The contract said
"a committed Python file holding your exact intended ledger invocations" and did
not say executable, so this is the contract underspecifying rather than the
agent departing from it.

**What was NOT done to make it pass.** The ids were **not** renumbered and no
pending file was edited. Renumbering is an allocation decision that belongs to
adjudication, the ids are cross-referenced inside note prose in both files
(Agent 1's `DRIVE-060` note says "filed as `MISC-034`"), and a rewrite that
missed one cross-reference would put a wrong id in a record whose whole premise
is provenance. `ledger.py check` refuses duplicate ids, so the collision fails
loudly at apply time rather than silently — the gate works.

**The recommended resolution, for whoever applies these.** Agent 1 is applied
first under §00.6.3, so Agent 1 keeps `MISC-034` / `TEST-114` / `TEST-115` and
Agent 2's five rows shift to the next free ids in each area, carrying their
cross-references with them. That ordering is the contract's, not a judgment
about which finding matters more.

**The fix the contract needs, so this does not recur.** An id is not free
because the ledger does not hold it; it is free because nothing has claimed it.
Either the pending-file contract assigns each agent a disjoint id block up
front, or `ledger.py` grows an `add` that allocates the next free id in an area
itself and the pending files stop naming ids at all. The second is better: it
removes the shared mutable resource instead of partitioning it, and it is the
same move `count_dom_writes.py` made for a number three documents disagreed
about.
