# TurboTab — the walking skeleton

The thinnest end-to-end slice that uses the real engine, real uploaded data, and the new
interface: **upload a CSV → real structural diagnosis → real profile → real ranked findings →
recorded decisions.**

No training, no job queue, no manuscript. Those are later slices. This one exists to prove the
shape — and, mainly, to test the riskiest assumption in `docs/turbotab/ARCHITECTURE.md` with
running code instead of a claim.

---

## Run it

From the repository root.

**Windows (PowerShell)**

```powershell
python -m venv turbotab\.venv
turbotab\.venv\Scripts\python.exe -m pip install -r turbotab\requirements.txt
turbotab\.venv\Scripts\python.exe -m uvicorn turbotab.api:app --port 8777
```

**macOS / Linux**

```bash
python3 -m venv turbotab/.venv
turbotab/.venv/bin/python -m pip install -r turbotab/requirements.txt
turbotab/.venv/bin/python -m uvicorn turbotab.api:app --port 8777
```

Then open **<http://127.0.0.1:8777/>** and drop a CSV on it.

`turbotab/sample_data/clinic_visits.csv` is there if you want something to try immediately — a
deliberately messy 140-row table the engine finds fifteen things in. Your own file is the better
test.

### Tests

```powershell
turbotab\.venv\Scripts\python.exe -m pytest turbotab\test_skeleton.py -v
```

44 tests, about three seconds. They need no server running.

### Driving it with the harness on

**Temporary development instrumentation, not a product feature.** Off by default.

```bash
TURBOTAB_DEV_CHECKS=1 turbotab/.venv/bin/python -m uvicorn turbotab.api:app --port 8777
```

Then drive normally. Every state transition is checked against the record — that
every number displayed traces to a value in it, that the seal's integrity holds
when recomputed rather than trusted, that a deferred transform left the working
table byte-identical, that exactly the right things went stale. **A violation
records and continues**: one bug must not end a drive, because the driver is
looking for the second and third bug too.

Each drive writes `turbotab/sessions/<timestamp>/` (gitignored):

| File | What it holds |
|---|---|
| `index.md` | **Read this first — it opens with violations, not narrative.** |
| `violations.jsonl` | one per line, each naming the check and the action |
| `swallowed.jsonl` | exceptions caught somewhere and never surfaced |
| `actions.jsonl` | every request and response, in order |
| `state/NNNN-{before,after,diff}.json` | the resolved project on both sides of each action |
| `dom/` | one snapshot per render, styles inline, opens in a browser |
| `console.jsonl` | browser errors and unhandled rejections |
| `replay.py` | re-runs the drive against a fresh server |

The state is written whole on both sides and diffed, so an unexpected change is
a **diff rather than a guess** — the difference between *"something went stale"*
and *"`selected_models` emptied when the recipe was set"*.

`swallowed.jsonl` deserves its own line. The ledger has a dozen open rows whose
text ends *"and nothing surfaces it"*, and a bug hunt through code whose failure
mode is silence finds nothing until the silence is removed. Two layers: sites we
own call `devchecks.swallowed()` with what the user therefore did not see, and
everything else is caught by `sys.monitoring`'s `EXCEPTION_HANDLED` filtered to
this repository's own files — **so nothing in the frozen engine modules is
edited**, which the freeze in `TRANSITION_PLAN.md` §05 would not permit anyway.

---

## The gate

> I can start the server, drop my own CSV on it, and see real findings about *my* data rendered in
> the new interface.

Met. Verified in Chromium against a real upload: the structural findings render, choosing a target
opens Explore with the real profile, deferring and flagging fill the rail docks from the server's
record, and there are no console errors in either color scheme.

**Preview slice, second gate:**

> The preview for a finding on a real CSV shows real changed cells, and declining leaves the
> project byte-identical.

Met. "Show me what this means" opens a before/after of the two real frames with only the cells that
actually differ highlighted — `male → Male`, `female → Female`, 47 of them, each quoted from the
frame it came from. Applying is a separate press; `revert` restores the table byte-for-byte, and
`test_declining_a_preview_leaves_the_project_byte_identical` previews every structural finding over
HTTP and asserts a content hash of values, row labels and dtypes is unchanged.

---

## What is actually here

| File | What it is |
|---|---|
| `engine.py` | The whole adapter. `import_doctor.diagnose`, `compute_dataset_profile`, `triage.detect_task_type`, plus JSON-safety and a merge of the two finding streams. It computes nothing. |
| `project.py` | `AnalysisProject`: a dataframe handle, the target, append-only decisions, findings. Row identity is the index **label**. Imports no engine code. |
| `api.py` | The endpoints, and the static mount for the frontend. Orchestrates; computes nothing. |
| `web/index.html` | The prototype, with its synthetic constants replaced by `fetch()`. The stylesheet is carried across byte for byte. |
| `test_skeleton.py` | Real CSV in, real findings out, compared against direct engine calls. |
| `sample_data/` | One messy table to try it on. |

```
POST /project                              upload a table, get a diagnosis
GET  /project/{id}                         what is currently true
POST /project/{id}/decision                record one answer
GET  /project/{id}/findings                the ranked findings
GET  /project/{id}/finding/{fid}/preview   what a repair would change, without changing it
```

Decision kinds: `set_target` · `set_task_type` · `apply` · `revert` · `defer` · `dismiss` ·
`undismiss` · `flag` · `unflag` · `note`. `apply` is the only one that changes the working table,
and `revert` puts it back byte-for-byte.

Projects live in memory. There is no disk path, because `ARCHITECTURE.md` §02 records a
`_NEVER_PERSIST` contract and §04 lists persistence as an open question — a skeleton that quietly
wrote to disk would settle it by accident. Restarting the server loses your projects; that is the
correct behavior for now, not an oversight.

---

## What this build found

Five things worth carrying forward. The first two were corrections to the transition documents and
**have since been fixed upstream** (`docs/turbotab/ARCHITECTURE.md`, commit `47c9f1b`); they are
kept here as the record of how the skeleton earned its keep. The third is a live bug, now tracked
as `T0-LIVE-004`.

### 1 · The engine runs headless — and `model_coach` is not tainted at import

`ARCHITECTURE.md` §01 used to list `ml.model_coach` as transitively coupled to Streamlit through
`utils.insight_ledger`. **It is not.** `model_coach` imports the ledger *lazily*, inside functions
(`ml/model_coach.py:634` and `:1080`), and its module-level imports are only `dataclasses`,
`typing` and `enum`, so it loads clean with Streamlit blocked. The coupling is real but deferred to
call time. Counting module-level imports only, 4 of 42 core modules are blocked, not 7 — and the
Router's basis is not one of them.

Nothing in `utils/insight_ledger.py` had to be cut, so **nothing outside `turbotab/` was modified**
to make this build work.

Related, and also now corrected: §01 said the ledger's only coupling was the `get_ledger()`
singleton. There is a module-level `import streamlit as st` at `utils/insight_ledger.py:46` above
it. Cutting the singleton alone would leave the module tainted.

### 2 · The reproduce snippet in `ARCHITECTURE.md` §01 could not fail

The snippet defined a meta-path finder with `find_module` / `load_module`. The import system
stopped consulting those in **Python 3.12** — this repo is on 3.12+ — so run as printed the blocker
was inert and reported success whether or not the module was coupled. The deeper problem was the
second one: on a machine with no Streamlit installed, the import fails for the wrong reason and the
check passes vacuously. A guard that cannot fail proves nothing either way. A working blocker
implements `find_spec`:

```python
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError(f"BLOCKED: {name}")
        return None
```

`test_engine_imports_and_runs_with_streamlit_blocked` uses that form, and guards against the other
way this test goes quietly wrong — on a machine with no Streamlit installed it would pass while
proving nothing, so it puts a stub `streamlit` on the path first and asserts the blocker actually
bites before importing the engine.

### 3 · pandas 3 silently turns a classification target into a regression one

Not a TurboTab issue — it is shipping in the current app.

`ml/triage.py:41` decides task type with `if target_series.dtype in ['object', 'category', 'bool']`.
pandas 3.0 makes `str` the default dtype for text columns, so a text target matches none of the
branches and falls through to the fallback at line 91: **`regression`, low confidence.**

Measured stage by stage on `sample_data/clinic_visits.csv`, across both majors:

| call | pandas 2.3.3 | pandas 3.0.5 |
|---|---|---|
| `diagnose` | 10 findings | 10 findings, byte-identical |
| `detect_task_type` | `classification` / **high** | **`regression` / low** |
| `profile(task=detected)` | ok | **`TypeError: Cannot perform reduction 'mean' with string dtype`** |
| `profile(task="classification")` | ok | ok |

Two things that sharpen the report. **The import doctor is unaffected** — structural diagnosis is
identical across majors. And **`compute_dataset_profile` is not independently broken**: it is
correct when told the truth. The damage is that one wrong answer poisons the next call — the
profiler takes the regression branch and tries to average a text column. The exception names the
string dtype, not the misdetection that caused it, so in Classic this surfaces as an unhandled
`TypeError` on the EDA page with no hint of the real cause.

Logged as `T0-LIVE-004`. The class is wider than the one line: the same dtype-identity comparison
appears at eleven sites — `ml/triage.py:41,47,75,165,166,170,189`, `ml/dataset_profile.py:189`
(`is_id_like`), `ml/eda_recommender.py:98,106,137` (which analyzes get offered),
`models/registry_wrappers.py:44` (model selection). **Two of this skeleton's three engine entry
points are on that list**, so the cap is load-bearing here, not hygiene.

`test_a_text_target_is_read_as_classification` is the canary: it asserts the behavior the pin
buys, so lifting the pin without doing the repair fails the suite rather than quietly changing an
answer.

> ### ⚠ The two requirements files must not drift
>
> `turbotab/requirements.txt` caps `pandas<3`. The root `requirements.txt` now does too — but the
> reason is worth stating, because it is not tidiness. If Classic ever installs pandas 3 while
> Guided installs 2, **the two doors return different task types for the same CSV**, and the "same
> modeling process behind both doors" promise breaks at the dependency layer, where no amount of
> shared code can catch it. Parity requires a shared dependency envelope, not just shared
> functions. Any change to the pandas or numpy bound in either file belongs in the same commit as
> the change to the other.

### 4 · Four of the nine fix kinds renumber your rows

Found by building the preview engine. `promote_header`, `drop_empty_rows` and `drop_rows` all end
in `.reset_index(drop=True)`, and `melt_repeated` rebuilds the index outright. This project keys
rows by **index label**, so those repairs silently repoint every label stored before them — a
sealed lockbox would survive the repair and afterwards name different rows.

This is `TRANSITION_PLAN.md` §02.2 reached from a new direction. §02.2 is about two *components*
disagreeing on a convention; this is one *repair* invalidating the convention mid-analysis, which
no amount of agreement between components would prevent.

The preview reports it rather than working around it, and detects it by content rather than by fix
kind: the surviving labels are looked up in the original frame and the untouched columns compared.
Dropping trailing footer rows from a clean `RangeIndex` renumbers nothing that matters and is
correctly reported safe; dropping from the middle is not, and neither is any drop on a frame that
has already been filtered once.

> A first version of that check asked whether *any* column still lined up. A constant column lines
> up under every renumbering there is, so it would have waved through exactly the corruption the
> check exists to catch. It now requires every untouched column to agree, and
> `test_row_identity_check_is_not_fooled_by_a_constant_column` builds a frame with a constant
> column in it to keep that honest.

### 5 · The diagnose → profile → detect path needs only pandas and numpy

No scikit-learn, no scipy, no statsmodels, no torch. The portability claim is not just true, it is
stronger than the census suggests — this slice installs in seconds.

---

## What it deliberately does not do

- **No training, jobs, or manuscript.** Out of scope by instruction, and the job queue is the
  component whose absence caused the migration; it deserves its own slice.
- **Only one file at a time.** Multi-file joining is a different set of questions, and
  `TRANSITION_PLAN.md` §05 freezes that path pending an open defect backlog.
- **Five of the eight steps in the rail are not built.** The rail says so rather than showing
  disabled scenery.

## Decisions taken while building, worth arguing with

- **Findings are ranked by the engine's own severity, then its confidence, then id.** That total
  order is the only ordering judgment `engine.py` makes; it introduces no new tiers.
- **Profile warnings carry no confidence**, so they are serialised with `confidence: null` and
  `auto_suggestable: false`. A `DataWarning` may never pre-select anything.
- **A duplicated target label is refused, not answered.** `df[target]` returns a frame there, and
  `detect_task_type` would confidently describe the wrong thing. Note that a CSV upload cannot
  currently reach that guard: `pd.read_csv` renames a repeated header (`bp`, `bp` becomes `bp`,
  `bp.1`). The guard is for the doors the transition plan actually worries about — joins, Excel
  sheets read without a header, frames built in code — and is tested through one of those rather
  than through the upload path, which cannot exercise it.
- **`read_table` uses plain pandas inference.** Pre-cleaning would delete the doctor's findings
  before it saw them; reading everything as text would make `check_numeric_stored_as_text` fire on
  every numeric column.
