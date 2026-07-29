# Do the intelligent features get ported?

Short answer: **they don't get ported — they get shared.** That distinction is the whole point of
the extraction, and it is also where the one real risk lives.

The question splits three ways, and each part has a different answer.

---

## 1 · Capability — shared, automatically

Almost every intelligent feature already lives in the engine, not in the UI. Both doors call the
same function. Nothing is copied, nothing is reimplemented, and a fix lands once for both.

| Capability | Home | Lines | Shared today |
|---|---|---:|---|
| Structural diagnosis + reversible repairs | `ml/import_doctor.py` | 1,029 | yes |
| Key detection, join confidence, relationship types | `ml/join_doctor.py` | 1,088 | yes |
| Model ranking, bucketing, viability | `ml/model_coach.py` | 1,443 | yes |
| Pre-training probe | `ml/coach_probe.py` | 243 | yes |
| Dataset diagnostics | `ml/dataset_profile.py` | 754 | yes |
| Task-type + cohort-structure detection | `ml/triage.py` | 282 | yes |
| EDA recommendations | `ml/eda_recommender.py` | 563 | yes |
| EDA actions | `ml/eda_actions.py` | 1,328 | yes (4 `st` refs to strip) |
| Table One | `ml/table_one.py` | 375 | yes |
| Statistical tests | `ml/stats_tests.py` | 169 | yes |
| Outlier detection | `ml/outliers.py` | 100 | yes |
| Calibration | `ml/calibration.py` | 338 | yes |
| Bootstrap CIs | `ml/bootstrap.py` | 263 | yes |
| Sensitivity analysis | `ml/sensitivity.py` | 132 | yes |
| Manuscript generation + `[AUTHOR REQUIRED]` | `ml/narrative_engine.py` | 1,975 | after detaint |
| Manuscript validation | `ml/manuscript_validator.py` | 426 | after detaint |
| LaTeX export | `ml/latex_report.py` | 1,066 | after detaint |
| PCA / UMAP / persistence / Mapper | `ml/macro_shape.py` | 723 | yes (5 `st` refs; also `T0-LIVE-001`) |
| Feature selection | `ml/feature_selection.py` | 295 | yes |
| NN configuration advice | `ml/nn_recommender.py` | 199 | yes |
| Plot narration | `ml/plot_narrative.py` | 469 | yes |
| Regime detection | `ml/regime.py` | 193 | yes |
| Clinical units + physiology reference | `ml/clinical_units.py`, `ml/physiology_reference.py` | 353 | yes |
| Insight lifecycle (the coach's memory) | `utils/insight_ledger.py` | 1,408 | after singleton cut |
| Provenance record | `utils/workflow_provenance.py` | 759 | after singleton cut |
| Test-set lockbox | `utils/test_lockbox.py` | 554 | after `st` reads removed |
| Cohort runs | `utils/cohorts.py` | 629 | after detaint |
| Replay engine | `utils/replay.py` | 405 | after detaint |

**~19,000 lines of intelligence, and essentially all of it is already engine code.** The "after
detaint" entries are the L7 work, and the deepest of them is a singleton at the bottom of a file.

## 2 · Orchestration — trapped, and this is the real work

What is *not* in the engine is the intelligence about **which analysis runs when, in what order,
with what defaults, and which options are offered.** That lives in `pages/`, and it is 19,835
lines of it.

| Trapped capability | Where | Risk |
|---|---|---|
| The whole split strategy — grouped / chronological / lockbox-respecting / stratified | `pages/06:380-760` (~370 loc) | **high** — safety-critical, untested, no `ml/splits.py` equivalent exists |
| Step-completion model + quick/advanced disclosure | `utils/theme.py:685` | **high** — the Router's readiness function, filed under CSS |
| Which EDA analyses run and what counts as notable | `pages/02` (222 logic markers) | high — this *is* the Router's raw material |
| Report assembly decisions | `pages/10` (24 local functions) | medium |
| SHAP orchestration + per-model applicability | `pages/07` (160 markers) | medium |
| Statistical test selection rules | `pages/09` (79 markers) | medium |
| Per-model pipeline defaults | `pages/05` (131 markers) | medium |
| Transform catalogue + applicability | `pages/03` (144 markers) | medium |

**Nothing here is lost — but nothing here is free either.** Each one has to be extracted to the
core before either door can share it, and until it is extracted it exists only in Streamlit.

## 3 · Exposure — a decision per feature

Even once a capability is shared, the Guided door only surfaces what its interview asks about.
That is a design choice, not an accident: the entire premise is that fewer, better-ordered
questions beat eleven pages of everything (`PRODUCT_VISION.md` §01).

So "is feature X in TurboTab?" is really three questions:

1. **Is the logic in the core?** Mostly yes today.
2. **Has the orchestration been extracted?** Mostly no.
3. **Does the Guided interview ask about it?** Per-feature decision, made when that step is built.

A capability can be fully shared and deliberately not surfaced in Guided — Classic remains the
door for it. That is a legitimate outcome, and it is not a regression, as long as it is
**recorded** rather than forgotten.

---

## The risk, stated plainly

> **A feature that exists only in `pages/` and is never touched will never reach the Guided door,
> and nothing will announce that.**

The lazy-migration policy (`ROADMAP.md` rule 4) says pages move to the core when you touch them.
That is the right policy for maintenance cost — and it means untouched pages stay Streamlit-only
indefinitely. Combined with the exposure decision in §3, a capability can go missing from Guided
for two entirely reasonable-sounding reasons at once, and nobody notices until a user asks.

### The mitigation

**A feature register, maintained like the ledger.** Every capability gets a row with five
states: `core` (extracted), `both` (extracted and exposed in Guided), `classic-only` (still
trapped, or deliberately not surfaced), `guided-only` (built in Guided **and owed back to
Classic**), and `guided-native` (belongs to Guided's design model; Classic is not expected to
gain it). Then:

The Guided pair is not symmetry for its own sake — **convergence runs both ways.**
Preview-before-apply and undo are Guided-first, and Classic today applies a repair from a single
button with no diff and no undo, which is precisely the blind consent `PRODUCT_VISION.md` §04
argues against. A `guided-only` row is a debt owed to Classic, and naming it stops the Guided
door quietly becoming the only place the product's own principles hold.

**Why the pair was split (L18).** `guided-only` originally meant *a debt owed back to Classic*.
That was true of the three constitutional capabilities it was coined for, and it stopped being
true as Guided accumulated things Classic will never have — a rendered skip, a read-as-draft
panel, a question grammar, deferral as a first-class disposition. Left merged, the debt count
inflates every loop and stops meaning anything: at the split it read 24, and 10 of those were
not debts at all.

The line, and it is a test rather than a feeling: **would a reasonable Classic still be missing
something?**

- **Debt** — a constitution clause binds both doors and Classic does not meet it; or Classic has
  a live defect Guided fixed; or Classic could have it and simply does not. `target-positive-class`
  is the clearest: Classic label-encodes a two-level target alphabetically, so the event is
  whichever level sorts last. That is a defect, not a difference.
- **Native** — the capability presupposes something Classic is not. `cross-deferral-resurface`
  needs a Router, and `TRANSITION_PLAN.md` §02.5 measured the coach as a pure annotator that can
  order questions but cannot gate them. Asking Classic for it is asking Classic to be Guided.

**The interesting cases are the ones where both doors have the feature and do different things
with it.** `feat-selection-methods` is the sharpest: Classic *runs* selection and stores a
result, Guided *records* what will be selected. Both have all five methods. Filing that as debt
says Classic owes itself a different implementation of something it already ships; the difference
is constitution §06's declaration-versus-execution split, which is design model, so it is
`guided-native`. The row's own text had said *"those are different things"* and then filed it as
debt anyway — which is exactly the drift the split fixes.

**`guided-native` still requires a reason.** *"It is just how Guided works"* is the shrug this
register exists to refuse, and it would be the easiest place to hide one.

Also note what the register caught on its very first use: `triage` returns low confidence and
tells the user to *verify or override*; Classic offers that override, Guided did not. Filing it
`classic-only` would have recorded **a governing-rule violation as a legitimate exclusion**. The
register works because it forces that comparison — but only if `classic-only` is treated as a
claim to be justified, never as a shrug.

**The register lives at [`FEATURE_REGISTER.md`](FEATURE_REGISTER.md), generated from
`data/register.json` via `tools/register.py` — never hand-edited.** It earned that structure the
hard way: the first register, written as a markdown table in this file, was destroyed by a branch
merge that blind-copied an older revision over it. The ledger survived the same merge because it
is data worked through a tool; the register now is too, with a `check` that fails when a built
step has no rows or the markdown goes stale.

- Building any Guided step **starts** by listing what the corresponding Classic page can do
  (`register.py add` per capability), and **ends** with every item dispositioned — and the step
  added to `BUILT_STEPS` so an empty step is a failure, not a silence.
- The parity harness covers `both` rows. `classic-only` rows are excluded from parity *by
  explicit entry*, never by omission — so the exclusion list is readable and arguable.
- "We forgot" stops being possible, because a capability with no row fails the register check.

This is the same trick as the findings ledger: the failure mode is silence, so make silence a
test failure.

**Corollary — the principle-locality rule** (from the T0-LIVE-005 close): *a principle written in
one place and applied in another is the same silence as a capability with no row.* The content-key
principle lived in a docstring on four caches while eight caches beside it violated it. When a
principle is worth a docstring, it is worth a test that applies it everywhere it binds — one
fingerprint for the page, not one per author.

**Corollary — the expiring-guarantee rule** (from the `T0-PREREG-002` close): *a protection that
depends on "X does not exist yet" expires the moment X exists, and nothing will tell you.*
`tests/integration/test_routing_baseline.py` wrote the pre-registered Classic baseline and said
why in its docstring — *"written now, while the Router does not exist, so it cannot be fitted
to."* True when written. False from the moment L8 landed, and no test, check or reviewer noticed:
for three loops afterwards every suite run re-measured the reference the value check is judged
against, with the Router present, and committed the new numbers over the old.

The sibling relationship to principle-locality is exact — both are **silence rather than
failure**, and both are invisible precisely because the thing that was true is still written
down. The difference is the axis: locality fails across *space* (stated here, violated there),
this one fails across *time* (true then, false now, unannounced).

Three defenses, in order of strength:

- **Do not write a temporal guarantee you cannot enforce.** Split the acts: measurement and
  comparison must not share a code path, or the comparison silently becomes a re-measurement.
- **Name the expiry condition in the artifact**, not in prose. A baseline carries `measured_at`;
  a test asserts it against the document that banks thresholds on it.
- **Audit every "before X exists" claim when X ships.** A loop that lands a component should grep
  for guarantees phrased against its absence. `router.py` landing at L8 should have triggered
  exactly that sweep and did not.

**Corollary — a check that nothing triggers is a check that does not exist** (from `GUIDED-019`).
Writing the check is half the work. The other half is naming the moment it runs, and that half is
skipped constantly because the check *passes* when you run it by hand, so it feels finished.

`tests/integration/test_routing_value_check.py` compares every run against the recorded result and
fails on divergence. It works. It fired correctly the whole time. And a metric regression
introduced at `4152020` rode through two loops, because nothing ran it: the pre-commit hook guards
four fast gates and this is not one of them, and the only other trigger was somebody choosing to
run the whole suite.

The sibling relationship to the expiring guarantee is exact, and the axis is different again.
Principle-locality fails across **space** — stated here, violated there. Expiring guarantees fail
across **time** — true then, false now. This one fails across **occasion**: correct always, and
consulted never.

Three defenses, in order of strength:

- **Give every check a trigger, and make the trigger structural.** Pre-commit for anything that
  fits in a couple of seconds; **pre-push** for the rest. A LOOP.md obligation is not a trigger —
  depending on memory is what failed at `8127101`, and `GUIDED-019` is the same failure a second
  time from a different direction.
- **Measure before designing the trigger.** The value check was assumed slow and is not: ~2.6 s
  warm, ~26 s cold. That measurement is the whole design. Under about ninety seconds, run it
  unconditionally — a gate that fires only when a *declared dependency set* changed is a gate that
  stops firing the day somebody adds a dependency and does not declare it, which is this same rule
  recurring one level up. Cleverness here buys latency and costs correctness.
- **A check whose trigger you cannot name is not done.** "Run it in CI" is a name; "run it when
  you touch the router" is not, unless something enforces the *when*.

**Corollary — the frozen-measurement rule** (from the same close, after the first "never edited,
ever" proved too blunt to state what it protected): *the measurements are frozen; the envelope may
gain labels, never lose or alter one.*

A pre-registered data file has two layers and only one of them is the experiment. The
`measurements` are the result and are immutable — a value that moves is a re-measurement, and a
re-measurement is a new row, never a restatement. Everything around them is envelope: schema
version, `measured_at`, `prereg`, and whatever provenance a later loop finds it needs.
`c8c5f51` added `pull_affordances` and `mode` to `routing-baseline.json` without moving a number,
and L9b added `measured_at`; both are legitimate, and a rule that forbade them would have forced
the provenance into a second file nobody checks.

**Why adding a label is safe here, stated because it is the whole justification:** a self-declared
stamp proves nothing on its own. Anyone who can swap the file can swap the stamp with it. So the
stamp is a convenience, and the load-bearing assertion is the git-read values check —
`test_the_frozen_baseline_is_the_one_the_prereg_names` reads the file as of the commit the
pre-registration names and compares every metric the prereg quotes. The envelope may be edited
precisely because nothing depends on trusting it.

**Corollary — the ephemeral-pointer rule** (from the L11 reconciliation): *a record that points at
ephemeral storage will eventually lie, and it lies toward "the work is gone."*

`docs/FINDINGS_LEDGER.md` pointed at `scratchpad/audit/orig48/` and a journal under `subagents/`
for the 48 confirmed import findings. Both paths were deleted. The durable copy existed the whole
time — `docs/audit/ORIGINAL_48_FINDINGS.md`, committed, and named in `TRANSITION_PLAN.md` §05 two
lines above the freeze rule — but the ledger's own pointer was the one anybody read, so the
findings were declared unrecoverable and a loop was spent reconstructing thirteen of them from
test names.

Note the direction of the lie. A stale pointer to durable storage says *"look here"* and fails
loudly when you do. A pointer to ephemeral storage fails **silently and pessimistically**: the path
is empty, the natural reading is that the work was lost, and the response is to redo it. That is
the expensive failure, and it is the one that looks like diligence.

- **Point at the repository, or do not point.** A path outside version control is a note about
  where something *was*, and it must be written as one.
- **When a record names a location, the location is part of the record** and is checked when the
  record is. A pointer nobody verifies is a claim nobody verifies.
- **"It is gone" is a finding, not a background fact.** It gets the same scepticism as any other
  claim, and the same demand for evidence — `git log --diff-filter=D`, a grep of the docs tree,
  and the neighbouring documents that might name it.

Fourth member of the family, with principle-locality, expiring guarantees, and fallback-path
survival. All four are silence rather than failure.

**Corollary — name every test after the defect it guards.** Thirteen findings whose text was
believed lost were reconstructed from test *names* alone: `TestBlowUpIsRefused`,
`TestSamePeopleDecidesGrouping`, `TestTheJoinDoesNotRetypeYourIdentifier`. Each name is a sentence
about what must remain true, so each was a recoverable statement of the finding.

That was an accident. It is now policy: **a regression test is named after the defect it guards, in
a sentence, not after the function it calls.** `test_join_keys_2` guards nothing anybody can read;
`test_leading_zero_ids_are_not_corrupted` is the finding. The test suite is the most durable record
in the project — it is executable, so it cannot silently stop being true — and naming it this way
makes it a readable one as well.

**Extension — a test that pins wrong behavior says so in its name.** A *pinning* test asserts what
the app currently does wrong, so that the gap stays visible and so that fixing it produces a red
test rather than silence. It is the exact opposite of a regression guard, and in CI output the two
are **indistinguishable**: both print one green line naming a behavior. A reader scanning

```
test_an_unrecognized_id_name_is_still_recorded_as_cross_sectional PASSED
```

reads the suite as *endorsing* that sentence, when it is in fact recording it as broken. That is
this project's governing failure — green output that asserts something false — reproduced in its
own instrumentation.

So: **prefix a pinning test `KNOWN_GAP_`, in the function name, and name the finding it pins in the
first line of the docstring.**

```
test_KNOWN_GAP_an_unrecognized_id_name_is_still_recorded_as_cross_sectional PASSED
```

Three consequences worth stating, because they are what the prefix buys:

- **The docstring is not enough.** CI prints names, not docstrings. A marker only a code reader
  sees does not reach the person reading a build log, who is the person at risk of misreading it.
- **`PASSED` on a `KNOWN_GAP_` line means the defect is still there.** That inversion has to be
  legible without opening the file, and the prefix is what makes it so.
- **The day it goes red is the day the row closes.** A `KNOWN_GAP_` test failing is not a
  regression and must not be "fixed" by editing the assertion. It is the signal to update the
  finding and the test together, which is what `IMPORT-022`'s docstring says in as many words.

`grep -r "def test_KNOWN_GAP_" tests/` is then the list of everything the suite knows is wrong —
a second, executable index of the ledger's open criticals.

**Corollary — the revert probe. A finding closes only against a test verified to fail when the fix
is reverted.** Topical proximity is not coverage.

`tools/ledger.py check` enforces that a `FIXED` row *names* a test. It cannot check that the test is
load-bearing, and that gap is where a false close lives. `IMPORT-108`/`118`/`124`/`128`/`148` were
held `OPEN` for "no named guard" while `tests/test_key_sampling_is_symmetric.py` sat in the tree
covering the same function — three readers took the file's existence as coverage. It was not: the
file tests three defects found *in* the fix, and reverting the defect the five findings actually
name left all seven of its tests green.

Two fixture coincidences hid it, and both are ordinary:

- **The fixtures agreed on row order**, so a positional truncation kept the same region of the key
  space on both sides, and `.sample(random_state=42)` on two equal-length columns drew the identical
  subset. Neither is a property of the code.
- **The repair was not where the reader assumed.** The rescale lives in `find_key_candidates`, not
  in `_key_tokens`, so it survived the revert and divided the answer back toward the truth.

So the procedure, and it is cheap — under a minute per finding:

1. `exec()` the module from source with the fix textually reverted, into a throwaway module name.
   (Register it in `sys.modules` first, or `@dataclass` cannot resolve its own module.)
2. Run the named test against both. It counts as a guard only if it is **green at `HEAD` and red
   under the revert**.
3. If nothing goes red, the finding is `OPEN` and the test that would catch it is the work.

**Write the revert down in the row.** Three findings this loop had a guard somewhere other than the
obvious line, and a reader tracing the row to the obvious line would have found code that looks
unfixed: `IMPORT-232`'s right-only IDs survive via the `_ORIGINAL_KEY` coalesce, *not* the
`drop(columns=[right_key])` the finding names; `IMPORT-244` needs **both** the restore and the
numeric re-coercion, so reverting either half alone reintroduces it; `IMPORT-252`'s
`if self.index_like:` appears **twice**, once in `score` and once in `confidence`, so a single-shot
replacement patches the wrong one and the revert looks harmless while the test stays green.

**Extension — a probe must verify the REASON for failure, not merely that it failed.** A revert
that turns a test red for some other reason has verified nothing, and it reads exactly like a
probe that worked.

This is the revert probe applied to itself. The probe answers *"is this test load-bearing?"* by
watching it go red; but a red test is a red test, and an import error, a fixture blowup, a
collection failure or a **different assertion in the same test** all produce one. Each of those
says the suite noticed *something*, which is not the claim. The claim is that this test guards
*this* defect.

So every revert declares the failure it expects, and the probe asserts that too:

```python
# (file, old, new, expect) — `expect` must appear in the failure output
("turbotab/features.py", '"p/n ≈ 0.39, which is the …"', '"which is fine; the …"',
 "the p/n argument is missing")
```

The precedent is `match=`. `pytest.raises(FeatureRefusal)` passes on *any* `FeatureRefusal`,
including one raised three lines earlier for an unrelated reason;
`pytest.raises(FeatureRefusal, match="not in the transform catalogue")` is the assertion somebody
actually meant. The probe needs the same discipline for the same reason, one level up.

Three consequences, all observed the first time it was used (L16, the polynomial routing message):

- **It catches assertions that pass on incidental substrings.** `"model" in message` survived
  deleting the entire route sentence, because *"45 pairwise products"* and an earlier *"a model"*
  both remained. `"55" in message` survived deleting the first of two arguments, because the second
  argument restated the count. Both looked like real assertions and neither was; the probe found
  them in one run, and the fix is to assert the **claim** — `"not a feature choice"` — rather than
  a word that happens to be nearby.
- **Assertion order becomes load-bearing.** A probe reads the *first* assertion to fire, so the
  most diagnostic one has to be first. With `"55" in message` ahead of `"did the routing fire at
  all?"`, every routing failure reported as a missing substring and the probe cheerfully verified
  the wrong reason.
- **An anchor that spans an implicit line continuation matches nothing**, and a revert that
  changes nothing leaves the test green — which reads as *not load-bearing* rather than as *the
  probe was broken*. The probe must fail loudly on an anchor that does not appear exactly once,
  and report that separately from a green test. Same failure the copy deck hit at L15, arriving
  from the other direction.

**Extension — assert on STRUCTURE, not on prose substrings.** *A substring of a message is a
wildcard wearing an assertion's clothes.*

`"model" in message` looks like an assertion and is a search. It passed on a message with the
route sentence deleted, because *"45 pairwise products"* and an earlier *"a model"* both remained.
`"55" in message` survived deleting the whole first argument, because the second argument restated
the count. Neither was a bad idea badly executed — both were the natural thing to write, and both
verified nothing.

The rule, in decreasing order of preference:

1. **Assert on the object, not on its rendering.** `record["n_excluded"] == 7`, `q.status ==
   "asked"`, `set(payload) & BANNED == set()`. A number, a key, a status, a type.
2. **When the prose IS the deliverable** — a disclosure, a refusal, a receipt — assert on the
   distinctive *claim*, not on a word inside it: `"not a feature choice" in message` rather than
   `"model" in message`, and quote enough of it that no other sentence in the file could satisfy
   it.
3. **Assert on the ABSENCE of a whole category** when the guarantee is a subtraction. The
   eligibility evidence is checked as *"no key in this payload is in `{median, quantiles,
   histogram, counts, …}`"* rather than as a sentence about withholding, because a median added
   later is one line and every sentence would still read correctly.

**And the self-referential round trip**, which is the same error in serialization clothing: a test
asserting `to_list(from_list(to_list(x))) == to_list(x)` compares the serializer against
**itself**, so a field the serializer never writes is absent from both sides and the equality
holds. `STATE-056` is the case — deleting `manuscript_text` from `to_dict` left the round trip
green. A round trip has to name at least one field explicitly, or compare against something the
serializer did not produce.

**Corollary — environment-dependent non-reproduction is not a fix.** If a finding does not reproduce
under a dependency version the repo does not pin to, it stays `OPEN` saying so.

**Corollary — a test that cannot run guards nothing, and a skip is invisible.** A `FIXED` row whose
test opens with `pytest.importorskip(...)` for a dependency the repo deliberately does not install
has a guarantee nobody in this environment can check. `MODELS-001` is the case: its guard is
`SKIPPED`, which in `-q` output is one character and in a summary line is a number nobody reads.

Fourth angle on the same family — principle-locality fails across **space**, expiring guarantees
across **time**, an untriggered check across **occasion**, and this one across **ENVIRONMENT**.
The repair is usually available and cheap: the dangerous behavior in `MODELS-001` is
`sklearn.base.clone` dropping a mark that is not a constructor parameter, which is testable
against a stub estimator with the same `get_params`/`set_params` shape — no network, no torch.

**And a warning about the sweep method itself, from running it.** Reconstructing the revert is the
hard part, not running it. Five of twenty-two reverts were wrong on the first attempt — a guard
that had moved, an entry point renamed instead of the behavior reverted, a no-op edit, an anchor
matching three places — and **every one of them produced a plausible-looking "NOT GUARDED"**. A
sweep that trusts its first revert reports false failures at roughly the rate it reports true ones.
So: a `NOT GUARDED` verdict is a hypothesis until the revert is confirmed to reintroduce the
defect the row actually names, and `docs/turbotab/tools/revertprobe.py`'s distinction between
*red for the wrong reason* and *green* exists for exactly this.

`IMPORT-265` is the case: `[1, '1']` column labels no longer crash `execute_stack`, and there is
nothing to close it with — `execute_stack` contains no duplicate-label handling at all, so there is
no change to revert, and the audit that recorded the crash ran on pandas 3.0.3 while
`requirements.txt` caps this repo below 3 for `T0-LIVE-004`. The cap is a ceiling this project
intends to lift.

This is `MINE-001`'s argument moved one layer out. There, a defect was unreachable because
`reconcile` raised first, and the row was correctly kept open — the accidental guard later moved and
the defect returned. A library version is the same kind of guard: incidental, external, and on a
schedule somebody else controls. **Record the version the measurement was taken on**, or the next
reader cannot tell a fix from a coincidence.

**Corollary — a specification clause with neither an implementation nor a tracked row is the same
silence the register exists to prevent.**

The register tracks what the doors **do**; the ledger tracks what they **should**. Keeping those
separate is what stops the register becoming a wishlist — when L13 tried to file "the impossibility
pass runs before the seal" as a register row, `register.py check` refused it, because every valid
state (`both` · `classic-only` · `core` · `guided-only`) answers *which door has this* and neither
door did. That refusal was right.

But it exposes the gap between the two artifacts. A clause of a constitution can be written, agreed,
and then land in neither: no implementation, and no finding either. Nothing fails, nobody is lying,
and the clause is simply not happening — which is exactly the shape of failure this whole family
describes.

**So every clause maps to a passing test or an open finding, and that mapping is itself checked.**
`tests/test_every_clause_is_tracked.py` reads the clause headings out of `ROADMAP.md`'s two
constitutions and `ASSEMBLY_SPEC.md`, and fails on any clause that names neither. The check is
deliberately dumb about *quality* — it cannot tell a good test from a weak one — and precise about
*existence*, which is the property that was missing.

Two things it must not become:

- **Not a coverage percentage.** A clause with one honest open finding is tracked; a clause with
  four vague tests may not be. Counting them would reward the wrong thing.
- **Not a reason to write a stub test.** A clause nobody has built gets an *open finding*, which is
  the truthful record. Satisfying the check with an empty test would be the silence rewritten as a
  green line — the `KNOWN_GAP_` problem one level up.

**Corollary — a helper that enforces an invariant needs a test asserting its call sites.**
Principle-locality says state a rule once. Its unstated half is that stating it once is only safe
when *using* it everywhere is checkable.

`_scoreable_here` in `utils/test_lockbox.py` is the fifth instance. It exists because held-out is
not the same as scoreable, its own comment says so in exactly the right words — *"reporting the
sealed count is a number a researcher would write down and be wrong about"* — and it is called at
**one** site, inside the cohort-run branch. The ordinary path prints the sealed count unconditionally
(`STATE-102`). The principle is stated correctly and applied in one place out of two, and nothing
fails when the second place forgets.

The family, all five of them silence rather than failure: principle-locality, expiring guarantees,
fallback-path survival, the ephemeral pointer, and now this. The fix shape is the same every time —
**make the locality executable.** A test that enumerates the sites which must call the helper, and
fails when a new one does not, converts "remember to call it" into a red line. Where enumeration is
impractical, invert the dependency so the invariant cannot be bypassed: the call site asks the
helper for the number rather than formatting one itself.

### Two specific things to watch

- **The pedagogy layer** — `utils/theory_anchors.py` (532 loc) and `utils/theory_demos.py`
  (869 loc) are a 19-key registry pair with **no test asserting the keys match**, plus a
  substring-matching fallback that silently drops a theory link when a finding string is reworded.
  It is the most fragile intelligent feature in the app and the most likely to quietly not
  survive a rewrite.
- **Cohort runs and the replay engine** — the newest subsystem (`utils/cohorts.py`,
  `utils/replay.py`), entangled with the lockbox through two import cycles. A `Project` that
  models "the working table" without modeling "the active cohort filter" deletes it silently.
  Already flagged in `TRANSITION_PLAN.md` §05; repeated here because it is a *feature*, not just
  a state-model detail.

---

## The register — state model (L5 / L6)

Not a Guided *step*, but the layer both doors read. Registered because "extract
the core so the UI becomes a choice" only holds if the **state model** is shared
too: two state models is the same failure as two engines, one level down.

| Capability | Classic | State | Reason |
|---|---|---|---|
| Step-completion model (ten predicates) | `utils/theme.py:685` | **core** | Extracted to `turbotab/readiness.py`; `theme.py` now asks instead of computing, and a test asserts the expressions are gone rather than merely unused. This is the Router's readiness function. |
| Quick / advanced disclosure split | `utils/theme.py:685` | **core** | Moved with the predicates — it decides which questions are optional, which the Router inherits. |
| Invalidation cascade | `utils/session_state.py`, plus a copy in `pages/03` | **core** | `turbotab/cascade.py` declares the graph once and reproduces the production function key for key across all four flag combinations. The `pages/03` copy is **not yet reconciled**: it misses fifteen result keys, and the DAG names them. |
| Partial invalidation (`clear_feature_*=False`) | `utils/session_state.py` | **core** | Expressed as `keep={stage}` and pinned per flag, because a naive full-cascade DAG cannot express it. |
| Session save / restore (zip archive) | `utils/session_manager.py` | **both** | `turbotab/archive.py` ports the schema — same members, same names, same version — so the doors read each other's archives. |
| Sealed lockbox, held by row label | `utils/test_lockbox.py` | **both** | Same dict shape, so one lockbox satisfies both doors; round-trips verbatim. |
| Active cohort filter | `utils/cohorts.py` | **both** | First-class field on the project, with `working_table` derived from it, so "which rows is this number about?" has one answer. |
| Per-model pipeline **specs** | `pages/05` (stores fitted objects) | **guided-only** | Guided stores serializable specs and has no global fallback slot. Classic still stores fitted pipelines behind a global slot that lets two models alias one instance (`TRANSITION_PLAN.md` §02.1). A debt owed back, and a live defect there. |
| Identity barrier (`T0-ID-001`) | — | **guided-only** | Classic has no phase rule: nothing stops a structural repair running after the lockbox is sealed, and the stale lockbox still looks well-formed. Owed back. |
| Serialization guard (no participant data in the record) | — | **guided-only** | New. `session_manager` drops derivatives for safety and cost; it does not check that what remains is free of cell values. Owed back. |

Three `guided-only` rows, all debts rather than luxuries — and two of them
(`pipeline specs`, `identity barrier`) describe defects that are **live in
Classic today**, not merely absent from it.

> **Note.** The Data & Target step register added in `9211566` is not in this
> file any more; it was dropped in `b4bff25`. Not restored here, because that
> may have been deliberate — but a step with no rows is the failure mode this
> register exists to prevent, so it needs either restoring or an explicit entry
> saying where it went.

---

## The one-line answer

**The algorithms are safe — they were always engine code and both doors will call the same
functions. The orchestration is not, and neither is exposure. Those two need a register, or
"ported" quietly becomes "most of it."**
