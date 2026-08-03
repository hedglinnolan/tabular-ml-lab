# Running this as an unsupervised loop

The operator's manual: how a loop is shaped, how to run one safely, how to judge the report that
comes back, and what has already run.

**If you are taking this project over, read in this order:** `README.md`, `PRODUCT_VISION.md`,
`ROADMAP.md`, then this file's §02 (the shape), §06 (adjudication) and §03 (the log). The rest is
reference.

`data/findings.json` is the source of truth. `FINDINGS_LEDGER.md` is generated. The agent edits the
JSON through `tools/ledger.py` and regenerates the markdown — never the reverse.

```bash
python docs/turbotab/tools/ledger.py stats            # progress
python docs/turbotab/tools/ledger.py next --n 15      # next batch, as JSON
python docs/turbotab/tools/ledger.py set ID --status OPEN --note "..." --evidence "file:line"
python docs/turbotab/tools/ledger.py regen            # rewrite the markdown
python docs/turbotab/tools/ledger.py check            # schema guard; non-zero on violation
```

`check` enforces what makes the ledger trustworthy: no duplicate ids, no invalid status, **no
`FIXED` without a named regression test**, and no `PARTIAL` / `NOT-A-DEFECT` / `WONTFIX` without a
written reason. It runs in the pre-commit hook, not on discipline — see §05.

---

## 01 · Two numbering systems, and why they are not renamed

`L1`–`L12` in `ROADMAP.md` name **phases**. `L1`–`L29` elsewhere name **loops**.

They coincided through `L8`, when one loop completed one phase. They diverged at `L9` — the
interaction layer — which is one phase and many loops. **Every loop from `L13` onward sits inside
roadmap phase `L9`**, alongside the domain track that `ROADMAP.md` §"The map" now carries.

The collision is documented rather than renamed because the references number ~100 each way and
reach generated files and source. A rename is churn with breakage risk; this paragraph is the fix.

**Reading a loop number in the wild.** An `L` tag on a finding marks **where the defect was
observed**, not where it landed — `GUIDED-046` carries both `L18` and `L24` because that is one
cause with two occurrences. To reconstruct a loop's contents:

```bash
python3 -c "import json,re,collections; d=json.load(open('docs/turbotab/data/findings.json')); \
r=d if isinstance(d,list) else d['findings']; \
[print(f['id'],f.get('item','')[:70]) for f in r if 'L24' in str(f)]"
```

`VALUE_CHECK_ADJUDICATION.md` carries a section per loop that moved a routing metric, and it is the
most detailed narrative record that exists. It is **not** a complete loop log — it only sees loops
that moved the value check — which is why §03 exists.

---

## 02 · The shape of a loop

A loop is **one prompt, run unattended, reporting once.** The prompt is four parts, and the shape
emerged from practice rather than design — it is written down here so it stops being rediscovered.

| Part | Role | Why it is there |
|---|---|---|
| **A** | close the previous loop's gap | The adjudicator names one thing the last report left. Doing it first means an accepted loop never quietly carries debt into the next. |
| **B** | the substantial build | The loop's reason for existing. One thing, deep. |
| **C** | a second build, deliberately different | Two builds of different shape expose an abstraction's seams. One build never does. |
| **D** | a probe, an audit, or a refusal | Something that tests whether what exists holds, rather than adding to it. |

Each part is independently gated: it lands with its own tests, its own ledger rows, its own commit.
**A loop that completes three of four parts is a loop that produced three parts**, not a failure —
which is why the parts are ordered by what would hurt most to lose.

**Rules that make the shape work, all learned by breaking them:**

- **Name the source, not the goal.** *"Build the nutrition detectors"* hands the agent the job of
  deciding what the science is. *"Implement the Atwater reconstruction, source of truth
  `research/NUTRITION_PACK.md` §01, read that section first"* does not. See §04.
- **Two, never three, when testing an abstraction.** The figure spec shipped with exactly two
  figures on instruction, and the second exposed two seams the first did not. A third before the
  spec survived both would have hardened the wrong shape.
- **And stop metering once it has stopped bending.** The rule above is about *discovery*, and read
  as a general cap it is expensive. The packs specify **eight** signature figures (`DOMAIN_SCIENCE.md`
  §02), **ten** import detectors (§03b) and ~150 anti-patterns, all written down to the annotation
  and the threshold. Releasing those two per loop is a schedule invented by the prompt rather than
  by the work. **The test for which phase you are in is not a count — it is whether the last example
  bent the abstraction.** L27's third figure did (`annotations` bent, `tier` did not fit), so figures
  were still in discovery at three; the next one of a *shape already seen* is fill-out, and a
  fill-out part should carry as many instances as the agent can hold. Product owner's framing:
  *"why stop at 2 plots when you know they could make all n plots that work well for a given card."*
- **Order a batch hardest-first.** The corollary that keeps a widened part honest. Five instances
  built easiest-first are five castings of a shape nobody stress-tested; built hardest-first, the
  seam shows while four are still unwritten. Judge "hardest" by what is most likely to break the
  abstraction, not by effort.
- **Say what not to build.** Every loop since `L25` has carried an explicit *deliberately unbuilt*
  clause. Agents finish things; the instruction to stop is load-bearing.
- **Scope note at the end.** When one part is allowed to run long, say which and what may be traded
  against it. *"If C runs long, ship the figure with fewer annotations rather than dropping the
  refusal."*

---

## 03 · The loop log

Terse by design. The narrative lives in commit messages, `VALUE_CHECK_ADJUDICATION.md`, and the
findings themselves; this is the index that makes them findable.

| # | What it did | Why those parts together |
|---|---|---|
| L1 | Verified 370 findings against current code | A backlog nobody has re-read is research, not a plan |
| L2 | Three live Streamlit bugs | Folded into L7 |
| L3 | Walking skeleton — upload → diagnosis → findings → decisions | Prove the architecture before building on it |
| L4–L7 | Characterization tests, AnalysisProject + DAG, split block, detaint + job queue | The engine had to be safe to call before a new door could call it |
| L8 | The Router, EDA only · **value check passed** | Routing is the differentiator; measure it before building on it |
| L13–L17 | `GUIDED-011`–`021` — early Guided hardening | Reconstruct via the snippet in §01 |
| L18 | Split "whether to scale" from "which scaling" (`GUIDED-022`) | Determinacy is a property of a **question**, not an operation. Added the `constitutional` question category rather than moving the coverage denominator |
| L19 | The lens becomes the third constitutional question | |
| L20 | The lens reaches `rank_findings` (`GUIDED-024`–`029`) | Reframing annotates and never deletes — a lens at generation would make presented and executed diverge |
| L21 | Pack benefit fell because the comparison became fair | A metric moving the wrong way for the right reason is recorded, not corrected |
| L22 | `GUIDED-031`–`033`, found by driving | |
| L23 | The drive bugs — `DRIVE-001`/`003`/`004`/`011` | The product owner drove the app; the lens was unreachable |
| L24 | Bulk repairs (`DRIVE-002`) — nine cards, one decision | Value check penalized the improvement twice; recorded, no threshold moved |
| L25 | Purpose question (`GUIDED-048`), evidence badge (`047`), sixth axis (`045`), SMOTE defect (`049`) | The spine primitives everything downstream needs, plus the first anti-pattern audit hit |
| L26 | Prereg Amendment 1 (`050`), figure spec + two figures (`051`), promotability (`052`), study-scoped finding (`053`) | The figure spine, tested by two deliberately different figures |
| L27 | Nutrition pack — Atwater, NHANES design, shrinkage plot, the EAR/AI **refusal** | The reference implementation. A pack that can only add findings has not been tested. **Accepted; all four parts correct and tested, and none of it is reachable from an upload** (`GUIDED-058`). The calibration fix is the part that ships — `weak_calibration` is live in Classic via `pages/06` |
| L28 | Reachability — the badge reaches the claim, the figure layer gets its first consumer, the detectors reach an upload, the refusal is probed from outside | L27 built content nothing could reach. **Accepted; `GUIDED-057`/`058`/`059`/`060` closed**, and an upload now draws two figures and triggers four refusals, verified through the API by the adjudicator rather than through the loop's own tests. The loop's sharpest finding came from the wire and no unit test could have: the dietary pack read a 107 kg body-weight column as a survey weight and asserted a partially-specified design on a clinical table. **Two seams found by widening, not by building** — `evidence.py` grew from one file to an AST walk over every emitter, and the `Callable[[df], Optional[Dict]]` detector contract was **split rather than widened**, on the argument that widening rests on one example |
| L29 | Three more figures, the matcher that read the wrong columns, and the first anti-pattern audit | **The first widened loop** — three figures at once rather than two, ordered hardest-first, on the rule added to §02 this cycle. It worked: the volcano bent the spec exactly where predicted (a precondition on *upstream data state*, which `when_applicable` cannot express) and the spline bent nothing, which is the other half of the result. Five of six figures now reach a user. **The audit is the headline**: 85 anti-patterns checked, 4 hit, and `AUDIT-001` is the SMOTE defect's exact shape — the generated manuscript reports how many tests were significant at raw *p* < 0.05 and names no correction |
| L30 | The audit turned on the app — the manuscript's uncorrected count, the leaking baseline, the exits that named no key | **The loop where the research stopped being content and became a bug report.** `AUDIT-001` removed a false statistical claim from the *generated manuscript* and `AUDIT-002` closed a real leak in the quick baseline, both in code Streamlit users run today. `GUIDED-064`/`072` fixed a class I had half-named: the machine-readable form was lossier than the sentence, in a badge and in a 409's exit. **The most valuable line was in prose and is now `AUDIT-008`** — four of the last four hits share a shape the registries were never written to find: the core already holds the capability and the path that needs it does not read it. Also the loop where **I** broke §05 by running `git add -A` mid-loop; see the rule added there |
| L31 | The recipe lattice prototyped, the purpose-blind refusal, the capability sweep, and the frontend harness | **The loop that measured a design question instead of arguing it.** The lattice prototype changes 6 of 16 cells when the lens moves — reproduced by the adjudicator — and establishes that whatever renders it must mutate cells in place, because `transitioncancel` never fires on a removed element, so a wholesale repaint cannot report what it interrupted. That narrows `GUIDED-073` without closing it. **And the harness earned itself on its first run**: the page never fetches `/figures` (`GUIDED-075`), which two loops of API verification — mine included — structurally could not see |
| L32 | The figure surface, the server's exits, two Classic leaks, and the copy deck walked | **The money maker became visible.** `GUIDED-075` closed and the adjudicator drove it — the page's own controller, live API responses, a real dietary project — and the figure surface renders three admitted figures with annotation boxes, scored checklists, captions and badges. `GUIDED-076` found something worse than discarding: `new Error(object)` stringifies to `[object Object]`, so a 409's exits were destroyed at the throw. **And the loop named the door's dominant defect class from three instances** — the server composes a user-facing string and the interface never renders it. Measured at `GUIDED-080`: six surfaces, and the sharpest is `nutrition/prevalence`, the refusal apparatus the whole domain-track ordering was justified by |
| L33 | The class closed — the prevalence refusal, the seal's basis, the Features step, the lattice, and a standing check | **The dominant defect class stopped being a finding and became a gate.** Five surfaces the server composes and the page never read now reach a person: `nutrition/prevalence` (the four refusals `LOOP.md` §04 says the whole domain-track ordering was justified by, plus the shrinkage plot the single-day refusal offers), the `disclosures` field, `/features` and `/selection/evidence`, and `/recipes`. **The seal was the closest to a governing-rule failure**: it tells a user whether their holdout was drawn by person or by row, and §03's `undetermined` had no renderer at all. Two findings the loop made rather than fixed: `GUIDED-081`, the harness silently dropped every `className` write, so any assertion about styling came back **passing** — the §03 claim would have passed against a page that got §03 exactly backwards; and `GUIDED-084`, `/capabilities` is served precisely so the page cannot claim an affordance the server lacks, and the page computes its own. **The measurement was wrong and the check corrected it**: `evidence/correlations` and `evidence/histograms` were counted unread because `runPull` composes their paths from a variable and no literal grep can see them — seven routes unread, not nine, five of them read by another door |
| L34 | The instrument fixed, then Preprocess, Train and the seal probed | **The loop where the harness stopped lying and immediately found an orphan.** `GUIDED-077`'s two holes closed at the source, and the sweep's first result was that L33-D's lattice claim had been measuring a node the page never attached — `latticeBox` held 0 characters while `latGrid` held 179. Three surfaces were repainting behind the same blind spot. **Two steps reached their end**: Preprocess, with §07's fork asserted as a property (no strategy on screen until the mechanism is answered), and Train — the first number this door computes rather than reads, on a watchable job with a cancel that stops it. `GUIDED-065` closed: the calibration plot is drawn from held-out predictions, eight loops after it was specified. **And the seal was probed rather than asserted for the first time** — seventeen probes, the fit clean to the last bit, and the leak beside it: `GUIDED-088`, the model shelf ranked on the whole table, so the order a user picks from was computed with the held-out rows in view. Nobody had thought of a ranking as a parameter estimated from data. It is one. Two more found by driving rather than reading: a categorical fill silently retyped a numeric column (`GUIDED-086`), and the blocker's safe exit rendered disabled beside a live *do it anyway* (`GUIDED-087`) |

| L35 | The connective tissue — a recorded decision reaches what it was recorded for | **The loop where the record stopped being a place decisions go to rest.** `pipeline_plan.py` is the executor clause §06's deferred half never had: the fold-fitted pipeline is composed from the missingness declarations, the recipe variants and the deferred transforms, with the sentence and the pipeline asserted as **identity** rather than agreement. Driven, not read — a user who answers *informative* and picks *indicator* gets a fit that leaves the blank on a model that reads one, and on `logreg`, which cannot, the run says *"For Logistic Regression only, missing values in `mz_0003` were filled with the training-fold median, because Logistic Regression cannot be fitted around a blank"* — per model, carrying both the recorded sentence and the true one, and it reaches the page. Undeclared columns are **counted** rather than dressed as choices. Three revert probes, each failing for its own reason. **The probe is the other half of the loop**: 41 decision kinds enumerated — the adjudicator's grep had said 36 — 23 probed, 6 reaching the fit, 11 allow-listed and 7 unconstructible, every reason in the file and the partition asserted exhaustive. On its first run it found `GUIDED-099`: the pack recipe table is process-global, so a clinical project sharing a process with a metabolomics one was fitted with Pareto scaling it never asked for — *a display defect until Part B made the trainer read the table*, which is the sharpest illustration of what wiring a capability does to the cost of a latent one. **And the loop declined to decide `GUIDED-096`**, correctly: post-seal Explore reads the held-out rows, five surfaces move under a poison probe, and two of them are orders a user picks from. It registered the exemptions so they fail if they quietly become masked, and found that `PRODUCT_VISION.md` §04b already draws the line for a different question. Accepted; `GUIDED-095` reopened to `PARTIAL` by the adjudicator because feature selection declares `scope=train_folds` and nothing fits it |

| L36 | The journey reaches its end — selection fits, Explain, Report, and a run that goes stale | **Upload to manuscript, without leaving the Guided door.** Selection became the last recorded decision to reach the fit, closing `GUIDED-095`: the selector sits between the shape stage and the model, probed by moving the held-out values and watching the chosen columns not move. Three defects it found rather than reasoned about — the selector ranked the shaped matrix and chose a top-3 from 244 one-hot columns rather than the six nominated; pandas output raised on a name collision numpy had hidden; and `mutual_info_*` draws from numpy's global state, so the same project selected different columns depending on what ran first. **Explain kept the register's promise** — it names which Features decision made the ranking harder to read, rendered *above* the table, because a reader who learns afterwards that the rows are principal components has already read it. **Report made the transcript the manuscript**: the preprocessing plan reaches the methods section quoting the record's own string, safe only because L35-B made that string true of the fit. **And the veil is per artifact rather than a flag** — a selection can be stale while a run fitted afterwards is not, and the explanation refits per request so its ranking is never stale while the scores beside it can be. Accepted; `GUIDED-104` filed beside it — the app records `train_folds` and fits `train_rows`, then corrects itself in a note, when `selection.declare` exists precisely so the weaker claim can be *said*. The loop also caught its own revert probe passing when it should not have, which is now the standard |

| L37 | The seal says what it can resolve, the fork was routed not built, and two coverage gates | **Three of four parts existed because a document said something the code did not.** `GUIDED-102`: the seal drew a constant 15% and stated only that the rows were held out — at n=80 that is 11 rows and a C-statistic on 11 rows answers nothing. The trigger is **derived, not picked**: the widest 95% interval this holdout can produce against the whole distance from a coin flip to a perfect classifier, `2·1.96·0.5/√n > 0.5 ⟺ n < 15.37`, and it fires on one of six fixtures — `PRODUCT_VISION.md`'s own worked case and nothing else. A third candidate trigger was **measured and rejected**: *parameters > training rows* is true arithmetic that fired on five of six, dominated by per-row identifier columns handed to the model as predictors, and was filed as `GUIDED-108` rather than laundered through a card. `MISC-014`: §03's claim that the sensitivity fork was *"absent from the app entirely"* was false — ~700 lines ship in Classic — corrected in place with the lesson attached, *a capability that ships in one surface is not absent, it is unrouted*, and neither of the Classic page's landmines inherited. **The gates found more than the findings did**: seven primitives measured where four were named, turning up a fifth with zero rows anywhere (figure tiering), and eleven pages where two were named. `GUIDED-109`–`112` filed. **Four existing guards caught this loop's own work** — the evidence gate refused five invented source anchors, the archive guard refused a serialized class label, the spelling gate refused a commit, and the surface guard failed on `/sensitivity` being served to nobody, which is `MISC-014`'s own shape arriving inside its fix |
| L38 | The resampling engine, the manuscript as data, and a refusal | **Three of six parts, reported plainly as three.** `GUIDED-103` closes: the whole pipeline including selection refits over `B` resamples, `B = 200` stated in both captions beside the 1000 Riley & Collins recommend. **The revert probe found a defect by not failing** — the first version passed each resample's seed to both the plan and the estimator, produced a convincing 17 distinct feature sets across 20 resamples, and the probe that should have destroyed it did not, because the signal was the estimator re-breaking ties on identical data. *An instability plot is an attribution claim*, so everything that is not the sample is held fixed now, and the finding closes on a **pair** because only a pair is evidence. §A4.8 writes "MAPE" unexpanded and the percentage reading returned 658% on risks near 0.02, so it reports in the prediction's own units. **Wiring the validator found three failures that are not formatting**: the methods prose states no analysis population, reconciles no split counts, and states no final predictor count while the abstract asserts one — `AUDIT-001` in the artifact that leaves the building, and `GUIDED-116` names the cause, that `draft.py` folds over decisions while both missing sections describe a *run*. **Kaplan–Meier refused and the refusal is the result**: the app cannot represent a time-to-event outcome, no survival target type was invented, and the entry names the anti-pattern before the build — §A4.6 is `SETTLED` that 1 − KM overestimates cumulative incidence under competing risks. Accepted; `GUIDED-104` **not** upgraded, against the prompt, and the agent was right — resampling measures how stable a selection is and does not make the reported model fold-local |

| L39 | The reassuring figure, the finished manuscript, and the zero that is the finding | **Three of four, with the fourth traded whole rather than half-built.** `GUIDED-114` closed by *measuring* the thing the finding predicted: the row bootstrap reports **21.4% less median interval width** than the cluster draw on `clinical_longitudinal.csv` — the understatement, in the predicted direction — and the scheme is now disclosed in the payload, both captions, the page and a checklist item, with a **second revert probe that draws clusters correctly and says nothing, and fails.** `GUIDED-108` closed by driving: a one-hot `record_id` plus a decision tree fits a coin flip at 1.00 apparent accuracy and collapses to one constant prediction on unseen ids. **Driving also overturned the adjudicator's premise for that row** — `is_id_like` needs an integer dtype and answers `False` for all four string columns in the row's own evidence (`GUIDED-120`), and *unique-per-row* is not the rule either, because it flagged 88 `mz_*` columns, the study's actual predictors. **The export audit changed the answer to a question the prompt only implied**: the first `to_latex` passed 9 of the exporter's 22 arguments while the app held seven more, so a Guided manuscript shipped a methods section and an abstract and silently dropped the metrics table, the predictor list, the recorded limitations, the importance ranking and the resampling results — *nothing failed; the document was thinner than the analysis.* **And the multiclass sweep's headline is a zero**: 14 surfaces driven, 11 correct, 3 silently wrong, **0 refusing** — nothing anywhere declines a multiclass target, so every failure looks exactly like a right answer. The sweep carried a binary control, which is what caught `GUIDED-124`, a defect that has nothing to do with multiclass |

| L40 | The trigger learns its arity, Table 1, eight figures, and the companion that was never registered | **Four of four, accepted, and the find is the best in twenty loops — but its characterization was wrong in the direction that flatters it.** The report said an unresolvable companion *removed* the calibration figure for six loops; **driven, it degraded it.** The figure was computed, rendered on the page with its title, caption and all seven annotations, under a `fig held` warning naming a companion `discrimination` that has never existed — so the app spent six loops telling every user it was withholding its flagship clinical figure pending a figure it does not have and no user action could supply. That is the **assert-something-false** branch of the governing rule rather than the silent one, which makes `GUIDED-128` worse than reported, not better. **`GUIDED-065` stands correctly closed**: *drawn* and *admitted* are different predicates, its test asserts the data path it was filed about, and scoping that test to `admitted` would have coupled a data-path regression to the companion registry. **The guard that should have caught it is the one that hid it** — the companion test satisfied the admitted branch with `{"discrimination": {}}`, a bare dict key `bundle()` skips on a line marked `# pragma: no cover`, so the test manufactured the production object whose absence was the defect. Named in a code comment, not filed; filed by the adjudicator as `GUIDED-134`, and it is **not** the guard-testing-its-own-description class — the assertion is right and the *fixture* is wrong, which is why reading assertions never finds it. **And the layer under all of it**: `GUIDED-131`, the admissibility rule has no consumer at the boundary it was written for — a held CONFIRMATORY figure promotes into the manuscript with `passed: True` and the document silent, because `promote_figure` reads the whole registry and never reads the bundle. `GUIDED-124`'s widened scope **accepted** — twelve literals, not two, resolved by derivation rather than by twelve edits, which is the right answer to a ruling that undercounted. `GUIDED-125`'s threshold move **accepted under §06.2**: the gated *quantity* changed, no assertion was relaxed, the one test edit preserved the count's history rather than shrinking a denominator, and the arithmetic re-derived independently — 15.4 / 8.6 / 6.8. **The zero that was reported as closed is still open one surface over**: `RANKS_AND_STATES` came back 0 because all seven surfaces swept are *output* surfaces and the middle rung is a property of *choice* surfaces — driven, the model shelf ranks a three-class target byte-identically to a binary one (`GUIDED-132`). Also `GUIDED-133`, found one line past L40's own net-benefit rounding fix: the decision curve's risk rug is capped at the first 200 rows in row order and nothing says so, on a figure whose whole argument is that a reader can tell 3 patients from 300 |

| L41 | Eight clinical detectors, the survey pair, the stand-in sweep — and two critical reachability defects found on the way past | **Four of four, accepted — and the loop's finding is not in the four parts.** Going outside them twice was correct and both were blockers for the part they were found in: `GUIDED-139`, `nudge()` deleted at `DRIVE-006` with seven call sites outliving it, so **every pull affordance in the door threw `ReferenceError`** — *push the notable, pull the rest* is one of five design principles and its half was dead; and `GUIDED-142`, **no pack finding in any pack had ever been rendered** — five packs, eighteen detectors, computed, served on `/project/{id}`, invisible. `DOMAIN_SCIENCE.md` §03's own correction says the content bin *is* the product and that scaffolding is what a researcher does not pay for; the content had reached zero users. **Two criticals, one loop, both reachability, both found by accident while doing something else** — which is the real result: the parts were fine and the door was not. Driven by the adjudicator on `clinical_labs.csv`: 8 of 8 clinical findings reach `profList` carrying SETTLED and CONVENTION badges. **And the drive corrected the adjudicator first** — a stubbed `interview?step=explore` left the page with no Explore step to paint into and `profList` came back empty, which is the absence-claim trap the handover names, arriving on the loop that had just closed the stand-in version of it. `GUIDED-143` **ruled and split**: the false assertion stops next loop via the lockbox constitution §03 pattern — a basis that states what was actually drawn, the way `undetermined` is first-class and never rendered as a clean lock — and the chronological grouped draw is a separate build, so the row stays `OPEN`. **The question is not removed; that would shorten the shelf.** `GUIDED-144` **ruled**: the conservative count stays, 140 mmHg is not adopted (§A1.2 is explicit that reference intervals are not disease thresholds), and reading the core rather than forking a second bounds table was right — **the defect is one layer down and is `MISC-018`**, `ml/physiology_reference.get_reference_interval` returns a p01/p99 pair under a name CLSI EP28-A3c defines as the central 95%, and `get_impossibility_band`'s own docstring says *improbable* where the function name says *reference interval*. The code knows what it holds; the name asserts otherwise, to both doors. **The divergence line corrected the adjudicator a fifth loop running, twice**: `GUIDED-135`'s premise was compressed rather than wrong — the checklist had passed against two numpy arrays and never against a payload from a project, which the row's own evidence said and the prompt's summary dropped — and the companion count is seven, not the four the prompt asserted |

| L42 | The field granularity, the delegated controls, and three standing checks catching their own author | **Four of four, accepted — and the report's headline is that the loop's own subject caught the loop.** Three checks fired during verification and all three would have shipped: the archive whitelist dropped the four `temporal_*` keys A1 had just built, so a restored project would show **a clean grouped seal over a holdout the user asked to have drawn chronologically** — the worst shape this archive has, because the keys that survive a lossy save are the honest ones; A1's tree sweep asserted only absences and would have reported a clean rename over a tree it never read, caught by `test_an_absence_assertion_carries_a_positive_control`; and a Part A script wrote `findings.json` at `indent=2` where `ledger.py` writes `indent=1`, so four commits carried an 18,844-line phantom diff over 38 real ones. **Nothing caught the third, and the adjudicator had done it first** — commit `5881aa8` carries a `docs(turbotab):` subject and 18,554 reformatted lines. The rule is now in §05 and `AGENT_ONBOARD.md` §06: *a file a tool owns has exactly one writer.* **`GUIDED-145` is the loop's best find and it is a third variant of the green-test-over-a-broken-thing class** — `test_temporal_prediction_routes_to_the_chronological_strategy` asserted a composer's string and its sentence, both true, and the word *routes* supplied a claim no assertion touched. Not trap #2: there the assertion is about the description; here the **name** promises what the body does not check. **B measured the gap it was built for**: 2,117 fields on the clinical lens, 358 reaching a person, 35 families with a fully-unread shape, 19 exempt with a named reader, 16 filed. Enumeration derived and tested by adding a live field. **And Part C — the tradeable part, not traded — is what established that B is insufficient**: B's positive verdicts are sound (a sentinel in the DOM is a sentinel in the DOM), its negatives rest on group-negative bisection, and its blind spot is a sentinel appearing inside an error render — which is `GUIDED-139`'s exact shape and which only C catches. **Neither instrument subsumes the other, and that is only knowable because the tradeable part was kept.** `GUIDED-147` **ruled one row** with two conditions — its `item` says fifteen and its `ev` lists sixteen, in a row whose whole content is a count; and it splits when someone looks rather than closing on a verdict over fifteen unexamined things. `GUIDED-140`'s static half **ruled wanted**, on the agent's own number: six of `nudge`'s seven call sites were on paths nothing drove, so the driven half catches one in seven. The unreachable `chronological_grouped` basis **stays** — §03's *three states, never two*, and a basis set omitting the honorable state cannot say the app is missing it — but its `# pragma: no cover` becomes a test, because a pragma is a claim with no guard and this project has been bitten by one before. **And re-counting the report produced `TEST-040`, which is the only place the report and the measurement disagree and the report is not at fault**: `turbotab/` came back 1 failed / 1422 passed against a reported 1423 / 0, on the same commit with the same ordering, because four job-polling sites spin a bounded loop with **no wait** and assert `status == "done"` — so under load the suite reports *the app had not answered yet* as *the app answered wrong*. The count this project reports every loop is load-dependent, which makes it a record that cannot settle anything; `TEST-030` is the same axis producing false passes instead of false failures |

| L43 | The anti-pattern audit run against shipped code, the chronological draw, and four of nine divergences that were findings | **Four of four, accepted — and the divergence section stopped being a caveat list and became a source of rows.** Item 1 is `TEST-042`: an `IndentationError` committed **green**, because none of the five pre-commit gates parses Python — and the loop's own new guard reads source with `read_text` and regexes, so *a file that does not parse is still text and still matches.* The generalization is the agent's own and it is the grep lesson at the gate layer: **a check whose input is a serialization of the thing it cares about will pass on a broken input.** Item 6 became trap **#3c** — `test_low_epv_keeps_lineup_small_and_cites_numbers` asserted `"class weights"` on a low-EPV profile, precisely where §A5.2 says the remedy is penalization, so **fixing the defect required a passing test to fail**; a loop that chased the red would have reverted the correction and recorded the revert as a success. Item 4 became `TEST-043`, three self-referential guards in one loop. And item 3 is the best use of the revert probe this project has recorded: removing the `temporal=` argument left the lockbox still reporting `chronological_grouped`, because `seal_lockbox` read *what was asked* rather than *what was drawn* — the probe found `GUIDED-143`'s own defect **inside its fix**, and the draw outranks the answer now. **The audit's headline is `AUDIT-014`**: `GUIDED-049` is `critical`, `FIXED`, and its fix reached **three of ten shipped surfaces** — `ml/publication.py` still shipped the removed sentence character for character, and a derived sweep found an eighth surface the original agents missed. **The class was not filed and the adjudicator filed it as `MISC-019`**, because it is L42's granularity lesson arriving at the ledger itself: `ledger.py check` enforces that a `FIXED` row *names* a test, and a named test proves the fix works at the site the test covers — so a row describing ten surfaces whose test covers three passes the gate and reads as done, with 286 of 758 closed and nothing telling one from the other. **Two corrections to the report, both in the rows.** `AUDIT-015`'s cause is *not* `GUIDED-131`: at `f2266a9`, the commit that wrote the gate, `figures` was already the whole registry, so **the gate was a tautology on the day it shipped** — trap #2 at birth rather than after drift, which is the harder version because there is no before-state in which it worked. And `GUIDED-049`'s note recorded none of its own incompleteness, corrected in place. Also `GUIDED-148` `high` — `/recipes`, 50 fields, **every one unread**, which is `GUIDED-142`'s shape at a route the route-level check cannot see; and `TEST-041`, the ordering twin of `TEST-040`, where **the full-suite green is the false one** |

| L44 | The sixth gate, eleven of twenty-one, and the first measurement of how much of *closed* is closed | **Four of four, accepted — and two of the loop's corrections are to rows the adjudicator filed or amplified.** `MISC-019`'s sweep put a number on the project's own progress measure for the first time: **124 closed rows examined, surfaces re-derived from code, 116 complete, 4 partial, 3 partial-claims refuted, 1 undecidable — a 3.2% partial rate**, with the frame stated (61 critical plus 63 plural-item, 161 unexamined) and the right caveat attached: *a rate measured on the highest-yield slice is an upper bound on the remainder, not an estimate of it.* Reassuring, and it is the first time anyone could say so. **`GUIDED-148` was false and the agent withdrew its own finding.** `/recipes` does reach a person — at the seal it renders *"Nothing is missing — the step that fills it has not happened"*, an honest placeholder that explains the gap in the user's own journey position, and after `select_models` it renders the full lattice. Driven by the adjudicator on both positions: 272 characters of placeholder, then **4,791 characters of per-model grid**. §08 check 5 firing on the loop that quoted it — *the sweep terminated where the sweeper's attention ended* — and **the adjudicator had repeated it to the product owner as fact without driving it**, which is the second time this session a finding was propagated because it fit the narrative. `MISC-018` was the adjudicator's row too, closed at L43 by a fix that renamed the identifiers and left the phrase in **five user-facing strings**, the sharpest an EDA header reading *'Reference Interval (NHANES p01–p99)'* — a label naming the central 95% printed beside the central 98%. Completed here, and it is `MISC-019`'s class finding its own filer. **`IMPORT-109` is the sharpest thing Part D found**: the row is about `join_doctor`, its named regression test is entirely about `import_doctor`, and `ledger.py check` was satisfied by a test that could not have failed if the fix had never been written — four entry points still raise the exact `AttributeError` the item names. Filed as `TEST-044`, the mechanically detectable variant, **with the obvious fix measured and rejected**: comparing test filenames against evidence modules fires on 127 of 284 FIXED rows, so the gate has to read what the test *imports*. **Part B is eleven of twenty-one and three were reverted after being applied** — `AUDIT-030` turned a green test red and the test is trap #3c, so adjudicating it means deciding what the manuscript should say instead; the agent reverted rather than ship what it could not verify, and reported eleven rather than claim fourteen |

**Adding a row is part of adjudicating a loop.** Two lines, written when the report is accepted.
This log decayed once because it lived only in chat; that is the failure this project has already
paid for twice, in two different places.

---

## 04 · Loops that build a domain pack

The five research threads in `docs/turbotab/research/` are **4,247 lines and are the authoritative
source** for every pack detector, coaching sentence, threshold and figure specification. They are
not background reading. A loop that builds pack content without citing them has invented its
content, which is the failure this whole apparatus exists to prevent.

Three problems, solved differently.

**Volume.** Nobody holds 4,247 lines. **The task block names the file and the section**, and the
agent reads that slice. A task block that says *"build the nutrition detectors"* without section
pointers is malformed.

**Provenance.** Every pack advisory, detector and figure spec carries a **`source`** naming file and
section, and an **`evidence_status`** of `SETTLED` / `CONVENTION` / `DISPUTED`. A checker verifies
the named section resolves, and runs in `.githooks/pre-commit` beside the other gates. Its limit,
stated wherever it is stated at all: it verifies a source is *named and resolvable*, never that the
claim is faithful — the same posture and the same honest limit as `ledger.py check` enforcing that a
test is named.

**Where the research file and the model's recollection disagree, the file wins.** The files were
built under a blocked egress proxy and say so; a threshold in the file is a recorded, checkable
claim, and one from memory is neither.

**`[verify-at-build]` is a hard stop.** Such a number ships as an `offered` item with its
uncertainty stated, or not at all — never as a hard-coded constant. Where the fact is unavailable,
that is a `BLOCKED.md` entry, not a guess.

**Sequencing.** Four parallel domain verticals is the wrong shape: the packs share the figure spec,
annotation engine, badge rendering and checklist engine, so built four times they are built four
ways. **One pack end-to-end first as the reference implementation** — discovering the abstractions
is the deliverable alongside the pack. Nutrition went first because the product owner can adjudicate
its content, the NHANES fixtures are real, and it is the one pack that forces a **refusal**.

---

## 05 · Guardrails

Append this to any unsupervised prompt.

> **Hard rules.** Stay on branch `TurboTab`. Never push to `main`, never force-push, never open a
> pull request. `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and
> `pages/01_Upload_and_Audit.py` are **frozen — see `TRANSITION_PLAN.md` §05 for the one statement
> of what that permits and the gates that lift it.** Never edit `FINDINGS_LEDGER.md` by hand; it is
> generated. Never mark a finding `FIXED` without a regression test **verified to fail when the fix
> is reverted** — see `FEATURE_PARITY.md`, "the revert probe". **First command in a fresh clone:**
> `git config core.hooksPath .githooks`. Commit after every batch so nothing is lost. **Domain
> science comes from `docs/turbotab/research/`, never from recollection** — where a research file
> and your memory disagree, the file wins, and a number marked `[verify-at-build]` may not ship as a
> hard-coded constant. If you are blocked or something looks structurally wrong, stop and write what
> you found to `docs/turbotab/BLOCKED.md` rather than guessing.

**One writer at a time, scoped to the shared data files** (`findings.json`, `register.json`, and
their generated markdown). A docs-only commit from another session may land mid-loop **only if** it
touches none of those files and its commit message says so; the loop agent rebases over it. Anything
touching the data files waits. That is the artifact this project has already lost once.

**The adjudicator never runs `git add -A`, and the rule is written here because it was broken here.**
Two commits during `L30` — `9ebf95d` and `7dd6aa6` — carry `docs(turbotab):` subjects about the
frontend stack and contain `turbotab/exits.py`, `turbotab/packs.py` and five other source files that
were the loop agent's uncommitted Part A at that moment. Nothing was lost and every gate stayed
green, but the subject line of a commit is a claim about its contents, and those two assert something
false about themselves — the governing rule failing in the record layer rather than in the app.
**Stage explicit paths, and run `git status` first, every time.** The failure was not carelessness
about the rule; it was not noticing that a conversation about architecture is a moment when a loop
may be running. The docs-only exemption above is what makes an adjudicator's commit safe mid-loop,
and `git add -A` silently converts a docs-only commit into a mixed one.

**A file a tool owns has exactly one writer, and it is the tool — and this binds the adjudicator
first, because the adjudicator broke it first.** `ledger.py` serializes `findings.json` at
`indent=1`. An adjudication script that dumps at `indent=2` reformats all nine thousand lines:
commit `5881aa8` carries a `docs(turbotab):` subject about the agent onboard and **18,554 changed
lines** in `findings.json`, thirty of them real. The loop agent then did the same thing in the other
direction at L42 and found it reading its own diffstat; nothing else caught either. A diffstat is a
claim about what changed, exactly as a subject line is a claim about contents — this is `git add -A`'s
lesson arriving through a different door. **Edit through the tool, or write the file back
byte-identically to how the tool writes it.** The older rule — *never edit `FINDINGS_LEDGER.md` by
hand* — failed to generalize because it named the **generated** file rather than the **owned** one.

**Verification loops run in a fresh session, not the builder's.** A builder verifying their own work
reads intent where the job is to read code — the same reason a review pattern never lets the finder
be the judge. And both write the same data files.

**The three gates are a hook, not an instruction.** `.githooks/pre-commit` runs `ledger.py check`,
`register.py check` and `tests/test_american_spelling.py`, and refuses the commit on any failure.
This replaced a line in the guardrails that enforced nothing: commit `8127101` went out with the
spelling test red because the gates were chained with a newline instead of `&&`, so a non-zero exit
did not stop the sequence. **An instruction a tired agent can skip by punctuation is not a gate.**
`core.hooksPath` is local config, so the one command above is the only part still carried by
discipline. Bypass with `--no-verify`, and say why.

**Run the documented setup path, or find out that nobody has.** The `Makefile` names
`./venv/bin/python` and nothing had created it in long enough that the spelling gate's skip list had
`.venv` and not `venv` — so the gate died on a compiled dependency the first time anyone followed
the instructions. A setup path is a claim like any other and decays the same way: silently, while
the people with working environments keep working.

**It decayed again, in the same file, and this paragraph did not catch it.** Found while adjudicating
L27 (`TEST-039`): `PYTEST_OPTS` passes `--timeout=60`, `pytest-timeout` is absent from `venv/`, and so
`make test` — and `make verify`, which line 13 of the `Makefile` calls the CI target — exits 4 having
run **zero** tests. Every count this project has reported came from a hand-rolled `pytest` invocation
instead: the numbers are real, and they are evidence about a command written down nowhere. The rule
above failed because it is prose. **Adjudicating a loop now includes running `make test` once** — the
only form of this rule that can actually fire.

**The freeze** and the three gates that lift it are stated **once**, in `TRANSITION_PLAN.md` §05. Do
not restate them; this file once said "never modify" while §05 said "engine-move-only", and a reader
following the stricter one could not do the work §05 permits.

### A capability ships with its consumer, or with a failing test that names the one it lacks

**The measurement that produced this rule.** Searching the ledger for findings whose text describes
a capability that exists beside a path that never reaches it returns **37 of 672** — four critical,
many high, several landmines. The count is not the finding. **The distribution is**, because it
spans all three eras of this codebase:

- `MODELS-005`, inherited Streamlit — the Cancel Training button is decorative; `cancel_training`
  is written and never read.
- `GUIDED-058`, `L27` — every surface the nutrition loop built was imported only by its own tests.
- `GUIDED-075`, `L31` — the page never fetches `/figures`.
- `GUIDED-094` / `GUIDED-095`, found adjudicating `L34` — `stale_downstream` written and never read;
  41 decision kinds recorded and 6 consumed by the thing that fits the model (the count read 36 until `L35-E` enumerated them properly; the adjudicator's grep had missed five).

Same defect, three codebases, years apart. This is not migration debt and it is not one bad loop.
**It is how this codebase has always been built**, and `AUDIT-008` named the shape without noticing
it was describing a habit rather than a cluster.

**The cause is an incentive gradient, and naming it is most of the fix.** A capability is gratifying
to build and *fully verifiable in isolation* — a green test can prove `figure_specs.py` correct
forever without anything ever calling it. Wiring requires the consumer to exist, and the consumer is
usually the next loop's work. So the pressure points at capabilities every single time, and the
suite stays green while the app cannot reach what was built.

**The rule.** A part that adds a capability ships **either** with the path that consumes it, **or**
with a test that names the missing consumer and **fails.** The second clause is the load-bearing
one: sometimes the consumer genuinely cannot exist yet — the calibration plot had no training step
for eight loops, and that was a correct sequencing call. The honest form of that is a red test with
a deadline, not a green suite over an unreachable module.

**The model already exists in this repository**, from `L34`:
`test_the_seal_holds.py::test_the_run_records_what_it_did_not_use` asserts the *limit* of the Train
slice, so the limit cannot be forgotten and closing `GUIDED-089` must change the test. That is the
pattern. It needs to be mandatory rather than admirable — which is the same move §06 made when
"adjudicate the report" stopped being a habit and became a written check.

**And the corollary for adjudication:** the §06 grep — *does anything outside a test file import
what the loop just built?* — is the detector for this class, and it only catches the import-shaped
instances. `stale_downstream` has importers; nothing renders it. Where the question is whether a
recorded thing reaches a consumer, **flip it and see if anything downstream moves.**

---

## 06 · Adjudicating the report

The half of this job that was never written down. A report is a **claim**, and the whole project
runs on the rule that a claim needs a record.

**Verify before accepting, and verify the load-bearing claim specifically.** Not everything — the
one thing the rest depends on. Pull the branch first; the agent's commits will not be in your tree.

```bash
git fetch -q origin TurboTab && git rebase origin/TurboTab
python docs/turbotab/tools/ledger.py stats          # do the counts match the report?
grep -c '"GUIDED-0NN"' docs/turbotab/data/findings.json   # was it actually filed?
make test                                           # does the documented gate still run? (§05)
```

**And one grep that is not about the report at all:** does anything outside a test file import what
the loop just built? A module reachable only from its own tests is a specification, and the report
will not say so, because from inside the loop it looks finished. This is how `GUIDED-058` was found.

**Then stop grepping and run it.** The adjudicator has now been wrong three times in the same way,
each time by searching for the shape expected rather than the shape written: `packs.findings` was
called from `engine.py` while the grep covered `api.py` and `ml/`; `AUDIT-001`'s sentence spanned two
f-string lines and a one-line pattern returned nothing; and `runPull` composes `"/evidence/" +
endpoint`, so a literal path search reported two surfaces unread that are fetched on every drive. A
grep answers *does this text appear*, and the question is almost always *does this run*.

The instrument already exists. `turbotab/pageharness.py` runs the page's real controller in node
against captured API responses, and `__harness.calls()` returns exactly which routes it fetched —
no pattern, no ambiguity. Feeding it live responses from a `TestClient` drive takes about thirty
lines and answers reachability questions that no search can. **Where a claim is about behavior,
drive it; reserve grep for claims that are genuinely about the file.**

**What to look for, in order of how often it has mattered:**

1. **Was a named defect *class* filed, or only its instance?** The highest-value finding of `L26`
   was a class the agent named in a docstring and did not file. A class that lives only in prose
   will be forgotten. **This is the single most common gap in an otherwise good report.**
2. **Did a threshold move?** Never accept a moved threshold in the same loop as the change that
   pressured it. If a gate is measuring the wrong thing, correct **which quantity is gated**, on a
   *passing* run, with the reasoning recorded before it is load-bearing. After a breach the same
   correction is indistinguishable from relaxing a gate under pressure.
3. **Does new numerical code have its own tests?** `weak_calibration` was hand-validated and
   exercised only through a figure test. A hand-check that is not a test is a claim without a
   record — the project's own rule, one level in.
4. **Does the code return a value where it should return nothing?** The strongest habit this
   project has: `(None, None)` rather than `(0.0, 1.0)`, because those are the values of *perfect*
   calibration and returning them from ignorance asserts perfection.
5. **Did a sweep terminate where the sweeper's attention ended?** Sweeps find the class they were
   pointed at. Ask what the same lens would find one surface over.
6. **Is a capability being deleted where it should be routed?** The SMOTE defect was fixed by
   routing behind purpose and keeping an offered path, not by removal. The shelf is never shortened.

**Then, before writing the next prompt:** add the §03 log row, and name in Part A the one gap you
found. If you found none, say so — an empty Part A is a real outcome and should be visible.

---

## 07 · Checking in

```bash
git -C . log --oneline TurboTab | head -30
python docs/turbotab/tools/ledger.py stats
python docs/turbotab/tools/ledger.py check
git diff main...TurboTab --stat
cat docs/turbotab/BLOCKED.md 2>/dev/null
```

Three questions worth asking of any result:

1. **Does the `FIXED` count have tests behind it?** `check` enforces that a test is *named*; it
   cannot verify the test is any good. Spot-check two or three.
2. **How many went `NOT-A-DEFECT`?** A high rate means either the agents over-reported or the
   verifier is credulous. Read those notes specifically — that is where a loop quietly goes wrong.
3. **Did anything land where it should not have?** `git diff main...TurboTab --stat` answers it in
   one line.

---

## 08 · What not to hand an unsupervised loop

- **Row identity, and choices like it.** Design decisions with consequences across the whole project
  model. An agent can gather the evidence; it should not make the call alone.
- **Large extractions of untested, safety-critical logic.** Supervised, with characterization tests
  already in place.
- **New construction under a governing rule about what may be asserted.** Design work, not loop work.

Loops are for verification, for well-specified builds with clear gates, and for writing tests
against behavior that already exists. They are not for decisions you would want to argue about.

But the reframe in `ROADMAP.md` holds: those items are not permanently off-limits. They are blocked
on **one decision each**. Make the decision and the execution behind it becomes ordinary loop work.

---

## 09 · Standing dispositions

Rules extracted from loop prompts that have since retired, kept because they still bind:

- **`FIXED` requires a named regression test, verified to fail on revert.** No test, no `FIXED` —
  the finding stays `OPEN` and the test is written later. Roughly one in five first-attempt reverts
  is wrong and produces a plausible false failure, which is why the probe must fail *for the stated
  reason*.
- **"Guided avoids it" is never closure.** Streamlit never retires, so a defect still present in
  Classic stays `OPEN` even where the core or the Guided door has structurally resolved it. Note it
  `resolved-in-core; closes at L11 convergence of <page>` — verbatim, because that phrase is the
  queue for the convergence loop.
- **Tag, don't fix, siblings of a known pattern.** Add `sibling-of: <ID>` and move on. They get one
  batched build, not twenty inline ones.
- **Ambiguity is `OPEN`, never `FIXED`.** A wrongly-closed finding is worse than an open one.
- **A record that points at ephemeral storage will eventually lie, and it lies toward "the work is
  gone."** The original 48 import findings were declared unrecoverable while sitting committed in
  `docs/audit/`. Cite paths that are in the repository.
