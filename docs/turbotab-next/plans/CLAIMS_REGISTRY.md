# The claims registry

**DRAFT 2 for Nolan's review, 2026-10-15.** It is not ruled yet, so by its own rule (§7) every part of it is proposed.

This draft was rewritten after an independent critique tested draft 1 against the backtest's 15 S1 defects. Draft 1 would have caught none of them outright and only partly exposed 5. Draft 1 aimed at invented precision and tests that pin the code to itself. Both problems exist in the code, but neither caused an S1. This draft is organized around the failures that did, and §10 says plainly what it still would not catch.

## 0. What the backtest says went wrong

The claims backtest is in notes/claims-backtest/ (e6e58809). Wherever a number was checked against an independent implementation, the number was right. The 15 S1 defects come from four failures, plus one condition that let them persist:

| failure | S1 cases |
|---|---|
| **A guard that never fired**, or fired only on the path its test took | B-435 (Cox monotone likelihood), B-281 (separation), C-123 (constant with blanks), B-211 (an exit path drops the design's refusals) |
| **A rule stated for a wider regime than its evidence** | D-058 (certified for linear outcomes, wrong for yes/no), A-102 (concern lines borrowed from the linear family), A-029 (a floor on the wrong scale for rare events), D-086 (the identity holds; the threshold does not) |
| **A grade nobody checked independently** | B-416 (Kutner, cited for positivity), A-124 ("calibrated" on 9 in-sample cases), C-170 (a judgment that contradicted the plan) |
| **A ruling that outran the code**, with the old behavior still live | A-001 (the small-sample standard was ruled, but the library defaults stayed in force) |
| **Domain facts** | BX-040 (UL facts wrong for 5 nutrients), C-143 (the 2021–23 phlebotomy weight), C-193 (the repo's own "ask" fixture missed) |

**The condition is sediment.** A claim today is spread across a module constant, an f-string, a hand-written relation sentence, a teaching section, a doc paragraph and, sometimes, a test. Nothing holds those pieces to one another:
- `trials.FEW_CLUSTERS = 10` is interpolated into one concern, while its relation says "fewer than 10 clusters" in hand-written words (methods/trials.py:121, :2047). `models/inference.py:46` has a different `FEW_CLUSTERS = 30`.
- `estimand.FIXED_EFFECTS_LOGIT_ROWS = 16` appears again as "16 rows per group" in teaching/content.py:1344.
- `tuning.SMALL_N = 300` appears as "300" in four family strings, while `boosted_trees.SMALL_N = 500` is a different rule with the same name.
- Seven grade vocabularies say how sure we are, and none is checked.

Sediment did not cause most S1s, but it is why no one could see them. A claim with no single home has no single place to check.

## 1. What a claim is

A **claim** is a statement the app relies on to set, rank, refuse or explain a decision. There are three kinds:
- **rule:** a guardrail, refusal, routing or default ("below 10 clusters, the corrected GEE names the few-clusters concern");
- **parameter:** a number with a unit ("10 clusters");
- **teaching:** an explanatory statement of fact about a method ("Kauermann–Carroll below a cluster-size CV of 0.6").

Interface copy with no factual content, purpose answers and labels are not claims.

The schema is the middle layer's (G, Q, R, D, O → V, M, with basis and test). This example is illustrative; its test names and citation key are not real yet.

```python
Claim(
    id="trials.few_clusters",        # stable and never reused; a retired id stays, marked retracted
    kind="rule",
    card="Fewer than {few_clusters} clusters: intervals can still be too narrow.",
    methods="With fewer than {few_clusters} clusters, the concern is named (Li & Redden 2015).",
    goal=("inference",),             # G
    regime=Regime(                   # R, the regime the EVIDENCE covers, not the one we hope for
        words="a cluster-randomized trial, GEE with a small-sample correction, a yes/no outcome",
        requires=("methods.trials:is_gee_corrected", "methods.trials:outcome_is_binary"),
    ),
    decision="trials.correction",    # D, a contract key or decision kind
    options=("gee_corrected", "mixed_kr"),          # O
    verdict="noted",                 # V, a leash rung, or "noted" for a concern
    parts=(                          # what the sentence asserts, each graded on its own (§3)
        Part("value", grade="sourced", basis=Basis(cites=("li2015small",), where="§3, design")),
        Part("direction", grade="sourced", basis=...),
        Part("scope", grade="judgment", basis=Basis(reason="Li & Redden simulated binary only")),
    ),
    params={"few_clusters": Param(10, unit="clusters")},
    tests=(Fires("…::test_nine_clusters_name_the_concern", truth="construction"),
           Silent("…::test_forty_clusters_say_nothing", truth="construction")),
    enforced_by=("methods.trials:estimate",),
    supersedes=(),
    graded_by="basis audit, 2026-10-15",            # not the author (§3)
    version=1,
)
```

Registering a claim takes about fifteen lines. The meta-tests (§6) run themselves.

### 1.1 Where it lives

The registry stands alone. Claims are declared in `core/claims/` modules, one per area (`claims/trials.py`, `claims/tuning.py`, …), which import nothing from the engine but `claims` itself:
- Predicates and enforcers are named as `"module:function"` strings, resolved when they are checked. A claims module therefore never creates an import edge, and registration has no import-order trap. The engine avoids top-level imports on purpose: contracts register lazily (trials.py:1837).
- The method contracts are the pattern to copy (stable ids, vocabulary checks at registration, `enforced_by`, a generated reference with `--check`). They are the wrong home: `DECLARING_MODULES` covers the home modules of only about 3 of the 15 S1 claims.

The engine points into the registry by id:
- A `Relation`, a family's `TuningDecl` reason and a teaching section each name a claim.
- Their `says` or reason text is **filled from the claim at registration** (`says=claims.card("trials.few_clusters")`). Every existing scanner (`cited_texts`, `test_plain_words`, the word budgets) therefore keeps reading real sentences, not ids, and none of them goes blind.
- A module constant keeps its name and takes its value from the claim (`FEW_CLUSTERS = claims.value("trials.few_clusters")`). The number then lives in one place.

**Two registers.** Each claim carries its card sentence and its methods sentence, so a fact worded twice for two readers (plain.py:23-28) is still one claim.

## 2. The four mechanisms, one per failure

### 2.1 Every decision-setting rule is tested both ways, whatever its grade

**For the guards that never fired.**

Every rule that refuses, settles, ranks or sets a default needs two known-truth tests:
- **`Fires`:** the rule fires where it should;
- **`Silent`:** it stays quiet where it should not fire.

This holds **at every grade, judgment included.** The grade is about the basis for the choice; the tests are about whether the code does what the claim says. Judgment was 46 of 98 sampled claims, and 6 of the 15 S1s (B-435, B-281 and C-193 among them).

Each test declares where its truth comes from:
- `construction`: data built so the answer is known;
- `independent:<library>`: R, statsmodels, or hand code that does not import the engine;
- `derivation:<notes path>`.

**A test whose expected value is read from the engine's own constant or output does not count.** A meta-test flags a claim test that imports the claim's constant. That is the self-pinning the backtest found (A-068's test could not fail; test_wp2_intervals.py parametrizes on `FEW_CLUSTERS`).

This is what stage 1 already did by hand: a failing test first, from known truth. Here it becomes a rule the code checks.

### 2.2 The regime is code, and a sentence cannot leave it

**For rules stated for a wider regime than their evidence.**

The regime the evidence covers is written as predicates (`regime.requires`). They are required at every grade above judgment, and they are recommended for judgment.

Two things read the predicates:
- **The app.** A claim's sentence renders, and its rule acts, only where every predicate holds. Elsewhere the app falls back to a named weaker claim (`fallback=`), typically "your call, here is what each choice does".
- **Strict mode.** In tests and CI, `claims.card(id, ctx)` raises when the context fails a predicate. An ordinal model rendering a linear family's concern line (A-102), or a yes/no outcome reaching a rule certified for linear outcomes (D-058), fails a test instead of reaching a user.

**This is where Lean enters.** A proved claim's theorem fixes its hypotheses, and those hypotheses become its predicates. The stratification-optimism formalization already corrected an informal claim this way: exact stratification needs the folds to divide the events, and E[KL] is infinite without clipping. Lean cannot prove empirical thresholds, domain facts or code bugs. For those, the predicates come from the evidence's own design, such as the regime a simulation or a paper covered.

### 2.3 Every path that reaches a decision checks its claims

**For guards bypassed by another path.**

The path fuzzer (core/tests/path_fuzzer.py) already walks seeded journeys and checks the surfacing policy's invariants at every state (I1–I11). One invariant is added:

> **I-claims:** at every state where a rule claim's decision is made and its regime holds, the rule's effect is present: the refusal stands, the concern is listed, or the default is the declared one.

B-211's exit path, which dropped the design's refusals, is exactly the case a direct `Fires` test misses and a journey finds.

### 2.4 Grades are assigned by someone else, per part, and checked

**For grades nobody checked independently.**

- **Graded by someone else.** The author of a claim may grade it judgment or unsupported. Any higher grade needs `graded_by` naming someone other than the author: the basis audit, a referee pair, a domain reviewer or Nolan. In the backtest, the independent code audit found 12 defects both blind referees missed, so both layers stay.
- **Graded per part.** One sentence often asserts several things: a value, a direction, a scope. Each part has its own grade, and the sentence shows the **weakest part it asserts**. D-086's identity is proved, but its "more than half" threshold is not. B-288's K − 2 is sourced, but the app's extension of it is not. The 30 clusters in inference.py sit inside Cameron & Miller's "20–50".
- **The requirements, by grade:**

| grade | requires |
|---|---|
| **proved** | A Lean theorem built in CI, or a published theorem. Its hypotheses are the claim's predicates (§2.2). A hand derivation counts once an independent reviewer has re-derived it, and its basis says "derivation". |
| **certified** | A test against an independent reference or known truth (§2.1) that would fail if the part were false. |
| **replicated** | Held-out evidence in the regime: cases not used to fit the rule, from at least two fixtures or datasets, with their rates recorded. Cases used to fit the bands do not count, and a rule without held-out cases is floored to judgment. This closes A-124 and A-125, which draft 1's mapping would have graded replicated. |
| **sourced** | A citation key that resolves, plus `where`: the passage, quoted by the grader, that says this in this regime. A key that resolves is not enough. Kutner's text exists; it says nothing about positivity (B-416). |
| **judgment** | Our reasoned call, with its reason, worded as a judgment ("we use", not "you should"). |
| **unsupported** | No basis found (§4). |

A grade moves up only through its requirement, and it moves down the moment that requirement fails. Each downgrade and retraction is recorded with its date and reason, never erased.

The seven old vocabularies map onto these grades:
- materiality's theorem → proved; calibrated → replicated only with held-out cases; convention → judgment; floored → unsupported;
- teaching's SETTLED → the grade of what it rests on. A disputed rule is never badged settled (D-037, D-021). An internal pack section is not a source: D-021's `CLIN_A53` cannot meet "sourced".

## 3. One home for numbers and words

**For sediment.**
- **Parameters** are read from the registry (§1.1).
- **Sentences** are templates over parameters, in two registers.
- **The digit gate** is static and typed. A test scans every template outside its `{placeholders}`, plus every `Relation.says` and teaching string, with `[0-9]`; `str.isdigit()` is true for "²". A digit must be:
  - a placeholder;
  - inside a citation;
  - or marked as a name (CR2, SHA-256, WTDRD1), a guideline item (TRIPOD+AI 13), an example, or a mathematical constant.

  On today's strings, about 6 of 30 digit-bearing relations are real thresholds, and about a third of teaching's are real parameters. The gate begins with an allowlist of today's false positives and refuses new ones. Spelled-out numbers ("half", "most") slip past it, so D-086's "more than half" needs its part graded (§2.4), not the gate.
- **The colliding names split first:**
  - `FEW_CLUSTERS` becomes `trials.few_clusters` (10, Li & Redden) and `inference.few_clusters` (30, Cameron & Miller).
  - `SMALL_N` becomes `tuning.small_sample_standard` (300) and `boosted_trees.…` (500).
  - The four r ≥ 0.3 copies serve four different decisions, so they need a ruling on whether they are one rule. They wait.

## 4. On screen, and calm

The calm ruling holds: one obvious click per screen, and every element earns its place.
- **The grade lives in the "why" layer,** as one phrase:
  - "proved";
  - "checked against R";
  - "Li & Redden 2015";
  - "our judgment";
  - "a convention we have not checked".

  It is never a chip on every card.
- **An unsupported claim may not refuse anything.** An unsupported refusal becomes "your call; here is what each choice does".
- **An unsupported default** is handled by its materiality, under SURFACING_POLICY:
  - high materiality: a Decide;
  - otherwise: kept as a labeled convention, shown as Confirm or For the record ("a convention we have not checked").

  Turning every unsupported default into a question would multiply Decides and undo the calm work.

## 5. Provenance

A project's provenance.json gains a **claims-in-force manifest**: the id, version, grade and content hash of every claim that acted on it.
- Decision records stay as they are. Stored sentences are still written once and quoted, and replay still writes decisions.jsonl byte for byte.
- The defects fire in stage outputs, not only in records (A-102, B-435, A-029, C-123), so the manifest is gathered from stage outputs as well.
- **Old versions are kept.** Each time a claim's hash changes, the reference generator appends the old version to `core/claims/history.jsonl`, and `--check` holds it current. An old project's manifest always resolves to the words and grade it ran under.

This is the standard Nolan applies to published papers' unstated decisions: every default that changes a number appears in the methods, with its basis. In the replication pilot, each paper's unstated decisions map onto registry claims. "The paper did not say" and "the app chose, because" become the same object.

**Materiality.** A rule claim names its materiality instrument, when one exists, and reads its M from materiality_calibration.json. That file keeps its generator, and claims only read it.

## 6. Status is derived, not declared

A claim is **proposed** until its `enforced_by` functions resolve and its tests exist. It is then **built**. **Retracted** is the only status anyone declares.

A ruling, Nolan's or mine, enters the registry as a claim with a pointer to where it was ruled. Until its code and tests exist:
- it shows as proposed;
- the generated reference lists it under "ruled, not built";
- any doc that describes it as current is visibly ahead of the code.

**Supersedes.** A claim that replaces another names it (`supersedes=("tuning.library_defaults",)`). A meta-test then fails while both are live on the same decision in overlapping regimes, as A-001's were: the small-sample standard was ruled while the library defaults stayed in force. The same meta-test catches two live claims on one decision with conflicting verdicts (C-170 against BEATS_PREDICT M4; A-114's 200 resamples against the app's own Carpenter rule).

**The meta-tests:**
1. every claim's tests exist and declare their truth;
2. no claim test imports its own claim's constant;
3. every grade meets its requirement, with `graded_by` set above judgment;
4. no proposed claim renders;
5. no two live claims conflict on one decision in overlapping regimes;
6. the digit gate.

## 7. Order of work

**R0, the scaffold, in one wave, after fix/s1-inference merges** (its A-102, B-281, B-416 and B-435 tests are needed):
- `core/claims/` with meta-tests 1–6 and the static digit gate;
- a generated CLAIMS_REFERENCE.md with `--check`, and history.jsonl;
- the 15 S1 claims, registered with their parts, regimes, tests and independent grades. They already have known-truth tests, so they exercise every field;
- the FEW_CLUSTERS and SMALL_N splits;
- A-001's "300" in the family reason strings, generated from its claim. The TuningDecl versioned fields stay put, and a test pins every tuning plan's seed before and after, so no fit changes.

**R1, the 39 S2 and 16 S3 fixes, done through the registry.** Each fix registers its claim and its sentence. That is why they waited: about 55 sentences are fixed once, not twice.
- The I-claims invariant joins the path fuzzer.
- The provenance manifest lands.

**R2, each W wave registers the claims it adds,** as part of its definition of done:
- The wave's reviewer checks them against the registry.
- The builder names its three most doubtful claims. In the backtest, 8 of the inventory agents' 9 suspect picks were defects.
- **The mutation check** joins here, CI-only and narrowed. It covers parameters graded above judgment. Each mutant runs in a fresh process (constants bind at import), and only assertion failures count as kills. The killing test must not import the constant or pin the boundary by hand. Results are cached by hash. It never runs on the dev machine.

**R3, the stage-3 audit** of the rest, ordered by materiality times weakness. Decoy set v2 is planted in the certification queue to measure the false-promotion rate.

**Seeding.** The backtest inventory's 991 rows are paraphrases (48 of 117 only partly faithful, 5 unfaithful), keyed by file:line that has drifted. Claims are seeded from what the user actually sees (`user_sees`) and from the audit's corrected statements, re-located in today's code. They are never copied in bulk.

**Not migrated:** the roughly 616 numeric constants wholesale. Only constants that reach a user-visible sentence, or that set a default, ranking or refusal, become claims. A numerical tolerance stays code unless it decides an outcome: COLLINEAR_TOL deciding a coding's tie is a claim; GEE_TOL is not.

## 8. The research swarm

The swarm writes observations to a corpus, not to the registry. Abundance belongs in the corpus, and authority belongs in the registry.

A swarm claim enters the registry as proposed and unsupported. It is promoted only through:
- its grade's requirement;
- the two-referee gate;
- the basis audit.

Decoys sit in the same queue, and the false-promotion rate is reported. Agents propose; only mechanical checks and independent graders certify.

## 9. Risks

| risk | handling |
|---|---|
| The static gates go blind to generated text | `says` is filled from the claim at registration, so the scanners read sentences, not ids |
| Stored sentences and byte-equal artifacts | Records unchanged; the manifest lives in provenance.json, which replay skips; METHODS_REFERENCE, the packets and refs.bib regenerate in the same commit |
| Same-named constants with different values | Split into separate ids in R0, before anything reads the registry |
| Moving tuning declarations changes seeds | Declarations stay; only their reason text is generated; a seed pin guards it |
| Import order and circularity | Claims modules import nothing from the engine; predicates are resolved lazily by name |
| Old projects' claim ids resolve to nothing | history.jsonl keeps every version |

## 10. What it would and would not have caught

The 15 S1s, under this draft:

| caught by | cases |
|---|---|
| tests both ways at every grade (§2.1) | B-435, B-281 |
| regime predicates and strict mode (§2.2) | D-058, A-102 |
| the path invariant (§2.3), given a case-control fixture among the fuzzer's journeys (today it has two: NHANES dietary and a metabolomics table) | B-211 |
| independent grading, per part, with held-out replication (§2.4) | B-416, A-124, D-086 |
| supersedes and the conflict test (§6), C-170 only if BEATS_PREDICT M4 is registered as a ruling | A-001, C-170 |
| **not caught** | A-029, C-123, BX-040, C-143, C-193 |

That is 8 to 10 of 15, against none for draft 1; the critique's own count, without the path invariant and strict mode, was 8. Each is "would have been surfaced for a person to see", not "prevented": a grader still has to read the source, and a test writer still has to choose the case.

**What still gets through:**
- **Domain facts:** BX-040's UL forms, C-143's phlebotomy weight. These are caught only by people who know the field:
  - the per-domain expert review packets before release;
  - the replication pilot, where real papers exercise real weights and real nutrients.
- **A known-truth case nobody thought to write:** A-029's rare events, C-193's fixture. "Ask the builder for its three doubts" and the path fuzzer's breadth narrow this gap; they do not close it.
- **A predicate with the same bug as the code:** C-123. Its predicate would call the same `constant_values`. Predicates must be written independently of the code they guard, and that is a review rule, not a mechanism.

The registry is the checker layer. The independent audit and the domain reviewers are the other layers, and the backtest showed all are needed.

## 11. For Nolan to rule

1. **Grades on screen:** in the "why" layer only, as one phrase (recommended), or visible on each card?
2. **Unsupported claims:** never a refusal; a default that changes much becomes a Decide; others stay as a labeled convention under SURFACING_POLICY (recommended). Or should every unsupported default become a question?
3. **The swarm's corpus:** kept apart from the registry, with promotion into it (recommended), or one ledger with statuses?

The rest are methods and architecture decisions, which Nolan has delegated to me. I will build R0 on approval.
