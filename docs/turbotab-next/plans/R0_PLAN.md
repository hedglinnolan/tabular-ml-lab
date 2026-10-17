# R0: the claims registry's scaffold and its first 15 claims

Planned 2026-10-16 on turbotab-next @ a78e535a, from CLAIMS_REGISTRY.md §7 (ruled 2026-10-15).
Two waves, each kept under Nolan's 3M line (ORCHESTRATOR §2): R0a builds the registry with no
real claim in it; R0b registers the 15 S1 claims and splits the colliding constants. R0b is sized
from R0a's measured spend before it starts. At stage 1's rate (about 0.3M fresh tokens per agent
run), R0a is about 1.2M and R0b about 2.5M.

## R0a · the scaffold (one package)

| Owns | Tier | Agents |
|---|---|---|
| `turbotab/core/claims/**` (new), `turbotab/core/reference/claims.py` (new), `docs/turbotab-next/reference/CLAIMS_REFERENCE.md` (generated), `turbotab/core/tests/test_claims_*.py` (new) | Opus builder, Opus adversarial verifier | builder, verifier, up to 2 repair rounds |

It touches no engine module, so no fitted number and no sentence a user reads can change.

**What it builds** (CLAIMS_REGISTRY.md §1, §1.1, §2.1, §2.2, §2.4, §3, §5's history, §6):
- The schema: `Claim`, `Part`, `Regime`, `Param`, `Basis`, `Fires` and `Silent` with their declared truth, kinds and grades as §1 and §2.4 give them.
- A `Registry` object, with one module-level default. Claims modules under `core/claims/` are found by a package scan, so no list of modules exists for two later packages to share.
- `card(id, ctx=None)`, `methods(id, ctx=None)` and `value(id, param)`. Strict mode, on in tests, raises when `ctx` fails a regime predicate. Outside it, the named fallback claim renders. A proposed claim never renders.
- Predicates and enforcers as `"module:function"` strings, resolved only when checked. Importing `turbotab.core.claims` loads no other `turbotab.core` module.
- Status derived, as §6 gives it: proposed, built, or retracted (the only declared one).
- Meta-tests 1–6 (§6) as functions over a registry. One test file runs them on the default registry, and each is also run on a planted registry that breaks it and on one that keeps it.
- The digit gate (§3), static and typed, over every claim template outside its placeholders, every `Relation.says` and every teaching string. It starts from an allowlist of today's digit-bearing strings, each tagged, and refuses a new one.
- The generator, `python -m turbotab.core.reference.claims [--check]`, which writes CLAIMS_REFERENCE.md and appends a claim's old version to `core/claims/history.jsonl` when its hash changes. "Ruled, not built" lists the proposed claims.

**Acceptance (test first, both ways):**
1. Each meta-test fails on a planted claim that breaks it and passes on a conforming one. Meta-test 2's planted test imports its claim's constant.
2. Strict mode raises outside the regime; non-strict renders the fallback; a proposed claim refuses to render.
3. A claim whose `enforced_by` does not resolve, or whose test does not exist, is proposed; resolving both makes it built.
4. A planted string with a new digit fails the gate, a placeholder passes, and a digit inside a citation or a tagged name passes.
5. `--check` exits 1 on a stale reference and 0 on a current one; a changed hash appends the old version to a temporary history file.
6. A subprocess import of `turbotab.core.claims` loads no other `turbotab.core` module.
7. The fast core tier passes, and `git diff` touches only the files R0a owns.

**Stop when** the seven hold and the verifier lists no problem; at most 2 repair rounds, then report.

## R0b · the 15 S1 claims and the splits (sized after R0a)

- The 15 S1 claims of notes/claims-backtest/FIXES.md, re-located in today's code (stage 1 fixed them), each with its parts, regime predicates, `Fires` and `Silent` tests (stage 1's known-truth tests, plus a `Silent` test wherever one is missing) and `enforced_by`. D-086 is a design-doc claim and stays proposed.
- The splits: `trials.few_clusters` (10, Li & Redden 2015) and `inference.few_clusters` (30, Cameron & Miller); `tuning.small_sample_standard` (300) and the boosted-trees and forest rule now named `SMALL_N` (500). Each constant takes its value from its claim, and each sentence that states it is filled from the claim.
- A-001's "300" in the family reason strings is generated from its claim. The TuningDecl fields stay, and a test pins every tuning plan's seed before and after.
- Grades: a builder grades only judgment or unsupported. Its verifier, who did not write the claim, proposes each higher part's grade with its evidence, and the repair round records it with `graded_by`.
- Probably two packages split by area, so no two own one engine file: inference, trials, causal, survival and designs in one; tuning, baseline, materiality, data recognizers, omics and usual intake in the other. Then one integrator, who regenerates METHODS_REFERENCE and the packets.

### Carried from R0a (landed 2026-10-16)

R0a took 8 agent runs and three repair rounds, against two planned. The orchestrator ruled the
third, on five items, and fixed the last two minors itself. It cost 1.74M fresh tokens (71M cached
re-reads), against about 1.2M planned. Each of its four verifier passes found new defects (5, 4,
4 and 2), none of them a narrower edge of an earlier one. At that rate R0b is about 3M to 3.5M,
over the 3M line, so Nolan is told before it starts.

**R0b does these, since its real claims need them:**
- Pass `ctx` wherever the data exists. With `ctx=None` a regime's predicates are skipped, even
  in strict mode.
- Engine code renders through `card()` and `methods()`. It never calls `claims.status(id)` at run
  time: that derives from the source, so in the shipped app, which holds no tests, it reads every
  tested claim as proposed.
- Add a test that the image and the release zip keep `turbotab/core/claims/status.json`. Both
  keep it today, which was checked by simulating them with rsync.
- Turn strict mode on for `turbotab/server/tests` too. Today only `turbotab/core/tests` has it.
- Re-tag the allowlist entries that a registered claim now covers. For example, "9 for don't know
  or 999 for not measured" (`scales.py:44` KNOWN_CODES) is tagged *example*, and "Hold out 10%"
  and "a 200 kcal swap" are tagged *name*.

**Filed in INBOX, not in R0b:**
- **Unconditional skips the status check misses:** `pytest.skip()` in a test's body,
  `pytest.param(..., marks=pytest.mark.skip)`, an always-true `skipif` (`skipif('True')`,
  `skipif(1 == 1)`), and a mark bound to an alias.
- **Derivation paths:** `truth='derivation:../x.md'` and `Basis.derivation` are not checked for
  leaving the root.
- **History integrity:** `--check` passes after an old history line is edited with its hash kept,
  or deleted. A changed hash does not bump the version, so `(id, version)` is ambiguous without
  the hash. An un-retracted id given new content reads as a revision, not a reuse.
- `check()` does not require a rule to name its decision, and meta-test 5 skips `decision=''`.
- **Meta-test 2:** it misses an id built by concatenation, and a constant imported through a
  helper module.
- **Digit gate edges:** "½" passes, since `str.isdigit` is false for it. "(Friedman 2001-5000
  trees)" passes through the year-span tail. A future `cited_as` pattern with a negated-class gap
  inside a work's name would be misread, and loudly.
- **Grader dates and parentheticals:**
  - Grader dates are not bounded: 2999-01-01 registers.
  - The author's own parenthetical is dropped, so the author "the R0a builder (Opus)" and the
    grader "Opus, 2026-10-16" pass. Decide this when real authors are written.
- **Parsing:**
  - The parse cache keys on `mtime_ns` and size, so a same-size rewrite within one timestamp
    tick can be read stale. This affects derivation only.
  - The test lookup's "last binding" flattens `if` and `try` branches.

### R0b landed (2026-10-16)

R0b registered 27 claims, not 15: the 15 S1 claims, their splits (`trials.few_clusters` and
`trials.few_clusters_extended`, `inference.few_clusters`, the four tuning claims, the forest's and
boosted trees' small-rows rules), and the fallbacks their regimes needed. 25 are built; D-086's
two (`data.tail_share`, `results.tail_share`) stay proposed.

**Cost:** 31 agent runs and 9.76M fresh tokens (about 570M cached re-reads), against about 3M to
3.5M planned. It ran as three packages (inference, models, data) with a build and three repair
rounds each, then one finishing builder and verifier, then one repair round. Its 14 verifier
passes found 10 blockers, 12 majors and 55 minors. Why it ran over:
- The claims were real engine guards, not wording. Writing a regime's predicates apart from the
  code it guards found code defects behind the claims: the trial core's estimating equations
  refused for the wrong reason, omics' name rule silently imported nothing, an infinite cut-off
  was accepted.
- One repair overreached: round 3's fix to start the estimating equations from standardized columns gave
  SEs up to 21% off on ill-conditioned columns. The pre-committed fallback (refuse, with a true
  reason) applied, and that commit (2806a492) stays on `feat/r0b-inference` for R1.
- Two of my own rulings were incomplete and cost a round each: the omics name rule (R2, re-ruled
  as omics' own list) and its NHANES names (RIDRETH3, RIDAGEMN).
- A package verifier that did not run the core shard missed a failure the integrator found.

**Settled on the way, for R1 and later waves:**
- A user-read refusal states only what its test shows: where the linear program cannot decide,
  the refusal says the fit did not settle, never that someone's risk runs to 0% or 100%.
- A claim's name list lives as a tuple in its claims module, which the engine imports, until a
  typed name-list parameter exists (INBOX).
- Any edit to an allowlisted template also changes `digit_allowlist.json`, so a package that
  rewords a claim owns that file too.
- Every package verifier runs the core shard once: `test_plain_words` scans the claims modules.

**Handed to R1** (INBOX, "From R0b"): V11, the few-clusters relation the digit gate cannot join;
the route to reporting trials whose raw-column start fails (2806a492 plus a standardized sandwich);
the claims the engine does not yet render with ctx; the unregistered readers of
`inference.few_clusters`; and the S1 engine items the claims surfaced, each for its lane's wave.

## Rulings

None needed from Nolan: the registry's three questions were ruled on 2026-10-15, and the rest is methods and architecture, which he delegated.
