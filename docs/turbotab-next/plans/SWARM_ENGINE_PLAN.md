# What the engine must serve the swarm (swarm path, step 2)

Written 2026-10-18 by the orchestrator, under Nolan's go-ahead of the same day: "Do everything
right up until we are ready to start the swarm and I will have a discussion with you about it." It
builds SWARM_PLAN §8, which was written for W2–W4 to carry. Nolan's swarm-first ruling of
2026-10-17 moves it ahead of them.

## TL;DR

- **What:** five things the server does not do today:
  1. keep a record of every fit;
  2. serve the results as one data document;
  3. say which decisions moved the result;
  4. run in an analyst mode, which refuses a re-seal and logs what the analyst saw;
  5. record where each data file came from.
- **Why:** the swarm's analysts work through the API and never see a screen. Each run's corpus
  record (SWARM_PLAN §4) is assembled from these, so each must be the engine's own data, not the
  analyst's word.
- **How:** three packages in parallel, split by the files each change lives in. Each gets an Opus
  builder, an adversarial verifier and at most two repairs, then one integration.
- **Cost:** about 7–9M fresh tokens planned; this week's rate says 10–13M.

## 1. What exists (a scout's map of 2026-10-18, file:line in its report; the code is the truth)

- **Fits:**
  - Every fit artifact is kept on disk under `cache/fit/<key>/`, and nothing deletes one. Nothing
    indexes them, and their meta holds no plan and no decisions.
  - `fit_pressed.json` keeps only the newest press.
  - The plan's digest is logged only at the Estimate lock.
  - `open_seal` logs each family's held-out scores.
- **Results:** `GET /projects/{pid}/stages/{stage}` serves the fit, effects and estimand artifacts
  as JSON, gated as the screens are. No route joins them into one result.
- **Materiality:**
  - `GET /projects/{pid}/materiality` serves the noticings' ledger, not the user's decisions.
  - `methods/spec_curve.py` ("Which of my decisions mattered?") is a finished method core wired
    to no stage or route.
  - `materiality.paired_folds` has no caller.
- **Analyst mode:**
  - None exists: `Settings.mode` is local or server.
  - `Reseal` is an ordinary decision, and an opened seal's refusal offers it as its first exit.
    A driver taking first exits would re-seal by itself.
  - Claims render through `Registry._render` with no log. Some render inside worker processes.
- **A premise of SWARM_PLAN §2 fails:** it lists the analyst's reason as `DecisionRecord.note`.
  The field exists, but no route accepts a note: every decision model forbids extra fields, and
  `POST /decisions` takes none.
- **Provenance:** the ingest records a blake2b fingerprint, the source's name and size. The
  export's `provenance.json` has each input's SHA-256. Nothing records a URL, a release date or a
  license.

## 2. The contracts (rulings)

### E1. A record of every fit
- **One record per distinct fit that completes:** a fit key that turns fresh, in completion
  order, written once and never rewritten.
- **Each record holds:**
  - the record's id;
  - the purpose;
  - the fit key;
  - the plan's digest (`plan_lock.digest(plan_lock.plan_of(state))`, defined under either goal);
  - the decisions in force (their seqs);
  - the press in force;
  - the families;
  - the fit's seconds and workers;
  - per family, either the cross-validated summary (Predict) or the exposure's estimate with its
    interval (Estimate).
- **Served at `GET /projects/{pid}/fits`.** Each number is withheld exactly where the stage route
  withholds it today: no held-out score before the seal, and no estimate before the lock under
  Estimate. The test reads both ways.
- **The movement under Predict:** a record whose folds are the previous record's (the same split
  and seed) carries each family's movement against it, by `materiality.paired_folds`. Where the
  folds differ, it carries no paired movement and says why.

### E2. Results as one document
`GET /projects/{pid}/results` joins what the screens show, gated as they gate it:
- **Under Predict:**
  - each family's cross-validated summary;
  - the declared result;
  - after the seal, the held-out scores with their intervals and the calibration.
- **Under Estimate:**
  - the estimand;
  - the exposure's effect with its interval, from the effects and fit stages;
  - the E-value;
  - the sequence fits.

Each field is the served artifact's own value, never recomputed. A test compares each one with
the stage route's.

### E3. Which decisions mattered, under Estimate
- **What it computes:** after the lock, the specification curve over the recorded choices, one at
  a time by default (`crossing="one_at_a_time"`), as a stage job, not in a request.
- **The alternatives (v0):**
  - each recorded exclusion rule, dropped;
  - each recorded adjustment covariate, dropped;
  - each covariate the adjustment card offered and the plan left out, added;
  - each alternative the analyst declared as examined (E4).
- **The primary stays the reported estimate** whatever the curve says (spec_curve's own rule).
- **Served** at `GET /projects/{pid}/mattered`, refused before the lock with the line saying when
  it opens.
- **Tests from known truth:**
  - statsmodels for the unweighted fits;
  - R's `survey` package for a survey design, honored as a domain, as spec_curve does.

### E4. Analyst mode
- **Switched on by** `Settings.analyst` (a CLI flag and `TURBOTAB_ANALYST=1`). Off by default.
  Nothing changes for a person.
- **In analyst mode:**
  - a `reseal` is refused, with a plain message and no exit offering one;
  - an opened seal's refusal drops its Reseal exit;
  - each claim rendered is logged (id, register, frozen status, grade, fallback used) to a
    project-side JSONL file, which works from worker processes;
  - each preview is logged (the decision previewed, when);
  - `GET /projects/{pid}/analyst-log` serves both logs.
- **In every mode,** `POST /decisions` accepts an optional `note`, the reason in at most 2,000
  characters. It is stored in `DecisionRecord.note` and served in the decision log, never in the
  methods.

### E5. Where each file came from
- An upload accepts optional `url`, `release_date` and `license` fields, kept on the file's and
  the project's meta and echoed into the export's `provenance.json` inputs.
- The ingest serves each input's SHA-256 beside its fingerprint.

## 3. Packages (files named by grep, per ORCHESTRATOR's ownership rule)

| Package | Contracts | Files (the builder confirms by grep) | Estimate |
|---|---|---|---|
| EA | E1, E2, E1's movement | `server/service.py` (press_fit and a record writer), `server/routes/projects.py`, `server/schemas.py`, a new `core/runs.py`, `core/materiality.py` (`paired_folds`'s caller) | 2–3M |
| EB | E3 | a new `core/mattered.py` and its stage in `core/stages/__init__.py`, `methods/spec_curve.py` (read), `server/routes/projects.py` and `schemas.py` for the route | 2–3M |
| EC | E4, E5 | `core/config.py`, `server/__main__.py`, `server/service.py` (decide's note, preview log, upload fields), `core/claims/_registry.py`, `core/seal.py` (the Reseal exit), `core/workspace.py`, `core/datastore.py`, `core/export/source.py`, `server/schemas.py` | 2–2.5M |

**Integration:**
- All three touch `service.py`, `schemas.py` and the routes, in different functions.
- The integrator merges them and regenerates `openapi.json` and `generated.ts` by their script,
  never by hand.
- EA's records carry E3's and E4's links by id only, so no package waits on another.

**Base:** turbotab-next once W1 part 2 lands.

**Out of scope:** the driver, the analyst agent type, the corpus record's schema and the intake
tooling (swarm path step 4). They are built on these.
