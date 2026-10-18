# R1: the registry's gates, its first batch and its fixes (swarm path, step 3)

Written 2026-10-18 by the orchestrator, under Nolan's go-ahead of the same day: "Do everything
right up until we are ready to start the swarm and I will have a discussion with you about it." It
carries out CLAIMS_REGISTRY §7's R1 and §8's gates, after R0's scaffold.

## TL;DR

- **What:** four parts, in this order:
  1. the gates a swarm claim must pass before it carries any authority;
  2. a list, in every export, of the claims that acted;
  3. the claims the NHANES Predict journey puts on screen, registered as the first batch;
  4. the 55 S2 and S3 fixes the backtest found, through the registry, the 34 real defects first.
- **Why:**
  - Nolan's biggest fear is slop in the middle layer. Today nothing stops a swarm claim from
    being promoted: no code implements a referee, a basis audit or a decoy check.
  - The analysts will read the app's cards and teaching. A false sentence there, such as
    "adjustment only sharpens precision" (D-032), would steer their decisions and so the corpus.
- **Cost:** honestly more than the 4–6M the swarm-first proposal gave R1. R0b spent 9.76M on 15
  claims. The gates and the batch are about 4–6M, and the 55 fixes another 10–15M.

## 1. What exists (a scout's map of 2026-10-18; the code is the truth)

- **R0's scaffold is complete:**
  - 27 claims (25 built, 2 proposed);
  - per-part grades;
  - six meta-tests, each tested both ways;
  - the digit gate;
  - the generated reference with `--check`.
- **Status and grades:** status is derived (proposed, built, retracted). A grade is an edit
  that meta-test 3 holds to "not graded by its author". A downgrade keeps no date or reason.
- **No promotion code exists:** no referee, no basis audit, no decoy queue and no false-promotion
  report. They exist only as the backtest's scripts in `notes/claims-backtest/scripts/`, with a
  hard-coded scratchpad path.
- **Few sentences carry a claim:** a sentence has no claim id on screen. Only 9 engine sites
  render through `claims.card` or `claims.methods`, and under Predict at most two of them fire.
- **The fixes are listed:** `notes/claims-backtest/FIXES.md` lists the 39 S2 and 16 S3 fixes:
  - 34 defects;
  - 12 conventions shown as findings;
  - 9 in design documents only.

  A sample of 10 still applies.
- **The path fuzzer** checks invariants on the fold, the route and the quest log over mocked
  artifacts. No claim's invariant is in it.

## 2. Rulings

### R1-G. The gates (CLAIMS_REGISTRY §8)
1. **Every claim has an `origin`:** engine, audit or swarm. A swarm claim enters `proposed`,
   every part `unsupported`, and renders nowhere (meta-test 4 already holds a proposed claim off
   screen).
2. **A grade above judgment on a swarm claim's part needs two referees.**
   - They must be two distinct graders, neither of them its author.
   - Neither may be a fork of the author (ORCHESTRATOR §3).
   - It also needs the grade's own requirement (CLAIMS_REGISTRY §2.4).
   - This is meta-test 7, tested both ways.
3. **Every grade change is a history line with its date, grader and reason,** downgrades
   included. The generator writes the lines, and meta-test 8 refuses a grade that differs from
   the last line.
4. **The basis audit:** a part graded sourced or above names who checked its basis against the
   source and when. The checker is not its author.
5. **The decoy queue:** a research tool, outside `turbotab/`.
   - It plants generated decoy claims beside real candidates in the grading queue.
   - It reports the false-promotion rate, generalized from the backtest's `score.py`.
   - It never uses the sealed decoy set v2, which is for the stage-3 audit and is never shown to
     any agent (ORCHESTRATOR §3).

### R1-M. The claims that acted
- The export's `provenance.json` gains `claims`: the id, version, grade and record hash of each
  claim that acted in the project.
- It is read from the claims render log, which SWARM_ENGINE_PLAN E4 writes in every mode and
  serves in analyst mode.
- A rule that acts without rendering (a refusal, a default) records itself through the same log.

### R1-B. The first batch
- After W1 part 2 lands, the server driver captures every sentence the NHANES glucose Predict
  journey shows, from the quest log, the stage artifacts and the methods.
- Each is indexed to the claim that rendered it, where one did.
- Each unmatched sentence is triaged by CLAIMS_REGISTRY §7's rule. It becomes a claim only if it
  states a number, sets a default, a ranking or a refusal, or teaches a method. Everything else
  is plain wording.
- Each new claim is registered with its parts, its tests both ways and an independent grade, and
  renders through `claims.card` or `claims.methods`.

### R1-F. The fixes
- The 34 defects come first, ordered by what an analyst reads:
  1. teaching and cards;
  2. the methods;
  3. the rest.
- Then the 12 conventions shown as findings.
- The 9 design-document fixes go to the documents' next edit.
- Each fix registers its claim and its sentence once (CLAIMS_REGISTRY §7), from a test that fails
  on the base commit.

## 3. Packages

| Package | Parts | Estimate |
|---|---|---|
| RG | R1-G and the decoy tool | 2–3M |
| RM | R1-M, after EC's render log merges | 0.5–1M |
| RB | R1-B, after W1 part 2 lands | 2–3M |
| RF-1, RF-2, RF-3 | R1-F's 34 defects in three batches by area; then one batch for the 12 conventions | 8–12M |

**Each package:** an Opus builder, an adversarial verifier and at most two repairs. Files are
named by grep before launch.

**Order:**
- RG runs first, beside the engine packages.
- RB waits for W1 part 2's landing, and RM for EC's merge.
- The RF batches run two at a time, after RG, since they register claims under its rules.

**Out of scope:**
- the mutation check (R2, CI-only);
- the stage-3 audit (R3).
