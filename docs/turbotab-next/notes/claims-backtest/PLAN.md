# Claims backtest: the plan, fixed before the inventory is read

Written 2026-10-14 at about 13:50 EDT, while the four inventory agents run and before any of their output is read. This file is not edited after the referees start; any change is added below the line at the end, dated, with its reason.

## Question

Nolan's biggest fear: slop in the middle layer undermines trust in the app's intelligence. The backtest asks two questions:

1. **The claims.** What is the app's existing middle layer worth? This covers every methods claim the app relies on to recommend, set a default, refuse, warn or explain. Each is graded by the standard we would apply to the swarm.
2. **The gate.** Can a referee process tell a true methods claim from a plausible false one? This is measured with planted decoys, so we have a false-promotion rate instead of a feeling.

## Phases

1. **Inventory.** Four agents (A prediction, B estimation, C gates and refusals, D teaching) state each claim in the claim schema with its location, its tier, the basis the repo gives, and any existing test that would fail if the claim were false. They do not judge.
2. **Selection.** The orchestrator dedupes. Every tier-1 claim (it sets, changes or blocks a user's decision or result) goes to referees. Tier 2 (text a user reads) goes too if the total allows. Otherwise tier 2 is sampled with a fixed rule (every k-th by id) and the count left out is reported. Tier 3 is listed and not refereed.
3. **Decoys.** Only the orchestrator writes them, after reading the inventory and before any referee starts. They are about 20% of the refereed set, with no fewer than 15. Each is a sibling of a real claim, perturbed in one of these ways:
   - **overreach:** the regime is widened past where the claim holds;
   - **direction:** an inequality, sign or ordering is reversed;
   - **constant:** a number is changed to a plausible wrong value;
   - **mechanism:** the conclusion is right but the reason is wrong;
   - **citation:** a false claim is given a real but misapplied source, or a plausible invented one;
   - **vacuous:** true, but it changes no decision (this tests the bite judgment, not truth).

   Every decoy except the vacuous ones must be false, and the orchestrator states why in the key. The key (id → real or decoy, type, why false) is written to `decoy_key.json` before the referees start, and is not shown to any agent. A decoy and its sibling never go to the same referee batch.
4. **Merits referees (blind).**
   - **What they see:** statement, goal, regime, decision, what the app does, and the claimed basis. They do not see location, tests or notes.
   - **What they cannot use:** the TurboTab repo, because the app's own documents are not evidence for the app's claims.
   - **What they can use:** derivation, primary literature (verified on the web when possible), and quick simulations (60 s CPU or less each, single process, in the scratchpad).
   - **Who:** two referees per item, of different models (Opus and Sonnet), in the same mixed batches of real claims and decoys. Neither knows which items are decoys. Both are told that some items may be wrong, which is true of the real ones as well.
   - **Output per item:**
     - truth ∈ {true, overstated, false, not_established};
     - the narrowest regime where it holds;
     - the decisive reason and the basis used;
     - bite ∈ {yes, no}, with the decision it changes;
     - confidence.
5. **Basis auditors (real claims only, with repo access).**
   - **Fidelity:** is the inventory's statement faithful to what the app actually does and says? This is the statement layer, and an unfaithful statement voids the merits verdict on it.
   - **Source:** does the cited source exist and say this, in this regime?
   - **Test:** would the named test really fail if the claim were false?
   - **Grade:**
     - proved (a derivation establishes it in the regime);
     - certified (a known-truth or independent-reference test in the repo would fail if it were false);
     - replicated (independent empirical work in this regime, published or ours, held out);
     - sourced (a primary source asserts it; not checked here);
     - judgment (a reasoned ruling with no external basis);
     - unsupported (no basis found, or the basis does not say this).
6. **Adjudication.** Every real claim that either merits referee marks false or overstated, or whose fidelity fails, is adjudicated. The orchestrator or an Opus adjudicator has all the evidence. The outcome is one of:
   - **defect:** the app's claim is wrong or overstated;
   - **referee error;**
   - **inventory misstatement.**

   Defects go to a fix list. They are not fixed during the backtest (usage pause; no build waves).

## Pre-registered measures

- **The gate under test:** a claim passes merits only if both referees say true.
- **False-promotion rate:**
  - the share of false decoys (excluding vacuous) that pass the gate;
  - each referee model's own single-referee pass rate on false decoys;
  - the vacuous decoys' bite = no rate, by model.
- **Thresholds for the gate, fixed now:**
  - 0 false decoys pass: the gate is fit to pilot;
  - 1 passes: fit to pilot only with an added check, which is named for the decoy type that got through;
  - 2 or more pass: not fit; the referee design changes before any swarm claim is refereed.
- **Agreement:** the two referees' agreement on real claims (the share with the same truth verdict), as a check on independence and noise.
- **The claims:** the grade distribution by tier; the adjudicated defects by tier; the share of tier-1 claims with a test that would fail if they were false; the share with bite = no ("rigorous trivia" or vacuous).
- **Reporting:** counts with their denominators, nothing smoothed. Every cap and sample is stated.

## Limits, stated now

- **Small sample.** With 15–25 decoys, a false-promotion rate of 0 is consistent with a true rate up to about 12–20% (the rule of three, 3/N at 95%). The backtest can show the gate is badly broken; it cannot show it is excellent.
- **Decoy realism.** The orchestrator writes the decoys, and is the same model family as the referees. Decoys from one mind may share blind spots with referees from the same family. A human-written decoy set (Nolan, or a colleague) would be the stronger test and is the recommended next step.
- **Basis.** The merits referees are LLM agents. Their "true" means only that it survived a capable adversarial reading. A claim graded certified or proved is stronger than one that passed merits.

---

## Amendment 1 — 2026-10-15, before any referee starts

**What happened:** the inventory is about five times larger than the plan assumed. Area A alone holds 135 tier-1 claims after dedupe, and area C's raw parts hold 312. Refereeing every tier-1 claim twice would take more than a hundred agents. That is against principle 6 (review capacity sets the pace), and it would turn the backtest into the volume problem it is meant to study. So phase 2's "every tier-1 claim" is replaced by a fixed sample, set only from the counts. It is written after area A's inventory arrived and before B, C or D did, and before any referee starts.

**The sample:**
- **Random stratum:** 25 tier-1 claims from each area (A, B, C, D), drawn with `random.Random(20261015 + k).sample(sorted_ids, 25)`, where k = 0, 1, 2, 3 for A, B, C, D. An area with fewer than 25 tier-1 claims gives all of them. Every claim-level estimate (grade distribution, defect rate) comes from this stratum only, weighted by area size, with a Wilson 95% interval.
- **Tier-2 stratum:** 10 tier-2 claims from the areas that have them, drawn the same way with seed 20261015 + 10 over the pooled sorted ids. Reported separately.
- **Targeted stratum:** each inventory agent's 3 most-suspect ids (12 in all). Reported separately and NEVER pooled into the rates, since chosen claims are not a sample. A claim that is in both the random and targeted strata counts in the random one and is flagged.
- **Duplicates across areas:** a claim inventoried in two areas has two chances of selection. If two sampled items are the same claim, the later id is dropped and the drop is reported.

**Decoys:** about 20% of the refereed set, so about 30. With N = 30, 0 promotions is consistent with a true rate up to 3/30 = 10%.

**Inventory findings:** code/doc disagreements the inventory agents found while extracting claims (for example A's small-sample standard not built, and F4-045's tree card) are recorded as findings of the inventory. They are not refereed again.

**Key sealed** 2026-10-15 13:36 EDT: 117 real items (random 98, tier-2 10, targeted 9; dropped duplicates C-163, D-036, D-048, C-242) and 30 decoys (26 false, 4 vacuous) in 12 merits batches. decoy_key.json sha256 53ccba6b5450f3b947ea2295869ca5a1755b2d9de134bce692806302f2b12880. Tier-2 pool: D's main file plus C's and B's sidecars (93 claims). The referee batches are in ref/, and the key is in sealed/; neither is shown to any agent.
