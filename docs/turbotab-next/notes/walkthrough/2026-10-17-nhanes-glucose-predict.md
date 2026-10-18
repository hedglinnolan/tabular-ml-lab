# Walkthrough: NHANES fasting glucose under Predict, the quest shell (2026-10-17)

## TL;DR

- **The walk:** the orchestrator walked Nolan's project end to end on a real server, on W1 part 2's integrated head (0e68942b):
  - NHANES fixture, 21,849 rows, the dietary and clinical lenses;
  - glucose under Predict;
  - every row kept;
  - linear and ridge;
  - Fit, the triage, the final model, the opening, the download.
- **What worked:**
  - the seal card's precision per option;
  - the missing-values card's recommendation;
  - Fit computing live while the scores wait;
  - Results' comparison against the outcome's average;
  - the refusal chain naming the final model;
  - the opened scores with calibration;
  - the TRIPOD-ordered manuscript and the 14-file bundle.
- **Results:** held-out R² 0.159 (0.136 to 0.181), RMSE 33.7 mg/dL, for ridge on 21 predictors. That matches the earlier light walk (0.155 to 0.18).
- **What has to change before Nolan walks it:**
  - **The biggest:** 16 of the 21 predictors are silently left out of the model unless each is confirmed on its own. These include blood pressure, waist, HDL and triglycerides. The shell offers no way to confirm them, and nothing says they are out.
  - **Next:** two record buttons that disagree; noticings that block Continue; cards without a record button; and three sentences that say Estimate things under Predict, one of them false.
- **Plan:** a fix round (W1-R, below) takes the blockers and majors, and the minors cheap enough to ride along. W2 takes the rest, with First look's missing views at the top.

## How the walk went

The walk drove the shell in a headless browser on this machine: step by step, with a screenshot after each move. The screenshots are in the session scratchpad, `walk_tools/shots/`, 63 of them.

| Stage | What happened |
|---|---|
| Start | A clean start screen. The heading is still the old serif. |
| Your data | 14 lines before the question can be asked: the lens, a SAS-zero repair, gender coding, the roles (waiting for the outcome) and 9 noticings. Continue stays disabled until every noticing is answered. |
| Your question | The outcome search is clean. The purpose card works. The intended-use card has no record button (Enter only) and offers decision-support thresholds for a numeric outcome. |
| First look | A list of noticings only, with no look at the data. |
| Who's in | Everything waits on the roles, which live back in Your data. Once the roles were confirmed, the noticings settled with them. The eligibility, impossible-value, missing-value and seal cards are good. |
| Models | No energy question (ruling 5 holds). The selection card is a form. The families card needs 4 readings first, and confirming them clears the chosen families. The families answer is a paragraph of every role. Fit computes live, and nothing is shown before the press. |
| Results | The comparison, the BBC-CV row and the refusal chain are good. A noticing says day-to-day noise "could bias the score". |
| After opening | Held-out R² 0.159 and its verdict. No false "changed after the opening" (W1-Q). Calibration is drawn, but the sparse tail misleads with nothing showing where the predictions fall. |
| Write-up | The TRIPOD+AI checklist (4 answered, 11 partly, 37 not of 52), the manuscript in sections, Download. |

## Findings

Severity follows verifier.md: a blocker gives a wrong number or a false sentence, or a dead end; a major is a stated requirement unmet or a serious obstacle; a minor is a defect in what the researcher reads. "→" names the fix and where it goes.

### Blockers

1. **Low-confidence roles leave the model in silence.**
   - **What happens:** the roles card records every proposal below high confidence as unconfirmed (`roles_unconfirmed`). Here that is 16 columns:
     - the cycle;
     - blood pressure, weight, height and waist;
     - HDL and triglycerides;
     - both medicine questions;
     - the six imputed flags.
   - **The rule behind it:** BLUEPRINT §14 rule 2, a proposal below high "needs its own confirmation; it never rides along in a bulk 'confirm all'".
   - **The effect:** until each is confirmed, no default reads it, and the model has 11 predictors, not 21.
   - **Why the shell hides it:** the confirmations live in the readings, which the shell draws as quiet rows (W1_PART2_PLAN ruling 7, the orchestrator's, now wrong). The quest log counts Your data complete. The only trace is a clause at the end of the roles sentence.
   - → **W1-R:**
     - each unconfirmed role is an open line, confirmed one at a time as rule 2 requires, on the roles card or its own card;
     - Your data is not complete while any waits;
     - the Columns panel says "N columns wait for their own confirmation and are not predictors yet".
2. **The roles sentence stays stale.** "…cycle_begin_year, bp_sys and 14 more were proposed below high confidence and wait for their own confirmation" is still in the record and the manuscript after each was confirmed. → **W1-R:** the methods say what holds now: confirmed columns are no longer listed as waiting.
3. **Estimate sentences under Predict.**
   - **(a) A false claim:** the day-to-day-noise noticing says "Could bias the score". Under Predict, the held-out rows carry the same noise the model meets in use, so the score is not biased. The noise limits what the predictor can predict. → **W1-R:** under Predict, it says that, or it is a For-the-record line.
   - **(b) Coefficient concerns:** "Concerns about this design" explains coefficient meaning and substitutions ("carb's coefficient is the remainder… A substitution through carb…"). Under Predict no coefficient is interpreted. → **W1-R:** silent under Predict (ruling 5 extended).
   - **(c) The roles card's wording:** it reads "Only study factors, covariates and the energy column can become predictors… 7 exposures", and the families answer says "what you study". This is BEATS_PREDICT P3 (C1), planned for W2. → **W1-R:** pulled forward for the wording only. Under Predict the card and the sentences say "predictors". The engine's roles stay as they are.
4. **Two record buttons that disagree.** The Stage's own record button stays in the tapestry beside the card's:
   - "Record the dietary intake lens" while the card says "Record these 2 lenses";
   - "Record this choice";
   - "Choose the ridge" while the card has two families chosen.

   Pressing the tapestry's records a different answer. → **W1-R:** the shell hides the Stage's record button. The card's foot is the one action (FOUNDATION §1).
5. **Overlapping drawing.** The lens preview's column lineage draws its labels over its own caption. W1-N's leftovers belong here too: the 1280 px matrix labels are cut, and a count is clipped at 400 px. → **W1-R.**

### Majors

6. **Noticings block Continue.** Continue is disabled until every noticing in the stage is answered: 9 in Your data, before the outcome exists. Under Predict, FOUNDATION §7 gives noticings their own sweep, before the held-out rows open, in Results. → **W1-R:** Continue is enabled once the stage's questions are answered. Open noticings travel to the triage, and the foot says "N noticings wait for the triage in Results".
7. **Continue goes to the next stage, not the next answerable line.** After the outcome, the next answerable line is the roles, back in Your data. Continue leads on to First look and then Who's in, where everything waits on the roles. → **W1-R:** Continue goes to the next open line that can be answered.
8. **Cards with no record button:** intended use, selection, shrinkage, the explanations (W1-F's cards). Only "Enter: record" is shown, so a mouse user cannot finish them. → **W1-R:** each card records from its own button.
9. **Intended use for a numeric outcome.** Decision support is offered with risk thresholds from 0 to 1, though the engine refuses decision support for a numeric outcome. The subgroup list offers every column, including the identifier, the imputed flags and continuous measures. → **W1-R:**
   - no decision support for a numeric outcome;
   - subgroups lead with the categorical and few-valued columns.
10. **Calibration's tail misleads.** Above about 150 mg/dL the smoothed curve bends far below the diagonal, though the slope is 1.0 (0.936 to 1.07): few rows sit there, and nothing shows it. This is the very place Nolan looked when he asked about high glucose. → **W1-R:** the engine serves the count of held-out rows behind each stretch of the curve, and the view draws a density strip under it. Ruling 5 of W1_PART2_PLAN (no binned points in v0) stands; this is a support strip, not bins.
11. **The families answer is one long paragraph.** It restates every role reading with its evidence ("bp_sys is an amount (one slope) (1,957 values with decimals that fill their grid…)"). → **W1-R:** the families sentence names the families. The readings keep their own sentences, collapsed under For the record.
12. **Readings shown twice, and a choice cleared.** The families card's readings show in the card and again in the tapestry as a wall of nine buttons. The families press is refused without a word until they are confirmed, and confirming them clears the chosen families. → **W1-R:**
    - the readings show once;
    - the refusal says what it waits for;
    - the choice survives the confirmation.

### Minors (W1-R where cheap, else INBOX)

13. The lens option's "offers energy adjustment and substitution curves" (now asked only under Estimate).
14. "152 cells in bp_di and 8 more" reads as either cells or columns, and the recorded sentence drops the 8.
15. "Joining another file" and "The data dictionary" show as dead "Not available yet" lines.
16. The sweep heading "Confirm · Here is the other choice set for you".
17. A numeric outcome's intended-use sentence says "no classification threshold was set".
18. "Keep as is" is recorded as "dismissed, with no reason given".
19. A "SETTLED" badge on an undecided repair.
20. Prose precision: "CV MSE of 1033.261", "held-out MSE 1133.621".
21. "Outcomes moved 1 times as far as the predictions did".
22. Results' Continue is another color.
23. The word "Prediction" in a description opens a tooltip where a click meant the option.
24. A recorded card stays expanded instead of moving to the next open line.
25. Write-up reads done before it is reached.

### For W2 (BEATS_PREDICT), not W1-R

- **First look shows no data:** no outcome distribution, no relationships. This is the gap with the vision ("understanding becomes an explicit object"), so it heads W2.
- **The seal by time:** for a model used on future patients, the honest held-out set is the latest cycle (2017–2018). The engine draws by time, but NHANES's seal card offers only random shares (BEATS_PREDICT's validation by cycle).
- **Diet conventions under Predict:** the eligibility card's dietary screens do not say that excluding rows also narrows who the model is for.
- **The start screen's serif heading.**

## W1-R, the fix round

Two packages in parallel, each with an adversarial verifier and at most two repair rounds; then the merge, CI, and Nolan's walk.

- **R-engine** (Opus). The engine side of:
  - 1: the unconfirmed roles as open lines, Your data incomplete while any waits, and "not predictors yet" served;
  - 2;
  - 3a, 3b and 3c (wording);
  - 10 (the support counts);
  - 11;
  - the engine-worded minors (14, 16, 17, 18, 20, 21).
- **R-frontend** (Opus). The shell side of:
  - 1: one-at-a-time confirmation;
  - 4, 5, 6, 7, 8, 9, 10 (the strip) and 12;
  - the frontend minors (15, 19, 22, 23, 24).

## The re-walk (2026-10-18, W1-U's head 712747ff)

The orchestrator walked the same path again on a fresh real server: the same fixture and lenses,
glucose under Predict, every row kept, blanks as a level, a 20% seal, linear and ridge, then
opened with ridge. Held-out MSE 1,131 for ridge. The cross-validated R² was 0.156 (0.141 to 0.171).

**Fixed: findings 1–17 and 19–25.**
- The 16 waiting roles are open lines: cycle_begin_year on its own, one group of 7 measures, one of
  2, and the six flags each on its own, since each has its own evidence.
- The model holds all 21 predictors once they are confirmed. Before, it held 11.
- Continue moves past open noticings to the next line that can be answered.
- There is one record button per card.
- The families choice survives the readings' confirmation.
- The calibration view draws its support strip.
- The noise noticing is honest under Predict.
- The design states no estimand.

Finding 18 was not exercised.

**Still open.** The Predict roles card's step notes are still there: W1-V removes them (one plain
sentence), with the "read as a covariate" quote of finding 3c's class.

**New, from this walk:**
- **N9 (major):**
  - **What happens:** the triage proposes "Act on it" for a noticing whose action is a change to an
    answered question, for example the energy screen at eligibility. "Confirm all 3" accepts it.
  - **The effect:** the seal then refuses until each is acted on or recorded as a limitation, and
    the record says "2 to act on".
  - **→ ruling:** where the deciding question is already answered, the proposal is to record it as
    a limitation, since the answer stands. Acting on it, by changing that answer, is the
    alternative.
- N1: a numeric outcome's only intended use is labeled "Risk estimation".
- N2: a waiting flag's line says it "cannot set the model's predictors yet", but a flag never
  enters the models.
- N3: the Columns lineage says "13 covariate columns" and "7 exposure columns" under Predict.
- N4: the families refusal is shown twice, and its only exit points to the roles question rather
  than the readings above it.
- N5: while the shelf re-ranks after the readings' confirmation, the families wait on "the event
  question", which a numeric outcome does not have.
- N6: "Confirm all 3" writes one combined sentence into each of the three lines, and it drops the
  validation scheme's value.
- N7: the banner "Your change to Your data reopened 2 questions in Who's in" stays while Who's in
  reads done.
- N8: in Results, a recorded card stays expanded with its record button.
- N10: the methods list each single confirmation as its own near-identical sentence (six flag
  lines).

→ W1-W takes N1–N10.
