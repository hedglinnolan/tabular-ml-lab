/**
 * The Predict declaration cards (plans/W1_PART2_PLAN.md §2, W1-F): one card for each declaration
 * line the Predict path has, keyed by the line's key, which is its decision kind. Each posts the
 * decision core/decisions.py defines through the existing decision hook, with exactly the values
 * its model accepts; payloads.ts holds them, and payloads.fixture.json is what the tests hold
 * the cards and the models to. The sensitivity analyses' card came with the design repair: a
 * Predict project with an exclusion rule meets that line as a counted Decide.
 *
 * The Estimate declaration cards are registered here too, since the shell reads one map
 * (src/quest/cards/estimate): the Model 1 question, without which an Estimate project's Models never
 * completes and Fit is never offered, how many tests are accounted for, the regression
 * calibration question, the usual-intake question, and the batch question.
 *
 * DECLARATION_DEFAULTS says, for a Confirm line the engine serves no reason for, the default it
 * confirms, from the engine's own value (src/quest/contracts.ts `DeclarationDefaults`).
 */
import type { DeclarationCards, DeclarationDefaults } from "../../contracts";
import { BatchCard } from "../estimate/BatchCard";
import { MeasurementErrorCard } from "../estimate/MeasurementErrorCard";
import { ModelOneCard } from "../estimate/ModelOneCard";
import { MultiplicityCard } from "../estimate/MultiplicityCard";
import { UsualIntakeCard } from "../estimate/UsualIntakeCard";
import { ExplainCard } from "./ExplainCard";
import { IntendedUseCard } from "./IntendedUseCard";
import { LeversCard, leversDefault } from "./LeversCard";
import { SelectionCard } from "./SelectionCard";
import { SensitivityCard } from "./SensitivityCard";
import { UpdatingCard } from "./UpdatingCard";
import { ValidationCard, validationDefault } from "./ValidationCard";

export const DECLARATION_CARDS: DeclarationCards = {
  set_intended_use: IntendedUseCard,
  set_selection: SelectionCard,
  set_validation: ValidationCard,
  set_levers: LeversCard,
  set_updating: UpdatingCard,
  set_explain: ExplainCard,
  set_sensitivity: SensitivityCard,
  set_model_sequence: ModelOneCard,
  set_multiplicity: MultiplicityCard,
  set_measurement_error: MeasurementErrorCard,
  set_usual_intake: UsualIntakeCard,
  set_batch: BatchCard,
};

export const DECLARATION_DEFAULTS: DeclarationDefaults = {
  set_validation: validationDefault,
  set_levers: leversDefault,
};
