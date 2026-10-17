/**
 * The Predict declaration cards (plans/W1_PART2_PLAN.md §2, W1-F): one card for each declaration
 * line the Predict path has, keyed by the line's key, which is its decision kind. Each posts the
 * decision core/decisions.py defines through the existing decision hook, with exactly the values
 * its model accepts; payloads.ts holds them, and payloads.fixture.json is what the tests hold
 * the cards and the models to.
 */
import type { DeclarationCards } from "../../contracts";
import { ExplainCard } from "./ExplainCard";
import { IntendedUseCard } from "./IntendedUseCard";
import { LeversCard } from "./LeversCard";
import { SelectionCard } from "./SelectionCard";
import { UpdatingCard } from "./UpdatingCard";
import { ValidationCard } from "./ValidationCard";

export const DECLARATION_CARDS: DeclarationCards = {
  set_intended_use: IntendedUseCard,
  set_selection: SelectionCard,
  set_validation: ValidationCard,
  set_levers: LeversCard,
  set_updating: UpdatingCard,
  set_explain: ExplainCard,
};
