/**
 * The seal's state machine, as the stage reads it (M2_CONTRACT §3; ROADMAP lockbox constitution
 * §01–§05) — pure, so the two rules that end up in a paper are tested without a browser:
 *
 *   none ─fit─▶ sealed ─open_seal─▶ opened ─a later change─▶ post_seal
 *          └──(nothing held out)──▶ cv_only
 *
 * 1. Held-out scores render only once the seal is opened. The server withholds them before that
 *    (`holdout: null`, `holdout_sealed: true`); the stage does not trust a stray number either: a
 *    score in an artifact the Record has not opened is never drawn, never printed, never saved.
 * 2. Only a grouped seal (or a verified one-row-per-unit seal) is drawn as a closed lock. An
 *    abandoned grouping is a ring with a gap; an undetermined basis — or one the stage cannot
 *    read — is a dashed ring around a question mark.
 */
import type { FitArtifact } from "../../../api/m1-stage-types";
import type { SealBasisState } from "../../../api/m2-stage-types";

export type SealPhase = "none" | "cv_only" | "sealed" | "opened" | "post_seal";

/** Which of the seal's states the Results are in, from the fit and whether `open_seal` is recorded. */
export function sealPhase(fit: Pick<FitArtifact, "n_holdout" | "holdout_sealed" | "changed_after_seal"> | null, opened: boolean): SealPhase {
  if (!fit) return "none";
  if (!fit.n_holdout) return "cv_only";
  // Both must hold: the Record says the seal was opened, and the served fit is no longer sealed.
  if (!opened || fit.holdout_sealed) return "sealed";
  return fit.changed_after_seal ? "post_seal" : "opened";
}

/** Whether held-out scores may be shown in this phase. */
export function scoresVisible(phase: SealPhase): boolean {
  return phase === "opened" || phase === "post_seal";
}

/** A held-out score as the stage may show it: the number once opened, else nothing at all. */
export function shownHoldout(phase: SealPhase, value: number | null | undefined): number | null {
  return scoresVisible(phase) && value !== null && value !== undefined && Number.isFinite(value) ? value : null;
}

export type GlyphState = "closed" | "abandoned" | "undetermined";

/** The glyph a basis earns. Anything but a verified basis is never a clean lock. */
export function glyphOf(state: SealBasisState | string | null | undefined): GlyphState {
  if (state === "grouped" || state === "one_row_per_unit") return "closed";
  if (state === "abandoned") return "abandoned";
  return "undetermined";
}

/** An exploratory basis (abandoned, undetermined, or unknown) carries its label wherever the seal shows. */
export function isExploratory(state: SealBasisState | string | null | undefined, exploratory?: boolean | null): boolean {
  return exploratory === true || glyphOf(state) !== "closed";
}
