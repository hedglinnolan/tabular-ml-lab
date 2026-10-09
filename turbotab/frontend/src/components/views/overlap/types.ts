/**
 * The overlap view's input: two groups' distributions over one scale, drawn mirrored (one above
 * the baseline, one below), with the trimmed region shown.
 *
 * Derived from the engine's OverlapView (`CausalArtifact.overlap`, turbotab/core: the propensity
 * histograms per exposure level, captured in src/mocks/fixtures/m3-causal.json) by `fromEngine`.
 * The trim itself is not on that artifact: it is the plan's `causal.trim` (Crump et al. 2009:
 * keep [trim, 1 − trim]) and the causal artifact's `n_trimmed`.
 *
 * Engine contract items (not served yet):
 *  - the kept range on the overlap artifact itself (`keep: [lo, hi]` and `n_trimmed` per group),
 *    so the view never joins the plan to the artifact;
 *  - bin edges aligned to the trim, or trimmed counts per bin, so a trim inside a bin is drawn
 *    exactly (today a bin the trim splits is drawn as kept and said in the caption);
 *  - covariate overlap (one column's distribution in each exposure group, on the same edges)
 *    for the balance views in Models; the input below already takes it (`scale: "covariate"`).
 */
import type { Purpose } from "../../stage/purposes";
import type { Slot } from "../common/frame";

/** The view's purpose entry (FOUNDATION §5 rule 9, in the purpose registry's form, BLUEPRINT §11.2). */
export const OVERLAP_PURPOSE: Purpose = {
  question: "matters",
  answer: "whether each group has comparable rows in the other, and which rows a trim removes",
};

export interface OverlapGroup {
  /** the group in plain words: "heavy_user = 1" */
  label: string;
  /** rows in each bin, one per pair of edges */
  counts: number[];
  /** the group's categorical slot, so its color follows it across views; default by its order */
  slot?: Slot;
}

export interface OverlapKeep {
  /** rows below `lo` or above `hi` are trimmed */
  lo: number;
  hi: number;
  /** the engine's count of trimmed rows; when null it is summed from the bins, if they allow */
  n_trimmed: number | null;
}

export interface OverlapInput {
  /** what the horizontal axis measures, in plain words */
  x_label: string;
  /** "propensity": a chance, drawn on 0 to 1; "covariate": the column's own range and unit */
  scale: "propensity" | "covariate";
  unit?: string | null;
  /** ascending bin edges, one more than each group's counts */
  edges: number[];
  /** declared order: the first is drawn above the baseline, the second below */
  groups: [OverlapGroup, OverlapGroup];
  /** the rows kept; null when nothing is trimmed */
  keep: OverlapKeep | null;
  /**
   * "choice": the trim is the option pointed at, so what it removes is indigo (what the choice
   * touches); "recorded": it is already the plan, so what it removed is gray, drawn lighter.
   */
  keep_state?: "choice" | "recorded";
}
