/** The engine's propensity overlap (CausalArtifact.overlap) as the overlap view's input. */
import type { CausalArtifact } from "../../../api/m3-types";
import type { OverlapInput } from "./types";

export type EngineOverlap = NonNullable<CausalArtifact["overlap"]>;

export function fromEngine(
  overlap: EngineOverlap,
  opts: {
    /** the exposure's levels in plain words, the exposed level first: ["heavy_user = 1", "heavy_user = 0"] */
    labels: [string, string];
    /** the plan's `causal.trim`: rows with a propensity outside [trim, 1 − trim] are trimmed */
    trim: number | null;
    /** the causal artifact's `n_trimmed` */
    n_trimmed?: number | null;
    keep_state?: "choice" | "recorded";
    x_label?: string;
  },
): OverlapInput {
  return {
    x_label: opts.x_label ?? "Each row's chance of being exposed, given the columns adjusted for",
    scale: "propensity",
    edges: overlap.bins,
    groups: [
      { label: opts.labels[0], counts: overlap.exposed },
      { label: opts.labels[1], counts: overlap.unexposed },
    ],
    keep: opts.trim === null ? null : { lo: opts.trim, hi: 1 - opts.trim, n_trimmed: opts.n_trimmed ?? null },
    keep_state: opts.keep_state ?? "choice",
  };
}
