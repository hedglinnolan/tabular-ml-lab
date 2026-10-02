/**
 * The stage's contract (M1_CONTRACT §4, §12–§13): thin aliases over the generated types
 * (src/api/generated.ts, from the server's OpenAPI document), plus one helper that builds the
 * substitution decision. Nothing here is hand-shaped; regenerate after a server model change.
 */
import type { components } from "./generated";
import type { SetSubstitution } from "./m1-types";
import type { Decision } from "./schema";

type S = components["schemas"];

export type {
  Baseline,
  BandEstimate,
  Coefficient,
  CohortArtifact,
  DesignArtifact,
  FitArtifact,
  FittedModel,
  Lineage,
  LineageLink,
  LineageNode,
  MetricSummary,
  Role,
  RowStep,
  ShelfArtifact,
  ShelfFamily,
  SplitArtifact,
  SubstitutionArtifact,
  SubstitutionBand,
  SubstitutionModel,
} from "./m1-types";
export type { Finding, FindingsArtifact } from "./schema";

// ── views and storyboards (§4, §12.1, §12.2) ─────────────────────────────────

export type HistogramData = S["HistogramData"];
export type Mark = S["Mark"];
export type TableRow = S["TableRow"];
export type FitLine = S["FitLine"];

export type RelationshipFrame = S["RelationshipFrame"];
export type DistributionFrame = S["DistributionFrame"];
export type LineageFrame = S["LineageFrame"];
export type TableFrame = S["TableFrame"];
export type RowFlowFrame = S["RowFlowFrame"];

export type RowFlowView = S["RowFlowView"];
export type LineageView = S["LineageView"];
export type TableFocusView = S["TableFocusView"];
export type DistributionView = S["DistributionView"];
export type RelationshipView = S["RelationshipView"];

export type PreviewResult = S["PreviewResult"];
export type ConsequenceView = PreviewResult["views"][number];
export type ViewKind = ConsequenceView["kind"];

// ── decisions the stage records (§12.7) ──────────────────────────────────────

/** A set_substitution decision; `nBoot` > 0 asks for the refit band. */
export function substitutionDecision(
  donor: string,
  recipient: string,
  stepKcal: number,
  nBoot = 0,
): Decision {
  const d: SetSubstitution = {
    kind: "set_substitution",
    donor,
    recipient,
    step_kcal: stepKcal,
    n_boot: Math.max(0, Math.round(nBoot)),
    acknowledged: false,
  };
  return d;
}
