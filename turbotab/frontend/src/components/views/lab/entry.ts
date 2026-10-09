/**
 * One view kind on /lab/views. Each kind's entries live in their own file under ./entries, which
 * the lab page gathers by glob: a new view kind adds a file and touches nothing shared.
 */
import type { ReactNode } from "react";
import type { Purpose } from "../../stage/purposes";

export interface LabSample {
  /** What this sample shows: the fixture, or the state (one point, no data, a closed gate). */
  label: string;
  render: () => ReactNode;
  /** Draw the sample across both theme columns' width (a wide view such as the page). */
  wide?: boolean;
}

export interface LabEntry {
  /** The view kind's name in FOUNDATION §5 rule 9's vocabulary. */
  kind: string;
  /** Its purpose entry (SIZING P0.3b): the question it answers and what it shows. */
  purpose: Purpose;
  samples: LabSample[];
  /** Orders the kinds on the page (rule 9's order: table, forest, curve, … page). */
  order: number;
}
