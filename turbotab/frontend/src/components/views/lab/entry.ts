/**
 * One entry on /lab/views: a view kind on one fixture. Each view folder lists its own entries in a
 * `*.lab.tsx` file, which the lab page gathers, so adding a view kind touches no shared file.
 */
import type { ReactNode } from "react";
import type { Purpose } from "../../stage/purposes";

export interface LabEntry {
  id: string;
  /** the exhibit view kind (FOUNDATION §5 rule 9) */
  kind: string;
  purpose: Purpose;
  title: string;
  /** where the fixture's numbers come from, in one line */
  source: string;
  render: () => ReactNode;
}
