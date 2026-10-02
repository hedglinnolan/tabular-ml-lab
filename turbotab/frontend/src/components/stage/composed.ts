/**
 * The pictures the stage composes from the closed vocabulary (BLUEPRINT §11 rule 2): no new view
 * kind on the wire, a specific picture where the data is that shape.
 *
 *   reshape  a table_focus whose frames carry the row map (many records → one row per unit)
 *   turn     a table_focus whose last frame is its first one transposed (orientation)
 *   seal     a row_flow carrying the seal's cells (the split's preview)
 *
 * Read from the data, never from the decision's kind, so a finding's evidence of the same shape
 * draws the same way. Cached per view object: a view is immutable once received.
 */
import type { ConsequenceView } from "../../api/m1-stage-types";
import { reshapeOf, type ReshapeModel } from "./reshape/reshape";
import { turnOf, type TurnModel } from "./reshape/TurnTable";

export type Composed =
  | { kind: "reshape_table"; model: ReshapeModel }
  | { kind: "turn_table"; model: TurnModel }
  | { kind: "seal_fork" }
  | null;

const cache = new WeakMap<object, Composed>();

export function composedOf(view: ConsequenceView): Composed {
  const hit = cache.get(view);
  if (hit !== undefined) return hit;
  let out: Composed = null;
  if (view.kind === "table_focus") {
    const reshape = reshapeOf(view);
    if (reshape) out = { kind: "reshape_table", model: reshape };
    else {
      const turn = turnOf(view);
      if (turn) out = { kind: "turn_table", model: turn };
    }
  } else if (view.kind === "row_flow" && view.seal) out = { kind: "seal_fork" };
  cache.set(view, out);
  return out;
}
