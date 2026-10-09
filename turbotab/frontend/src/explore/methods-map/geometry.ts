/**
 * Where everything on the map sits: the regions (sections of the methods), the decision rail, the
 * participant ribbon and the column lanes, for a given width. Pure, so the map stays a drawing.
 *
 * The map reads left to right like a figure in a paper: the table, then each decision where it
 * acts, then the estimate. Columns run as lanes beneath the rail and end, fork or change where a
 * decision says so; the ribbon above them carries the rows.
 */
import { REGIONS, type NodeId, type Purpose } from "./model";

export const H = 300;
export const RAIL_Y = 98;
export const RIBBON_Y = 46;
export const LANE_TOP = 158;
export const LANE_GAP = 16;
export const LABEL_W = 132;

export type LaneId = "outcome" | "exposure" | "nutrients" | "energy" | "demo" | "body" | "unguessed" | "ids";

export const LANES: LaneId[] = ["outcome", "exposure", "nutrients", "energy", "demo", "body", "unguessed", "ids"];

export const laneY = (l: LaneId) => LANE_TOP + LANES.indexOf(l) * LANE_GAP;

/** Region widths, as shares of the drawing (by how much each region has to say). */
const SHARES: Record<Purpose, number[]> = {
  inference: [0.17, 0.19, 0.15, 0.31, 0.18],
  prediction: [0.2, 0.15, 0.13, 0.34, 0.18],
};

export interface RegionBox {
  id: string;
  title: string;
  items: string;
  x0: number;
  x1: number;
}

export interface Layout {
  w: number;
  regions: RegionBox[];
  x: Partial<Record<NodeId, number>>;
  /** The width each node's column has for its label. */
  room: Partial<Record<NodeId, number>>;
}

/** `hidden`: nodes not drawn (a silent decision); the rest of its region closes the gap. */
export function layout(w: number, purpose: Purpose, hidden: ReadonlySet<NodeId> = new Set()): Layout {
  const defs = REGIONS[purpose].map((r) => ({ ...r, nodes: r.nodes.filter((n) => !hidden.has(n)) }));
  const shares = SHARES[purpose];
  const regions: RegionBox[] = [];
  let x = 0;
  defs.forEach((r, i) => {
    const width = w * shares[i]!;
    regions.push({ id: r.id, title: r.title, items: r.items, x0: x, x1: x + width });
    x += width;
  });
  const pos: Partial<Record<NodeId, number>> = {};
  const room: Partial<Record<NodeId, number>> = {};
  defs.forEach((r, i) => {
    const box = regions[i]!;
    let nodes = r.nodes;
    if (r.nodes[0] === "source") {
      // The table sits over the lane labels; the region's other nodes share the rest.
      pos.source = LABEL_W / 2 + 6;
      room.source = LABEL_W;
      nodes = r.nodes.slice(1);
      const from = LABEL_W + 22;
      const span = box.x1 - from;
      nodes.forEach((n, j) => {
        pos[n] = from + (span * (j + 0.5)) / nodes.length;
        room[n] = span / nodes.length;
      });
      return;
    }
    const pad = 6;
    const span = box.x1 - box.x0 - 2 * pad;
    nodes.forEach((n, j) => {
      pos[n] = box.x0 + pad + (span * (j + 0.5)) / nodes.length;
      room[n] = span / nodes.length;
    });
  });
  return { w, regions, x: pos, room };
}

/** A smooth step from (x0, y0) to (x1, y1): horizontal, a short S-curve at the gate, horizontal. */
export function step(x0: number, y0: number, x1: number, y1: number, bend = 28): string {
  if (y0 === y1) return `M${x0},${y0}H${x1}`;
  const m = Math.min(bend, (x1 - x0) / 2);
  return `M${x0},${y0}C${x0 + m},${y0} ${x1 - m},${y1} ${x1},${y1}`;
}
