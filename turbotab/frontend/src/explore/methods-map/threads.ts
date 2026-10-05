/**
 * The column lanes and the row ribbon for a set of answers: which columns each decision passes,
 * ends, forks or rewrites, drawn where the decision sits. Pure: the map draws what this returns,
 * so a hovered option's answers (a preview) draw the same way as recorded ones.
 *
 * Every count here is a count of the fixture's columns or rows; every label names a column or an
 * engine label (an energy method, a derived role), never a claim of the prototype's own.
 */
import { INF, PRED, fmtInt } from "./fixture";
import { derive, routeOf, rowsKept, type Answers, type Purpose } from "./model";
import { LABEL_W, laneY, type LaneId, type Layout, RAIL_Y } from "./geometry";

export type Tone = "ink" | "accent" | "muted" | "ok";

export interface Seg {
  key: string;
  d: string;
  tone: Tone;
  width: number;
  dashed?: boolean;
  faint?: boolean;
}

export interface Cap {
  key: string;
  x: number;
  y: number;
  label: string;
  tone: Tone;
}

export interface Marker {
  key: string;
  x: number;
  y: number;
  label: string;
}

export interface LaneLabel {
  key: string;
  y: number;
  label: string;
  count: number;
  tone: Tone;
}

export interface Bracket {
  key: string;
  x: number;
  y0: number;
  y1: number;
  label: string;
  sub: string;
  dashed: boolean;
}

export interface Ribbon {
  d: string;
  start: string;
  drops: { x: number; label: string }[];
  end: string;
  /** Prediction: the held-out rows peeling off at the seal. */
  sealed: { d: string; label: string } | null;
}

export interface Drawing {
  segs: Seg[];
  caps: Cap[];
  markers: Marker[];
  labels: LaneLabel[];
  brackets: Bracket[];
  ribbon: Ribbon;
  result: { x0: number; x: number; y: number; d: string; live: boolean } | null;
}

const W = (n: number) => Math.min(4.2, 1.2 + 0.42 * n);

const ENERGY_MARK: Record<string, string> = {
  residual: "adj",
  residual_energy_dropped: "adj",
  density_multivariate: "per kcal",
  density: "per kcal",
};
const ENERGY_DROPS = new Set(["residual_energy_dropped", "density"]);

function seg(key: string, x0: number, y0: number, x1: number, y1: number, tone: Tone, width: number, extra: Partial<Seg> = {}): Seg {
  if (y0 === y1) return { key, d: `M${x0.toFixed(1)},${y0}H${x1.toFixed(1)}`, tone, width, ...extra };
  const m = Math.min(26, (x1 - x0) / 2);
  return {
    key,
    d: `M${x0.toFixed(1)},${y0}C${(x0 + m).toFixed(1)},${y0} ${(x1 - m).toFixed(1)},${y1} ${x1.toFixed(1)},${y1}`,
    tone,
    width,
    ...extra,
  };
}

/** The adjustment groups, as the lanes that carry them. */
const GROUP_LANE: Record<string, LaneId> = { demographic: "demo", dietary: "nutrients", body: "body", unguessed: "unguessed" };

export function draw(L: Layout, a: Answers, purpose: Purpose): Drawing {
  return purpose === "inference" ? inference(L, a) : prediction(L, a);
}

function inference(L: Layout, a: Answers): Drawing {
  const x = L.x as Record<string, number>;
  const x0 = LABEL_W + 8;
  const xMatrix = x.family! + 18;
  const xEnd = x.estimate!;
  const segs: Seg[] = [];
  const caps: Cap[] = [];
  const markers: Marker[] = [];
  const labels: LaneLabel[] = [];
  const brackets: Bracket[] = [];
  const intoModel: number[] = [];
  const intoModel3: number[] = [];
  let columns = 0;

  // the outcome runs straight to the estimate
  labels.push({ key: "outcome", y: laneY("outcome"), label: "glucose", count: 1, tone: "muted" });
  segs.push(seg("outcome", x0, laneY("outcome"), xMatrix, laneY("outcome"), "muted", 1.4));

  // the nutrients: one bundle until the exposure is declared, then the exposure forks out
  const yE = laneY("exposure");
  const yN = laneY("nutrients");
  const yMid = (yE + yN) / 2;
  const xExp = x.exposure!;
  const nutrients = INF.adjustment.groups.find((g) => g.key === "dietary")!.columns;
  labels.push({ key: "nutrients", y: yMid, label: `${nutrients.length + 1} nutrients`, count: nutrients.length + 1, tone: "ink" });
  segs.push(seg("nut-in", x0, yMid, xExp, yMid, "ink", W(nutrients.length + 1)));
  const energy = a.energy;
  const mark = ENERGY_MARK[energy];
  if (a.exposure) {
    segs.push(seg("sugar-fork", xExp, yMid, xExp + 34, yE, "accent", 2.4));
    segs.push(seg("sugar", xExp + 34, yE, xMatrix, yE, "accent", 2.4));
    caps.push({ key: "sugar-name", x: xExp + 40, y: yE - 6, label: "sugar · exposure", tone: "accent" });
    intoModel.push(yE);
    columns += 1;
    if (mark) markers.push({ key: "sugar-mark", x: x.energy!, y: yE, label: mark });
    if (a.form === "spline") markers.push({ key: "sugar-form", x: x.form!, y: yE, label: "spline" });
    // the other nutrients ride to the adjustment gate, where the answers place them
    segs.push(seg("nut-fork", xExp, yMid, xExp + 34, yN, "ink", W(nutrients.length)));
    lane(segs, caps, markers, "dietary", "nutrients", xExp + 34, x, xMatrix, a, intoModel, intoModel3, mark);
    columns += countInto(a, "dietary");
  } else {
    segs.push(seg("nut-wait", xExp, yMid, xMatrix, yMid, "muted", W(nutrients.length + 1), { dashed: true, faint: true }));
  }

  // total energy
  const yK = laneY("energy");
  labels.push({ key: "energy", y: yK, label: "kcal", count: 1, tone: "ink" });
  if (ENERGY_DROPS.has(energy)) {
    segs.push(seg("kcal", x0, yK, x.energy!, yK, "ink", 1.8));
    caps.push({ key: "kcal-out", x: x.energy!, y: yK, label: "leaves the model", tone: "muted" });
  } else {
    segs.push(seg("kcal", x0, yK, xMatrix, yK, "ink", 1.8));
    intoModel.push(yK);
    columns += 1;
  }

  // the covariates: each group's lane to the adjustment gate, then where its answers send it
  for (const g of INF.adjustment.groups) {
    if (g.key === "dietary") continue;
    const id = GROUP_LANE[g.key]!;
    const y = laneY(id);
    labels.push({ key: id, y, label: g.key === "unguessed" ? `no guess · ${g.columns.length}` : groupLabel(g.key, g.columns), count: g.columns.length, tone: "ink" });
    segs.push(seg(`${id}-in`, x0, y, x.adjustment!, y, "ink", W(g.columns.length), a.exposure ? {} : {}));
    lane(segs, caps, markers, g.key, id, x.adjustment!, x, xMatrix, a, intoModel, intoModel3);
    columns += countInto(a, g.key);
  }

  // the identifier and the flags end at the readings
  const yI = laneY("ids");
  const ids = INF.roles.filter((r) => r.proposed === "identifier" || r.proposed === "flag");
  labels.push({ key: "ids", y: yI, label: `SEQN, ${ids.length - 1} flags`, count: ids.length, tone: "muted" });
  const settled = a.unit && INF.readings.items.every((i) => a.readings[i.key]);
  segs.push(seg("ids", x0, yI, x.readings!, yI, "muted", W(ids.length), settled ? {} : { dashed: true }));
  caps.push({ key: "ids-out", x: x.readings!, y: yI, label: settled ? "not predictors" : "waits on the readings", tone: "muted" });

  // the model's columns meet in its matrix; Model 3 sits beside it
  const live = !!a.locked;
  {
    const open = !a.exposure || !INF.adjustment.groups.every((g) => !!a.adjustment[g.key]);
    const y0 = laneY("outcome");
    const y1 = open ? laneY("unguessed") : Math.max(...intoModel, y0);
    brackets.push({
      key: "m2",
      x: xMatrix,
      y0,
      y1,
      label: "Model 2",
      sub: open ? "waits on the plan" : `${columns} columns`,
      dashed: open,
    });
  }
  if (intoModel3.length) {
    const yb = Math.max(...intoModel3);
    brackets.push({ key: "m3", x: xMatrix + 44, y0: Math.min(...intoModel3) - 6, y1: yb + 6, label: "Model 3", sub: "beside", dashed: true });
  }
  const yR = yE;
  const result = { x0: xMatrix, x: xEnd, y: yR, d: `M${(xMatrix + 6).toFixed(1)},${yR}H${xEnd.toFixed(1)}`, live };

  // the rows
  const kept = rowsKept(a);
  const base = INF.exclusions.n_base;
  const ribbon = ribbonPath(x0, x.exclusions!, xEnd, kept.n / base);
  return {
    segs,
    caps,
    markers,
    labels,
    brackets,
    ribbon: {
      d: ribbon,
      start: `${fmtInt(base)} rows`,
      drops: kept.dropped ? [{ x: x.exclusions!, label: `−${fmtInt(kept.dropped)}` }] : [],
      end: `n ${fmtInt(kept.n)}`,
      sealed: null,
    },
    result,
  };
}

function groupLabel(key: string, cols: string[]): string {
  if (key === "demographic") return cols.join(", ");
  if (key === "body") return `body size · ${cols.length}`;
  return `${cols.length} columns`;
}

function countInto(a: Answers, group: string): number {
  const g = INF.adjustment.groups.find((x) => x.key === group)!;
  const rec = a.adjustment[group];
  if (!rec) return 0;
  return g.columns.filter((c) => derive(rec[c]!).adjusted).length;
}

/** One adjustment group's lane from the gate on: unanswered, it runs on dashed; answered, its
 *  columns split by where the criterion sends them (the model, Model 3 beside it, or out). */
function lane(
  segs: Seg[],
  caps: Cap[],
  markers: Marker[],
  group: string,
  id: LaneId,
  xFrom: number,
  x: Record<string, number>,
  xMatrix: number,
  a: Answers,
  intoModel: number[],
  intoModel3: number[],
  mark?: string,
) {
  const g = INF.adjustment.groups.find((k) => k.key === group)!;
  const y = laneY(id);
  const xAdj = x.adjustment!;
  if (xFrom < xAdj) segs.push(seg(`${id}-pre`, xFrom, y, xAdj, y, "ink", W(g.columns.length)));
  const rec = a.adjustment[group];
  if (!rec) {
    segs.push(seg(`${id}-wait`, xAdj, y, xMatrix, y, "muted", W(g.columns.length), { dashed: true, faint: true }));
    return;
  }
  const by: Record<string, string[]> = {};
  for (const c of g.columns) {
    const r = routeOf(a, c)!;
    (by[r] ??= []).push(c);
  }
  const routes = Object.keys(by);
  const off = (i: number) => (routes.length > 1 ? (i - (routes.length - 1) / 2) * 5 : 0);
  routes.forEach((r, i) => {
    const cols = by[r]!;
    const yy = y + off(i);
    if (r === "model") {
      segs.push(seg(`${id}-model`, xAdj, y, xAdj + 30, yy, "ink", W(cols.length)));
      segs.push(seg(`${id}-model2`, xAdj + 30, yy, xMatrix, yy, "ink", W(cols.length)));
      intoModel.push(yy);
      if (mark && group === "dietary") markers.push({ key: `${id}-mark`, x: x.energy!, y: yy, label: mark });
      if (cols.length < g.columns.length)
        caps.push({ key: `${id}-name-model`, x: xAdj + 36, y: yy - 6, label: cols.join(", "), tone: "ink" });
    } else if (r === "secondary") {
      segs.push(seg(`${id}-sec`, xAdj, y, xAdj + 30, yy, "ink", W(cols.length), { dashed: true }));
      segs.push(seg(`${id}-sec2`, xAdj + 30, yy, xMatrix + 44, yy, "muted", W(cols.length), { dashed: true }));
      intoModel3.push(yy);
    } else {
      const words = derive(rec[cols[0]!]!).words;
      segs.push(seg(`${id}-out`, xAdj, y, xAdj + 30, yy, "muted", W(cols.length)));
      caps.push({
        key: `${id}-out`,
        x: xAdj + 30,
        y: yy,
        label: `${cols.length === 1 ? cols[0] : `${cols.length}`} ${cols.length === 1 ? words : plural(words)}: left out`,
        tone: "muted",
      });
    }
  });
}

function plural(words: string): string {
  if (words === "confounder") return "confounders";
  if (words === "mediator") return "mediators";
  if (words === "timing unknown") return "of unknown timing";
  return words;
}

function ribbonPath(x0: number, xCut: number, x1: number, share: number): string {
  const full = 11;
  const y = 46;
  const h1 = full;
  const h2 = Math.max(2, full * share);
  const m = 18;
  return [
    `M${x0},${y - h1 / 2}`,
    `H${xCut - m}`,
    `C${xCut},${y - h1 / 2} ${xCut},${y - h2 / 2} ${xCut + m},${y - h2 / 2}`,
    `H${x1}`,
    `V${y + h2 / 2}`,
    `H${xCut + m}`,
    `C${xCut},${y + h2 / 2} ${xCut},${y + h1 / 2} ${xCut - m},${y + h1 / 2}`,
    `H${x0}Z`,
  ].join("");
}

// ── prediction ───────────────────────────────────────────────────────────────

function prediction(L: Layout, a: Answers): Drawing {
  const x = L.x as Record<string, number>;
  const x0 = LABEL_W + 8;
  const xModels = x.p_models! + 10;
  const xEnd = x.p_score!;
  const segs: Seg[] = [];
  const caps: Cap[] = [];
  const markers: Marker[] = [];
  const labels: LaneLabel[] = [];
  const brackets: Bracket[] = [];
  const into: number[] = [];
  labels.push({ key: "outcome", y: laneY("outcome"), label: "glucose", count: 1, tone: "muted" });
  segs.push(seg("outcome", x0, laneY("outcome"), xModels, laneY("outcome"), "muted", 1.4));
  const groups: [LaneId, string, number][] = [
    ["exposure", "7 nutrients", 7],
    ["energy", "kcal", 1],
    ["demo", "age, gender", 2],
    ["body", "body size · 4", 4],
    ["unguessed", "7 more covariates", 7],
  ];
  const missingCols = INF.missing.columns.map((c) => c.column);
  for (const [id, label, n] of groups) {
    const y = laneY(id);
    labels.push({ key: id, y, label, count: n, tone: "ink" });
    // Under prediction every predictor enters the models: no adjustment set is asked.
    segs.push(seg(`${id}`, x0, y, xModels, y, "ink", W(n), a.pSeal ? {} : {}));
    into.push(y);
  }
  // the blanks in the medication answers are filled in each training fold (or their rows leave)
  markers.push({
    key: "fill",
    x: x.p_missing!,
    y: laneY("unguessed"),
    label: a.pMissing ? "filled in-fold" : `${missingCols.length} with blanks`,
  });
  const yI = laneY("ids");
  labels.push({ key: "ids", y: yI, label: "SEQN, 6 flags", count: 7, tone: "muted" });
  segs.push(seg("ids", x0, yI, x.readings!, yI, "muted", W(7)));
  caps.push({ key: "ids-out", x: x.readings!, y: yI, label: "not predictors", tone: "muted" });
  brackets.push({
    key: "shelf",
    x: xModels,
    y0: laneY("outcome"),
    y1: Math.max(...into),
    label: "the shelf",
    sub: "waits on the seal",
    dashed: true,
  });
  const base = INF.exclusions.n_base;
  const hold = a.pSeal ? (PRED.seal.options.find((o) => String(o.holdout) === a.pSeal)?.n_holdout ?? 0) : 0;
  const share = (base - hold) / base;
  const xs = x.p_seal!;
  const ribbon = ribbonPath(x0, xs, xEnd, share);
  const sealed = hold
    ? {
        d: `M${xs + 18},${46 + 5.5}C${xs + 50},${46 + 5.5} ${xs + 50},${RAIL_Y - 30} ${xs + 90},${RAIL_Y - 30}H${xEnd - 10}`,
        label: `${fmtInt(hold)} held out, sealed`,
      }
    : null;
  return {
    segs,
    caps,
    markers,
    labels,
    brackets,
    ribbon: {
      d: ribbon,
      start: `${fmtInt(base)} rows`,
      drops: [], // the sealed rows peel off with their own label (`sealed`)
      end: `train ${fmtInt(base - hold)}`,
      sealed,
    },
    result: null,
  };
}
