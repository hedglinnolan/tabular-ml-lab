/**
 * The forest view kind (FOUNDATION §5 rule 9, designed once here): effect estimates with their
 * intervals on one axis, beside the rows of the table they come from.
 *
 * Design decisions, recorded:
 * - One axis, one scale (`scale.ts`): a difference on a linear axis, a ratio on a log axis so that
 *   halving and doubling sit the same distance from 1. The reference (0 or 1) is always inside the
 *   axis, always a labeled tick, and drawn as the one stronger vertical line; the other ticks are
 *   hairline grid, recessive.
 * - Two ways to sit on a tapestry. Alone, the forest carries its own stub, named by the data
 *   ("Model", "Subgroup"), and its printed estimates, in the same grid row as each mark. Beside its
 *   table (`TableWithForest`), the forest is the table's last column: the marks sit on the table's
 *   own rows, and the labels and estimates print once, in the table.
 * - Marks: a 2 px interval line and a dot of 8 px (10 px for the locked primary), with a 2 px ring
 *   in the tapestry's color painted outside the dot, so the dot keeps its full size and reads where
 *   it crosses its line. One series is drawn in the fitted-data ink; two to five take the comparison
 *   palette in their declared order (sage, plum, ochre, steel, clay), never by rank, with a legend
 *   and, up to four, a direct label. More than five series, or a row naming a series the forest
 *   does not declare, is refused in one line rather than drawn in a wrong or shared color.
 * - No indigo: an estimate is not a choice being made.
 * - What each side of the reference means is said in plain words under the axis, one phrase at
 *   each end ("← Lower mean glucose", "Higher mean glucose →"), set as text that wraps, so neither
 *   is ever dropped for want of room.
 * - Hover or keyboard focus on a row shows its value and label; the hit target is the whole row,
 *   and the whole row lights: stub, estimate and band.
 * - The table alternative sits behind one quiet disclosure.
 * - Under a closed gate (rule 6) the forest draws nothing and says one line; the gate has no
 *   default. With no rows, no row with an estimate, or a ratio at or below zero, it says which.
 */
import type { Purpose } from "../stage/purposes";
import { fmtTick } from "../stage/format";
import { ExhibitTable, type TablePlot } from "./ExhibitTable";
import { fmtEstimate } from "./format";
import { forestScale, valuesOf, type ForestScale } from "./scale";
import { CATEGORICAL, OneLine, TableAlternative, useTip, useWidth } from "./shared";
import type { ForestData, ForestRow, Gate, TableData, TableRow } from "./types";
import v from "./views.module.css";

export const FOREST_PURPOSE: Purpose = {
  question: "matters",
  answer: "each estimate and its interval against no effect, row for row with the table",
};

/** Each row's height and the tick band's height under the plot (px). */
export const ROW = 48;
export const AXIS = 24;
const INSET = 16;
/** A rough width of the axis text per character, to keep labels inside the plot. */
const CHAR = 6.4;
/** The dot's radius, and the ring's width painted outside it. */
export const DOT = { r: 4, primary: 5, ring: 2 } as const;

export interface ForestProps {
  data: ForestData;
  /** Rule 6: the line said instead of any estimate while the gate is closed; null when open. */
  gate: Gate;
  lit?: string | null;
  onLit?: (key: string | null) => void;
  /** The plot's width before it is measured (and in tests). */
  width?: number;
}

/** The forest's rows as an exhibit table: its table alternative. */
export function forestTable(data: ForestData): TableData {
  const series = new Map((data.series ?? []).map((s) => [s.key, s.label]));
  const many = series.size > 1;
  return {
    number: "Estimates",
    title: data.measure,
    stub: data.stub,
    columns: [
      ...(many ? [{ key: "series", label: "Series" }] : []),
      { key: "est", label: "Estimate (95% CI)" },
    ],
    rows: data.rows.map((r) => ({
      kind: "row" as const,
      key: r.key,
      label: r.label,
      sub: r.sub,
      primary: r.primary,
      cells: {
        ...(many ? { series: { kind: "text" as const, text: series.get(r.series ?? "") ?? "" } } : {}),
        est: { kind: "estimate" as const, est: r.est, lo: r.lo, hi: r.hi },
      },
    })),
    footnotes: [{ text: `No effect is ${fmtTick(data.reference)}${data.axis === "log" ? ", on a ratio scale" : ""}.` }],
  };
}

/** Where a text of `chars` characters centered at x must anchor to stay inside [0, W]. */
export function anchorInside(x: number, chars: number, W: number): "start" | "middle" | "end" {
  const half = (chars * CHAR) / 2;
  if (x - half < 0) return "start";
  if (x + half > W) return "end";
  return "middle";
}

/** Why the series cannot be colored truthfully, or null when they can: the palette holds five,
 *  and a row naming an undeclared series would otherwise take another series' color. */
export function seriesProblem(data: ForestData): string | null {
  const declared = data.series ?? [];
  if (declared.length > CATEGORICAL.length)
    return `${declared.length} series are more than the comparison palette's ${CATEGORICAL.length} colors, so the forest is not drawn; the table below lists them.`;
  if (declared.length < 2) return null;
  const keys = new Set(declared.map((s) => s.key));
  const stray = data.rows.find((r) => !r.series || !keys.has(r.series));
  if (stray) return `The row “${stray.label}” names no declared series, so the forest is not drawn rather than colored as another; the table below lists every row.`;
  return null;
}

/** Why there is nothing to draw, by its cause. */
export function forestEmpty(data: ForestData): string | null {
  if (!data.rows.length) return data.empty ?? "No estimate to draw yet.";
  if (!valuesOf(data.rows).length) return "No row has an estimate to draw.";
  if (data.axis === "log") return "A ratio at or below zero cannot sit on a ratio axis, so nothing is drawn.";
  return "No row has an estimate to draw.";
}

/** Each row's color: the fitted-data ink for one series, the palette by declared order for more. */
function colorOf(data: ForestData): (series?: string) => string {
  const order = (data.series ?? []).map((x) => x.key);
  if (order.length < 2) return () => "var(--data-fit)";
  return (series) => CATEGORICAL[order.indexOf(series ?? "")]!;
}

/** The ticks' grid and the reference, from y1 to y2 (numbers, or percents of a table cell). */
function Grid({ s, reference, y1, y2 }: { s: ForestScale; reference: number; y1: number | string; y2: number | string }) {
  return (
    <>
      {s.ticks.map((t) =>
        t === reference ? null : <line key={`g${t}`} x1={s.x(t)} x2={s.x(t)} y1={y1} y2={y2} style={{ stroke: "var(--canvas-line)" }} strokeWidth={1} />,
      )}
      <line x1={s.x(reference)} x2={s.x(reference)} y1={y1} y2={y2} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} data-testid="reference" />
    </>
  );
}

/** One row's interval and dot at height y. */
function Mark({ s, r, y, c }: { s: ForestScale; r: ForestRow; y: number | string; c: string }) {
  const rad = r.primary ? DOT.primary : DOT.r;
  return (
    <>
      {r.lo !== null && r.hi !== null ? (
        <line x1={s.x(r.lo)} x2={s.x(r.hi)} y1={y} y2={y} style={{ stroke: c }} strokeWidth={2} strokeLinecap="round" pointerEvents="none" />
      ) : null}
      {r.est !== null ? (
        <>
          <circle data-ring="true" cx={s.x(r.est)} cy={y} r={rad + DOT.ring} style={{ fill: "var(--canvas)" }} pointerEvents="none" />
          <circle data-dot="true" cx={s.x(r.est)} cy={y} r={rad} style={{ fill: c }} pointerEvents="none" />
        </>
      ) : null}
    </>
  );
}

/** The tick labels, inside the plot's bounds. */
function Ticks({ s, W, y }: { s: ForestScale; W: number; y: number }) {
  return (
    <>
      {s.ticks.map((t) => {
        const label = fmtTick(t);
        return (
          <text key={`t${t}`} className={v.axis} x={s.x(t)} y={y} textAnchor={anchorInside(s.x(t), label.length, W)} data-tick={t}>
            {label}
          </text>
        );
      })}
    </>
  );
}

/** What each side of the reference means, one phrase at each end; text that wraps, never dropped. */
function Sides({ sides }: { sides?: [string, string] }) {
  if (!sides) return null;
  return (
    <div className={v.sides} data-testid="forest-sides">
      <span>← {sides[0]}</span>
      <span>{sides[1]} →</span>
    </div>
  );
}

function Legend({ data, color }: { data: ForestData; color: (s?: string) => string }) {
  return (
    <div className={v.key} data-testid="forest-legend">
      {data.series!.map((x) => (
        <span key={x.key}>
          <i style={{ background: color(x.key) }} />
          {x.label}
        </span>
      ))}
    </div>
  );
}

/** A series' name beside its first mark, in the ink (never the series color), or left of the
 *  mark when the right would run out of the plot. */
function DirectLabel({ x, y, text, W }: { x: number; y: number; text: string; W: number }) {
  const room = W - x;
  const right = text.length * CHAR <= room;
  return (
    <text className={v.axis} x={right ? x : W} y={y - 8} textAnchor={right ? "start" : "end"} pointerEvents="none" data-direct-label="true">
      {text}
    </text>
  );
}

/** The series whose first drawn row gets a direct label, up to four series. */
function directLabels(data: ForestData): Map<string, string> {
  const out = new Map<string, string>();
  const n = data.series?.length ?? 0;
  if (n < 2 || n > 4) return out;
  for (const r of data.rows) if (r.series && !out.has(r.series) && r.est !== null) out.set(r.series, r.key);
  return out;
}

/** A forest that cannot be drawn truthfully: one line, with its rows still in a table. */
function Refused({ data, text }: { data: ForestData; text: string }) {
  return (
    <div className={v.view} data-exhibit-view="forest" data-state="line">
      <p className={v.line} role="note">
        {text}
      </p>
      <TableAlternative>
        <ExhibitTable data={forestTable(data)} gate={null} />
      </TableAlternative>
    </div>
  );
}

export function Forest({ data, gate, lit, onLit, width = 360 }: ForestProps) {
  const [ref, W] = useWidth<HTMLDivElement>(width);
  const tip = useTip();
  if (gate !== null) return <OneLine view="forest" text={gate} />;
  if (!data.rows.length) return <OneLine view="forest" text={forestEmpty(data)!} />;
  const problem = seriesProblem(data);
  if (problem) return <Refused data={data} text={problem} />;
  const s = forestScale(data.rows, { axis: data.axis, reference: data.reference, width: W, inset: INSET });
  if (!s) return <OneLine view="forest" text={forestEmpty(data)!} />;
  const color = colorOf(data);
  const many = (data.series?.length ?? 0) > 1;
  const labelled = directLabels(data);
  const n = data.rows.length;
  const H = n * ROW;
  const linked = (key: string) =>
    onLit ? { onPointerEnter: () => onLit(key), onPointerLeave: () => onLit(null), onFocus: () => onLit(key), onBlur: () => onLit(null) } : {};

  return (
    <div className={v.view} data-exhibit-view="forest">
      {many ? <Legend data={data} color={color} /> : null}
      <div className={v.forest} style={{ gridTemplateRows: `auto repeat(${n}, ${ROW}px) auto` }}>
        <div className={v.fhead}>{data.stub}</div>
        <div className={v.fhead}>Estimate (95% CI)</div>
        <div className={v.fhead}>{data.measure}</div>
        {data.rows.map((r, i) => {
          const hint = tip.on(fmtEstimate(r.est, r.lo, r.hi), r.label);
          const on = lit === r.key ? "true" : undefined;
          return [
            <div
              key={`${r.key}-l`}
              className={v.fcell}
              style={{ gridRow: i + 2, gridColumn: 1 }}
              data-primary={r.primary ? "true" : undefined}
              data-row={r.key}
              data-lit={on}
              tabIndex={0}
              {...hint}
              {...linked(r.key)}
            >
              <span>{r.label}</span>
              {r.sub ? <small>{r.sub}</small> : null}
            </div>,
            <div
              key={`${r.key}-e`}
              className={`${v.fcell} ${v.fest}`}
              style={{ gridRow: i + 2, gridColumn: 2 }}
              data-primary={r.primary ? "true" : undefined}
              data-est={r.key}
              data-lit={on}
              {...hint}
              {...linked(r.key)}
            >
              {fmtEstimate(r.est, r.lo, r.hi)}
            </div>,
          ];
        })}
        <div className={v.plot} ref={ref} style={{ gridRow: `2 / span ${n + 1}` }}>
          <svg width={W} height={H + AXIS} viewBox={`0 0 ${W} ${H + AXIS}`} role="img" aria-label={`${data.measure}: ${n} estimates with 95% intervals; no effect at ${fmtTick(data.reference)}`}>
            <Grid s={s} reference={data.reference} y1={0} y2={H} />
            {data.rows.map((r, i) => {
              const y = i * ROW + ROW / 2;
              return (
                <g key={r.key} data-mark={r.key}>
                  <rect className={v.band} data-lit={lit === r.key ? "true" : undefined} x={0} y={i * ROW} width={W} height={ROW} {...tip.on(fmtEstimate(r.est, r.lo, r.hi), r.label)} {...linked(r.key)} />
                  <Mark s={s} r={r} y={y} c={color(r.series)} />
                  {r.series && labelled.get(r.series) === r.key ? (
                    <DirectLabel x={s.x(r.hi ?? r.est!) + 8} y={y} text={data.series!.find((x) => x.key === r.series)!.label} W={W} />
                  ) : null}
                </g>
              );
            })}
            <line x1={0} x2={W} y1={H} y2={H} style={{ stroke: "var(--canvas-line)" }} strokeWidth={1} />
            <Ticks s={s} W={W} y={H + 16} />
          </svg>
          <Sides sides={data.sides} />
        </div>
      </div>
      {tip.node}
      <TableAlternative>
        <ExhibitTable data={forestTable(data)} gate={null} />
      </TableAlternative>
    </div>
  );
}

export interface TableWithForestProps {
  table: TableData;
  /** The same rows under the same keys as the table's: each mark sits on its table row. */
  forest: ForestData;
  gate: Gate;
  lit?: string | null;
  onLit?: (key: string | null) => void;
  /** The whole view's width before it is measured (and in tests). */
  width?: number;
}

/** The plot column's width for a view `W` px wide: about two fifths, within reason. */
export const plotWidth = (W: number) => Math.max(180, Math.min(360, Math.round(W * 0.38)));

/** A table with its forest as the last column: labels and estimates print once, in the table, and
 *  each mark sits on its own table row. Pointing at or focusing a row lights the whole row. The
 *  table's stub names every row, so the series need the legend only, no direct label. */
export function TableWithForest({ table, forest, gate, lit, onLit, width = 720 }: TableWithForestProps) {
  const [ref, W] = useWidth<HTMLDivElement>(width);
  const tip = useTip();
  if (gate !== null) return <ExhibitTable data={table} gate={gate} />;
  const P = plotWidth(W);
  const problem = seriesProblem(forest);
  const s = problem ? null : forestScale(forest.rows, { axis: forest.axis, reference: forest.reference, width: P, inset: INSET });
  if (!s || !table.rows.some((r) => r.kind === "row")) {
    const why = problem ?? (forest.rows.length ? forestEmpty(forest) : null);
    return (
      <div ref={ref} className={v.view} data-exhibit-view="table-forest">
        <ExhibitTable data={table} gate={null} lit={lit} onLit={onLit} />
        {why && table.rows.length ? (
          <p className={v.line} role="note">
            {why}
          </p>
        ) : null}
      </div>
    );
  }
  const color = colorOf(forest);
  const many = (forest.series?.length ?? 0) > 1;
  const byKey = new Map(forest.rows.map((r) => [r.key, r]));
  const plot: TablePlot = {
    width: P,
    head: forest.measure,
    cell: (row: TableRow) => {
      const r = row.kind === "row" ? byKey.get(row.key) : undefined;
      return (
        <svg width={P} height="100%" overflow="visible" data-mark={r?.key} aria-hidden="true">
          <Grid s={s} reference={forest.reference} y1={0} y2="100%" />
          {r ? (
            <>
              <rect className={v.band} x={0} y={0} width={P} height="100%" {...tip.on(fmtEstimate(r.est, r.lo, r.hi), r.label)} />
              <Mark s={s} r={r} y="50%" c={color(r.series)} />
            </>
          ) : null}
        </svg>
      );
    },
    foot: (
      <>
        <svg width={P} height={AXIS} viewBox={`0 0 ${P} ${AXIS}`} role="img" aria-label={`${forest.measure}: no effect at ${fmtTick(forest.reference)}`}>
          <Ticks s={s} W={P} y={16} />
        </svg>
        <Sides sides={forest.sides} />
      </>
    ),
  };
  return (
    <div ref={ref} className={v.view} data-exhibit-view="table-forest">
      {many ? <Legend data={forest} color={color} /> : null}
      <ExhibitTable data={table} gate={null} lit={lit} onLit={onLit} plot={plot} />
      {tip.node}
    </div>
  );
}
