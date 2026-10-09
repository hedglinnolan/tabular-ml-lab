/**
 * The forest view kind (FOUNDATION §5 rule 9, designed once here): effect estimates with their
 * intervals on one axis, beside the rows of the table they come from.
 *
 * Design decisions, recorded:
 * - One axis, one scale (`scale.ts`): a difference on a linear axis, a ratio on a log axis so that
 *   halving and doubling sit the same distance from 1. The reference (0 or 1) is always inside the
 *   axis, always a labeled tick, and drawn as the one stronger vertical line; the other ticks are
 *   hairline grid, recessive.
 * - Rows align to the table: the stub and the printed estimate sit in the same grid row as the
 *   mark, in the table's order and under its keys, so pointing at a row lights it here and in the
 *   table beside it.
 * - Marks: a 2 px interval line and an 8 px dot (10 px for the locked primary) with a 2 px ring in
 *   the tapestry's color so the dot reads where it crosses its line. One series is drawn in the
 *   fitted-data ink; two or more take the comparison palette in their declared order (sage, plum,
 *   ochre, steel, clay), never by rank, with a legend and, up to four, a direct label.
 * - No indigo: an estimate is not a choice being made.
 * - What each side of the reference means is said in plain words under the axis ("Lower mean
 *   glucose"), so the reader never has to work out the direction from the sign.
 * - Hover or keyboard focus on a row shows its value and label; the hit target is the whole row.
 * - The table alternative sits behind one quiet disclosure.
 * - Under a closed gate (rule 6) the forest draws nothing and says one line; with no rows, or a
 *   ratio at or below zero, it says why in one line.
 */
import type { Purpose } from "../stage/purposes";
import { fmtTick } from "../stage/format";
import { ExhibitTable } from "./ExhibitTable";
import { fmtEstimate } from "./format";
import { forestScale } from "./scale";
import { CATEGORICAL, OneLine, TableAlternative, useTip, useWidth } from "./shared";
import type { ForestData, Gate, TableData } from "./types";
import v from "./views.module.css";

export const FOREST_PURPOSE: Purpose = {
  question: "matters",
  answer: "each estimate and its interval against no effect, row for row with the table",
};

/** Each row's height, the axis band's height, and the plot's inset at either end (px). */
export const ROW = 48;
export const AXIS = 42;
const INSET = 16;
/** A rough width of the axis text per character, to keep labels inside the plot. */
const CHAR = 6.4;

export interface ForestProps {
  data: ForestData;
  gate?: Gate;
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
    stub: "Row",
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

export function Forest({ data, gate, lit, onLit, width = 360 }: ForestProps) {
  const [ref, W] = useWidth<HTMLDivElement>(width);
  const tip = useTip();
  if (gate) return <OneLine view="forest" text={gate} />;
  if (!data.rows.length) return <OneLine view="forest" text={data.empty ?? "No estimate to draw yet."} />;
  const s = forestScale(data.rows, { axis: data.axis, reference: data.reference, width: W, inset: INSET });
  if (!s) {
    const why = data.axis === "log" ? "A ratio at or below zero cannot sit on a ratio axis, so nothing is drawn." : "No row has an estimate to draw.";
    return <OneLine view="forest" text={why} />;
  }
  const order = (data.series ?? []).map((x) => x.key);
  const many = order.length > 1;
  const color = (series?: string) => (many ? CATEGORICAL[Math.max(0, order.indexOf(series ?? "")) % CATEGORICAL.length]! : "var(--data-fit)");
  const firstOf = new Map<string, string>();
  for (const r of data.rows) if (r.series && !firstOf.has(r.series) && r.est !== null) firstOf.set(r.series, r.key);
  const n = data.rows.length;
  const H = n * ROW;
  const xRef = s.x(data.reference);
  const linked = (key: string) =>
    onLit ? { onPointerEnter: () => onLit(key), onPointerLeave: () => onLit(null), onFocus: () => onLit(key), onBlur: () => onLit(null) } : {};
  const [below, above] = data.sides ?? [null, null];
  const fitsLeft = below && below.length * CHAR + 18 <= xRef;
  const fitsRight = above && above.length * CHAR + 18 <= W - xRef;

  return (
    <div className={v.view} data-exhibit-view="forest">
      {many ? (
        <div className={v.key} data-testid="forest-legend">
          {data.series!.map((x) => (
            <span key={x.key}>
              <i style={{ background: color(x.key) }} />
              {x.label}
            </span>
          ))}
        </div>
      ) : null}
      <div className={v.forest} style={{ gridTemplateRows: `auto repeat(${n}, ${ROW}px) ${AXIS}px` }}>
        <div className={v.fhead}>Model</div>
        <div className={v.fhead}>Estimate (95% CI)</div>
        <div className={v.fhead}>{data.measure}</div>
        {data.rows.map((r, i) => {
          const hint = tip.on(fmtEstimate(r.est, r.lo, r.hi), r.label);
          return [
            <div
              key={`${r.key}-l`}
              className={v.fcell}
              style={{ gridRow: i + 2, gridColumn: 1 }}
              data-primary={r.primary ? "true" : undefined}
              data-row={r.key}
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
            >
              {fmtEstimate(r.est, r.lo, r.hi)}
            </div>,
          ];
        })}
        <div className={v.plot} ref={ref} style={{ gridRow: `2 / span ${n + 1}` }}>
          <svg width={W} height={H + AXIS} viewBox={`0 0 ${W} ${H + AXIS}`} role="img" aria-label={`${data.measure}: ${n} estimates with 95% intervals; no effect at ${fmtTick(data.reference)}`}>
            {s.ticks.map((t) =>
              t === data.reference ? null : <line key={`g${t}`} x1={s.x(t)} x2={s.x(t)} y1={0} y2={H} style={{ stroke: "var(--canvas-line)" }} strokeWidth={1} />,
            )}
            <line x1={xRef} x2={xRef} y1={0} y2={H + 4} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} data-testid="reference" />
            {data.rows.map((r, i) => {
              const y = i * ROW + ROW / 2;
              const c = color(r.series);
              return (
                <g key={r.key} data-mark={r.key}>
                  <rect className={v.band} data-lit={lit === r.key ? "true" : undefined} x={0} y={i * ROW} width={W} height={ROW} {...tip.on(fmtEstimate(r.est, r.lo, r.hi), r.label)} {...linked(r.key)} />
                  {r.lo !== null && r.hi !== null ? (
                    <line x1={s.x(r.lo)} x2={s.x(r.hi)} y1={y} y2={y} style={{ stroke: c }} strokeWidth={2} strokeLinecap="round" pointerEvents="none" />
                  ) : null}
                  {r.est !== null ? (
                    <circle cx={s.x(r.est)} cy={y} r={r.primary ? 5 : 4} style={{ fill: c, stroke: "var(--canvas)" }} strokeWidth={2} pointerEvents="none" />
                  ) : null}
                  {many && order.length <= 4 && r.series && firstOf.get(r.series) === r.key && r.est !== null ? (
                    <DirectLabel x={s.x(r.hi ?? r.est) + 8} y={y} text={(data.series ?? []).find((x) => x.key === r.series)?.label ?? ""} W={W} />
                  ) : null}
                </g>
              );
            })}
            <line x1={0} x2={W} y1={H} y2={H} style={{ stroke: "var(--canvas-line)" }} strokeWidth={1} />
            {s.ticks.map((t) => {
              const label = fmtTick(t);
              return (
                <text key={`t${t}`} className={v.axis} x={s.x(t)} y={H + 16} textAnchor={anchorInside(s.x(t), label.length, W)} data-tick={t}>
                  {label}
                </text>
              );
            })}
            {fitsLeft ? (
              <text className={v.axisTitle} x={xRef - 6} y={H + 33} textAnchor="end">
                ← {below}
              </text>
            ) : null}
            {fitsRight ? (
              <text className={v.axisTitle} x={xRef + 6} y={H + 33} textAnchor="start">
                {above} →
              </text>
            ) : null}
          </svg>
        </div>
      </div>
      {tip.node}
      <TableAlternative>
        <ExhibitTable data={forestTable(data)} />
      </TableAlternative>
    </div>
  );
}

/** A series' name beside its first mark, in the ink (never the series color), or left of the
 *  mark when the right would run out of the plot. */
function DirectLabel({ x, y, text, W }: { x: number; y: number; text: string; W: number }) {
  const room = W - x;
  const right = text.length * CHAR <= room;
  return (
    <text className={v.axis} x={right ? x : W} y={y - 8} textAnchor={right ? "start" : "end"} pointerEvents="none">
      {text}
    </text>
  );
}
