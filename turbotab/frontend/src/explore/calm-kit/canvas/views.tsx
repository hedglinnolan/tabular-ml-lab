/**
 * The canvas's views, drawn with React SVG and d3 math only. Gray is your data now; indigo is
 * exactly what the pointed or chosen option touches (FOUNDATION §4). Every mark keeps its identity
 * across options, so switching options morphs (CSS transitions on the geometry; reduced motion
 * makes them instant).
 */
import { scaleLinear } from "d3-scale";
import { linkHorizontal } from "d3-shape";
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { HistogramData, LineageView, RelationshipView, RowFlowView, TableFocusView } from "../fixture";
import { routingOf } from "../router";
import { fmtInt, fmtR, fmtTick, plain } from "../text";
import k from "../kit.module.css";

/** Linked views: the column (or step) pointed at anywhere is lit everywhere. */
export interface Linked {
  lit: string | null;
  setLit: (c: string | null) => void;
}

function Tip({ text, at }: { text: string | null; at: { x: number; y: number } | null }) {
  if (!text || !at) return null;
  return (
    <div className={k.tip} style={{ left: at.x, top: at.y }} role="status">
      {text}
    </div>
  );
}

function useTip() {
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  return {
    node: <Tip text={tip?.text ?? null} at={tip} />,
    on: (text: string) => ({
      onPointerMove: (e: React.PointerEvent) => setTip({ text, x: e.clientX, y: e.clientY }),
      onPointerLeave: () => setTip(null),
    }),
  };
}

/** The container's width in CSS pixels, so a view's text stays at its set size at any width. */
export function useWidth(fallback = 560): [React.RefObject<HTMLElement | null>, number] {
  const ref = useRef<HTMLElement | null>(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setW(Math.max(260, Math.round(width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

// ── distribution ─────────────────────────────────────────────────────────────

/** A histogram: the engine's (HistogramData) or the Strip's derived ones. */
export type Bins = Pick<HistogramData, "edges" | "counts">;

export interface HistProps {
  before: Bins;
  after: Bins;
  /** "cut": after is who stays, before − after is who leaves. "transform": after replaces before. */
  mode: "cut" | "transform";
  after_on: boolean;
  marks?: { value: number; label: string; group: string | null }[];
  unit?: string;
  title?: ReactNode;
}

export function Hist({ before, after, mode, after_on, marks = [], unit, title }: HistProps) {
  const tip = useTip();
  const [ref, W] = useWidth();
  const H = 132;
  const T = 14;
  const sameEdges = before.edges.length === after.edges.length && before.edges.every((e, i) => Math.abs(e - after.edges[i]!) < 1e-9);
  const showEdges = mode === "transform" && after_on && !sameEdges ? after.edges : before.edges;
  const lo = showEdges[0]!;
  const hi = showEdges[showEdges.length - 1]!;
  const x = scaleLinear().domain([lo, hi]).range([0, W]);
  const shown = mode === "transform" && after_on ? after.counts : before.counts;
  const max = Math.max(1, ...before.counts, ...(mode === "transform" ? after.counts : []));
  const y = scaleLinear().domain([0, max]).range([0, H]);
  const ticks = x.ticks(5);
  // The marks are the choice's (a cut, a range in the new unit): drawn with it, never on "now".
  const visibleMarks = after_on ? marks : [];
  return (
    <figure className={k.panel} style={{ margin: 0 }} ref={ref as React.RefObject<HTMLElement>}>
      {title ? <h3>{title}</h3> : null}
      <svg viewBox={`0 0 ${W} ${H + T + 22}`} role="img" aria-label={`Histogram${unit ? ` of ${unit}` : ""}`}>
        <g>
          {shown.map((c, i) => {
            const x0 = x(showEdges[i]!);
            const x1 = x(showEdges[i + 1]!);
            const left = mode === "cut" ? after.counts[i]! : c;
            const gone = mode === "cut" && after_on ? Math.max(0, before.counts[i]! - after.counts[i]!) : 0;
            const stay = mode === "cut" ? (after_on ? left : before.counts[i]!) : c;
            const hStay = Math.max(stay ? 1 : 0, y(stay));
            const hGone = gone ? Math.max(1, y(gone)) : 0;
            const range = `${fmtTick(showEdges[i]!)}–${fmtTick(showEdges[i + 1]!)}${unit ? ` ${unit}` : ""}`;
            const hitColor = mode === "transform" && after_on;
            return (
              <g key={i}>
                <rect
                  className={`${k.bar} ${hitColor ? k.hit : k.ctx}`}
                  style={{ x: x0 + 0.5, y: T + H - hStay, width: Math.max(0.5, x1 - x0 - 1), height: hStay }}
                  rx={1}
                  {...tip.on(`${range} · ${fmtInt(stay)} ${mode === "cut" && after_on ? "stay" : "rows"}`)}
                />
                <rect
                  className={`${k.bar} ${k.hit}`}
                  style={{ x: x0 + 0.5, y: T + H - hStay - hGone, width: Math.max(0.5, x1 - x0 - 1), height: hGone }}
                  rx={1}
                  {...tip.on(`${range} · ${fmtInt(gone)} leave`)}
                />
              </g>
            );
          })}
        </g>
        <line x1={0} x2={W} y1={T + H} y2={T + H} style={{ stroke: "var(--canvas-line)" }} />
        {ticks.map((t) => (
          <text key={t} className={k.axis} x={x(t)} y={T + H + 15} textAnchor="middle">
            {fmtTick(t)}
          </text>
        ))}
        {visibleMarks.map((m, i) => {
          if (m.value < lo || m.value > hi) return null;
          const groups = [...new Set(marks.map((x) => x.group))];
          const row = groups.length > 1 ? groups.indexOf(m.group) : 0;
          return (
            <g key={`${m.value}-${i}`}>
              <line x1={x(m.value)} x2={x(m.value)} y1={T} y2={T + H} style={{ stroke: "var(--canvas-ink)" }} strokeWidth={1.25} />
              <text className={k.cutlabel} x={x(m.value) + 4} y={T + 10 + row * 13}>
                {m.group ? `${m.group} ` : ""}
                {plain(m.label)}
              </text>
            </g>
          );
        })}
      </svg>
      {tip.node}
    </figure>
  );
}

// ── relationship ─────────────────────────────────────────────────────────────

export function Scatter({ view, after_on, frame }: { view: RelationshipView; after_on: boolean; frame: number | null }) {
  const [ref, W] = useWidth();
  const H = Math.min(250, Math.max(200, W * 0.45));
  const ML = 44;
  const MB = 24;
  const MT = 18;
  const story = view.story ?? [];
  const f = after_on && frame !== null ? story[frame] : null;
  const pts = !after_on ? view.points_before : f ? f.points : view.points_after.length ? view.points_after : view.points_before;
  const all = [...view.points_before, ...(view.points_after ?? []), ...story.flatMap((s) => s.points)];
  const xs = all.map((p) => p[0]);
  const ysNow = pts.map((p) => p[1]);
  const x = scaleLinear()
    .domain([Math.min(...xs), Math.max(...xs)])
    .nice()
    .range([ML, W - 6]);
  const y = scaleLinear()
    .domain([Math.min(0, ...ysNow), Math.max(...ysNow)])
    .nice()
    .range([H - MB, MT]);
  const line = f?.fit_line;
  // A frame that names no axis of its own and still draws the values as recorded (the residual's
  // fit step) is labeled as recorded, not with the result's name.
  const recorded = !!f && (f.points === view.points_before || (f.points.length === view.points_before.length && f.points.every((p, i) => p[0] === view.points_before[i]![0] && p[1] === view.points_before[i]![1])));
  const yLabel = !after_on ? view.y_label_before : (f?.y_label ?? (recorded ? view.y_label_before : view.y_label_after));
  const r = !after_on ? view.r_before : f ? f.r : view.r_after;
  return (
    <figure className={k.panel} style={{ margin: 0 }} ref={ref as React.RefObject<HTMLElement>}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${plain(yLabel)} against ${plain(view.x_label)}, r ${fmtR(r)}`}>
        {y.ticks(4).map((t) => (
          <g key={t}>
            <line x1={ML} x2={W} y1={y(t)} y2={y(t)} style={{ stroke: "var(--canvas-line)" }} />
            <text className={k.axis} x={ML - 6} y={y(t) + 4} textAnchor="end">
              {fmtTick(t)}
            </text>
          </g>
        ))}
        {x.ticks(5).map((t) => (
          <text key={t} className={k.axis} x={x(t)} y={H - 6} textAnchor="middle">
            {fmtTick(t)}
          </text>
        ))}
        <text className={k.axis} x={ML} y={10}>
          {plain(yLabel)}
        </text>
        <text className={k.axis} x={W} y={H - 6 - 12} textAnchor="end">
          {plain(view.x_label)}
        </text>
        <g>
          {pts.map((p, i) => (
            <circle key={i} className={`${k.dot} ${after_on ? k.hit : k.ctx}`} r={2.3} style={{ cx: x(p[0]), cy: y(p[1]) }} />
          ))}
        </g>
        {line ? (
          <line
            className={k.fit}
            strokeWidth={1.75}
            x1={x.range()[0]}
            x2={x.range()[1]}
            y1={y(line.intercept + line.slope * x.invert(x.range()[0]!))}
            y2={y(line.intercept + line.slope * x.invert(x.range()[1]!))}
          />
        ) : null}
      </svg>
    </figure>
  );
}

// ── the participant flow ─────────────────────────────────────────────────────

export function FlowBars({ view, after_on, linked, bare = false }: { view: RowFlowView; after_on: boolean; linked?: Linked; bare?: boolean }) {
  const tip = useTip();
  const steps = after_on ? view.after : view.before;
  // Indigo is only what the choice changes: a step's drop that differs from the drop it has now.
  // Rows that left before this question are your data as it is: gray, drawn lighter.
  const was = new Map(view.before.map((s) => [s.key, s.dropped]));
  const total = Math.max(1, ...view.before.map((s) => s.n), ...view.after.map((s) => s.n + s.dropped));
  const x = scaleLinear().domain([0, total]).range([0, 1]);
  return (
    <figure className={k.panel} style={{ margin: 0 }}>
      {bare ? null : <h3>{plain(view.title)}</h3>}
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 8 }}>
        {steps.map((s) => {
          const lit = linked?.lit === s.key;
          const changed = after_on && s.dropped !== (was.get(s.key) ?? 0);
          return (
            <li
              key={s.key}
              onPointerEnter={() => linked?.setLit(s.key)}
              onPointerLeave={() => linked?.setLit(null)}
              style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: "2px 12px", fontSize: 15 }}
            >
              <span style={{ fontWeight: lit || changed ? 600 : 400 }}>{plain(s.label)}</span>
              <span>
                <b>{fmtInt(s.n)}</b>
                {s.dropped ? (
                  <span style={{ color: changed ? "var(--data-affected)" : "var(--canvas-muted)", fontWeight: changed ? 600 : 400 }}> −{fmtInt(s.dropped)}</span>
                ) : null}
              </span>
              <svg viewBox="0 0 400 10" preserveAspectRatio="none" style={{ gridColumn: "1 / 3", height: 10 }} aria-hidden="true">
                <rect className={`${k.bar} ${k.ctx}`} style={{ x: 0, y: 0, width: 400 * x(s.n), height: 10 }} rx={2} {...tip.on(`${fmtInt(s.n)} rows`)} />
                <rect
                  className={`${k.bar} ${changed ? k.hit : k.gone}`}
                  style={{ x: 400 * x(s.n), y: 0, width: 400 * x(s.dropped), height: 10 }}
                  rx={2}
                  {...tip.on(`${fmtInt(s.dropped)} rows ${changed ? "leave here" : "left here"}`)}
                />
              </svg>
            </li>
          );
        })}
      </ol>
      {tip.node}
    </figure>
  );
}

// ── the lineage: raw columns → roles → the model's inputs ───────────────────

interface Row {
  id: string;
  label: string;
  touched: boolean;
  /** One of the question's own columns, named at rest (in gray). */
  named?: boolean;
  /** the node ids this row stands for */
  members: string[];
  count: number;
}

const LANES: { id: "raw" | "adjusted" | "matrix"; title: string; short: string }[] = [
  { id: "raw", title: "Your columns", short: "Columns" },
  { id: "adjusted", title: "Roles", short: "Roles" },
  { id: "matrix", title: "The model's inputs", short: "Inputs" },
];

function compact(nodes: LineageView["after"]["nodes"], touched: Set<string>, keep: Set<string>, limit = 9): Row[] {
  const rows: Row[] = [];
  const quiet = new Map<string, LineageView["after"]["nodes"]>();
  const isTouched = (n: (typeof nodes)[number]) => !!n.column && touched.has(n.column);
  const isKept = (n: (typeof nodes)[number]) => !!n.column && keep.has(n.column);
  const many = nodes.length > limit;
  for (const n of nodes) {
    if (isTouched(n) || isKept(n) || !many) {
      rows.push({ id: n.id, label: n.column && n.lane !== "adjusted" ? n.column : plain(n.label), touched: isTouched(n), named: isKept(n), members: [n.id], count: n.count || 1 });
      continue;
    }
    const key = `${n.role ?? ""}|${n.group ?? ""}`;
    quiet.set(key, [...(quiet.get(key) ?? []), n]);
  }
  for (const [key, ns] of quiet) {
    const named = ns.filter((n) => n.column).map((n) => n.column!);
    const count = ns.reduce((c, n) => c + (n.count || 1), 0);
    const label =
      named.length === ns.length
        ? named.length <= 2
          ? named.join(", ")
          : `${named.slice(0, 2).join(", ")} and ${named.length - 2} more`
        : ns.map((n) => plain(n.label)).join(", ");
    rows.push({ id: `q:${key}:${ns[0]!.lane}`, label, touched: false, members: ns.map((n) => n.id), count });
  }
  return rows;
}

const link = linkHorizontal<unknown, [number, number]>()
  .source((d: unknown) => (d as { s: [number, number] }).s)
  .target((d: unknown) => (d as { t: [number, number] }).t);

export function Lineage({
  view,
  after_on,
  touch = [],
  frame = null,
  linked,
  keep = [],
}: {
  view: LineageView;
  after_on: boolean;
  touch?: string[];
  frame?: number | null;
  linked?: Linked;
  /** Columns named even where the others are grouped: the question's own, at rest. */
  keep?: string[];
}) {
  const [ref, W] = useWidth();
  const story = view.story ?? [];
  const lineage = !after_on ? (view.before ?? view.after) : frame !== null && story[frame] ? story[frame]!.lineage : view.after;
  // What the choice touches: what the engine emphasizes, what a panel names, and every column the
  // choice moves into or out of the model's inputs.
  const touched = new Set<string>(after_on ? [...view.emphasis, ...touch, ...routingOf(view).map((r) => r.column)] : []);
  if (linked?.lit) touched.add(linked.lit);
  const lanes = LANES.filter((l) => lineage.nodes.some((n) => n.lane === l.id));
  const kept = new Set(keep);
  const byLane = lanes.map((l) => compact(lineage.nodes.filter((n) => n.lane === l.id), touched, kept));
  const rowOf = new Map<string, { lane: number; i: number }>();
  byLane.forEach((rows, li) => rows.forEach((r, i) => r.members.forEach((m) => rowOf.set(m, { lane: li, i }))));
  const RH = 22;
  const HEAD = 30;
  const n = Math.max(...byLane.map((r) => r.length));
  const H = HEAD + n * RH + 6;
  const colW = W / lanes.length;
  const xText = (li: number) => li * colW + 12;
  const xOut = (li: number, rows: Row[], i: number) => xText(li) + Math.min(colW - 30, 8 + Math.min(rows[i]!.label.length, Math.floor((colW - 34) / 6.6)) * 6.6);
  const yRow = (i: number) => HEAD + i * RH + RH / 2;
  // aggregate links between rows
  const seen = new Map<string, { s: [number, number]; t: [number, number]; touched: boolean }>();
  for (const l of lineage.links) {
    const a = rowOf.get(l.source);
    const b = rowOf.get(l.target);
    if (!a || !b || a.lane === b.lane) continue;
    const ra = byLane[a.lane]!;
    const key = `${a.lane}:${a.i}>${b.lane}:${b.i}`;
    const t = ra[a.i]!.touched || byLane[b.lane]![b.i]!.touched;
    const prev = seen.get(key);
    seen.set(key, { s: [xOut(a.lane, ra, a.i) + 4, yRow(a.i)], t: [xText(b.lane) - 8, yRow(b.i)], touched: t || !!prev?.touched });
  }
  const count = (li: number) => byLane[li]!.reduce((c, r) => c + r.count, 0);
  const inputs = (view as LineageView & { inputs_label?: string }).inputs_label;
  const maxChars = Math.max(10, Math.floor((colW - 34) / 6.6));
  return (
    <figure className={k.panel} style={{ margin: 0 }} ref={ref as React.RefObject<HTMLElement>}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Which columns feed which role and the model's inputs">
        {lanes.map((l, li) => (
          <text key={l.id} className={k.axis} x={xText(li)} y={14} style={{ fontSize: 12, fontWeight: 600 }}>
            {l.id === "matrix" ? (inputs ?? (colW < 150 ? l.short : l.title)) : colW < 150 ? l.short : l.title} · {count(li)}
          </text>
        ))}
        {[...seen.values()]
          .sort((a, b) => Number(a.touched) - Number(b.touched))
          .map((e, i) => (
            <path
              key={i}
              className={k.path}
              d={link({ s: e.s, t: e.t } as never) ?? ""}
              fill="none"
              style={{ stroke: e.touched ? "var(--data-affected)" : "var(--data-context)" }}
              strokeWidth={e.touched ? 2 : 1}
            />
          ))}
        {byLane.map((rows, li) =>
          rows.map((r, i) => (
            <g
              key={r.id}
              onPointerEnter={() => linked?.setLit(r.members.length === 1 ? (lineage.nodes.find((x) => x.id === r.id)?.column ?? null) : null)}
              onPointerLeave={() => linked?.setLit(null)}
            >
              <circle cx={xText(li) - 6} cy={yRow(i)} r={3} style={{ fill: r.touched ? "var(--data-affected)" : "var(--data-context)" }} />
              <text
                x={xText(li)}
                y={yRow(i) + 4}
                style={{
                  fontFamily: "var(--font)",
                  fontSize: 13,
                  fill: r.touched || r.named ? "var(--canvas-ink)" : "var(--canvas-muted)",
                  fontWeight: r.touched ? 600 : 400,
                }}
              >
                {r.label.length > maxChars ? `${r.label.slice(0, maxChars - 1)}…` : r.label}
              </text>
            </g>
          )),
        )}
      </svg>
    </figure>
  );
}

// ── cells ────────────────────────────────────────────────────────────────────

/** A recorded value as recorded: whole numbers without a thousands separator (a year or a code is
 *  not an amount: 2001, not 2,001), other numbers at drawing precision. */
function cell(v: unknown): string {
  if (typeof v !== "number") return String(v ?? "—");
  return Number.isInteger(v) ? String(v) : fmtTick(v);
}

export function Cells({ view, after_on, bare = false }: { view: TableFocusView; after_on: boolean; bare?: boolean }) {
  const cols = after_on ? view.columns_after : view.columns_before;
  const hit = new Set(view.changed.map(([r, c]) => `${r}|${c}`));
  return (
    <figure className={k.panel} style={{ margin: 0 }}>
      {bare ? null : <h3>{plain(view.title)}</h3>}
      <div className={k.tableWrap}>
        <table className={k.cells}>
          <thead>
            <tr>
              <th scope="col">Row</th>
              {cols.map((c) => (
                <th key={c} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.rows.map((r) => (
              <tr key={r.row_id}>
                <th scope="row" style={{ fontWeight: 400, color: "var(--canvas-muted)" }}>
                  {fmtInt(r.row_id)}
                </th>
                {cols.map((c) => {
                  const v = (after_on ? r.after : r.before)[c];
                  const changed = after_on && hit.has(`${r.row_id}|${c}`);
                  return (
                    <td key={c} data-hit={changed || undefined}>
                      {cell(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
