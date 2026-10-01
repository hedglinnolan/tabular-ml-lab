/**
 * Journal-format figures of the stage's views (DESIGN_LANGUAGE §07): serif, grayscale-safe, series
 * told apart by dash pattern, literal colors, a numbered-figure caption that carries provenance.
 *
 * A saved figure is drawn from a real state's data, never copied from the screen, so it can never
 * be an interpolated mid-animation frame: `figureForTrack` refuses any state that is not a whole
 * step of the view's storyboard.
 */
import type {
  ConsequenceView,
  DistributionView,
  HistogramData,
  Lineage,
  Mark,
  RowStep,
  TableFocusView,
} from "../../../api/m1-stage-types";
import { fmtInt, fmtR, fmtTick, plain } from "../format";
import {
  clipTail,
  type DistributionState,
  type LineageState,
  type RelationshipState,
  type RowFlowState,
  type TableState,
  type Track,
} from "../tracks";
import { placeLineage } from "../views/lineageLayout";
import { J, el, esc, hatch, linear, niceDomain, niceTicks, text, wrap, type Box } from "./svg";

export interface Panel {
  label: string;
  body: (box: Box, uid: string) => string;
}

export interface FigureSpec {
  title: string;
  panels: Panel[];
  /** What the figure shows (the view's caption). */
  caption: string;
  /** Which choice produced it and on what rows. */
  provenance: string;
  width?: number;
  panelHeight?: number;
}

const M = 24;
const GAP = 28;

/** The whole figure as a standalone SVG document. */
export function figureSvg(spec: FigureSpec): string {
  const n = spec.panels.length;
  const W = spec.width ?? (n > 1 ? 1100 : 720);
  const H = spec.panelHeight ?? 380;
  const pw = (W - 2 * M - GAP * (n - 1)) / n;
  const titleLines = wrap(plain(spec.title), W - 2 * M, 15);
  const top = M + titleLines.length * 19 + 6;
  const capText = `${plain(spec.caption)} ${plain(spec.provenance)}`.trim();
  const capLines = wrap(capText, W - 2 * M, 11.5);
  const Htot = top + H + 18 + capLines.length * 15 + M;
  const parts: string[] = [];
  parts.push(el("rect", { x: 0, y: 0, width: W, height: Htot, fill: J.paper }));
  titleLines.forEach((l, i) => parts.push(text(M, M + 12 + i * 19, l, { size: 15, weight: 700 })));
  const defs: string[] = [];
  spec.panels.forEach((p, i) => {
    const x0 = M + i * (pw + GAP);
    const uid = `p${i}`;
    defs.push(hatch(`hatch-${uid}`));
    if (n > 1 || p.label) {
      const letter = n > 1 ? `${String.fromCharCode(97 + i)}  ` : "";
      parts.push(text(x0, top + 10, `${letter}${plain(p.label)}`, { size: 11.5, weight: 700 }));
    }
    parts.push(el("g", {}, p.body({ x0, x1: x0 + pw, y0: top + 22, y1: top + H }, uid)));
  });
  const capY = top + H + 22;
  capLines.forEach((l, i) => {
    const s = i === 0 ? `Figure. ${l}` : l;
    parts.push(text(M, capY + i * 15, s, { size: 11.5, fill: J.dark }));
  });
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${Htot}" viewBox="0 0 ${W} ${Htot}" ` +
    `font-family="${esc(J.font)}">` +
    el("defs", {}, defs) +
    parts.join("") +
    `</svg>`
  );
}

// ── axes ─────────────────────────────────────────────────────────────────────

function axisLeft(y: (v: number) => number, ticks: number[], box: Box): string {
  return ticks
    .map(
      (t) =>
        el("line", { x1: box.x0, x2: box.x1, y1: y(t), y2: y(t), stroke: t === 0 ? J.light : J.wash, "stroke-width": 0.8 }) +
        text(box.x0 - 6, y(t) + 3.5, fmtTick(t), { size: 9.5, anchor: "end", fill: J.dark }),
    )
    .join("");
}

function axisBottom(x: (v: number) => number, ticks: number[], box: Box, title?: string): string {
  return (
    el("line", { x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1, stroke: J.dark, "stroke-width": 0.8 }) +
    ticks
      .map(
        (t) =>
          el("line", { x1: x(t), x2: x(t), y1: box.y1, y2: box.y1 + 3, stroke: J.dark, "stroke-width": 0.8 }) +
          text(x(t), box.y1 + 14, fmtTick(t), { size: 9.5, anchor: "middle", fill: J.dark }),
      )
      .join("") +
    (title ? text((box.x0 + box.x1) / 2, box.y1 + 29, title, { size: 10.5, anchor: "middle" }) : "")
  );
}

// ── relationship ─────────────────────────────────────────────────────────────

function ols(points: [number, number][]): [number, number] {
  let sx = 0,
    sy = 0,
    sxx = 0,
    sxy = 0;
  for (const [x, y] of points) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const n = points.length;
  const den = n * sxx - sx * sx;
  if (!n || !den) return [0, n ? sy / n : 0];
  const slope = (n * sxy - sx * sy) / den;
  return [slope, (sy - slope * sx) / n];
}

export function relationshipPanel(st: RelationshipState, xLabel: string, xRange: [number, number]) {
  return (b: Box) => {
    const box = { x0: b.x0 + 44, x1: b.x1 - 8, y0: b.y0 + 18, y1: b.y1 - 34 };
    let lo = 0;
    let hi = 0;
    for (const [, y] of st.points) {
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
    }
    const yd = niceDomain(lo, hi || 1);
    const xd = niceDomain(xRange[0], xRange[1] || 1);
    const x = linear(xd, [box.x0, box.x1]);
    const y = linear(yd, [box.y1, box.y0]);
    const dots = st.points
      .map(([px, py]) => `M${(x(px) - 1.6).toFixed(1)},${y(py).toFixed(1)}a1.6,1.6 0 1,0 3.2,0a1.6,1.6 0 1,0 -3.2,0`)
      .join("");
    const [slope, ic] = st.fit ? [st.fit.slope, st.fit.intercept] : ols(st.points);
    return [
      axisLeft(y, niceTicks(yd[0], yd[1], 5), box),
      el("path", { d: dots, fill: J.dark, "fill-opacity": 0.42 }),
      el("line", {
        x1: x(xd[0]),
        y1: y(ic + slope * xd[0]),
        x2: x(xd[1]),
        y2: y(ic + slope * xd[1]),
        stroke: J.ink,
        "stroke-width": st.fit ? 1.6 : 1.1,
        "stroke-dasharray": st.fit ? undefined : "5 3",
      }),
      axisBottom(x, niceTicks(xd[0], xd[1], 6), box, xLabel),
      text(box.x0, b.y0 + 8, st.yLabel, { size: 10.5 }),
      el("text", { x: box.x1, y: b.y0 + 8, "text-anchor": "end", "font-size": 11, fill: J.ink }, [
        el("tspan", { "font-style": "italic" }, "r"),
        esc(` = ${fmtR(st.r)}`),
      ]),
    ].join("");
  };
}

// ── distribution ─────────────────────────────────────────────────────────────

const GROUP_NAME: Record<string, string> = { female: "women", male: "men" };

export function distributionPanel(
  st: DistributionState,
  marks: Mark[],
  levels: string[] | undefined,
  base: HistogramData | null,
) {
  return (b: Box, uid: string) => {
    const h = st.hist;
    if (levels) {
      const rows = levels.map((l, i) => ({ label: l, n: h.counts[i] ?? 0 }));
      if (h.n_missing) rows.push({ label: "blank", n: h.n_missing });
      const top = Math.max(1, ...rows.map((r) => r.n));
      const rowH = Math.min(30, (b.y1 - b.y0) / Math.max(1, rows.length));
      const x0 = b.x0 + 110;
      const x1 = b.x1 - 70;
      return rows
        .map((r, i) => {
          const y = b.y0 + i * rowH + rowH / 2;
          return (
            text(x0 - 8, y + 4, r.label, { size: 11, anchor: "end" }) +
            el("rect", { x: x0, y: y - 7, width: ((x1 - x0) * r.n) / top, height: 14, fill: J.light }) +
            text(x1 + 8, y + 4, fmtInt(r.n), { size: 11 })
          );
        })
        .join("");
    }
    const box = { x0: b.x0 + 6, x1: b.x1 - 6, y0: b.y0 + (marks.some((m) => m.group) ? 34 : 22), y1: b.y1 - 40 };
    const clip = clipTail(h, marks.map((m) => m.value));
    const lo = h.edges[0] ?? 0;
    const x = linear([Math.abs(lo) < 1e-12 ? 0 : lo, clip.hi], [box.x0, box.x1]);
    let top = 1;
    for (let k = 0; k < clip.bins; k++) top = Math.max(top, h.counts[k] ?? 0, base?.counts[k] ?? 0);
    const y = linear([0, top], [box.y1, box.y0]);
    const bars: string[] = [];
    for (let k = 0; k < clip.bins; k++) {
      const c = h.counts[k] ?? 0;
      const xa = x(h.edges[k]!) + 0.4;
      const xb = x(h.edges[k + 1]!) - 0.4;
      if (c > 0) bars.push(el("rect", { x: xa, y: y(c), width: Math.max(0, xb - xa), height: box.y1 - y(c), fill: J.pale, stroke: J.mid, "stroke-width": 0.5 }));
      const gone = base ? Math.max(0, (base.counts[k] ?? 0) - c) : 0;
      if (gone > 0)
        bars.push(el("rect", { x: xa, y: y(c + gone), width: Math.max(0, xb - xa), height: y(c) - y(c + gone), fill: `url(#hatch-${uid})`, stroke: J.mid, "stroke-width": 0.4 }));
    }
    const markSvg = marks
      .filter((m) => m.value >= (h.edges[0] ?? 0) && m.value <= clip.hi)
      .map((m) => {
        const row = m.group === "male" ? 1 : 0;
        const ly = b.y0 + 10 + row * 12;
        const name = m.group ? `${GROUP_NAME[m.group] ?? m.group} ` : "";
        return (
          el("line", { x1: x(m.value), x2: x(m.value), y1: ly + 3, y2: box.y1, stroke: J.ink, "stroke-width": 1, "stroke-dasharray": m.group === "male" ? "2 2" : "5 3" }) +
          text(x(m.value) + 3, ly, `${name}${m.label}`, { size: 9.5 })
        );
      })
      .join("");
    const note =
      clip.over > 0
        ? text(box.x0, b.y1 - 4, `Not drawn: ${fmtInt(clip.over)} rows from ${fmtTick(clip.hi)} to ${fmtTick(clip.max)}.`, { size: 9.5, italic: true, fill: J.dark })
        : "";
    return [
      text(box.x0, b.y0 + 8, st.label, { size: 10.5, weight: 700 }),
      ...bars,
      axisBottom(x, niceTicks(Math.abs(lo) < 1e-12 ? 0 : lo, clip.hi, 6), box),
      markSvg,
      note,
    ].join("");
  };
}

// ── lineage ──────────────────────────────────────────────────────────────────

export function lineagePanel(lineage: Lineage) {
  return (b: Box) => {
    const placed = placeLineage(lineage);
    const label = (n: (typeof placed.nodes)[number]) => (n.node.count > 1 ? n.node.label : (n.node.column ?? n.node.label));
    const fs = placed.rows > 16 ? 9.5 : 10.5;
    const cw = fs * 0.5;
    const rawW = Math.max(0, ...placed.nodes.filter((n) => n.lane === "raw").map((n) => label(n).length)) * cw + 10;
    const mxW = Math.max(0, ...placed.nodes.filter((n) => n.lane === "matrix").map((n) => label(n).length), ...placed.notes.map((n) => n.text.length)) * cw + 12;
    const x0 = b.x0 + rawW;
    const x2 = b.x1 - mxW;
    const x1 = (x0 + x2) / 2;
    const top = b.y0 + 22;
    const rowH = Math.min(24, (b.y1 - top) / Math.max(1, placed.rows));
    const yOf = (r: number) => top + r * rowH + rowH / 2;
    const xOf = (lane: string) => (lane === "raw" ? x0 : lane === "adjusted" ? x1 : x2);
    const out: string[] = [
      text(x0, b.y0 + 8, `RAW (${placed.counts.raw})`, { size: 9, anchor: "end", weight: 700 }),
      text(x1, b.y0 + 8, "ADJUSTED", { size: 9, anchor: "middle", weight: 700 }),
      text(x2, b.y0 + 8, `MATRIX (${placed.counts.matrix})`, { size: 9, weight: 700 }),
    ];
    for (const k of placed.links) {
      const xa = xOf(k.from.lane);
      const xb = xOf(k.to.lane);
      const ya = yOf(k.from.row);
      const yb = yOf(k.to.row);
      const m = (xa + xb) / 2;
      out.push(
        el("path", {
          d: `M${xa.toFixed(1)},${ya.toFixed(1)}C${m.toFixed(1)},${ya.toFixed(1)} ${m.toFixed(1)},${yb.toFixed(1)} ${xb.toFixed(1)},${yb.toFixed(1)}`,
          fill: "none",
          stroke: k.rewrite ? J.ink : J.light,
          "stroke-width": k.rewrite ? 1.1 : 0.8,
        }),
      );
    }
    for (const n of placed.nodes) {
      const x = xOf(n.lane);
      const y = yOf(n.row);
      out.push(el("circle", { cx: x, cy: y, r: n.node.count > 1 ? 3.6 : 2.3, fill: n.changed ? J.ink : J.dark }));
      const weight = n.changed || (n.lane === "matrix" && placed.nodes.some((m) => m.changed && m.identity === n.identity)) ? 700 : undefined;
      if (n.lane === "raw") out.push(text(x - 6, y + 3.5, label(n), { size: fs, anchor: "end" }));
      else if (n.lane === "matrix") out.push(text(x + 6, y + 3.5, label(n), { size: fs, weight }));
      else if (n.changed) out.push(text(x, y - 4, label(n), { size: fs - 1, anchor: "middle", italic: true }));
    }
    for (const note of placed.notes) out.push(text(x2 + 6, yOf(note.row) + 3.5, note.text, { size: fs, italic: true, fill: J.mid }));
    return out.join("");
  };
}

// ── row flow ─────────────────────────────────────────────────────────────────

export function rowFlowPanel(steps: RowStep[]) {
  return (b: Box, uid: string) => {
    const n0 = Math.max(1, steps[0]?.n ?? 1);
    const rowH = Math.min(30, (b.y1 - b.y0) / Math.max(1, steps.length));
    const lx = b.x0 + Math.min(240, (b.x1 - b.x0) * 0.38);
    const rx = b.x1 - 120;
    return steps
      .map((st, i) => {
        const y = b.y0 + i * rowH + rowH / 2;
        const kept = ((rx - lx) * st.n) / n0;
        const gone = ((rx - lx) * st.dropped) / n0;
        return (
          text(b.x0, y + 4, plain(st.label), { size: 11 }) +
          el("rect", { x: lx, y: y - 6, width: rx - lx, height: 12, fill: J.wash }) +
          el("rect", { x: lx, y: y - 6, width: kept, height: 12, fill: st.key === "holdout" ? J.pale : J.light }) +
          (gone > 0 ? el("rect", { x: lx + kept, y: y - 6, width: gone, height: 12, fill: `url(#hatch-${uid})` }) : "") +
          text(rx + 10, y + 4, fmtInt(st.n), { size: 11 }) +
          (st.dropped ? text(b.x1, y + 4, `−${fmtInt(st.dropped)}`, { size: 10, anchor: "end", fill: J.mid }) : "")
        );
      })
      .join("");
  };
}

// ── table ────────────────────────────────────────────────────────────────────

export function tablePanel(st: TableState, view: TableFocusView, showChanges: boolean) {
  return (b: Box) => {
    const cols = st.columns.slice(0, 8);
    const rows = view.rows.map((r) => r.row_id);
    const changed = new Set(view.changed.map(([r, c]) => `${r}|${c}`));
    const colW = (b.x1 - b.x0 - 60) / Math.max(1, cols.length);
    const rowH = Math.min(24, (b.y1 - b.y0 - 24) / Math.max(1, rows.length + 1));
    const out: string[] = [text(b.x0, b.y0 + 12, "row", { size: 10, fill: J.mid })];
    cols.forEach((c, j) => out.push(text(b.x0 + 60 + (j + 1) * colW - 6, b.y0 + 12, c, { size: 10, anchor: "end", weight: 700 })));
    out.push(el("line", { x1: b.x0, x2: b.x1, y1: b.y0 + 18, y2: b.y0 + 18, stroke: J.ink, "stroke-width": 0.8 }));
    rows.forEach((id, i) => {
      const y = b.y0 + 18 + (i + 1) * rowH;
      out.push(text(b.x0, y - 6, String(id), { size: 10, fill: J.mid }));
      cols.forEach((c, j) => {
        const v = st.values.get(id)?.[c];
        const s = v === null || v === undefined ? "blank" : typeof v === "number" ? fmtTick(v) : String(v);
        const x = b.x0 + 60 + j * colW;
        const hit = showChanges && (changed.has(`${id}|${c}`) || changed.has(`${id}|${view.columns_before[j] ?? ""}`));
        if (hit) out.push(el("rect", { x: x + 2, y: y - rowH + 3, width: colW - 4, height: rowH - 4, fill: J.wash }));
        out.push(text(x + colW - 6, y - 6, st.gone ? `${s} (leaves)` : s, { size: 10, anchor: "end", italic: v === null || v === undefined }));
      });
    });
    return out.join("");
  };
}

// ── a view's track as a figure ───────────────────────────────────────────────

/** A saved state is a whole step of the view's own storyboard, or the save is refused. */
export function assertRealState(index: number, last: number): void {
  if (!Number.isInteger(index) || index < 0 || index > last) {
    throw new Error(`Only a real state can be saved, not position ${index}.`);
  }
}

export function panelFor(track: Track, index: number): Panel {
  const view = track.view as ConsequenceView;
  const st = track.states[index]!;
  switch (view.kind) {
    case "relationship": {
      let lo = 0;
      let hi = 0;
      for (const s of track.states as RelationshipState[]) for (const [x] of s.points) {
        lo = Math.min(lo, x);
        hi = Math.max(hi, x);
      }
      return { label: st.label, body: relationshipPanel(st as RelationshipState, view.x_label, [lo, hi]) };
    }
    case "distribution": {
      const d = view as DistributionView;
      const marks = d.marks.length ? d.marks : (d.cuts ?? []).map((value) => ({ value, label: fmtInt(value), group: null }));
      const first = (track.states as DistributionState[])[0]!.hist;
      const s = st as DistributionState;
      const sameEdges = s.hist.edges.length === first.edges.length && s.hist.edges.every((e, i) => e === first.edges[i]);
      return {
        label: st.label,
        body: distributionPanel(s, sameEdges ? marks : [], d.levels, index > 0 && sameEdges ? first : null),
      };
    }
    case "lineage":
      return { label: st.label, body: lineagePanel((st as LineageState).lineage) };
    case "row_flow":
      return { label: st.label, body: rowFlowPanel((st as RowFlowState).steps) };
    case "table_focus":
      return { label: st.label, body: tablePanel(st as TableState, view, index > 0) };
  }
}

export function figureForTrack(
  track: Track,
  indices: number[],
  meta: { title: string; caption: string; provenance: string },
): string {
  const last = track.states.length - 1;
  indices.forEach((i) => assertRealState(i, last));
  return figureSvg({
    title: meta.title,
    caption: meta.caption,
    provenance: meta.provenance,
    panels: indices.map((i) => panelFor(track, i)),
  });
}
