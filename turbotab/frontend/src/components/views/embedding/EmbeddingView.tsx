/**
 * Embedding (FOUNDATION §5 rule 9): each row placed on two derived axes (PCA or UMAP), colored by
 * a declared grouping in the categorical slots' fixed order. It serves First look before the fit.
 * Above `EMBED.densityAbove` rows it draws density: squares colored by the group most of their rows
 * belong to, darker for more rows. Up to four groups are named on the drawing, each by its rows;
 * more take a legend. Pointing at a group's name isolates it; the arrow keys read each mark.
 *
 * The axes carry names and, for PCA, each component's share of the spread; never numbers, which
 * would imply a unit.
 */
import { useState } from "react";
import { fmtInt, fmtNum } from "../../stage/format";
import { Legend, ViewFrame, fitText, rowsWord, slotColor, useKeyMarks, useTip, useWidth, type Slot } from "../common/frame";
import v from "../common/views.module.css";
import { EMBED, LABEL_CHAR_PX, OPACITY, keyOf, layoutEmbedding, nearest, type EmbedLayout } from "./layout";
import type { EmbeddingInput } from "./types";

const TABLE_ROWS = 300;

function slotOfKey(key: string): Slot | null {
  const m = /^g(\d)$/.exec(key);
  return m ? ((Number(m[1]) + 1) as Slot) : null;
}

function explanation(input: EmbeddingInput, mode: EmbedLayout["mode"]): string {
  const axes =
    input.method === "pca"
      ? "Each axis mixes the columns and has no unit; rows near each other are alike across them."
      : "Rows near each other are alike; the axes have no unit, and gaps between clusters mean little.";
  const by = input.grouping ? ` Colored by ${input.grouping.name}.` : "";
  const density = mode === "density" ? " Darker squares hold more rows." : "";
  return `${input.basis ? `${input.basis}.` : ""}${by} ${axes}${density}`.trim();
}

export function EmbeddingView({ input, title }: { input: EmbeddingInput; title?: string }) {
  const [ref, W] = useWidth();
  const [focus, setFocus] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const tip = useTip();
  const res = layoutEmbedding(input, W, focus);
  const lay = "layout" in res ? res.layout : null;
  const levels = input.grouping?.levels ?? [];
  const levelOf = (i: number) => {
    const key = keyOf(input.groups?.[i], levels.length);
    return key === "none" ? "not recorded" : levels[input.groups![i]!]!;
  };
  const nameOf = (i: number) => input.ids?.[i] ?? `Row ${fmtInt(i + 1)}`;
  const legendLabel = (key: string) => lay?.legend.find((it) => it.key === key)?.label ?? key;
  const pointText = (i: number) => (input.grouping ? `${nameOf(i)} · ${input.grouping.name}: ${levelOf(i)}` : nameOf(i));
  const cellText = (c: { count: number; by: Record<string, number> }) => {
    const parts = Object.entries(c.by)
      .map(([k, n]) => `${legendLabel(k)} ${fmtInt(n)}`)
      .join(", ");
    return `${rowsWord(c.count)}${input.grouping ? ` · ${parts}` : ""}`;
  };
  // The keyboard reads the marks left to right.
  const marks = !lay
    ? []
    : lay.mode === "points"
      ? [...lay.points].sort((a, b) => a.px - b.px || a.py - b.py).map((p) => ({ id: p.i, x: p.px, y: p.py, text: pointText(p.i) }))
      : [...lay.cells].sort((a, b) => a.cx - b.cx || a.cy - b.cy).map((c) => ({ id: -1, x: c.cx, y: c.cy, text: cellText(c) }));
  const keys = useKeyMarks(marks, tip, lay?.width ?? W, (id) => setHover(id !== null && id >= 0 ? id : null));
  if (!lay) {
    return <ViewFrame title={title} empty={"empty" in res ? res.empty : null} table={() => null} frameRef={ref} kind="embedding">{null}</ViewFrame>;
  }
  const l = lay;

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const svg = e.currentTarget.ownerSVGElement!;
    const box = svg.getBoundingClientRect();
    const sx = l.width / (box.width || l.width);
    const m = nearest(l, (e.clientX - box.left) * sx, (e.clientY - box.top) * sx);
    if (!m) {
      tip.hide();
      setHover(null);
      return;
    }
    if ("i" in m) {
      setHover(m.i);
      tip.show(pointText(m.i), e.clientX, e.clientY);
    } else {
      setHover(null);
      tip.show(cellText(m), e.clientX, e.clientY);
    }
  };

  const table = () => {
    const shown = Math.min(TABLE_ROWS, input.xs.length);
    return (
      <>
        {l.legend.length ? (
          <table className={v.table} style={{ marginBottom: 12 }}>
            <thead>
              <tr>
                <th scope="col">{input.grouping!.name}</th>
                <th scope="col">Rows</th>
              </tr>
            </thead>
            <tbody>
              {l.legend.map((it) => (
                <tr key={it.key}>
                  <td>{it.label}</td>
                  <td>{fmtInt(it.count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        <table className={v.table}>
          {shown < input.xs.length ? (
            <caption className={v.caption} style={{ textAlign: "left" }}>
              The first {fmtInt(shown)} of {fmtInt(input.xs.length)} rows
            </caption>
          ) : null}
          <thead>
            <tr>
              <th scope="col">Row</th>
              {input.grouping ? <th scope="col">{input.grouping.name}</th> : null}
              <th scope="col">{input.axes[0].label}</th>
              <th scope="col">{input.axes[1].label}</th>
            </tr>
          </thead>
          <tbody>
            {input.xs.slice(0, shown).map((x, i) => (
              <tr key={i}>
                <td>{nameOf(i)}</td>
                {input.grouping ? <td>{levelOf(i)}</td> : null}
                <td>{fmtNum(x)}</td>
                <td>{fmtNum(input.ys[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    );
  };

  const fillOf = (key: string) => (input.grouping ? slotColor(slotOfKey(key)) : "var(--data-context)");
  const { plot } = l;
  const hovered = hover !== null ? l.points.find((p) => p.i === hover) : undefined;

  return (
    <ViewFrame
      title={title}
      caption={explanation(input, l.mode)}
      legend={l.labels.length ? null : <Legend items={l.legend.map((it) => ({ ...it, label: `${it.label} · ${rowsWord(it.count)}` }))} focus={focus} onFocus={setFocus} />}
      table={table}
      frameRef={ref}
      kind="embedding"
    >
      <svg className={v.svg} viewBox={`0 0 ${l.width} ${l.height}`} role="img" aria-label={`${rowsWord(l.n)} on ${l.axisTitles[0]} and ${l.axisTitles[1]}${input.grouping ? `, colored by ${input.grouping.name}` : ""}; the arrow keys read each mark`} data-mode={l.mode} {...keys.props}>
        <text className={v.axisTitle} x={plot.x0} y={14}>
          <title>{l.axisTitles[1]}</title>
          {fitText(l.axisTitles[1], plot.x1 - plot.x0)}
        </text>
        <line x1={plot.x0} x2={plot.x0} y1={plot.y0} y2={plot.y1} style={{ stroke: "var(--canvas-line)" }} />
        <line x1={plot.x0} x2={plot.x1} y1={plot.y1} y2={plot.y1} style={{ stroke: "var(--canvas-line)" }} />
        <text className={v.axisTitle} x={plot.x1} y={l.height - 6} textAnchor="end">
          <title>{l.axisTitles[0]}</title>
          {fitText(l.axisTitles[0], plot.x1 - plot.x0)}
        </text>
        {l.mode === "points" ? (
          <g>
            {l.points.map((p) => (
              <circle
                key={p.i}
                cx={p.px}
                cy={p.py}
                r={EMBED.radius}
                data-key={p.key}
                style={{
                  fill: fillOf(p.key),
                  stroke: "var(--canvas)",
                  strokeWidth: 1,
                  opacity: focus === null || focus === p.key ? 0.85 : 0.12,
                }}
              />
            ))}
            {hovered ? <circle cx={hovered.px} cy={hovered.py} r={EMBED.radius + 2.5} fill="none" style={{ stroke: "var(--canvas-ink)" }} strokeWidth={1.5} /> : null}
          </g>
        ) : (
          <g>
            {l.cells.map((c) => (
              <rect
                key={`${c.col},${c.row}`}
                x={c.cx - EMBED.cell / 2 + 0.5}
                y={c.cy - EMBED.cell / 2 + 0.5}
                width={EMBED.cell - 1}
                height={EMBED.cell - 1}
                rx={1.5}
                data-key={c.key}
                style={{ fill: fillOf(c.key), opacity: OPACITY[c.step - 1] }}
              />
            ))}
          </g>
        )}
        <rect
          x={plot.x0}
          y={plot.y0}
          width={plot.x1 - plot.x0}
          height={plot.y1 - plot.y0}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => {
            tip.hide();
            setHover(null);
          }}
        />
        {/* direct labels: each group by its rows, a square key in its color, the text in ink; pointing isolates it */}
        {l.labels.map((lb) => {
          const half = (lb.text.length * LABEL_CHAR_PX) / 2;
          return (
            <g
              key={lb.key}
              data-label={lb.key}
              onPointerEnter={() => setFocus(lb.key)}
              onPointerLeave={() => setFocus(null)}
              style={{ opacity: focus === null || focus === lb.key ? 1 : 0.45 }}
            >
              {/* a square key, as in a legend, so it never reads as one more point */}
              <rect x={lb.x - half - 9} y={lb.y - 9} width={10} height={10} rx={2} style={{ fill: fillOf(lb.key), stroke: "var(--canvas)", strokeWidth: 1.5 }} />
              <text className={`${v.direct} ${v.halo}`} x={lb.x + 4} y={lb.y} textAnchor="middle">
                {lb.text}
              </text>
            </g>
          );
        })}
      </svg>
      {tip.node}
    </ViewFrame>
  );
}
