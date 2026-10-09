/**
 * Overlap (FOUNDATION §5 rule 9): two groups' distributions on one scale, mirrored about one
 * baseline, with the trimmed region shown. It serves Models before any estimate (propensity
 * overlap, covariate overlap) and the positivity exhibit after Fit. It draws no outcome and no
 * estimate, so it may open before the lock (§5 rule 6).
 *
 * Each group keeps its categorical slot (sage above, plum below, by declared order) and is named
 * on the drawing with its rows (two series: direct labels, no legend). Bars the trim removes are
 * indigo while the trim is the option pointed at (what the choice touches), and gray, your data now,
 * once it is recorded; a one-item key names them. The arrow keys read each bar.
 */
import { fmtInt, fmtTick } from "../../stage/format";
import { DATA_CONTEXT_STRONG, Legend, ViewFrame, barPath, fitText, fmtPct, rowsWord, slotColor, useKeyMarks, useTip, useWidth, type Slot } from "../common/frame";
import v from "../common/views.module.css";
import { keepLine, labelSide, labelY, layoutOverlap, OVERLAP } from "./layout";
import type { OverlapInput } from "./types";

export function OverlapView({ input, title }: { input: OverlapInput; title?: string }) {
  const [ref, W] = useWidth();
  const tip = useTip();
  const res = layoutOverlap(input, W);
  const slots = input.groups.map((g, i) => g.slot ?? ((i + 1) as Slot));
  const unit = input.unit ? ` ${input.unit}` : "";
  const range = (lo: number, hi: number) => `${fmtTick(lo)} to ${fmtTick(hi)}${unit}`;
  const lay = "layout" in res ? res.layout : null;
  const barText = (b: NonNullable<typeof lay>["bins"][number], gi: number) => {
    const g = b.groups[gi]!;
    return `${input.groups[gi]!.label} · ${range(b.lo, b.hi)} · ${rowsWord(g.count)}, ${fmtPct(g.share)} of the group${b.trimmed ? " · trimmed" : ""}`;
  };
  // The keyboard reads the bins left to right, the upper group then the lower in each.
  const keys = useKeyMarks(
    (lay?.bins ?? []).flatMap((b) =>
      ([0, 1] as const).map((gi) => ({ id: `${b.i}-${gi}`, x: (b.x0 + b.x1) / 2, y: gi === 0 ? lay!.mid - b.groups[0]!.h : lay!.mid, text: barText(b, gi) })),
    ),
    tip,
    lay?.width ?? W,
  );
  if (!lay) {
    return <ViewFrame title={title} empty={"empty" in res ? res.empty : null} table={() => null} frameRef={ref} kind="overlap">{null}</ViewFrame>;
  }
  const l = lay;
  const choice = (input.keep_state ?? "choice") === "choice";
  const line = keepLine(input, l);
  const axisTitle = `${input.x_label}${input.unit ? ` (${input.unit})` : ""}`;

  const table = () => (
    <table className={v.table}>
      <caption className={v.caption} style={{ textAlign: "left" }}>
        Rows in each bin, and each group's share of its own rows
      </caption>
      <thead>
        <tr>
          <th scope="col">{input.scale === "propensity" ? "Chance" : "Value"}</th>
          {input.groups.map((g) => (
            <th key={g.label} scope="col">
              {g.label}
            </th>
          ))}
          {input.keep ? <th scope="col">Trimmed</th> : null}
        </tr>
      </thead>
      <tbody>
        {l.bins.map((b) => (
          <tr key={b.i}>
            <td>{range(b.lo, b.hi)}</td>
            {b.groups.map((g, gi) => (
              <td key={gi}>
                {fmtInt(g.count)} ({fmtPct(g.share)})
              </td>
            ))}
            {input.keep ? <td>{b.trimmed ? "yes" : b.split ? "partly" : ""}</td> : null}
          </tr>
        ))}
        <tr>
          <th scope="row" style={{ textAlign: "left" }}>
            All
          </th>
          {l.totals.map((t, i) => (
            <td key={i}>{fmtInt(t)}</td>
          ))}
          {input.keep ? <td>{l.keep?.nTrimmed === null || !l.keep ? "" : fmtInt(l.keep.nTrimmed)}</td> : null}
        </tr>
      </tbody>
    </table>
  );

  const trimFill = choice ? "var(--data-affected)" : DATA_CONTEXT_STRONG;
  const fillOf = (gi: number, trimmed: boolean) => (trimmed ? trimFill : slotColor(slots[gi]!));

  return (
    <ViewFrame
      title={title}
      caption={line ?? `Each group's share of its own rows, by ${input.x_label.charAt(0).toLowerCase()}${input.x_label.slice(1)}.`}
      legend={
        l.keep && l.bins.some((b) => b.trimmed) ? (
          <Legend keyOnly items={[{ key: "trim", label: choice ? "would be trimmed" : "trimmed", slot: null, color: trimFill }]} />
        ) : null
      }
      table={table}
      frameRef={ref}
      kind="overlap"
    >
      <svg
        className={v.svg}
        viewBox={`0 0 ${l.width} ${l.height}`}
        role="img"
        aria-label={`${input.groups[0].label} above and ${input.groups[1].label} below, by ${input.x_label}; the arrow keys read each bar`}
        {...keys.props}
      >
        <text className={v.axisTitle} x={l.left} y={12}>
          Share of each group
        </text>
        {/* recessive grid: the shares above and below, one scale */}
        {l.yTicks.map((t) => (
          <g key={t.value}>
            <line x1={l.left} x2={l.right} y1={t.up} y2={t.up} style={{ stroke: "var(--canvas-line)" }} />
            <line x1={l.left} x2={l.right} y1={t.down} y2={t.down} style={{ stroke: "var(--canvas-line)" }} />
            <text className={v.axis} x={l.left - 6} y={t.up + 4} textAnchor="end">
              {t.label}
            </text>
            <text className={v.axis} x={l.left - 6} y={t.down + 4} textAnchor="end">
              {t.label}
            </text>
          </g>
        ))}
        {l.bins.map((b) => {
          const w = Math.max(0.5, b.x1 - b.x0 - 2 * OVERLAP.gap);
          return b.groups.map((g, gi) => {
            const dir = gi === 0 ? -1 : 1;
            const base = l.mid + dir * 0.5;
            return (
              <g key={`${b.i}-${gi}`} data-bin={b.i} data-group={gi} data-trimmed={b.trimmed || undefined}>
                <path
                  d={barPath(b.x0 + OVERLAP.gap, base, w, g.count ? Math.max(1, g.h) : 0, dir)}
                  style={{ fill: fillOf(gi, b.trimmed) }}
                />
                {keys.active === `${b.i}-${gi}` ? (
                  <rect x={b.x0 + 0.5} width={Math.max(1, b.x1 - b.x0 - 1)} y={gi === 0 ? l.top : l.mid} height={l.half} rx={2} fill="none" style={{ stroke: "var(--canvas-ink)" }} strokeWidth={1.5} />
                ) : null}
                {/* the hit target: the bin's whole half-column, larger than the bar */}
                <rect
                  x={b.x0}
                  width={b.x1 - b.x0}
                  y={gi === 0 ? l.top : l.mid}
                  height={l.half}
                  fill="transparent"
                  {...tip.on(barText(b, gi))}
                />
              </g>
            );
          });
        })}
        <line x1={l.left} x2={l.right} y1={l.mid} y2={l.mid} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} />
        {l.keep ? (
          <g aria-hidden="true">
            {[l.keep.xlo, l.keep.xhi].map((x, i) =>
              x > l.left && x < l.right ? (
                <line
                  key={i}
                  x1={x}
                  x2={x}
                  y1={l.top}
                  y2={l.bottom}
                  style={{ stroke: choice ? "var(--data-affected)" : "var(--canvas-muted)" }}
                  strokeWidth={2}
                />
              ) : null,
            )}
          </g>
        ) : null}
        {/* direct labels with each group's rows, in ink, on the side where its bars are low, inside the
            kept range and clear of the gridlines: two series, so no legend repeats them */}
        {([0, 1] as const).map((gi) => {
          const side = labelSide(l.bins, gi);
          const x = side === "start" ? Math.max(l.left, l.keep?.xlo ?? l.left) + 6 : Math.min(l.right, l.keep?.xhi ?? l.right) - 6;
          const text = `${input.groups[gi]!.label} · ${rowsWord(l.totals[gi]!)}`;
          const shown = fitText(text, (l.right - l.left) / 2);
          return (
            <text key={gi} className={`${v.direct} ${v.halo}`} x={x} y={labelY(l, gi)} textAnchor={side} data-direct={gi}>
              {shown === text ? null : <title>{text}</title>}
              {shown}
            </text>
          );
        })}
        {l.xTicks.map((t) => (
          <text key={t.value} className={v.axis} x={t.px} y={l.bottom + 15} textAnchor={t.px <= l.left + 1 ? "start" : t.px >= l.right - 1 ? "end" : "middle"}>
            {t.label}
          </text>
        ))}
        <text className={v.axisTitle} x={l.right} y={l.height - 4} textAnchor="end">
          <title>{axisTitle}</title>
          {fitText(axisTitle, l.right - l.left)}
        </text>
      </svg>
      {tip.node}
    </ViewFrame>
  );
}
