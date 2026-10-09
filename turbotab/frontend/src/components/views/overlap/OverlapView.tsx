/**
 * Overlap (FOUNDATION §5 rule 9): two groups' distributions on one scale, mirrored about one
 * baseline, with the trimmed region shown. It serves Models before any estimate (propensity
 * overlap, covariate overlap) and the positivity exhibit after Fit. It draws no outcome and no
 * estimate, so it may open before the lock (§5 rule 6).
 *
 * Each group keeps its categorical slot (sage above, plum below, by declared order). Bars the
 * trim removes are indigo while the trim is the option pointed at (what the choice touches), and
 * gray, drawn lighter, once it is recorded.
 */
import { fmtInt, fmtTick } from "../../stage/format";
import { Legend, ViewFrame, barPath, fitText, fmtPct, slotColor, useTip, useWidth, type Slot } from "../common/frame";
import v from "../common/views.module.css";
import { keepLine, labelSide, layoutOverlap, OVERLAP } from "./layout";
import type { OverlapInput } from "./types";

export function OverlapView({ input, title }: { input: OverlapInput; title?: string }) {
  const [ref, W] = useWidth();
  const tip = useTip();
  const res = layoutOverlap(input, W);
  const slots = input.groups.map((g, i) => g.slot ?? ((i + 1) as Slot));
  if ("empty" in res) {
    return <ViewFrame title={title} empty={res.empty} table={() => null} frameRef={ref} kind="overlap">{null}</ViewFrame>;
  }
  const l = res.layout;
  const choice = (input.keep_state ?? "choice") === "choice";
  const unit = input.unit ? ` ${input.unit}` : "";
  const range = (lo: number, hi: number) => `${fmtTick(lo)} to ${fmtTick(hi)}${unit}`;
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

  const fillOf = (gi: number, trimmed: boolean) =>
    trimmed ? (choice ? "var(--data-affected)" : "var(--data-context)") : slotColor(slots[gi]!);

  return (
    <ViewFrame
      title={title}
      caption={line ?? `Each group's share of its own rows, by ${input.x_label.charAt(0).toLowerCase()}${input.x_label.slice(1)}.`}
      legend={<Legend items={input.groups.map((g, i) => ({ key: String(i), label: `${g.label} · ${fmtInt(l.totals[i]!)} rows`, slot: slots[i]! }))} />}
      table={table}
      frameRef={ref}
      kind="overlap"
    >
      <svg className={v.svg} viewBox={`0 0 ${l.width} ${l.height}`} role="img" aria-label={`${input.groups[0].label} above and ${input.groups[1].label} below, by ${input.x_label}`}>
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
            const label = `${input.groups[gi]!.label} · ${range(b.lo, b.hi)} · ${fmtInt(g.count)} ${g.count === 1 ? "row" : "rows"}, ${fmtPct(g.share)} of the group${b.trimmed ? " · trimmed" : ""}`;
            return (
              <g key={`${b.i}-${gi}`} data-bin={b.i} data-group={gi} data-trimmed={b.trimmed || undefined}>
                <path
                  d={barPath(b.x0 + OVERLAP.gap, base, w, g.count ? Math.max(1, g.h) : 0, dir)}
                  style={{ fill: fillOf(gi, b.trimmed), opacity: b.trimmed && !choice ? 0.45 : 1 }}
                />
                {/* the hit target: the bin's whole half-column, larger than the bar */}
                <rect
                  x={b.x0}
                  width={b.x1 - b.x0}
                  y={gi === 0 ? l.top : l.mid}
                  height={l.half}
                  fill="transparent"
                  {...tip.on(label)}
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
                  strokeWidth={1.5}
                />
              ) : null,
            )}
          </g>
        ) : null}
        {/* direct labels, in ink, on the side where each group's bars are low, inside the kept range */}
        {([0, 1] as const).map((gi) => {
          const side = labelSide(l.bins, gi);
          const x = side === "start" ? Math.max(l.left, l.keep?.xlo ?? l.left) + 6 : Math.min(l.right, l.keep?.xhi ?? l.right) - 6;
          const y = gi === 0 ? l.top + 12 : l.bottom - 6;
          return (
            <text key={gi} className={v.direct} x={x} y={y} textAnchor={side}>
              {input.groups[gi]!.label}
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
