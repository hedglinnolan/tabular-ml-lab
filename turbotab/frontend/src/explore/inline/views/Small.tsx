/** Evidence pictures for findings, at sparkline scale: shares of a column and level counts. */
import { useId } from "react";
import { fmtInt } from "../format";
import s from "./views.module.css";

export function ShareBar({
  parts,
  width,
  height = 10,
  title,
}: {
  parts: { label: string; n: number; tone: "on" | "off" | "blank" }[];
  width: number;
  height?: number;
  title: string;
}) {
  const id = useId().replace(/:/g, "");
  const total = parts.reduce((a, p) => a + p.n, 0) || 1;
  const starts = parts.map((_, i) =>
    parts.slice(0, i).reduce((a, p) => a + (p.n / total) * width, 0),
  );
  return (
    <svg width={width} height={height} className={s.share} role="img" aria-label={title}>
      <defs>
        <pattern id={`blank-hatch-${id}`} width="4" height="4" patternUnits="userSpaceOnUse">
          <rect width="4" height="4" className={s.hatchGround} />
          <path d="M0 4 L4 0" className={s.hatchLine} />
        </pattern>
      </defs>
      {parts.map((p, i) => {
        const w = (p.n / total) * width;
        return (
          <rect
            key={p.label}
            x={starts[i]}
            y={0}
            width={Math.max(0, w - 1)}
            height={height}
            rx={1.5}
            data-tone={p.tone}
            style={p.tone === "blank" ? { fill: `url(#blank-hatch-${id})` } : undefined}
          >
            <title>{`${p.label}: ${fmtInt(p.n)}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export function LevelBars({
  levels,
  width,
  height,
  title,
}: {
  levels: { label: string; n: number }[];
  width: number;
  height: number;
  title: string;
}) {
  const max = Math.max(1, ...levels.map((l) => l.n));
  const barH = Math.min(15, (height - 4) / levels.length - 5);
  const labelW = 44;
  return (
    <svg width={width} height={height} className={s.levels} role="img" aria-label={title}>
      {levels.map((l, i) => {
        const y = 4 + i * (barH + 7);
        return (
          <g key={l.label}>
            <text x={0} y={y + barH / 2} dy="0.34em" className={s.levelName}>
              {l.label}
            </text>
            <rect x={labelW} y={y} width={(l.n / max) * (width - labelW - 2)} height={barH} rx={2}>
              <title>{`${l.label}: ${fmtInt(l.n)}`}</title>
            </rect>
            <text x={labelW + 5} y={y + barH / 2} dy="0.34em" className={s.levelQ}>
              = ?
            </text>
          </g>
        );
      })}
    </svg>
  );
}
