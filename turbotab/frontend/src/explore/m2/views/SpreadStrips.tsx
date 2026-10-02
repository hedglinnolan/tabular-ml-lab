/**
 * The orientation reading made visible (turbotab/orientation.py): the spread of row means against
 * the spread of column means, on a log scale. Every dot is one feature's mean or one sample's
 * mean; turning the table does not change a single value, only which axis it is a mean along —
 * so the feature dots cross from the rows lane to the columns lane at the same x, and the ratio
 * that made the question fire (23, above the threshold of 4) inverts.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { scaleLinear } from "d3-scale";
import { eased } from "../../../components/stage/player";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { dotPath, useSize } from "../../../components/stage/views/geometry";
import { fmtNum } from "../data";
import type { OrientationFixture } from "../types";
import s from "./views.module.css";

const LANE_A = 26;
const LANE_B = 74;
const H = 104;

function jitter(i: number): number {
  const x = Math.sin(i * 12.9898) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 16;
}

export function SpreadStrips({ fx, last }: { fx: OrientationFixture; last: number }) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const [ref, { w }] = useSize<HTMLDivElement>();
  const at = Math.min(3, Math.round((ui.nearest * 3) / Math.max(1, last)));
  const width = Math.max(200, w);
  const feats = fx.before.row_means; // one per feature; equal to the after table's column means
  const samps = fx.before.col_means; // one per sample; equal to the after table's row means
  const x = useMemo(() => {
    const all = [...feats, ...samps];
    return scaleLinear()
      .domain([Math.floor(Math.min(...all)), Math.ceil(Math.max(...all))])
      .range([132, width - 10]);
  }, [feats, samps, width]);

  const featPath = useRef<SVGPathElement>(null);
  const sampPath = useRef<SVGPathElement>(null);
  const draw = useCallback(
    (pos: number) => {
      const t = Math.max(0, Math.min(1, (pos * 3) / Math.max(1, last) - 1));
      const fv = new Float64Array(feats.length * 2);
      feats.forEach((v, i) => {
        fv[i * 2] = x(v);
        fv[i * 2 + 1] = LANE_A + (LANE_B - LANE_A) * t + jitter(i);
      });
      const sv = new Float64Array(samps.length * 2);
      samps.forEach((v, i) => {
        sv[i * 2] = x(v);
        sv[i * 2 + 1] = LANE_B + (LANE_A - LANE_B) * t + jitter(i + 1000) * 0.6;
      });
      featPath.current?.setAttribute("d", dotPath(fv, feats.length, 1.8));
      sampPath.current?.setAttribute("d", dotPath(sv, samps.length, 2.2));
    },
    [feats, samps, x, last],
  );
  useEffect(() => {
    draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  const shape = at >= 2 ? fx.after : fx.before;
  const ticks = x.ticks(6);
  return (
    <div className={s.ss} ref={ref} data-view="spread_strips" data-state={at}>
      {w > 0 ? (
        <svg width={width} height={H + 18} className={s.svg} role="img"
          aria-label="Spread of row means and column means, log scale">
          <text x={0} y={LANE_A + 4} className={s.laneLabel}>across rows</text>
          <text x={0} y={LANE_B + 4} className={s.laneLabel}>across columns</text>
          <text x={96} y={LANE_A + 4} className={s.laneValue}>{fmtNum(shape.s_rows, 2)}</text>
          <text x={96} y={LANE_B + 4} className={s.laneValue}>{fmtNum(shape.s_cols, 2)}</text>
          <line x1={132} x2={width - 10} y1={H} y2={H} className={s.axisLine} />
          {ticks.map((t) => (
            <text key={t} x={x(t)} y={H + 14} textAnchor="middle" className={s.tick}>
              {`10${superscript(t)}`}
            </text>
          ))}
          <path ref={featPath} className={s.dotFeature} />
          <path ref={sampPath} className={s.dotSample} />
        </svg>
      ) : null}
      <div className={s.spreadLegend}>
        <span className={s.keyFeature} /> a feature&apos;s mean · {fmtNum(feats.length)}
        <span className={s.keySample} /> a sample&apos;s mean · {fmtNum(samps.length)}
        <span className={s.keyAxis}>
          ratio {fmtNum(shape.ratio, 2)} · reads feature-major above {fmtNum(fx.threshold)}
        </span>
      </div>
    </div>
  );
}

const SUP: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
function superscript(n: number): string {
  return String(n)
    .split("")
    .map((c) => SUP[c] ?? c)
    .join("");
}
