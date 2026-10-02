/**
 * The seal moment (lifted from /lab/m2): the row flow forks into training and held out, and the
 * seal states its basis (M2_CONTRACT §3). One cell per row, each keeping its identity from the
 * file to its side; the draw's own steps are the storyboard. A grouped seal moves whole units; a
 * seal whose grouping was abandoned leaves a hole in each unit it split, joined by an amber line to
 * the row that crossed; an undetermined seal is drawn with a dashed frame, never a clean lock.
 *
 * The cells and counts come from the split preview (`RowFlowView.seal`, turbotab/core/seal_picture.py).
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import type { RowFlowView } from "../../../api/m1-stage-types";
import type { SealCells } from "../../../api/m2-stage-types";
import { useTheme } from "../../../theme";
import { fmtInt } from "../format";
import { eased } from "../player";
import { Rich } from "../text";
import { localPos, type Track } from "../tracks";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../usePlayer";
import { mix, mixRGB, prepare, rgba, token } from "../views/canvas";
import { useSize } from "../views/geometry";
import { RowFlow } from "../views/RowFlow";
import { useMorph } from "../views/useMorph";
import { basisText, forkPlan, lerpGeom, type SealGeom } from "./forkPlan";
import { glyphOf, isExploratory } from "./phase";
import { SealGlyph } from "./SealGlyph";
import s from "./seal.module.css";

const unitWord = (c: SealCells) => (c.column ? `\`${c.column}\` units` : "units");

export function SealFork({
  cells,
  recorded,
  globalLast,
}: {
  cells: SealCells;
  recorded: boolean;
  globalLast: number;
}) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const { theme } = useTheme();
  const [ref, { w }] = useSize<HTMLDivElement>();
  const p = useMemo(() => forkPlan(cells, w || 760), [cells, w]);
  const last = p.states.length - 1;
  const canvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const glyph = glyphOf(cells.state);
  const exploratory = isExploratory(cells.state, cells.exploratory);
  const at = Math.min(last, Math.round(localPos(ui.nearest, globalLast, last)));

  const paint = useCallback(
    (g: SealGeom) => {
      const el = canvas.current;
      if (!el || theme === undefined) return;
      const ctx = prepare(el, p.width, p.height);
      if (!ctx) return;
      const c4 = token(el, "--c4");
      const c1 = token(el, "--c1");
      const ink = token(el, "--ink");
      const line = token(el, "--line");
      const warn = token(el, "--warn");
      const ok = token(el, "--ok");
      const surface2 = token(el, "--surface-2");
      const c = p.cell;
      const n = g.x.length;
      if (g.boxes > 0.01) {
        const pad = 7;
        const bottom = p.splitHeight + 2;
        ctx.globalAlpha = g.boxes;
        ctx.lineWidth = 1;
        ctx.strokeStyle = rgba(line, 1);
        ctx.beginPath();
        ctx.roundRect(p.train[0] - 0.5, p.top - pad, p.train[1] - p.train[0] + pad - 2, bottom - p.top + pad - 1, 8);
        ctx.stroke();
        const tone = exploratory ? warn : recorded ? ok : ink;
        ctx.fillStyle = rgba(surface2, 0.5 + 0.5 * g.sealed);
        ctx.strokeStyle = rgba(tone, 0.55 + 0.45 * g.sealed);
        ctx.lineWidth = 1 + 0.8 * g.sealed;
        // An exploratory basis is never drawn as a clean lock: its frame stays dashed.
        ctx.setLineDash(exploratory ? [5, 4] : []);
        ctx.beginPath();
        ctx.roundRect(p.held[0] - pad, p.top - pad, p.held[1] - p.held[0] + pad - 1, bottom - p.top + pad - 1, 8);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      if (p.boundaryX !== null && g.boundary > 0.01) {
        ctx.globalAlpha = g.boundary * (1 - g.boxes);
        ctx.strokeStyle = rgba(ink, 0.7);
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(p.boundaryX + 0.5, 4);
        ctx.lineTo(p.boundaryX + 0.5, p.height - 4);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      if (p.holes.size) {
        ctx.strokeStyle = rgba(warn, 0.9);
        ctx.setLineDash([2, 2]);
        for (const [row, [hx, hy]] of p.holes) {
          const a = g.placed[row] ?? 0;
          if (a < 0.02) continue;
          ctx.globalAlpha = a;
          ctx.strokeRect(hx + 0.5, hy + 0.5, c - 1, c - 1);
        }
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      if (g.straddle > 0.01) {
        ctx.strokeStyle = rgba(warn, 0.95);
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 3]);
        ctx.globalAlpha = g.straddle;
        for (const [row, mate] of p.straddles) {
          const hole = p.holes.get(row);
          const t = g.placed[row] ?? 0;
          const fx = hole ? mix(g.x[mate]!, hole[0], t) : g.x[mate]!;
          const fy = hole ? mix(g.y[mate]!, hole[1], t) : g.y[mate]!;
          ctx.beginPath();
          ctx.moveTo(fx + c / 2, fy + c / 2);
          ctx.lineTo(g.x[row]! + c / 2, g.y[row]! + c / 2);
          ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      for (let i = 0; i < n; i++) {
        const mk = g.mark[i]!;
        const pl = g.placed[i]!;
        const held = cells.hold[i] === 1;
        const col = held ? mixRGB(c4, ink, mk) : mixRGB(c4, c1, pl);
        const a = held ? 0.5 + 0.35 * mk : 0.5 + 0.35 * pl;
        ctx.fillStyle = rgba(col, a);
        ctx.beginPath();
        ctx.roundRect(g.x[i]!, g.y[i]!, c, c, c > 12 ? 3 : 1.5);
        ctx.fill();
      }
      const o = overlay.current;
      if (o) {
        o.style.setProperty("--boxes", g.boxes.toFixed(3));
        o.style.setProperty("--sealed", g.sealed.toFixed(3));
        o.style.setProperty("--boundary", (g.boundary * (1 - g.boxes)).toFixed(3));
      }
    },
    [p, cells, recorded, exploratory, theme],
  );

  const target = useCallback(
    (pos: number) => {
      const lp = Math.max(0, Math.min(last, localPos(pos, globalLast, last)));
      const i = Math.floor(lp);
      return i >= last ? p.states[last]! : lerpGeom(p.states[i]!, p.states[i + 1]!, lp - i);
    },
    [p, last, globalLast],
  );
  // Another holdout size on the same side morphs from what is drawn (no storyboard replay).
  const redrawRef = useRef<() => void>(() => {});
  const blend = useMorph<SealGeom>(
    lerpGeom,
    useCallback(() => redrawRef.current(), []),
  );
  const morphKey = `${cells.hold.length}-${cells.n_holdout}-${cells.state}-${p.width}`;
  const draw = useCallback((pos: number) => paint(blend(target(pos), morphKey)), [paint, blend, target, morphKey]);
  useEffect(() => {
    redrawRef.current = () => draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  const nTrain = cells.n_rows - cells.n_holdout;
  const trainUnits = cells.n_units !== null && cells.n_holdout_units !== null ? cells.n_units - cells.n_holdout_units : null;
  const both =
    cells.straddle === null
      ? "on both sides: not known"
      : `${fmtInt(cells.straddle)} ${cells.straddle === 1 ? "unit" : "units"} on both sides`;
  return (
    <div className={s.fork} ref={ref} data-view="seal_fork" data-state={at} data-basis={cells.state}>
      <canvas ref={canvas} style={{ width: p.width, height: p.height }} />
      <div ref={overlay} className={s.forkOverlay} style={{ width: p.width, height: p.height }}>
        <span className={s.laneHead} style={{ left: p.train[0] - 2, opacity: "var(--boxes)" }}>
          Training · {fmtInt(nTrain)} rows
          {trainUnits !== null && cells.state !== "one_row_per_unit" ? ` · ${fmtInt(trainUnits)} units` : ""}
        </span>
        <span className={s.laneHead} style={{ left: p.held[0] - 2, opacity: "var(--boxes)" }}>
          Held out · {fmtInt(cells.n_holdout)} rows
          {cells.n_holdout_units !== null && cells.state !== "one_row_per_unit" ? ` · ${fmtInt(cells.n_holdout_units)} units` : ""}
        </span>
        <span className={s.sealBadge} style={{ left: p.held[1] - 24, top: 0 }}>
          <SealGlyph state={glyph} recorded={recorded} />
        </span>
        {p.axis.map((a, i) => (
          <span
            key={`${a.label}-${i}`}
            className={s.axisLabel}
            data-align={i === 0 ? "start" : i === 1 ? "end" : "middle"}
            style={{ left: a.x, opacity: "var(--boundary)" }}
          >
            {a.label}
          </span>
        ))}
      </div>
      <p className={s.basisLine} data-exploratory={exploratory || undefined} data-shown={at === last || undefined} data-testid="seal-basis">
        <span className={s.basisKicker}>{exploratory ? "Sealed by row · exploratory" : recorded ? "Sealed" : "Would seal"}</span>
        <span className={s.basisText}>
          <Rich text={basisText(cells)} />
        </span>
        <span className={s.basisFact}>{both}</span>
      </p>
      {cells.evidence ? (
        <p className={s.evidence} data-testid="seal-evidence">
          <Rich text={cells.evidence} />
        </p>
      ) : null}
      {cells.hold.length < cells.n_rows ? (
        <p className={s.sampled}>
          Drawn: the first {fmtInt(cells.hold.length)} of {fmtInt(cells.n_rows)} rows
          {cells.unit ? <>, whole <Rich text={unitWord(cells)} /></> : null}; the counts are over every row.
        </p>
      ) : null}
    </div>
  );
}

/** The split's preview: the fork, then the row flow it ends in. */
export function SealForkTrack({
  track,
  globalLast,
  compact,
}: {
  track: Track<RowFlowView>;
  globalLast: number;
  compact?: boolean;
}) {
  const ui = usePlayerUi(usePlayerStore());
  const cells = track.view.seal!;
  const localLast = track.states.length - 1;
  const shown = Math.min(localLast, Math.round(localPos(ui.nearest, globalLast, localLast)));
  if (compact) {
    return (
      <div data-state={shown}>
        <RowFlow steps={track.states[shown]!.steps} compact preview={shown > 0} emphasis={track.view.emphasis} />
      </div>
    );
  }
  return (
    <div className={s.forkTrack} data-state={shown}>
      <SealFork cells={cells} recorded={false} globalLast={globalLast} />
      <RowFlow
        steps={track.states[shown]!.steps}
        compact
        preview={shown > 0}
        emphasis={track.view.emphasis}
        coach={track.view.coach}
      />
    </div>
  );
}
