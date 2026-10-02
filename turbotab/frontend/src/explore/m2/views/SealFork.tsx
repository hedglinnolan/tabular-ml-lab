/**
 * The seal moment: the row flow forks into training and held out, and the seal states its basis
 * (M2_CONTRACT §3). One cell per row of the table, each keeping its identity from the file to its
 * side; the draw's own steps are the storyboard. A grouped seal moves whole units; a seal whose
 * grouping was abandoned leaves a hole in each unit it split, joined by an amber line to the row
 * that crossed; an undetermined seal is drawn with a dashed frame, never a clean lock.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { eased } from "../../../components/stage/player";
import { usePlayerFrame, usePlayerStore, usePlayerUi } from "../../../components/stage/usePlayer";
import { useSize } from "../../../components/stage/views/geometry";
import { Rich } from "../../../components/stage/text";
import { useTheme } from "../../../theme";
import { fmtInt } from "../data";
import { lerpGeom, plan, type SealGeom } from "../seal";
import type { SealVariant } from "../types";
import { useMorph } from "../useMorph";
import { mix, mixRGB, prepare, rgba, token } from "./canvas";
import { SealGlyph, sealStateOf } from "./SealGlyph";
import s from "./views.module.css";

export function basisLabel(v: SealVariant): string {
  if (v.basis === "grouped")
    return v.chronological
      ? `chronological, grouped by \`${v.group_column}\``
      : `grouped by \`${v.group_column}\``;
  if (v.basis === "repetition_found_grouping_abandoned") return "repetition found but grouping abandoned";
  return "undetermined";
}

export function SealFork({ v, recorded }: { v: SealVariant; recorded: boolean }) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const { theme } = useTheme();
  const [ref, { w }] = useSize<HTMLDivElement>();
  const p = useMemo(() => plan(v, w || 760), [v, w]);
  const last = p.states.length - 1;
  const canvas = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const seal = sealStateOf(v.basis);
  const exploratory = v.exploratory;
  const at = Math.min(last, ui.nearest);

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
      // The lanes' frames.
      if (g.boxes > 0.01) {
        const pad = 7;
        const bottom = p.splitHeight + 2;
        ctx.globalAlpha = g.boxes;
        ctx.lineWidth = 1;
        ctx.strokeStyle = rgba(line, 1);
        ctx.beginPath();
        ctx.roundRect(p.train[0] - 0.5, p.top - pad, p.train[1] - p.train[0] + pad - 2, bottom - p.top + pad - 1, 8);
        ctx.stroke();
        const sealTone = exploratory ? warn : recorded ? ok : ink;
        ctx.fillStyle = rgba(surface2, 0.5 + 0.5 * g.sealed);
        ctx.strokeStyle = rgba(sealTone, 0.55 + 0.45 * g.sealed);
        ctx.lineWidth = 1 + 0.8 * g.sealed;
        ctx.setLineDash(exploratory ? [5, 4] : []);
        ctx.beginPath();
        ctx.roundRect(p.held[0] - pad, p.top - pad, p.held[1] - p.held[0] + pad - 1, bottom - p.top + pad - 1, 8);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      // The chronological boundary.
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
      // Holes: where a split unit's row sat.
      if (p.holes.size && g.placed[0] !== undefined) {
        ctx.strokeStyle = rgba(warn, 0.9);
        ctx.setLineDash([2, 2]);
        for (const [row, [hx, hy]] of p.holes) {
          const a = g.placed[row]!;
          if (a < 0.02) continue;
          ctx.globalAlpha = a;
          ctx.strokeRect(hx + 0.5, hy + 0.5, c - 1, c - 1);
        }
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      // Straddle lines: a unit on both sides.
      if (g.straddle > 0.01) {
        ctx.strokeStyle = rgba(warn, 0.95);
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 3]);
        ctx.globalAlpha = g.straddle;
        for (const [row, mate] of p.straddles) {
          const hole = p.holes.get(row);
          const t = g.placed[row]!;
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
      // The rows.
      for (let i = 0; i < n; i++) {
        const mk = g.mark[i]!;
        const pl = g.placed[i]!;
        const held = v.hold[i] === 1;
        const col = held ? mixRGB(c4, ink, mk) : mixRGB(c4, c1, pl);
        const a = held ? 0.5 + 0.35 * mk : 0.5 + 0.35 * pl;
        ctx.fillStyle = rgba(col, a);
        ctx.beginPath();
        ctx.roundRect(g.x[i]!, g.y[i]!, c, c, c > 12 ? 3 : 1.5);
        ctx.fill();
      }
      // Overlays follow the same position.
      const o = overlay.current;
      if (o) {
        o.style.setProperty("--boxes", g.boxes.toFixed(3));
        o.style.setProperty("--sealed", g.sealed.toFixed(3));
        o.style.setProperty("--boundary", (g.boundary * (1 - g.boxes)).toFixed(3));
      }
    },
    [p, v, recorded, exploratory, theme],
  );

  const target = useCallback(
    (pos: number) => {
      const i = Math.max(0, Math.min(last, Math.floor(pos)));
      return i >= last ? p.states[last]! : lerpGeom(p.states[i]!, p.states[i + 1]!, pos - i);
    },
    [p, last],
  );
  // Another holdout size on the same side morphs from what is drawn (no storyboard replay).
  const redrawRef = useRef<() => void>(() => {});
  const blend = useMorph<SealGeom>(
    lerpGeom,
    useCallback(() => redrawRef.current(), []),
  );
  const morphKey = `${v.key}-${v.fraction}-${p.width}`;
  const draw = useCallback((pos: number) => paint(blend(target(pos), morphKey)), [paint, blend, target, morphKey]);
  useEffect(() => {
    redrawRef.current = () => draw(eased(store.get().pos));
  }, [draw, store]);
  useEffect(() => {
    draw(eased(store.get().pos));
  }, [draw, store]);
  usePlayerFrame(store, draw);

  const nTrainUnits = v.n_train_units;
  const nHoldUnits = v.n_hold_units;
  return (
    <div className={s.fork} ref={ref} data-view="seal_fork" data-state={at} data-basis={v.basis}>
      <canvas ref={canvas} style={{ width: p.width, height: p.height }} />
      <div ref={overlay} className={s.forkOverlay} style={{ width: p.width, height: p.height }}>
        <span className={s.laneHead} style={{ left: p.train[0] - 2, opacity: "var(--boxes)" }}>
          Training · {fmtInt(v.n_train_rows)} {v.row_noun}
          {nTrainUnits !== null ? ` · ${fmtInt(nTrainUnits)} ${v.unit_noun}` : ""}
        </span>
        <span className={s.laneHead} style={{ left: p.held[0] - 2, opacity: "var(--boxes)" }}>
          Held out · {fmtInt(v.n_hold_rows)} {v.row_noun}
          {nHoldUnits !== null ? ` · ${fmtInt(nHoldUnits)} ${v.unit_noun}` : ""}
        </span>
        <span className={s.sealBadge} data-tone={exploratory ? "warn" : recorded ? "ok" : "ink"}
          style={{ left: p.held[1] - 24, top: 0 }}>
          <SealGlyph state={seal} recorded={recorded} />
        </span>
        {p.axis.map((a, i) => (
          <span
            key={a.label}
            className={s.axisLabel}
            data-align={i === 0 ? "start" : i === 1 ? "end" : "middle"}
            style={{ left: a.x, opacity: "var(--boundary)" }}
          >
            {a.label}
          </span>
        ))}
      </div>
      <p className={s.basisLine} data-exploratory={exploratory || undefined} data-shown={at === last || undefined}>
        <span className={s.basisKicker}>{exploratory ? "Sealed by row · exploratory" : recorded ? "Sealed" : "Would seal"}</span>
        <span className={s.basisText}>
          <Rich text={basisLabel(v)} />
        </span>
        <span className={s.basisFact}>
          {v.straddle === null ? "on both sides: unknown" : `${v.straddle} ${v.unit_noun} on both sides`}
        </span>
      </p>
    </div>
  );
}
