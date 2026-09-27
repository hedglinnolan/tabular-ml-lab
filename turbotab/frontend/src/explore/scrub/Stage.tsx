/**
 * The stage — the pipeline panel in preview mode. Top to bottom:
 *
 *   PipelineStrip  Rows · Columns · Results: the lane a choice touches lights up, its number changes
 *   ScrubBar       your data now ←→ with this choice; the thumb is the scrub, the ends are its legend
 *   (views)        the canonical before→after picture, then the working table
 *   RecordBar      what it was computed on, and the one outcome-labeled action
 */
import { useCallback, useEffect, useMemo, useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { Prose } from "../../components/Prose";
import { cx } from "../../util/format";
import { SceneBuilder, cached, unionKeys, type Scene } from "./engine/morph";
import { place, useMorph, useRegistry, useScrub } from "./engine/scrub";
import s from "./Stage.module.css";

// ── pipeline strip ───────────────────────────────────────────────────────────

export type LaneKey = "rows" | "columns" | "results";
export interface LaneValue {
  value: string;
  sub: string;
}
export type StripState = Record<LaneKey, LaneValue> & { touched: LaneKey[] };

const LANES: { key: LaneKey; label: string }[] = [
  { key: "rows", label: "Rows" },
  { key: "columns", label: "Columns" },
  { key: "results", label: "Results" },
];

export function PipelineStrip({ states }: { states: Record<string, StripState> }) {
  const { map, reg } = useRegistry<HTMLElement>();
  const sceneOf = useMemo(() => {
    return cached((state: string): Scene => {
      const st = states[state] ?? states.now!;
      const sb = new SceneBuilder();
      for (const { key } of LANES) {
        sb.set(`v:${key}:${st[key].value}`, { o: 1, slide: 6 });
        sb.set(`s:${key}:${st[key].sub}`, { o: 1 });
        if (state !== "now" && st.touched.includes(key)) sb.set(`hot:${key}`, { o: 1 });
      }
      return sb.scene;
    });
  }, [states]);
  const keys = useMemo(() => unionKeys(Object.keys(states).map((k) => sceneOf(k))), [states, sceneOf]);
  const apply = useCallback(
    (sc: Scene) => {
      for (const [k, el] of map.current) place(el, sc.items.get(k), { x: false, y: false });
    },
    [map],
  );
  useMorph({ sceneOf, apply, span: [0.25, 1] });
  const of = (p: string, lane: string) =>
    keys.filter((k) => k.startsWith(`${p}:${lane}:`)).map((k) => [k, k.slice(p.length + lane.length + 2)] as const);
  return (
    <div className={s.strip} role="group" aria-label="Pipeline">
      {LANES.map(({ key, label }) => (
        <div key={key} className={s.lane}>
          <span ref={reg(`hot:${key}`)} className={s.laneHot} style={{ opacity: 0 }} />
          <span className={s.laneLabel}>{label}</span>
          <span className={s.laneValue}>
            {of("v", key).map(([k, t]) => (
              <span key={k} ref={reg(k)} className={s.stackItem} style={{ opacity: 0 }}>
                {t}
              </span>
            ))}
          </span>
          <span className={s.laneSub}>
            {of("s", key).map(([k, t]) => (
              <span key={k} ref={reg(k)} className={s.stackItem} style={{ opacity: 0 }}>
                <Prose text={t} />
              </span>
            ))}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── scrub bar ────────────────────────────────────────────────────────────────

interface ScrubBarProps {
  /** "With Willett residual" — the right end names the choice. */
  afterLabel: string | null;
  recorded: boolean;
  /** When the previewed choice changes nothing, say so instead of offering an empty scrub. */
  unchanged?: boolean;
  refused?: boolean;
}

export function ScrubBar({ afterLabel, recorded, unchanged = false, refused = false }: ScrubBarProps) {
  const { t, phase, active, dragging, flip, scrubTo, beginDrag, release } = useScrub();
  const track = useRef<HTMLDivElement>(null);
  const thumb = useRef<HTMLDivElement>(null);
  const fill = useRef<HTMLDivElement>(null);
  const disabled = !active || refused;

  // A refused choice has no "after": the thumb stays on the user's data.
  useEffect(() => {
    const draw = (v: number) => {
      const x = refused ? 0 : v;
      if (thumb.current) thumb.current.style.left = `${x * 100}%`;
      if (fill.current) fill.current.style.width = `${x * 100}%`;
    };
    draw(t.get());
    return t.on("change", draw);
  }, [t, refused]);

  const fromPointer = (e: PointerEvent) => {
    const r = track.current?.getBoundingClientRect();
    if (!r) return 0;
    return (e.clientX - r.left) / r.width;
  };

  const onKey = (e: KeyboardEvent) => {
    if (disabled) return;
    const step = e.shiftKey ? 0.1 : 1;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      if (step === 1) flip("now");
      else scrubTo(t.get() - step);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      if (step === 1) flip("after");
      else scrubTo(t.get() + step);
    } else if (e.key === "Home") {
      e.preventDefault();
      flip("now");
    } else if (e.key === "End") {
      e.preventDefault();
      flip("after");
    }
  };

  const tag = recorded
    ? { text: "Recorded", tone: "ok" }
    : !active
      ? null
      : refused
        ? { text: "Not applicable", tone: "muted" }
        : dragging || phase === "moving"
          ? { text: "Scrubbing", tone: "muted" }
          : phase === "after"
            ? { text: unchanged ? "Preview · no change" : "Preview · not recorded", tone: "preview" }
            : { text: "Your data now", tone: "muted" };

  return (
    <div className={s.scrub} data-disabled={disabled || undefined}>
      <button
        type="button"
        className={cx(s.end, s.endNow)}
        data-on={phase === "now" || undefined}
        onClick={() => flip("now")}
        disabled={disabled}
        tabIndex={-1}
      >
        <span className={s.dotNow} aria-hidden="true" />
        Your data now
      </button>
      <div
        ref={track}
        className={s.track}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label="Scrub between your data now and this choice"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={phase === "now" ? 0 : phase === "after" ? 100 : 50}
        aria-valuetext={phase === "after" ? (afterLabel ?? "") : "Your data now"}
        aria-disabled={disabled || undefined}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          beginDrag();
          scrubTo(fromPointer(e));
        }}
        onPointerMove={(e) => {
          if (dragging) scrubTo(fromPointer(e));
        }}
        onPointerUp={() => dragging && release()}
        onPointerCancel={() => dragging && release()}
      >
        <div className={s.rail} />
        <div ref={fill} className={s.fill} />
        <div ref={thumb} className={s.thumb} data-dragging={dragging || undefined} />
      </div>
      <button
        type="button"
        className={cx(s.end, s.endAfter)}
        data-on={phase === "after" || undefined}
        onClick={() => flip("after")}
        disabled={disabled}
        tabIndex={-1}
      >
        <span className={s.dotAfter} aria-hidden="true" />
        {afterLabel ?? "With a choice"}
      </button>
      <span className={s.tag} data-tone={tag?.tone} aria-live="polite">
        {tag?.text ?? ""}
      </span>
    </div>
  );
}

// ── frame ────────────────────────────────────────────────────────────────────

export function StageFrame({ children, label }: { children: ReactNode; label: string }) {
  return (
    <section className={s.stage} aria-label={label}>
      {children}
    </section>
  );
}

export function StageSection({
  children,
  kicker,
  aside,
  className,
}: {
  children: ReactNode;
  kicker?: ReactNode;
  aside?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx(s.section, className)}>
      {kicker || aside ? (
        <div className={s.sectionHead}>
          {kicker ? <span className={s.kicker}>{kicker}</span> : <span />}
          {aside ? <span className={s.aside}>{aside}</span> : null}
        </div>
      ) : null}
      {children}
    </div>
  );
}

export function RecordBar({
  basis,
  action,
  onRecord,
  recorded,
  onChange,
  disabled,
}: {
  basis: string;
  action: string | null;
  onRecord: () => void;
  recorded: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <div className={s.recordBar}>
      <span className={s.basis}>{basis}</span>
      {recorded ? (
        <button type="button" className={s.secondary} onClick={onChange}>
          Change the answer
        </button>
      ) : action ? (
        <button type="button" className={s.primary} onClick={onRecord} disabled={disabled}>
          {action}
        </button>
      ) : null}
    </div>
  );
}
