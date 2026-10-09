/**
 * The canvas: the production stage's pieces (StageBar, the transform player, PreviewGrid and its
 * views) driven by a captured preview instead of the server. One flip, *Your data now ⇄ With this
 * choice*, plays the method's storyboard; another option of the same question morphs straight to
 * its result at the flip's side (BLUEPRINT §11.1). Results (Table 2, the decisions that mattered)
 * take the same pane once the plan is locked.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { ViewKind } from "../../api/m1-stage-types";
import { AnimatePresence, motion } from "motion/react";
import type { PreviewResult } from "../../api/m1-stage-types";
import { initial } from "../../components/stage/player";
import { PreviewGrid } from "../../components/stage/PreviewGrid";
import { PlayerControls, StageBar } from "../../components/stage/StageBar";
import { ColumnsContext, Rich } from "../../components/stage/text";
import { readoutOf, storyboardOf, trackOf } from "../../components/stage/tracks";
import { createPlayerStore, PlayerContext } from "../../components/stage/usePlayer";
import { useMotionPrefs, useTransitions } from "../../motion/prefs";
import { INF } from "./fixture";
import s from "../../components/stage/Stage.module.css";
import c from "./screen.module.css";

export type Scene =
  | { kind: "preview"; group: string; key: string; label: string; result: PreviewResult; recorded: boolean }
  | { kind: "refusal"; group: string; key: string; label: string; message: string; exits: string[] }
  | { kind: "note"; group: string; key: string; label: string; note: string; basis?: string; aside?: ReactNode }
  | { kind: "panel"; group: string; key: string; label: string; pill?: string | null; aside?: ReactNode; body: ReactNode };

const AUTOPLAY_DELAY_MS = 320;

const KNOWN = new Set<string>([
  ...INF.roles.map((r) => r.column),
  "glucose",
  "fat_total_adj",
  "sugar_adj",
  "kcal_from_other",
]);

interface Props {
  scene: Scene | null;
  /** The record button: records the previewed option (the touch path). */
  onRecord?: (() => void) | null;
  recordLabel?: string;
  empty: ReactNode;
}

export function Canvas({ scene, onRecord, recordLabel = "Record this choice", empty }: Props) {
  const { reduced } = useMotionPrefs();
  const t = useTransitions();
  const store = useMemo(() => createPlayerStore(), []);
  useEffect(() => store.setReduced(reduced), [store, reduced]);
  useEffect(() => {
    // Review captures step the player by hand (the production stage's own hook).
    const w = window as unknown as { __turbotabStage?: unknown };
    w.__turbotabStage = {
      manual: (v: boolean) => store.setManual(v),
      advance: (ms: number) => store.advance(ms),
      state: () => store.get(),
    };
    return () => {
      delete w.__turbotabStage;
    };
  }, [store]);

  const [promoted, setPromoted] = useState<Record<string, ViewKind>>({});
  const showing = scene?.kind === "preview" ? scene.result : null;
  const tracks = useMemo(() => (showing ? showing.views.map((v) => trackOf(v)) : []), [showing]);
  const story = useMemo(() => storyboardOf(tracks), [tracks]);
  const readout = useMemo(() => (showing ? readoutOf(showing.views).slice(0, 3) : []), [showing]);
  const still = tracks.every((tr) => tr.still);

  // A new question's preview starts at "your data now" and plays forward once its views arrive;
  // another option of the same question holds the flip's side and lands on its result.
  const prev = useRef<{ group: string; key: string; still: boolean } | null>(null);
  const autoplay = useRef(0);
  useLayoutEffect(() => {
    if (!showing || scene?.kind !== "preview") {
      prev.current = null;
      return;
    }
    const p = prev.current;
    prev.current = { group: scene.group, key: scene.key, still };
    // Rifling from an option that changes nothing shown (the recorded one) to one that does plays
    // the new storyboard, as a new scene would: there was no side worth holding (as the stage does).
    if (p && p.group === scene.group && !(p.still && !still)) {
      if (p.key !== scene.key) store.dispatch({ type: "options", last: story.last });
      return;
    }
    window.clearTimeout(autoplay.current);
    store.reset(initial(story.last, "now"));
    if (still) return;
    autoplay.current = window.setTimeout(
      () => store.dispatch({ type: "show", side: "with" }),
      reduced ? 0 : AUTOPLAY_DELAY_MS,
    );
  }, [showing, scene, story.last, still, store, reduced]);
  useEffect(() => () => window.clearTimeout(autoplay.current), []);

  // Space flips while a preview is on the canvas (not while typing, not on a button).
  const flippable = !!showing && !still;
  useEffect(() => {
    if (!flippable) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " " || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(el.tagName))) return;
      if (el?.closest('[role="button"]')) return;
      e.preventDefault();
      window.clearTimeout(autoplay.current);
      store.dispatch({ type: "flip" });
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [flippable, store]);

  let pill: string | null = null;
  let aside: ReactNode = null;
  if (scene?.kind === "preview") {
    pill = scene.recorded ? null : "Preview";
    aside = scene.recorded ? "as recorded" : "nothing is recorded";
  } else if (scene?.kind === "refusal") {
    pill = "Preview";
    aside = "not available";
  } else if (scene?.kind === "note") {
    pill = "Preview";
    aside = scene.aside ?? "nothing is recorded";
  } else if (scene?.kind === "panel") {
    pill = scene.pill ?? null;
    aside = scene.aside ?? null;
  }
  const record =
    onRecord && scene && (scene.kind === "preview" || scene.kind === "note") && !(scene.kind === "preview" && scene.recorded) ? (
      <button type="button" className={s.record} onClick={onRecord} data-testid="canvas-record">
        {recordLabel}
      </button>
    ) : null;

  return (
    <ColumnsContext.Provider value={KNOWN}>
      <PlayerContext.Provider value={store}>
        <section className={`${s.stage} ${c.canvas}`} data-testid="stage" data-scene={scene?.kind ?? "empty"} aria-label="Canvas">
          {scene ? (
            <StageBar pill={pill} label={scene.label} aside={aside} loading={false} action={record}>
              {showing ? (
                <PlayerControls
                  story={story}
                  readout={readout}
                  still={still}
                  stillLabel={still ? "Unchanged by this choice" : null}
                />
              ) : null}
            </StageBar>
          ) : null}
          <div className={s.body}>
            <AnimatePresence initial={false} mode="popLayout">
              <motion.div
                key={scene?.group ?? "empty"}
                className={s.scene}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: t.reduced ? 0 : 0.12 } }}
                transition={t.arrive}
              >
                {!scene ? empty : null}
                {scene?.kind === "refusal" ? (
                  <div className={s.refusal} role="alert" data-testid="refusal">
                    <p className={s.refusalText}>
                      <Rich text={`Not available: ${scene.message}`} />
                    </p>
                    {scene.exits.length ? (
                      <ul className={c.exits}>
                        {scene.exits.map((x) => (
                          <li key={x}>
                            <Rich text={x} />
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                ) : null}
                {scene?.kind === "note" ? (
                  <div className={c.noteScene}>
                    <p className={s.note} data-testid="stage-note">
                      <Rich text={scene.note} />
                    </p>
                    {scene.basis ? <p className={c.basis}>{scene.basis}</p> : null}
                  </div>
                ) : null}
                {scene?.kind === "panel" ? scene.body : null}
                {showing ? (
                  <div className={c.viewsHost}>
                    {showing.note ? (
                      <p className={s.note}>
                        <Rich text={showing.note} />
                      </p>
                    ) : null}
                    {showing.caution ? (
                      <div className={s.caution} role="note">
                        <p className={s.cautionText}>
                          <Rich text={showing.caution.text} />
                        </p>
                      </div>
                    ) : null}
                    {/* the views keep the size they draw at; the canvas scrolls instead of squeezing them */}
                    <div className={c.gridHost} data-views={tracks.length}>
                      <PreviewGrid
                        tracks={tracks}
                        story={story}
                        promoted={promoted[scene!.group] ?? null}
                        onPromote={(k) => setPromoted((p) => ({ ...p, [scene!.group]: k }))}
                        basis={showing.basis}
                        provenance={`Preview, not recorded: ${scene!.label}. ${showing.basis}`}
                      />
                    </div>
                  </div>
                ) : null}
              </motion.div>
            </AnimatePresence>
          </div>
        </section>
      </PlayerContext.Provider>
    </ColumnsContext.Provider>
  );
}
