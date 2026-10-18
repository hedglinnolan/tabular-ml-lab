/**
 * The stage — the right half of the working window (M1_CONTRACT §10–§13).
 *
 * It shows exactly one thing, chosen by the focus the Record, the banner and the stage share:
 *   option   what recording that option would do to the user's own data (POST /preview), with
 *            the transform player: one flip, *Your data now ⇄ With this choice*, that plays the
 *            method's storyboard and drives every view at once
 *   finding  the views that show why the finding was raised
 *   banner   a pipeline segment full size: the row flow, the lineage, the shelf, the Results
 *   live     the Results once fitted; before that, your data now (rows and columns as recorded)
 *
 * What is on screen stays until the next thing is ready: the stage never flashes empty. A new
 * scene arrives choreographed (cards rise in turn, then the storyboard plays forward); switching
 * options within a question morphs straight between results at the flip's current side.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { isRefusalError } from "../../api/client";
import type { PreviewResult, ViewKind } from "../../api/m1-stage-types";
import type { StageFocus } from "../../state/focus";
import { useDecide } from "../../api/queries";
import type { Decision, ProjectView, Refusal } from "../../api/schema";
import { useMotionPrefs, useTransitions } from "../../motion/prefs";
import { besideAsk } from "../record/besideAsk";
import { stateSeq, useEvidence, useM1Stage, usePreview, type Answer } from "./data";
import { ColumnsScene, ModelsScene, NowScene, ResultsScene, RowsScene, type LiveData } from "./LiveScenes";
import { FitHeld } from "./results/Results";
import { initial } from "./player";
import { PreviewGrid } from "./PreviewGrid";
import { PlayerControls, StageBar } from "./StageBar";
import { composedOf } from "./composed";
import { fmtInt } from "./format";
import { ColumnsContext, Rich } from "./text";
import { readoutOf, storyboardOf, trackOf, type ReadoutItem } from "./tracks";
import { createPlayerStore, PlayerContext } from "./usePlayer";
import s from "./Stage.module.css";

export interface StageProps {
  pid: string;
  view: ProjectView;
  focus: StageFocus;
  onFocus: (focus: StageFocus) => void;
  /**
   * Why the screen shows no fit yet, though one may be computed (the quest shell under Predict
   * until Fit is pressed: FOUNDATION §7, "what waits is what is shown"), or null. While it is set
   * the stage draws no fit: no Results scene (the comparison, the coefficients, the substitution,
   * the seal), no fitted family's concern on the shelf, and no Fit of its own, since the screen
   * places Fit itself; the live Results card says this reason. Today's screen passes none.
   */
  fitWithheld?: string | null;
  /**
   * Whether a choice can be recorded from the stage (default true, today's screen): the record
   * button beside a preview, the touch equivalent of Enter in the Record, and a refusal's or a
   * preview caution's ways forward, each of which previews the decision it carries for that button
   * to record. A screen whose card is the one place a choice is recorded (the quest shell:
   * FOUNDATION §1, the card's foot is the one action) passes false: the stage draws none of them
   * (a caution keeps its words, W1-R R3 P2), so no second record button
   * disagrees with the card's and a refusal's readings are not drawn again beside the card that
   * asks them: the refusal still says why there is no preview, in the consumer's own words up to
   * the ledger's question, which the card asks (record/besideAsk.ts, as the card's own answer to a
   * refused press does).
   */
  recordHere?: boolean;
  /**
   * How many columns proposed as predictors wait for their role's confirmation, counted from the
   * quest log the screen already reads (the quest shell: roles.ts `waitingPredictorColumns`), which
   * the Columns card says; null (today's screen, a prototype drawn from fixtures) says nothing of
   * them, and no card asks the server for it (U9).
   */
  waitingPredictors?: number | null;
  /** Under Predict, the predictor groups the server serves for the project, which the Columns card
   *  names the lineage's groups by (LiveData `predictorGroups`; the re-walk's N3); null draws the
   *  engine's words. */
  predictorGroups?: LiveData["predictorGroups"];
}

type Scene =
  | { kind: "preview"; group: string; key: string; label: string; decision: Decision; result: PreviewResult }
  | { kind: "refusal"; group: string; key: string; label: string; decision: Decision; refusal: Refusal }
  | { kind: "evidence"; group: string; key: string; label: string; result: PreviewResult }
  | { kind: "now" | "rows" | "columns" | "models" | "results"; group: string; key: string };

/** Views arrive (≈ 300 ms, staggered) before the storyboard starts to play. */
const AUTOPLAY_DELAY_MS = 320;

function answerScene(a: Answer): Scene {
  const group = `preview:${a.decision.kind}`;
  if (a.refusal) return { kind: "refusal", group, key: a.key, label: a.label, decision: a.decision, refusal: a.refusal };
  return { kind: "preview", group, key: a.key, label: a.label, decision: a.decision, result: a.result };
}

function liveScene(kind: "now" | "rows" | "columns" | "models" | "results"): Scene {
  return { kind, group: kind, key: kind };
}

const LIVE_TITLE: Record<string, [string, string]> = {
  now: ["Your data now", "what is recorded so far"],
  rows: ["Rows", "the participant flow, as recorded"],
  columns: ["Columns", "the column lineage, as recorded"],
  models: ["Models", "the shelf for this table"],
  results: ["Results", "what the recorded pipeline fitted"],
};

export function Stage({
  pid,
  view,
  focus,
  onFocus,
  fitWithheld = null,
  recordHere = true,
  waitingPredictors = null,
  predictorGroups = null,
}: StageProps) {
  const seq = stateSeq(view);
  const { reduced } = useMotionPrefs();
  const t = useTransitions();
  const store = useMemo(() => createPlayerStore(), []);
  useEffect(() => store.setReduced(reduced), [store, reduced]);
  useEffect(() => {
    // Review captures: step the player by hand for a frame strip. Dev builds always; a production
    // build only when the review harness set `window.__turbotabReview` before the page loaded.
    if (typeof window === "undefined") return;
    const w = window as unknown as { __turbotabStage?: unknown; __turbotabReview?: boolean };
    if (!import.meta.env.DEV && !w.__turbotabReview) return;
    w.__turbotabStage = {
      manual: (v: boolean) => store.setManual(v),
      advance: (ms: number) => store.advance(ms),
      state: () => store.get(),
    };
    return () => {
      delete w.__turbotabStage;
    };
  }, [store]);

  // ── data ──
  const option = focus.kind === "option" ? { decision: focus.decision, label: focus.label } : null;
  const preview = usePreview(pid, option, seq);
  const evidence = useEvidence(pid, focus.kind === "finding" ? focus.findingId : null, seq);
  const cohort = useM1Stage(pid, view, "cohort");
  const split = useM1Stage(pid, view, "split");
  const shelf = useM1Stage(pid, view, "shelf");
  const design = useM1Stage(pid, view, "design");
  const fit = useM1Stage(pid, view, "fit");
  const substitution = useM1Stage(pid, view, "substitution");
  const findings = useM1Stage(pid, view, "findings", focus.kind === "finding");

  const live: LiveData = {
    pid,
    cohort: { artifact: cohort.artifact, veil: cohort.veil },
    splitVeil: split.veil,
    designVeil: design.veil,
    split: split.artifact,
    shelf: shelf.artifact,
    design: design.artifact,
    // Computed or not, a withheld fit reaches no scene.
    fit: { artifact: fitWithheld ? null : fit.artifact, veil: fit.veil },
    substitution: { artifact: substitution.artifact, veil: substitution.veil },
    fitWithheld,
    waitingPredictors,
    predictorGroups,
  };

  // ── what the stage wants to show, and whether it is ready ──
  const fitted = !!live.fit.artifact;
  const segment = focus.kind === "banner" ? focus.segment : null;
  const findingList = findings.artifact?.findings;
  const target = useMemo<Scene | null>(() => {
    if (focus.kind === "option") return preview.answer ? answerScene(preview.answer) : null;
    if (focus.kind === "finding") {
      if (!evidence.answer) return null;
      const f = findingList?.find((x) => x.id === evidence.answer!.key);
      return {
        kind: "evidence",
        group: `evidence:${evidence.answer.key}`,
        key: evidence.answer.key,
        label: f?.summary ?? f?.title ?? "Evidence",
        result: evidence.answer.result,
      };
    }
    if (segment) return liveScene(segment === "result" ? "results" : segment);
    return liveScene(fitted ? "results" : "now");
  }, [focus.kind, preview.answer, evidence.answer, findingList, segment, fitted]);
  const loading =
    focus.kind === "option" ? preview.loading : focus.kind === "finding" ? evidence.loading : false;

  const [shown, setShown] = useState<Scene | null>(null);
  const [lastPreview, setLastPreview] = useState<Extract<Scene, { kind: "preview" }> | null>(null);
  // What is on screen stays until the next thing is ready (state adjusted while rendering).
  if (target && target !== shown) setShown(target);
  if (target?.kind === "preview" && target !== lastPreview) setLastPreview(target);
  // Held while the next answer loads — unless it is another question's preview: that is not a
  // real state of this question, so the recorded pipeline stays on screen until this one lands.
  const foreign =
    focus.kind === "option" &&
    (shown?.kind === "preview" || shown?.kind === "refusal") &&
    shown.group !== `preview:${focus.decision.kind}`;
  const scene = target ?? (foreign ? null : shown) ?? liveScene(fitted ? "results" : "now");

  // ── the player's material ──
  const showing: PreviewResult | null =
    scene.kind === "preview" || scene.kind === "evidence"
      ? scene.result
      : scene.kind === "refusal" && lastPreview?.group === scene.group
        ? lastPreview.result
        : null;
  const tracks = useMemo(() => (showing ? showing.views.map((v) => trackOf(v)) : []), [showing]);
  const story = useMemo(() => storyboardOf(tracks), [tracks]);
  const nRows = view.summary.n_rows;
  const nCols = view.summary.n_cols;
  const readout = useMemo(() => {
    if (!showing) return [];
    const out: ReadoutItem[] = readoutOf(showing.views);
    // A turn: the table's rows become its columns (one row per sample, one column per feature).
    if (showing.views.some((v) => composedOf(v)?.kind === "turn_table") && nRows !== null && nCols !== null) {
      out.unshift(
        { key: "turn-rows", name: "rows", before: fmtInt(nRows), after: fmtInt(nCols - 1) },
        { key: "turn-cols", name: "columns", before: fmtInt(nCols), after: fmtInt(nRows + 1) },
      );
    }
    return out.slice(0, 3);
  }, [showing, nRows, nCols]);
  // Nothing to flip when every view shows one state (evidence), or a preview has no views at all.
  const still = tracks.every((tr) => tr.still);
  // Say truly what that one state is: the data as loaded (evidence), the choice's own picture (a
  // lineage with no "before", as for the first roles), or the data the choice leaves unchanged.
  const stillLabel =
    scene.kind === "evidence"
      ? "Your data as loaded"
      : !tracks.length || scene.kind === "refusal"
        ? null
        : (scene.kind === "preview" && scene.decision.kind === "select_models") ||
            tracks.some((tr) => tr.view.kind === "lineage" && tr.view.before === null)
          ? "With this choice (preview)"
          : "Unchanged by this choice";

  // A new scene starts at "your data now" and plays forward once its views have arrived; another
  // option of the same question holds the flip's side and lands on its result directly.
  const prev = useRef<{ group: string; key: string; last: number; still: boolean } | null>(null);
  const autoplay = useRef(0);
  useLayoutEffect(() => {
    if (!showing) {
      // Left the previews: coming back to a question plays its storyboard again.
      prev.current = null;
      return;
    }
    if (scene.kind === "refusal") return;
    const p = prev.current;
    prev.current = { group: scene.group, key: scene.key, last: story.last, still };
    // Rifling from an option that changes nothing (keep every row) to one that does plays the
    // new storyboard, as a new scene would: there was no side worth holding.
    if (p && p.group === scene.group && !(p.still && !still)) {
      if (p.key !== scene.key || p.last !== story.last) store.dispatch({ type: "options", last: story.last });
      return;
    }
    window.clearTimeout(autoplay.current);
    if (still) {
      // One recorded state (a finding's evidence): drawn as the data now, nothing plays.
      store.reset(initial(story.last, "now"));
      return;
    }
    store.reset(initial(story.last, "now"));
    autoplay.current = window.setTimeout(
      () => store.dispatch({ type: "show", side: "with" }),
      reduced ? 0 : AUTOPLAY_DELAY_MS,
    );
  }, [showing, scene.kind, scene.group, scene.key, story.last, still, store, reduced]);
  useEffect(() => () => window.clearTimeout(autoplay.current), []);

  // Space flips while a preview is on the stage (not while typing, not on the stage's own buttons).
  const root = useRef<HTMLElement>(null);
  const flippable = !!showing && !still && scene.kind !== "refusal";
  useEffect(() => {
    if (!flippable) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " " || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      if (el && root.current?.contains(el) && el.tagName === "BUTTON") return;
      // In a multi-select list (the lens, the models) Space chooses; the Record keeps it.
      if (el?.closest('[aria-multiselectable="true"]')) return;
      e.preventDefault();
      window.clearTimeout(autoplay.current);
      store.dispatch({ type: "flip" });
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [flippable, store]);

  // ── recording from the stage (the touch path: tap previews, the record button records) ──
  const decide = useDecide(pid);
  const [ack, setAck] = useState<{ text: string; refused: boolean } | null>(null);
  useEffect(() => {
    if (!ack) return;
    const id = window.setTimeout(() => setAck(null), 4000);
    return () => window.clearTimeout(id);
  }, [ack]);
  const recordable =
    focus.kind === "option" && scene.kind === "preview" && !preview.loading && scene.key === preview.answer?.key;
  // In a multi-select list the button records what Enter records there (the chosen set).
  const toRecord = focus.kind === "option" ? (focus.record ?? null) : null;
  const recordLabel = (focus.kind === "option" ? focus.recordLabel : undefined) ?? "Record this choice";
  const record = () => {
    if (scene.kind !== "preview") return;
    decide.mutate(toRecord ?? scene.decision, {
      onSuccess: (next) => {
        const rec = [...next.decisions].sort((a, b) => b.seq - a.seq)[0];
        setAck({ text: `Recorded: ${rec?.sentence ?? scene.label}`, refused: false });
        onFocus({ kind: "live" });
      },
      onError: (e) =>
        setAck({ text: isRefusalError(e) ? e.refusal.error.message : e.message, refused: true }),
    });
  };

  // Columns known to the project: bare names in titles become data chips.
  const known = useMemo(() => {
    const set = new Set<string>(Object.keys(view.state.roles ?? {}));
    if (view.state.target) set.add(view.state.target);
    return set;
  }, [view.state.roles, view.state.target]);

  const [promoted, setPromoted] = useState<Record<string, ViewKind>>({});

  // ── the bar ──
  let pill: string | null = null;
  let label: string | null;
  let aside: React.ReactNode;
  if (scene.kind === "preview" || scene.kind === "refusal") {
    pill = "Preview";
    label = scene.label;
    aside = scene.kind === "refusal" ? "not available" : "nothing is recorded";
  } else if (scene.kind === "evidence") {
    pill = "Evidence";
    label = scene.label;
    aside = "your data as loaded";
  } else {
    [label, aside] = LIVE_TITLE[scene.kind] ?? ["", ""];
  }
  const recordButton =
    recordHere && focus.kind === "option" && (scene.kind === "preview" || scene.kind === "refusal") ? (
      <button
        type="button"
        className={s.record}
        disabled={!recordable || decide.isPending}
        onClick={record}
        data-testid="stage-record"
        title={`${recordLabel} (Enter in the Record)`}
      >
        {decide.isPending ? "Recording…" : recordLabel}
      </button>
    ) : null;

  const provenance =
    scene.kind === "preview"
      ? `Preview, not recorded: ${scene.label}. ${scene.result.basis}`
      : scene.kind === "evidence"
        ? `Evidence for the finding: ${scene.label} ${scene.result.basis}`
        : "";

  return (
    <ColumnsContext.Provider value={known}>
      <PlayerContext.Provider value={store}>
        <section
          ref={root}
          className={s.stage}
          data-testid="stage"
          data-focus={focus.kind}
          data-scene={scene.kind}
          data-group={scene.group}
          aria-label="Stage"
        >
          <StageBar pill={pill} label={label} aside={aside} loading={loading} action={recordButton}>
            {showing ? (
              <PlayerControls
                story={story}
                readout={readout}
                still={still || scene.kind === "refusal"}
                stillLabel={stillLabel}
              />
            ) : null}
          </StageBar>
          {ack ? (
            <p className={ack.refused ? s.ackRefused : s.ack} role="status" data-testid="stage-ack">
              <Rich text={ack.text} />
            </p>
          ) : null}
          {/* Fit waits for the person when it is expected to take long: it is offered wherever the
              recorded pipeline is on the stage, never while a preview or evidence is, and once:
              where the Results show an older fit withheld for Fit, they offer it themselves. A
              screen that withholds the fit places Fit itself. */}
          {!fitWithheld &&
          (scene.kind === "now" ||
            scene.kind === "models" ||
            (scene.kind === "results" && live.fit.artifact?.withheld_for !== "fit")) ? (
            <FitHeld pid={pid} status={view.stages.fit} />
          ) : null}
          <div className={s.body}>
            <AnimatePresence initial={false} mode="popLayout">
              <motion.div
                key={scene.group}
                className={s.scene}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: t.reduced ? 0 : 0.12 } }}
                transition={t.arrive}
              >
                {scene.kind === "refusal" ? (
                  <div className={s.refusal} role="alert" data-testid="refusal" data-purpose="refusal">
                    <p className={s.refusalText}>
                      <Rich
                        text={`Not available: ${(recordHere ? scene.refusal : besideAsk(scene.refusal)).error.message}`}
                      />
                    </p>
                    {recordHere && scene.refusal.error.exits.length ? (
                      <div className={s.exits}>
                        {scene.refusal.error.exits.map((x, i) =>
                          x.decision ? (
                            <button
                              key={i}
                              type="button"
                              className={s.exit}
                              onClick={() =>
                                onFocus({ kind: "option", decision: x.decision!, label: x.label.replace(/`/g, "") })
                              }
                              data-testid="refusal-exit"
                            >
                              <Rich text={x.label} />
                            </button>
                          ) : null,
                        )}
                      </div>
                    ) : null}
                  </div>
                ) : null}
                {showing && (scene.kind === "preview" || scene.kind === "evidence" || scene.kind === "refusal") ? (
                  <div className={scene.kind === "refusal" ? s.veiled : s.views} inert={scene.kind === "refusal"}>
                    {showing.note ? (
                      <p className={s.note} data-testid="stage-note" data-purpose="stage_note">
                        <Rich text={showing.note} />
                      </p>
                    ) : null}
                    {showing.caution ? (
                      <div className={s.caution} role="note" data-testid="stage-caution" data-purpose="stage_caution">
                        <p className={s.cautionText}>
                          <Rich text={showing.caution.text} />
                        </p>
                        {/* Its ways forward preview a decision for the stage's record button, so
                            where the card records (recordHere false) they would lead nowhere. */}
                        {recordHere && showing.caution.exits.length ? (
                          <div className={s.exits}>
                            {showing.caution.exits.map((x) => (
                              <button
                                key={x.label}
                                type="button"
                                className={s.exit}
                                onClick={() =>
                                  onFocus({
                                    kind: "option",
                                    decision: x.decision as unknown as Decision,
                                    label: x.label.replace(/`/g, ""),
                                  })
                                }
                              >
                                <Rich text={x.label} />
                              </button>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    {!showing.views.length && scene.kind === "preview" && live.cohort.artifact ? (
                      // Nothing on the rows to draw for this choice: what is recorded stays in view.
                      <NowScene view={view} data={live} />
                    ) : null}
                    <PreviewGrid
                      tracks={tracks}
                      story={story}
                      promoted={promoted[scene.group] ?? null}
                      onPromote={(k) => setPromoted((p) => ({ ...p, [scene.group]: k }))}
                      basis={showing.basis}
                      provenance={provenance}
                    />
                  </div>
                ) : null}
                {scene.kind === "now" ? <NowScene view={view} data={live} /> : null}
                {scene.kind === "rows" ? <RowsScene view={view} data={live} /> : null}
                {scene.kind === "columns" ? <ColumnsScene view={view} data={live} /> : null}
                {scene.kind === "models" ? <ModelsScene view={view} shelf={live.shelf} fit={live.fit.artifact} /> : null}
                {scene.kind === "results" ? <ResultsScene pid={pid} view={view} data={live} /> : null}
              </motion.div>
            </AnimatePresence>
          </div>
        </section>
      </PlayerContext.Provider>
    </ColumnsContext.Provider>
  );
}
