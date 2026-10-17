/**
 * The quest shell v0 (FOUNDATION §3; plans/W1_PART2_PLAN.md §2 "W1-E"), at /p/:pid: the header with
 * the work chips, the quest line and progress bar, then the manuscript rail, the card and the
 * tapestry. Every stage and line comes from GET /quest; each line's card goes through CardFor; the
 * answers come from GET /methods by `decision_id`, the manuscript from GET /methods/sections. Today's
 * screen stays at /p/:pid/classic until W2 closes (ruling 8), one link away in the header.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { StreamState } from "../api/events";
import { useMethods, useMethodsSections, useProjectView, useQuest } from "../api/queries";
import type { DecisionRecord, ProjectView, QuestLog } from "../api/schema";
import { Header } from "../components/Header";
import { JobChips } from "../components/JobChips";
import { ManuscriptOverlay, ManuscriptRail, kit as k, newestSeq, useManuscriptRail } from "../kit";
import { Link, classicPath } from "../router";
import { StageFocusProvider, useStageFocus } from "../state/focus";
import { fitPosition, heldFor, landingStageKey, stageOf } from "./log";
import { StageCard } from "./StageCard";
import { StageLine } from "./StageLine";
import { Tapestry } from "./Tapestry";
import q from "./quest.module.css";

/** `stream`: the project's event stream, held by the route above (App.tsx) and shared with
 *  today's screen. */
export function QuestScreen({ pid, stream }: { pid: string; stream: StreamState }) {
  const viewQ = useProjectView(pid);
  const questQ = useQuest(pid);
  const header = (
    <Header>
      <Link href={classicPath(pid)} className={q.classicLink} data-testid="to-classic">
        Classic screen
      </Link>
    </Header>
  );
  if (viewQ.isError || questQ.isError) {
    const error = viewQ.error ?? questQ.error;
    return (
      <>
        {header}
        <main className={q.message}>
          <p>This project could not be opened: {error?.message}</p>
          <Link href="/">Back to the start</Link>
        </main>
      </>
    );
  }
  if (!viewQ.data || !questQ.data) {
    return (
      <>
        {header}
        <main className={q.message}>Opening the project…</main>
      </>
    );
  }
  return (
    <StageFocusProvider key={pid}>
      <Shell pid={pid} view={viewQ.data} log={questQ.data} stream={stream} />
    </StageFocusProvider>
  );
}

function Shell({ pid, view, log, stream }: { pid: string; view: ProjectView; log: QuestLog; stream: StreamState }) {
  const [chosen, setChosen] = useState<string | null>(null);
  const fit = fitPosition(log, view.stages.fit);
  const shown = chosen && stageOf(log, chosen) ? chosen : landingStageKey(log, fit);
  const stage = stageOf(log, shown)!;
  const methods = useMethods(pid).data;
  const sections = useMethodsSections(pid).data;
  const rail = useManuscriptRail();
  const answers = useMemo(
    () => new Map((methods?.lines ?? []).map((l) => [l.record_id, l.sentence])),
    [methods],
  );
  useResetOnRecord(view.decisions);
  // Pressing Fit opens Results once the fit is computed (FOUNDATION §7): at once for a short fit
  // computed before the press; a fit that must run keeps the card on Models, whose flowchart shows
  // its progress and a Cancel where Fit was (ruling 3), and Results opens when it is computed, if
  // the screen is still there. The press reaches the quest log before the view's fit job does.
  const pressed = log.fit?.pressed ?? false;
  const computed = view.stages.fit?.status === "fresh";
  const [seenPressed, setSeenPressed] = useState(pressed);
  const [awaiting, setAwaiting] = useState(pressed && !computed);
  if (seenPressed !== pressed) {
    setSeenPressed(pressed);
    setAwaiting(pressed && !computed);
    if (pressed) setChosen(computed ? "results" : "models");
  }
  if (awaiting && pressed && computed) {
    setAwaiting(false);
    if (shown === "models") setChosen("results");
  }

  // The arrival line: the newest line the manuscript gained since the screen opened.
  const newest = sections ? newestSeq(sections.sections) : null;
  const [firstSeen, setFirstSeen] = useState<number | null | undefined>(undefined);
  if (sections && firstSeen === undefined) setFirstSeen(newest);
  const arrival = ((): { text: string; kind: string | null } | null => {
    if (!sections || newest === null || firstSeen === undefined) return null;
    if (firstSeen !== null && newest <= firstSeen) return null;
    for (const sec of sections.sections)
      for (const e of sec.entries) if (e.seq === newest) return { text: e.text, kind: e.kind };
    return null;
  })();

  const openKind = (kind: string) => {
    const at = log.kinds[kind];
    if (at) setChosen(at);
  };
  const fitting = fit.kind === "running" ? (heldFor(log.fit) ? `fitting, ${heldFor(log.fit)}` : "fitting") : null;

  return (
    <div className={q.screen}>
      <Header jobs={<JobChips pid={pid} stages={view.stages} />}>
        <span className={q.name} title={view.summary.source_name}>
          {view.summary.name}
        </span>
        <Link href={classicPath(pid)} className={q.classicLink} data-testid="to-classic">
          Classic screen
        </Link>
        {stream === "reconnecting" ? (
          <span className={q.stream} role="status">
            reconnecting to the server…
          </span>
        ) : null}
      </Header>
      <StageLine log={log} shown={shown} onOpen={setChosen} answers={answers} fitting={fitting} />
      <div className={`${k.zones} ${q.zones}`}>
        <div className={k.zoneRail} data-kit>
          <ManuscriptRail count={sections?.count ?? null} open={rail.open} setOpen={rail.setOpen} />
        </div>
        <div className={k.zoneCard}>
          <StageCard
            key={shown}
            pid={pid}
            log={log}
            stage={stage}
            answers={answers}
            fit={fit}
            onOpenStage={setChosen}
            arrival={arrival}
            onOpenManuscript={() => rail.setOpen(true)}
          />
          <ManuscriptOverlay methods={sections} open={rail.open} setOpen={rail.setOpen} onOpen={openKind} />
        </div>
        <div className={k.zoneCanvas}>
          <Tapestry pid={pid} view={view} log={log} shown={shown} fit={fit} methods={sections} onOpenKind={openKind} />
        </div>
      </div>
    </div>
  );
}

/** The decision log's newest state: a record added (or taken back) changes it. */
const logMark = (decisions: readonly DecisionRecord[]): string =>
  `${decisions.length}:${decisions.reduce((m, r) => Math.max(m, r.seq), 0)}`;

/**
 * A recorded answer returns the tapestry to live, as the Record's recording does (record/Record.tsx
 * resets the stage focus on success), so an option previewed before it is not left on the tapestry
 * as "nothing is recorded", with a button that records it again. A card records through its own
 * mutation (AskFor's plain wiring, W1-F's and W1-G's cards), so the shell reads the decision log,
 * whichever card recorded. A press refused or failed records nothing and keeps the preview.
 */
function useResetOnRecord(decisions: readonly DecisionRecord[]) {
  const { reset } = useStageFocus();
  const mark = logMark(decisions);
  const seen = useRef(mark);
  useEffect(() => {
    if (seen.current === mark) return;
    seen.current = mark;
    reset();
  }, [mark, reset]);
}
