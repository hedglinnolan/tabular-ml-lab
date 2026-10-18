/**
 * The quest line and the progress bar (FOUNDATION §3): the seven stages by name, the open one in
 * the text color, and seven segments at the right, one per stage, filled with its objectives
 * (empty when not reached, never "0 of N"; a stage that asks nothing reads as done only once the
 * analysis comes to it, log.ts `readsDone`, and as not done, never as not reached, before). A
 * stage reopened by an earlier change drops back and says why beside the bar. Pointing at a stage
 * or its segment, or focusing it, opens a quiet panel of its lines by label, each answered line
 * with its answer (GET /methods by `decision_id`); leaving closes it after the kit's grace.
 * Clicking a stage opens it on the card.
 */
import { useEffect, useId, useRef, useState, type FocusEvent } from "react";
import type { QuestLine, QuestLog, QuestStage } from "../api/schema";
import { POINT_GRACE_MS, Plain, kit as k } from "../kit";
import { LABELS, byLabel, filled, readsDone, segments } from "./log";
import { said } from "./words";
import q from "./quest.module.css";

export interface StageLineProps {
  log: QuestLog;
  /** The stage shown on the card. */
  shown: string;
  onOpen: (stage: string, line?: string) => void;
  /** Each answered line's answer, by its decision record's id (the methods' sentences). */
  answers: ReadonlyMap<string, string>;
  /** While the fit runs: "fitting", with the engine's estimate when it is known. */
  fitting: string | null;
}

/** Whether an element's focus came from the keyboard (`:focus-visible`); true where the browser
 *  cannot say, so a panel is never unreachable. */
function keyboardFocus(el: HTMLElement): boolean {
  try {
    return el.matches(":focus-visible");
  } catch {
    return true;
  }
}

export function StageLine({ log, shown, onOpen, answers, fitting }: StageLineProps) {
  const [pointed, setPointed] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancel = () => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  };
  useEffect(() => cancel, []);
  const at = (key: string) => {
    cancel();
    setPointed(key);
  };
  const leave = (key: string) => {
    cancel();
    timer.current = setTimeout(() => {
      timer.current = null;
      setPointed((cur) => (cur === key ? null : cur));
    }, POINT_GRACE_MS);
  };
  // Focus from the keyboard opens the panel; the focus a click leaves behind does not, so the
  // stage the click opened is not covered by its own panel.
  const point = (key: string) => ({
    onPointerEnter: () => at(key),
    onPointerLeave: () => leave(key),
    onFocus: (e: FocusEvent<HTMLElement>) => {
      if (keyboardFocus(e.currentTarget)) at(key);
    },
    onBlur: () => leave(key),
  });
  const open = (key: string) => {
    cancel();
    setPointed(null);
    onOpen(key);
  };
  const segs = segments(log);
  const reopened = segs.flatMap((s) => s.reopened.map((sentence) => ({ key: s.key, sentence })));
  const panelStage = pointed ? log.stages.find((s) => s.key === pointed) : undefined;

  return (
    <div className={q.questBar} data-kit data-testid="quest-bar">
      <nav aria-label="The analysis, stage by stage" className={q.questNav}>
        <ol className={k.chain} data-testid="quest-line">
          {log.stages.map((s) => {
            const status = s.key === shown ? "current" : readsDone(log, s) ? "done" : "todo";
            return (
              <li key={s.key} data-status={status}>
                <button
                  type="button"
                  className={k.chainBtn}
                  data-status={status}
                  data-testid={`stage-${s.key}`}
                  aria-current={s.key === shown ? "step" : undefined}
                  onClick={() => open(s.key)}
                  {...point(s.key)}
                >
                  <span className={k.chainName}>
                    {s.name}
                    {s.key === "results" && fitting ? ` · ${fitting}` : ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </nav>
      <div className={q.barSide}>
        {reopened.length ? (
          <p className={q.reopenedNote} data-testid="reopened-note">
            <Plain text={reopened[0]!.sentence} />
          </p>
        ) : null}
        <ol className={q.bar} aria-label="Progress, stage by stage" data-testid="progress-bar">
          {segs.map((seg) => (
            <li
              key={seg.key}
              className={q.segment}
              data-reached={seg.reached || undefined}
              data-complete={seg.complete || undefined}
              data-done={seg.done || undefined}
              data-reopened={seg.reopened.length ? true : undefined}
              data-answered={seg.answered ?? undefined}
              data-required={seg.required ?? undefined}
              data-testid={`segment-${seg.key}`}
              {...point(seg.key)}
            >
              <span className={q.segmentFill} style={{ width: `${filled(seg) * 100}%` }} />
              {/* "Not reached" only where the engine says so; a reached stage that asks nothing
                  and does not read as done is not done (W1-R B1). */}
              <span className={q.srOnly}>
                {seg.name}:{" "}
                {!seg.reached
                  ? "not reached yet"
                  : seg.done
                    ? "done"
                    : !seg.required
                      ? "not done yet"
                      : `${seg.answered} of ${seg.required} answered`}
              </span>
            </li>
          ))}
        </ol>
      </div>
      {panelStage ? (
        <HoverPanel
          stage={panelStage}
          answers={answers}
          onOpen={(key, line) => {
            setPointed(null);
            onOpen(key, line);
          }}
          keep={() => at(panelStage.key)}
          leave={() => leave(panelStage.key)}
        />
      ) : null}
    </div>
  );
}

/** The stage's lines by label (FOUNDATION §3, "Hover to see a stage's questions"). */
export function HoverPanel({
  stage,
  answers,
  onOpen,
  keep,
  leave,
}: {
  stage: QuestStage;
  answers: ReadonlyMap<string, string>;
  onOpen: (stage: string, line?: string) => void;
  keep?: () => void;
  leave?: () => void;
}) {
  const id = useId();
  const groups = byLabel(stage);
  return (
    <div
      className={q.panel}
      role="region"
      aria-labelledby={id}
      data-testid={`panel-${stage.key}`}
      onPointerEnter={keep}
      onPointerLeave={leave}
    >
      <h3 id={id} className={q.panelHead}>
        {stage.name}
      </h3>
      {!stage.reached ? (
        <>
          <p className={q.panelQuiet}>May change as you answer</p>
          <ul className={q.panelList}>
            {stage.lines.map((l) => (
              <li key={`${l.source}:${l.key}`} data-testid={`panel-line-${l.key}`} data-status={l.status}>
                {said(l.name)}
              </li>
            ))}
          </ul>
        </>
      ) : stage.lines.length === 0 ? (
        <p className={q.panelQuiet}>Nothing to decide here.</p>
      ) : (
        LABELS.filter((label) => groups[label].length).map((label) => (
          <section key={label}>
            <h4 className={q.panelLabel}>{label}</h4>
            <ul className={q.panelList}>
              {groups[label].map((l) => (
                <PanelLine key={`${l.source}:${l.key}`} line={l} answers={answers} onOpen={onOpen} />
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  );
}

function PanelLine({
  line,
  answers,
  onOpen,
}: {
  line: QuestLine;
  answers: ReadonlyMap<string, string>;
  onOpen: (stage: string, line?: string) => void;
}) {
  const answer = line.decision_id ? answers.get(line.decision_id) : undefined;
  // An answered noticing reads as its answer, as on the card: its open-tense name is settled.
  const settled = line.source === "finding" && line.status === "answered" && !!answer;
  return (
    <li data-testid={`panel-line-${line.key}`} data-status={line.status}>
      <span className={q.panelName}>{settled ? <Plain text={answer} /> : said(line.name)}</span>
      {line.status === "answered" ? (
        answer && !settled ? (
          <span className={q.panelAnswer}>
            <Plain text={answer} />
          </span>
        ) : null
      ) : line.status === "waiting" ? (
        <span className={q.panelQuiet}>
          Waiting for:{" "}
          {line.waiting_for.length
            ? line.waiting_for.map((w, i) => (
                <span key={w.key}>
                  {i ? ", " : ""}
                  <button type="button" className={k.linkish} onClick={() => onOpen(w.stage, w.key)}>
                    {w.name}
                  </button>
                </span>
              ))
            : "a result still computing"}
        </span>
      ) : line.status === "set_for_you" ? (
        <span className={q.panelQuiet}>
          Set for you{line.reason ? ": " : ""}
          {line.reason ? <Plain text={line.reason} /> : null}
        </span>
      ) : null}
    </li>
  );
}
