/**
 * The card (FOUNDATION §3): the stage shown, its name, why it dropped back, then its lines under
 * their labels. The open Decide line is expanded through the dispatcher (CardFor); answered lines
 * collapse to their answers (a noticing to its answer alone, so its open-tense headline is not left
 * beside it); lines still to come wait below, quieter, each "Waiting for" what it needs. A line no
 * card can answer yet is listed after the rest, "Not available yet", and never opened. The stage's
 * Confirm sweep comes after its Decides, each default with its reason (or, where the engine serves
 * none, the default itself, from its value) and what another choice would change, and any can be
 * opened to change it; under Predict, Who's in's draw waits for that sweep (§3, "two departures").
 * Results, once reached, has W1-G's ResultsCards in place of its lines, mounted once
 * (src/quest/contracts.ts). First look lists every noticing with the stage that decides it. For
 * the record sits collapsed at the foot (GET /record). The one primary action is at the very foot
 * (FOUNDATION §1); a disabled one says why and links to what it waits for.
 *
 * A stage opened by a request (the quest line, a "Waiting for" link, a noticing's lever, the foot's
 * link) brings the line it names, else its open line, into view and gives it focus (§1, the
 * five-second test).
 */
import { useEffect, useRef, useState, type ReactNode } from "react";
import { isRefusalError } from "../api/client";
import {
  useChecklist,
  useConfirmSweep,
  useDecide,
  useExport,
  useProjectView,
  useRecord,
} from "../api/queries";
import type { Decision, ProjectState, QuestLine, QuestLog, QuestStage, Refusal, SweepStage } from "../api/schema";
import { Foot, Plain, foot as f, kit as k } from "../kit";
import { CardFor } from "./cards/CardFor";
import { DECLARATION_CARDS, DECLARATION_DEFAULTS } from "./cards/predict";
import { columnOf, recordable } from "./cards/renderers";
import { FootSlot } from "./contracts";
import { type FitPosition, type Hint, DRAW, byLabel, drawWaitsForSweep, openLine, primaryFor, sweptLines } from "./log";
import { ResultsCards } from "./results";
import { said } from "./words";
import q from "./quest.module.css";

const idOf = (l: QuestLine) => `${l.source}:${l.key}`;
const canAnswer = (l: QuestLine) => recordable(l, DECLARATION_CARDS);

/** A request to bring a line into view: the line's key (null: the stage's open line), and a count
 *  that changes with every request, so asking again for the same line still answers. */
export interface OpenRequest {
  line: string | null;
  n: number;
}

export interface StageCardProps {
  pid: string;
  log: QuestLog;
  stage: QuestStage;
  answers: ReadonlyMap<string, string>;
  fit: FitPosition;
  /** Open a stage, with the line named (a lever's question, a link's line) brought into view. */
  onOpenStage: (stage: string, line?: string) => void;
  /** The request that opened this stage, while it stands; null when the screen opened it itself. */
  request?: OpenRequest | null;
  /** The newest line the manuscript gained since this screen opened (FOUNDATION §3, the rail). */
  arrival: { text: string; kind: string | null } | null;
  onOpenManuscript: () => void;
}

/** Scroll an element into view only when it is not already near the top of the window, below the
 *  sticky header, then focus it. */
function bringIntoView(el: HTMLElement) {
  const top = el.getBoundingClientRect().top;
  const below = document.querySelector("header")?.getBoundingClientRect().bottom ?? 0;
  const near = top >= below && top <= window.innerHeight * 0.6;
  if (!near && typeof el.scrollIntoView === "function") el.scrollIntoView({ block: "start" });
  el.focus({ preventScroll: true });
}

export function StageCard({ pid, log, stage, answers, fit, onOpenStage, request = null, arrival, onOpenManuscript }: StageCardProps) {
  const open = openLine(stage, canAnswer);
  const openId = open ? idOf(open) : null;
  /** The line a request names, by key: a question or a declaration before a noticing of the same key. */
  const targetOf = (r: OpenRequest | null): string | null => {
    if (!r?.line) return openId;
    const named = stage.lines.filter((l) => l.key === r.line);
    const line = named.find((l) => l.source !== "finding") ?? named[0];
    return line ? idOf(line) : openId;
  };
  const [expanded, setExpanded] = useState<string | null>(() => targetOf(request));
  // A new open line (the last one was answered) is the one expanded.
  const [seenOpen, setSeenOpen] = useState(openId);
  if (seenOpen !== openId) {
    setSeenOpen(openId);
    setExpanded(openId);
  }
  // A new request, in the stage already shown (a lever whose question is here): its line.
  const [seenRequest, setSeenRequest] = useState(request?.n ?? null);
  if (request && request.n !== seenRequest) {
    setSeenRequest(request.n);
    setExpanded(targetOf(request));
  }
  const root = useRef<HTMLElement>(null);
  const [slot, setSlot] = useState<HTMLDivElement | null>(null);
  const requested = request?.n ?? null;
  useEffect(() => {
    if (requested === null || !root.current) return;
    const lines = [...root.current.querySelectorAll<HTMLElement>("[data-line]")];
    const el = lines.find((x) => x.dataset.line === expanded) ?? root.current.querySelector<HTMLElement>("h2");
    if (el) bringIntoView(el);
    // Only a request moves the screen; the expanded line changing as answers arrive does not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requested]);

  const state = useProjectView(pid).data?.state ?? null;
  const groups = byLabel(stage);
  // A line no card can answer yet comes after those that can (FOUNDATION §3, "Not available yet").
  const decide = [...groups.Decide.filter(canAnswer), ...groups.Decide.filter((l) => !canAnswer(l))];
  const drawHeld = drawWaitsForSweep(stage);
  // Results, once reached, is W1-G's column whole: it draws every line of the stage itself.
  const results = columnOf(stage) === "results";
  const back = !results && expanded !== openId ? () => setExpanded(openId) : null;
  const toggle = (l: QuestLine) => setExpanded((cur) => (cur === idOf(l) ? openId : idOf(l)));
  const line = (l: QuestLine, confirm = false) => (
    <Line
      key={idOf(l)}
      pid={pid}
      line={l}
      expanded={expanded === idOf(l)}
      onToggle={() => toggle(l)}
      answers={answers}
      onOpenStage={onOpenStage}
      answerable={canAnswer(l)}
      held={drawHeld && l.key === DRAW}
      state={state}
      confirm={confirm}
    />
  );

  return (
    <FootSlot.Provider value={slot}>
      <section ref={root} className={q.card} aria-labelledby={`card-${stage.key}`} data-testid={`card-${stage.key}`}>
        <div data-kit className={q.cardHead}>
          <h2 id={`card-${stage.key}`} className={q.cardTitle} tabIndex={-1}>
            {stage.name}
          </h2>
          {stage.reopened.map((r) => (
            <p key={r.decision_id} className={q.reopened} data-testid="card-reopened">
              <Plain text={r.sentence} />
            </p>
          ))}
          {!stage.reached ? <p className={q.quietLine}>May change as you answer</p> : null}
          {stage.reached && stage.lines.length === 0 ? (
            <p className={q.quietLine} data-testid="nothing-here">
              Nothing to decide here.
            </p>
          ) : null}
        </div>

        {stage.key === "first_look" && stage.reached ? (
          <Noticed log={log} answers={answers} onOpenStage={onOpenStage} />
        ) : null}

        {results ? (
          <div data-testid="results-column">
            <ResultsCards pid={pid} />
          </div>
        ) : null}

        {!results && decide.length ? <Group label="Decide">{decide.map((l) => line(l))}</Group> : null}

        {!results && groups.Confirm.length ? (
          <Group label="Confirm" heading={stage.sweep?.heading}>
            {groups.Confirm.map((l) => line(l, true))}
          </Group>
        ) : null}

        {stage.reached ? <ForTheRecord pid={pid} stage={stage} /> : null}
        {stage.key === "writeup" && stage.reached ? <Checklist pid={pid} /> : null}

        {back ? (
          <button type="button" className={k.linkish} data-kit onClick={back} data-testid="back-to-stage">
            Back to {stage.name}
          </button>
        ) : null}

        <Primary pid={pid} log={log} stage={stage} fit={fit} onOpenStage={onOpenStage} slot={setSlot} />
        {arrival ? (
          <button type="button" className={f.arrival} data-kit onClick={onOpenManuscript} data-testid="arrival">
            <span className={f.arrivalWhere}>Manuscript</span>
            <span className={f.arrivalText}>
              <Plain text={arrival.text} />
            </span>
          </button>
        ) : null}
      </section>
    </FootSlot.Provider>
  );
}

function Group({ label, heading, children }: { label: string; heading?: string | undefined; children: ReactNode }) {
  return (
    <section className={q.group} data-testid={`group-${label}`}>
      <h3 className={q.groupLabel} data-kit>
        {label}
        {heading ? <span className={q.groupHeading}> · {heading}</span> : null}
      </h3>
      <ul className={q.lines}>{children}</ul>
    </section>
  );
}

function Line({
  pid,
  line,
  expanded,
  onToggle,
  answers,
  onOpenStage,
  answerable,
  held,
  state,
  confirm = false,
}: {
  pid: string;
  line: QuestLine;
  expanded: boolean;
  onToggle: () => void;
  answers: ReadonlyMap<string, string>;
  onOpenStage: (stage: string, line?: string) => void;
  /** A card can record this line's answer. */
  answerable: boolean;
  /** The draw, waiting for the sweep it reads. */
  held: boolean;
  state: ProjectState | null;
  confirm?: boolean;
}) {
  const answer = line.decision_id ? answers.get(line.decision_id) : undefined;
  const testId = `line-${line.key}`;
  const id = idOf(line);
  if (expanded && answerable && !held && line.status !== "waiting") {
    return (
      <li className={q.lineOpen} data-testid={testId} data-status={line.status} data-line={id} tabIndex={-1} data-expanded>
        <CardFor pid={pid} line={line} onOpenStage={onOpenStage} />
      </li>
    );
  }
  // An answered noticing reads as its answer: the engine names a noticing by its open-tense
  // summary ("…is not asked yet"), which its answer settles.
  const settled = line.source === "finding" && line.status === "answered" && !!answer;
  const offer = answerable && !held && line.status !== "waiting";
  // A default the engine states no reason for is said by its value (DECLARATION_DEFAULTS).
  const reason =
    line.status === "set_for_you" ? (line.reason ?? (state ? (DECLARATION_DEFAULTS[line.key]?.(state) ?? null) : null)) : null;
  return (
    <li
      className={q.line}
      data-kit
      data-testid={testId}
      data-status={line.status}
      data-line={id}
      data-unavailable={answerable ? undefined : true}
      data-held={held || undefined}
      tabIndex={-1}
    >
      <div className={q.lineRow}>
        <span className={q.lineName}>
          {settled ? <Plain text={answer} /> : said(line.name)}
        </span>
        {offer ? (
          <button type="button" className={k.linkish} onClick={onToggle} data-testid={`${testId}-toggle`}>
            {line.status === "answered" || line.status === "set_for_you" ? "Change" : "Answer"}
          </button>
        ) : !answerable ? (
          <span className={q.lineLabel} data-testid={`${testId}-unavailable`}>
            Not available yet
          </span>
        ) : null}
      </div>
      {line.status === "answered" && answer && !settled ? (
        <p className={q.lineAnswer}>
          <Plain text={answer} />
        </p>
      ) : null}
      {reason ? (
        <p className={q.lineReason} data-testid={`${testId}-reason`}>
          <Plain text={reason} />
        </p>
      ) : null}
      {confirm && line.would_change ? (
        <p className={q.lineQuiet}>
          <Plain text={line.would_change} />
        </p>
      ) : null}
      {!answerable ? (
        <p className={q.lineQuiet}>
          {line.status === "answered" || line.status === "set_for_you"
            ? "This version has no card to change it here yet."
            : "This version has no card to answer it here yet."}
        </p>
      ) : null}
      {held && line.status !== "waiting" ? (
        <p className={q.lineQuiet} data-testid={`${testId}-held`}>
          The rows are drawn once the choice set for you below is confirmed, since the draw reads it.
        </p>
      ) : null}
      {line.status === "waiting" ? (
        <p className={q.lineQuiet} data-testid={`${testId}-waiting`}>
          Waiting for:{" "}
          {line.waiting_for.length
            ? line.waiting_for.map((w, i) => (
                <span key={w.key}>
                  {i ? ", " : ""}
                  <button type="button" className={k.linkish} onClick={() => onOpenStage(w.stage, w.key)}>
                    {w.name}
                  </button>
                </span>
              ))
            : "a result still computing"}
        </p>
      ) : null}
      {line.changed_since ? (
        <p className={q.lineQuiet}>
          <Plain text={line.changed_since.reason} />
        </p>
      ) : null}
    </li>
  );
}

/**
 * First look's noticings (FOUNDATION §3; WALKABLE_PLAN §2, "lists its noticings by where each is
 * decided"): First look decides nothing itself; each noticing is decided in the stage whose answer
 * it changes, and each links there. A settled one reads as its answer, as on its own stage's card.
 */
function Noticed({
  log,
  answers,
  onOpenStage,
}: {
  log: QuestLog;
  answers: ReadonlyMap<string, string>;
  onOpenStage: (stage: string, line?: string) => void;
}) {
  const noticed = log.stages.flatMap((s) => s.lines.filter((l) => l.source === "finding").map((l) => ({ s, l })));
  if (!noticed.length) return null;
  return (
    <Group label="Noticed" heading="each is decided where it changes an answer">
      {noticed.map(({ s, l }) => {
        const answer = l.status === "answered" && l.decision_id ? answers.get(l.decision_id) : undefined;
        const where =
          l.label === "For the record" ? "For the record, in" : l.status === "answered" ? "Settled in" : "Decided in";
        return (
          <li key={`${s.key}:${l.key}`} className={q.line} data-kit data-testid={`noticed-${l.key}`} data-status={l.status}>
            <span className={q.lineName}>{answer ? <Plain text={answer} /> : said(l.name)}</span>
            <p className={q.lineQuiet}>
              {where}{" "}
              <button type="button" className={k.linkish} onClick={() => onOpenStage(s.key, l.key)} data-testid={`noticed-${l.key}-to`}>
                {s.name}
              </button>
            </p>
          </li>
        );
      })}
    </Group>
  );
}

/** For the record (FOUNDATION §3): one quiet disclosure at the stage's foot; it never counts. */
function ForTheRecord({ pid, stage }: { pid: string; stage: QuestStage }) {
  const record = useRecord(pid).data;
  const lines = record?.stages.find((s) => s.key === stage.key)?.lines ?? [];
  if (!lines.length) return null;
  return (
    <details className={q.record} data-kit data-testid="for-the-record">
      <summary>For the record · {lines.length}</summary>
      <ul className={q.recordList}>
        {lines.map((l, i) => (
          <li key={`${l.key}-${i}`}>
            <Plain text={l.text} />
          </li>
        ))}
      </ul>
    </details>
  );
}

/** Write-up's checklist: the reporting guideline's items, by how far the record answers each. */
function Checklist({ pid }: { pid: string }) {
  const report = useChecklist(pid).data;
  if (!report) return null;
  const { counts } = report;
  return (
    <section className={q.group} data-kit data-testid="checklist">
      <h3 className={q.groupLabel}>{report.title}</h3>
      <p className={q.lineQuiet}>
        {counts.answered} answered, {counts.partly_answered} partly answered, {counts.unanswered} unanswered,
        of {counts.items} items.
      </p>
      {report.waiting.length ? (
        <ul className={q.recordList}>
          {report.waiting.map((w) => (
            <li key={w}>
              <Plain text={w} />
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

/** The foot's line beside a waiting action: what it waits for, its name a link to that line. */
function HintLine({ hint, onOpenStage }: { hint: Hint; onOpenStage: (stage: string, line?: string) => void }) {
  const to = hint.to;
  return (
    <span data-testid="primary-hint">
      {hint.lead}
      {hint.name && to ? (
        <button type="button" className={k.linkish} onClick={() => onOpenStage(to.stage, to.line ?? undefined)} data-testid="primary-hint-link">
          {hint.name}
        </button>
      ) : (
        hint.name
      )}
      {hint.tail}
    </span>
  );
}

function Primary({
  pid,
  log,
  stage,
  fit,
  onOpenStage,
  slot,
}: {
  pid: string;
  log: QuestLog;
  stage: QuestStage;
  fit: FitPosition;
  onOpenStage: (stage: string, line?: string) => void;
  /** Where a part draws the primary action itself (FootSlot). */
  slot: (el: HTMLDivElement | null) => void;
}) {
  const confirm = useConfirmSweep(pid);
  const decide = useDecide(pid);
  const bundle = useExport(pid);
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const action = primaryFor(log, stage, fit);
  const onRefused = (err: Error) => setRefusal(isRefusalError(err) ? err.refusal : { error: { code: "failed", message: err.message, exits: [] } });

  let node: ReactNode;
  switch (action.kind) {
    case "fit":
      node = (
        <div className={q.footQuiet} data-kit data-testid="primary-fit">
          Fit is at the end of your analysis, beside the card.
        </div>
      );
      break;
    case "open":
      // The opening, with its chain of refusals, is W1-G's column's; it draws its action here.
      node = <div ref={slot} data-testid="primary-slot" />;
      break;
    case "confirm":
      node = (
        <Foot
          label={action.label}
          disabled={confirm.isPending}
          onClick={() => {
            setRefusal(null);
            confirm.mutate(
              { stage: stage.key as SweepStage, sweep: "defaults", lines: sweptLines(stage) },
              { onError: onRefused },
            );
          }}
        />
      );
      break;
    case "download":
      node = (
        <Foot
          label="Download"
          disabled={bundle.isPending}
          onClick={() => {
            setRefusal(null);
            bundle.mutate(undefined, {
              onSuccess: ({ blob, filename }) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = filename;
                a.click();
                URL.revokeObjectURL(url);
              },
              onError: onRefused,
            });
          }}
        />
      );
      break;
    case "continue":
      node = (
        <Foot
          label={action.label}
          disabled={action.disabled}
          left={action.hint ? <HintLine hint={action.hint} onOpenStage={onOpenStage} /> : null}
          onClick={() => action.to && onOpenStage(action.to)}
        />
      );
      break;
  }
  return (
    <div className={q.foot}>
      {node}
      {refusal ? (
        <div className={q.refusal} data-kit role="alert" data-testid="primary-refusal">
          <Plain text={refusal.error.message} />
          {refusal.error.exits.length ? (
            <span className={q.exits}>
              {refusal.error.exits.map((exit) => (
                <button
                  key={exit.label}
                  type="button"
                  className={k.linkish}
                  onClick={() => {
                    setRefusal(null);
                    if (exit.decision) decide.mutate(exit.decision as Decision, { onError: onRefused });
                  }}
                >
                  {exit.label}
                </button>
              ))}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
