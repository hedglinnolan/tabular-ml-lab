/**
 * The card (FOUNDATION §3): the stage shown, its name, why it dropped back, then its lines under
 * their labels. The open Decide line is expanded through the dispatcher (CardFor); answered lines
 * collapse to their answers (a noticing to its answer alone, so its open-tense headline is not left
 * beside it, or, where a question's answer settled it, to what it was about and that question, so
 * the answer is printed once); lines still to come wait below, quieter, each "Waiting for" what it
 * needs. A line a change asked again says which change (an answer taken back says only that),
 * until it is answered (`reopened_by`). A line no card can answer yet is listed after the rest,
 * "Not available yet", and never opened; those that count toward nothing are said together in one
 * quiet line. A record on an expanded line collapses it, and the open line opens. The stage's
 * Confirm sweep comes after its Decides, each default with its reason (or, where the engine serves
 * none, the default itself, from its value) and what another choice would change, and any can be
 * opened to change it; under Predict, Who's in's draw waits for that sweep (§3, "two departures").
 * Results, once reached, has W1-G's ResultsCards in place of its lines, mounted once
 * (src/quest/contracts.ts). First look lists every noticing with the stage that decides it. For
 * the record sits collapsed at the foot (GET /record); Write-up's checklist below it lists what the
 * author still owes for each item answered in part (GET /checklist). The one primary action is at
 * the very foot (FOUNDATION §1); a disabled one says why and links to what it waits for. Continue
 * goes on to the next open line a card can answer, naming it where that is not the next stage;
 * under Predict open noticings hold it in no stage but the triage's own, and the foot, or Fit's
 * beside the flowchart, says how many wait for the triage in Results (log.ts). Under Estimate the
 * triage of the open noticings (cards/TriageCard.tsx) is drawn at Models' end, after its Confirm
 * sweep, before For the record and the foot: before the plan's lock and the analysis flowchart
 * beside the card (§7; G05). While it is unconfirmed, the foot beside Fit says that Fit fixes the
 * plan and the noticings above are confirmed first (log.ts `triageBeforeFit`).
 *
 * A stage opened by a request (the quest line, a "Waiting for" link, a noticing's lever, the foot's
 * link) brings the line it names, else its open line, into view and gives it focus (§1, the
 * five-second test).
 */
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { isRefusalError } from "../api/client";
import {
  useChecklist,
  useConfirmSweep,
  useDecide,
  useExport,
  useProjectView,
  useRecord,
  useStageResult,
  useTriage,
} from "../api/queries";
import type {
  ChecklistItem,
  Decision,
  DecisionRecord,
  Finding,
  ProjectState,
  QuestLine,
  QuestLog,
  QuestStage,
  Refusal,
  SweepStage,
} from "../api/schema";
import { Foot, Plain, foot as f, kit as k, plain } from "../kit";
import { listJoin } from "../util/format";
import { oneRoleAtATime } from "../components/record/besideAsk";
import { CardFor } from "./cards/CardFor";
import { DECLARATION_CARDS, DECLARATION_DEFAULTS } from "./cards/predict";
import { columnOf, recordable } from "./cards/renderers";
import { TriageCard } from "./cards/TriageCard";
import { FootSlot } from "./contracts";
import {
  type FitPosition,
  type Hint,
  DRAW,
  byLabel,
  drawWaitsForSweep,
  openLine,
  primaryFor,
  runningName,
  settledBy,
  sweptLines,
  triageStage,
} from "./log";
import { ResultsCards } from "./results";
import { said } from "./words";
import q from "./quest.module.css";

const idOf = (l: QuestLine) => `${l.source}:${l.key}`;
const canAnswer = (l: QuestLine) => recordable(l, DECLARATION_CARDS);
/** A line no card can answer yet that counts toward nothing and holds no answer to show (joining
 *  another file, the data dictionary): said once, with the others like it, in one quiet line. */
const unavailableQuiet = (l: QuestLine) => !canAnswer(l) && !l.counted && (l.status === "open" || l.status === "waiting");

/** The findings artifact's noticings, once the stage has started (as FindingCard reads them). */
function useNoticings(pid: string): readonly Finding[] | undefined {
  const status = useProjectView(pid).data?.stages.findings?.status;
  const started = status !== undefined && status !== "idle" && status !== "blocked";
  return useStageResult(pid, "findings", started).data?.artifact?.findings;
}

/**
 * A noticing settled by another line's answer (log.ts `settledBy`), in a line of its own: what it
 * was about, by its columns (else its title, else its name), and the question that settled it, so
 * the answer is printed once, on that question's line (FOUNDATION §2). Null for any other line.
 */
function settledText(log: QuestLog, line: QuestLine, noticings: readonly Finding[] | undefined): string | null {
  const by = settledBy(log, line);
  if (!by) return null;
  const f = noticings?.find((x) => x.id === line.key);
  const cols = f?.affected_columns ?? [];
  const about = cols.length
    ? cols.length <= 3
      ? listJoin(cols)
      : `${cols.slice(0, 2).join(", ")} and ${cols.length - 2} more`
    : runningName(f?.title ?? line.name);
  return `${said(about)}: settled by ${runningName(by.name)}`;
}

/**
 * How many reverts record `id` is from the answer it finally changes, following each revert to the
 * record it undoes as the engine does (core/quest.py `_Log.undone`): 1 for a revert of an answer,
 * 2 for a revert of that revert. Null where the project's records cannot finish the walk (a record
 * they do not hold, or a loop).
 */
function revertDepth(id: string, records: readonly DecisionRecord[]): number | null {
  const byId = new Map(records.map((r) => [r.id, r]));
  const seen = new Set<string>();
  let depth = 0;
  let record = byId.get(id);
  while (record && record.decision.kind === "revert") {
    if (seen.has(record.id)) return null;
    seen.add(record.id);
    depth += 1;
    record = byId.get(record.decision.decision_id);
  }
  return record ? depth : null;
}

/**
 * The change that asked a line again (`reopened_by`), in the words of the stage's own reason
 * (`QuestStage.reopened`, "Your change to … reopened"): by the stage it was made in, or as another
 * answer in this one, since any other cause is a record after the line's last answer. A revert
 * (the Record's "undo") that takes an answer back is said as only that: the engine names one the
 * cause whichever answer it took back, the line's own included, and by the stage the undone kind
 * sits in (every noticing's disposition in Your data), so neither "another answer" nor the stage
 * holds. The log cancels latest first (core/decisions.py `reverted`), so a revert of an answer
 * takes it back, a revert of that revert puts it back, and so on in turn: one that puts an answer
 * back is said as the change it restored, by that answer's stage, which is the stage the engine
 * names (`_Log.stage_of`). A revert the project's records (`records`) cannot follow says nothing.
 * Null on an answered line, which reads as its answer, and where the log names no such stage.
 */
function reopenedText(
  log: QuestLog,
  stage: QuestStage,
  line: QuestLine,
  records: readonly DecisionRecord[] | undefined,
): string | null {
  const by = line.reopened_by;
  if (!by || line.status === "answered") return null;
  if (by.kind === "revert") {
    const depth = records ? revertDepth(by.decision_id, records) : null;
    if (depth === null) return null;
    if (depth % 2 === 1) return "Reopened because an answer was taken back.";
  }
  if (by.stage === stage.key) return `Reopened by your change to another answer in ${stage.name}.`;
  const where = log.stages.find((s) => s.key === by.stage);
  return where ? `Reopened by your change to ${where.name}.` : null;
}

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
  // A new open line is the one expanded once the last one was answered (or is gone), and where
  // nothing is expanded. Only an answer moves the expansion: an open line that went waiting while
  // its stage computes and comes back with no answer leaves the line the person has expanded as it
  // is (a noticing's card collapsed under the Enter of its previewed repair, which then recorded
  // nothing: the own-csv journey's race).
  const openRecord = open?.decision_id ?? null;
  const [seenOpen, setSeenOpen] = useState({ id: openId, record: openRecord });
  if (seenOpen.id !== openId || seenOpen.record !== openRecord) {
    setSeenOpen({ id: openId, record: openRecord });
    if (seenOpen.id !== openId) {
      const was = seenOpen.id === null ? null : (stage.lines.find((l) => idOf(l) === seenOpen.id) ?? null);
      const answered =
        seenOpen.id !== null &&
        (was === null || (was.status !== "open" && was.status !== "waiting") || (was.decision_id ?? null) !== seenOpen.record);
      if (answered || expanded === null) setExpanded(openId);
    }
  }
  // A new request, in the stage already shown (a lever whose question is here): its line.
  const [seenRequest, setSeenRequest] = useState(request?.n ?? null);
  if (request && request.n !== seenRequest) {
    setSeenRequest(request.n);
    setExpanded(targetOf(request));
  }
  // A record on the expanded line that is not the open one (an answer changed, a later line
  // answered first) collapses it, and the open line opens (finding 24): the card does not stay
  // expanded on what was just recorded.
  const recordOf = (id: string | null) => stage.lines.find((l) => idOf(l) === id)?.decision_id ?? null;
  const [seenRecord, setSeenRecord] = useState({ id: expanded, record: recordOf(expanded) });
  const nowRecord = recordOf(expanded);
  if (seenRecord.id !== expanded || seenRecord.record !== nowRecord) {
    setSeenRecord({ id: expanded, record: nowRecord });
    if (seenRecord.id === expanded && nowRecord !== null && expanded !== openId) setExpanded(openId);
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

  const view = useProjectView(pid).data;
  const state = view?.state ?? null;
  const noticings = useNoticings(pid);
  const groups = byLabel(stage);
  // A line no card can answer yet comes after those that can (FOUNDATION §3, "Not available yet").
  // Those that count toward nothing and hold no answer to show are one quiet line (finding 15).
  const quiet = groups.Decide.filter(unavailableQuiet);
  const decide = [
    ...groups.Decide.filter(canAnswer),
    ...groups.Decide.filter((l) => !canAnswer(l) && !unavailableQuiet(l)),
  ];
  const drawHeld = drawWaitsForSweep(stage);
  // Results, once reached, is W1-G's column whole: it draws every line of the stage itself.
  const results = columnOf(stage) === "results";
  // The triage of the open noticings, where the shell draws it outside Results' column: under
  // Estimate, at Models' end (log.ts `triageStage`).
  const triageHere = !results && stage.reached && triageStage(log) === stage.key;
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
      settledAs={settledText(log, l, noticings)}
      reopenedAs={reopenedText(log, stage, l, view?.decisions)}
      swept={!!stage.sweep?.confirmed_by && l.label === "Confirm" && l.decision_id === stage.sweep.confirmed_by}
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
          <Noticed log={log} answers={answers} noticings={noticings} onOpenStage={onOpenStage} />
        ) : null}

        {results ? (
          <div data-testid="results-column">
            <ResultsCards pid={pid} />
          </div>
        ) : null}

        {!results && (decide.length || quiet.length) ? (
          <Group label="Decide">
            {decide.map((l) => line(l))}
            {quiet.length ? (
              <li className={q.line} data-kit data-testid="lines-unavailable" data-unavailable>
                <p className={q.lineQuiet}>Not available in this version yet: {listJoin(quiet.map((l) => plain(l.name)))}.</p>
              </li>
            ) : null}
          </Group>
        ) : null}

        {!results && groups.Confirm.length ? (
          <Group label="Confirm" heading={stage.sweep?.heading}>
            {groups.Confirm.map((l) => line(l, true))}
          </Group>
        ) : null}

        {triageHere ? <StageTriage pid={pid} stage={stage.key} /> : null}

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
  settledAs = null,
  reopenedAs = null,
  swept = false,
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
  /** A noticing another line's answer settled, as it reads then (`settledText`). */
  settledAs?: string | null;
  /** The change that asked it again, while it is not answered (`reopenedText`). */
  reopenedAs?: string | null;
  /** Answered by its stage's "Confirm all", one record for every default it confirmed. */
  swept?: boolean;
}) {
  // A default "Confirm all" answered reads as its own value, as stated (the engine's reason, which
  // the sweep recorded), never the record's one sentence for all of them (the re-walk's N6).
  const own = swept ? (line.reason ?? (state ? (DECLARATION_DEFAULTS[line.key]?.(state) ?? null) : null)) : null;
  const answer = own ?? (line.decision_id ? answers.get(line.decision_id) : undefined);
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
  // summary ("…is not asked yet"), which its answer settles. Settled by another line's answer, it
  // names what it was about and that line, which prints the answer once.
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
          {settledAs ?? (settled ? <Plain text={answer} /> : said(line.name))}
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
      {reopenedAs ? (
        <p className={q.lineQuiet} data-testid={`${testId}-reopened`}>
          {reopenedAs}
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
 * it changes, and each links there. A settled one reads as it does on its own stage's card: its
 * answer, or what it was about and the question whose answer settled it.
 */
function Noticed({
  log,
  answers,
  noticings,
  onOpenStage,
}: {
  log: QuestLog;
  answers: ReadonlyMap<string, string>;
  noticings: readonly Finding[] | undefined;
  onOpenStage: (stage: string, line?: string) => void;
}) {
  const noticed = log.stages.flatMap((s) => s.lines.filter((l) => l.source === "finding").map((l) => ({ s, l })));
  if (!noticed.length) return null;
  return (
    <Group label="Noticed" heading="each is decided where it changes an answer">
      {noticed.map(({ s, l }) => {
        const answer = l.status === "answered" && l.decision_id ? answers.get(l.decision_id) : undefined;
        const settledAs = settledText(log, l, noticings);
        const where =
          l.label === "For the record" ? "For the record, in" : l.status === "answered" ? "Settled in" : "Decided in";
        return (
          <li key={`${s.key}:${l.key}`} className={q.line} data-kit data-testid={`noticed-${l.key}`} data-status={l.status}>
            <span className={q.lineName}>{settledAs ?? (answer ? <Plain text={answer} /> : said(l.name))}</span>
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

/**
 * The triage of the open noticings in `stage` (GET /triage; FOUNDATION §7), drawn only from a
 * triage the engine serves for that stage: under Estimate, Models, before the plan's lock (G05).
 * Its own "Confirm all" records every disposition; the stage's foot stays the shell's, which says
 * beside Fit, while the triage is unconfirmed, that it is confirmed before the press (`Primary`).
 */
function StageTriage({ pid, stage }: { pid: string; stage: string }) {
  const served = useTriage(pid).data;
  const id = useId();
  return served && served.stage === stage ? <TriageCard pid={pid} triage={served} id={id} /> : null;
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

/** An item's number as the bundle's checklist labels it (core/export/checklists.py `markdown`):
 *  STROBE's own items under STROBE-nut by "STROBE". */
function itemLabel(item: ChecklistItem): string {
  return item.kind === "STROBE" ? `${item.kind} ${item.id}` : item.id;
}

/**
 * TRIPOD+AI's scope codes in plain words, as the source's own legend defines them (Collins et al.
 * 2024, Table 2's footnote, which core/export/data/tripod_ai.json holds): "D=items relevant only to
 * the development of a prediction model; E=items relating solely to the evaluation of a prediction
 * model; D;E=items applicable to both the development and evaluation of a prediction model". The
 * report serves the code alone; a code the legend does not define is shown as served, with no
 * words.
 */
const SCOPE_WORDS: Readonly<Record<string, string>> = {
  D: "Applies only to developing a model.",
  E: "Applies only to evaluating a model.",
  "D;E": "Applies to both developing and evaluating a model.",
};

/**
 * Write-up's checklist: the reporting guideline's items, by how far the record answers each, then,
 * behind one quiet disclosure as For the record is (FOUNDATION §2, one click away), each item the
 * record answers only in part, with what the author must still supply (`owed`), as the bundle's
 * checklist says it. Every item the record answers in part owes something, and only those do
 * (core/export/checklists.py `fill`), so the disclosure is counted by the items it holds.
 * TRIPOD+AI's scope is said in plain words (`SCOPE_WORDS`), its code the quiet label.
 */
function Checklist({ pid }: { pid: string }) {
  const report = useChecklist(pid).data;
  if (!report) return null;
  const { counts } = report;
  const owed = report.items.filter((i) => i.owed);
  return (
    <section className={q.group} data-kit data-testid="checklist">
      <h3 className={q.groupLabel}>{report.title}</h3>
      <p className={q.lineQuiet}>
        {counts.answered} answered, {counts.partly_answered} partly answered, {counts.unanswered} unanswered,
        of {counts.items} items.
      </p>
      {owed.length ? (
        <details className={q.record} data-testid="checklist-owed">
          <summary>Partly answered · {owed.length}</summary>
          <ul className={q.lines}>
            {owed.map((i) => {
              const words = i.scope ? SCOPE_WORDS[i.scope] : undefined;
              return (
                <li key={i.id} className={q.line}>
                  <div className={q.lineRow}>
                    <span className={q.lineName}>
                      {itemLabel(i)} · <Plain text={i.topic} />
                    </span>
                    {i.scope ? (
                      <span className={q.lineLabel} data-testid={`checklist-scope-code-${i.id}`}>
                        {i.scope}
                      </span>
                    ) : null}
                  </div>
                  {words ? (
                    <p className={q.lineQuiet} data-testid={`checklist-scope-${i.id}`}>
                      {words}
                    </p>
                  ) : null}
                  <p className={q.lineReason}>
                    The author must supply <Plain text={i.owed} />.
                  </p>
                </li>
              );
            })}
          </ul>
        </details>
      ) : null}
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

/** The foot's line beside a waiting action: what it waits for, its name underlined. Where it can
 *  go there, the whole sentence is the link, so it wraps as one sentence, its period with it. */
function HintLine({ hint, onOpenStage }: { hint: Hint; onOpenStage: (stage: string, line?: string) => void }) {
  const to = hint.to;
  if (hint.name && to) {
    return (
      <span data-testid="primary-hint">
        <button type="button" className={q.hintLink} onClick={() => onOpenStage(to.stage, to.line ?? undefined)} data-testid="primary-hint-link">
          {hint.lead}
          <span className={q.hintName}>{hint.name}</span>
          {hint.tail}
        </button>
      </span>
    );
  }
  return (
    <span data-testid="primary-hint">
      {hint.lead}
      {hint.name}
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
  // Models' foot beside Fit reads the triage at its end, where the shell draws it (log.ts
  // `triageBeforeFit`); no other foot reads it.
  const triage = useTriage(pid, stage.key === "models" && triageStage(log) === "models").data;
  const action = primaryFor(log, stage, fit, canAnswer, triage);
  // No exit confirms a role in a block (§14 rule 2; besideAsk.ts `oneRoleAtATime`).
  const onRefused = (err: Error) =>
    setRefusal(isRefusalError(err) ? oneRoleAtATime(err.refusal) : { error: { code: "failed", message: err.message, exits: [] } });

  let node: ReactNode;
  switch (action.kind) {
    case "fit":
      node = (
        <div className={q.footQuiet} data-kit data-testid="primary-fit">
          {action.waits ?? "Fit is at the end of your analysis, beside the card."}
          {action.note ? (
            <>
              {" "}
              <span data-testid="primary-note">{action.note}</span>
            </>
          ) : null}
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
          left={
            action.hint || action.note ? (
              <>
                {action.hint ? <HintLine hint={action.hint} onOpenStage={onOpenStage} /> : null}
                {action.hint && action.note ? " " : null}
                {action.note ? <span data-testid="primary-note">{action.note}</span> : null}
              </>
            ) : null
          }
          onClick={() => action.to && onOpenStage(action.to, action.line)}
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
