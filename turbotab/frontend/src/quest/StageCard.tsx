/**
 * The card (FOUNDATION §3): the stage shown, its name, why it dropped back, then its lines under
 * their labels. The open Decide line is expanded through the dispatcher (CardFor); answered lines
 * collapse to their answers; lines still to come wait below, quieter, each "Waiting for" what it
 * needs. The stage's Confirm sweep comes after its Decides, each default with its reason and what
 * another choice would change, and any can be opened to change it. For the record sits collapsed at
 * the foot (GET /record). The one primary action is at the very foot (FOUNDATION §1).
 */
import { useState, type ReactNode } from "react";
import { isRefusalError } from "../api/client";
import {
  useChecklist,
  useConfirmSweep,
  useDecide,
  useExport,
  useRecord,
} from "../api/queries";
import type { Decision, QuestLine, QuestLog, QuestStage, Refusal, SweepStage } from "../api/schema";
import { Foot, Plain, foot as f, kit as k } from "../kit";
import { CardFor } from "./cards/CardFor";
import { type FitPosition, byLabel, openLine, primaryFor, sweptLines } from "./log";
import { said } from "./words";
import q from "./quest.module.css";

const idOf = (l: QuestLine) => `${l.source}:${l.key}`;

export interface StageCardProps {
  pid: string;
  log: QuestLog;
  stage: QuestStage;
  answers: ReadonlyMap<string, string>;
  fit: FitPosition;
  onOpenStage: (stage: string) => void;
  /** The newest line the manuscript gained since this screen opened (FOUNDATION §3, the rail). */
  arrival: { text: string; kind: string | null } | null;
  onOpenManuscript: () => void;
}

export function StageCard({ pid, log, stage, answers, fit, onOpenStage, arrival, onOpenManuscript }: StageCardProps) {
  const open = openLine(stage);
  const openId = open ? idOf(open) : null;
  const [expanded, setExpanded] = useState<string | null>(openId);
  // A new open line (the last one was answered) is the one expanded.
  const [seenOpen, setSeenOpen] = useState(openId);
  if (seenOpen !== openId) {
    setSeenOpen(openId);
    setExpanded(openId);
  }

  const groups = byLabel(stage);
  const back = open && expanded !== openId ? () => setExpanded(openId) : null;
  const toggle = (l: QuestLine) => setExpanded((cur) => (cur === idOf(l) ? openId : idOf(l)));

  return (
    <section className={q.card} aria-labelledby={`card-${stage.key}`} data-testid={`card-${stage.key}`}>
      <div data-kit className={q.cardHead}>
        <h2 id={`card-${stage.key}`} className={q.cardTitle}>
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

      {groups.Decide.length ? (
        <Group label="Decide">
          {groups.Decide.map((l) => (
            <Line
              key={idOf(l)}
              pid={pid}
              line={l}
              stage={stage}
              expanded={expanded === idOf(l)}
              onToggle={() => toggle(l)}
              answers={answers}
              onOpenStage={onOpenStage}
            />
          ))}
        </Group>
      ) : null}

      {groups.Confirm.length ? (
        <Group label="Confirm" heading={stage.sweep?.heading}>
          {groups.Confirm.map((l) => (
            <Line
              key={idOf(l)}
              pid={pid}
              line={l}
              stage={stage}
              expanded={expanded === idOf(l)}
              onToggle={() => toggle(l)}
              answers={answers}
              onOpenStage={onOpenStage}
              confirm
            />
          ))}
        </Group>
      ) : null}

      {stage.reached ? <ForTheRecord pid={pid} stage={stage} /> : null}
      {stage.key === "writeup" && stage.reached ? <Checklist pid={pid} /> : null}

      {back ? (
        <button type="button" className={k.linkish} data-kit onClick={back} data-testid="back-to-stage">
          Back to {stage.name}
        </button>
      ) : null}

      <Primary pid={pid} log={log} stage={stage} fit={fit} onOpenStage={onOpenStage} />
      {arrival ? (
        <button type="button" className={f.arrival} data-kit onClick={onOpenManuscript} data-testid="arrival">
          <span className={f.arrivalWhere}>Manuscript</span>
          <span className={f.arrivalText}>
            <Plain text={arrival.text} />
          </span>
        </button>
      ) : null}
    </section>
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
  stage,
  expanded,
  onToggle,
  answers,
  onOpenStage,
  confirm = false,
}: {
  pid: string;
  line: QuestLine;
  stage: QuestStage;
  expanded: boolean;
  onToggle: () => void;
  answers: ReadonlyMap<string, string>;
  onOpenStage: (stage: string) => void;
  confirm?: boolean;
}) {
  const answer = line.decision_id ? answers.get(line.decision_id) : undefined;
  const testId = `line-${line.key}`;
  if (expanded && line.status !== "waiting") {
    return (
      <li className={q.lineOpen} data-testid={testId} data-status={line.status} data-expanded>
        <CardFor pid={pid} line={line} stage={stage.key} onOpenStage={onOpenStage} />
      </li>
    );
  }
  return (
    <li className={q.line} data-kit data-testid={testId} data-status={line.status}>
      <div className={q.lineRow}>
        <span className={q.lineName}>{said(line.name)}</span>
        {line.status === "waiting" ? null : (
          <button type="button" className={k.linkish} onClick={onToggle} data-testid={`${testId}-toggle`}>
            {line.status === "answered" || line.status === "set_for_you" ? "Change" : "Answer"}
          </button>
        )}
      </div>
      {line.status === "answered" && answer ? (
        <p className={q.lineAnswer}>
          <Plain text={answer} />
        </p>
      ) : null}
      {line.status === "set_for_you" && line.reason ? (
        <p className={q.lineReason}>
          <Plain text={line.reason} />
        </p>
      ) : null}
      {confirm && line.would_change ? (
        <p className={q.lineQuiet}>
          <Plain text={line.would_change} />
        </p>
      ) : null}
      {line.status === "waiting" ? (
        <p className={q.lineQuiet} data-testid={`${testId}-waiting`}>
          Waiting for:{" "}
          {line.waiting_for.length
            ? line.waiting_for.map((w, i) => (
                <span key={w.key}>
                  {i ? ", " : ""}
                  <button type="button" className={k.linkish} onClick={() => onOpenStage(w.stage)}>
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

function Primary({
  pid,
  log,
  stage,
  fit,
  onOpenStage,
}: {
  pid: string;
  log: QuestLog;
  stage: QuestStage;
  fit: FitPosition;
  onOpenStage: (stage: string) => void;
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
          label="Continue"
          disabled={action.disabled}
          left={action.hint}
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
