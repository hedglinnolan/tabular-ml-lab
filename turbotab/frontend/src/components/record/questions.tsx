/**
 * The four M0 questions' answer areas. Each owns only its draft answer; the
 * Record posts the decision and decides which silhouette is on screen.
 * Detection suggests beside an option; it never pre-selects one.
 */
import { useId, useState } from "react";
import type {
  ColumnInfo,
  ColumnSummary,
  Lens,
  LensHint,
  Purpose,
  Refusal,
  TargetInfoArtifact,
  Task,
} from "../../api/schema";
import { LENSES } from "../../api/schema";
import { Prose, V } from "../Prose";
import { ColumnPicker } from "./ColumnPicker";
import { LENS_LABEL, PURPOSE_TEXT, TASK_TEXT } from "./sentences";
import c from "./controls.module.css";

interface Common {
  pending: boolean;
  /** Present when reopening an answered question: leaves the record unchanged. */
  onKeep?: () => void;
}

function Keep({ onKeep }: { onKeep?: () => void }) {
  if (!onKeep) return null;
  return (
    <button
      type="button"
      className={c.ghost}
      onClick={onKeep}
      title="Closes the question and leaves the recorded answer as it is."
    >
      Keep the recorded answer
    </button>
  );
}

// ─── lens ────────────────────────────────────────────────────────────────────
export function LensAnswers({
  current,
  hints,
  onSubmit,
  pending,
  onKeep,
}: Common & { current: Lens[] | null; hints: LensHint[]; onSubmit: (l: Lens[]) => void }) {
  // Reopening shows the user's own recorded answer; hints are never selected for them.
  const [chosen, setChosen] = useState<Lens[]>(current ?? []);
  const hintFor = (l: Lens) => hints.find((h) => h.lens === l);
  const toggle = (l: Lens) =>
    setChosen((cur) => (cur.includes(l) ? cur.filter((x) => x !== l) : [...cur, l]));
  const ordered = LENSES.filter((l) => chosen.includes(l));
  return (
    <>
      <ul className={c.options} aria-label="Lenses">
        {LENSES.map((l) => {
          const hint = hintFor(l);
          return (
            <li key={l} className={c.option}>
              <button
                type="button"
                className={c.chip}
                aria-pressed={chosen.includes(l)}
                onClick={() => toggle(l)}
                data-testid={`lens-${l}`}
              >
                <span className={c.check} aria-hidden="true" />
                {LENS_LABEL[l]}
                <span className={c.key}>{l}</span>
              </button>
              {hint ? (
                <span className={c.hint}>
                  <span className={c.hintLabel}>suggested</span>
                  because <Prose text={hint.because} />
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      <div className={c.actions}>
        <button
          type="button"
          className={c.primary}
          disabled={ordered.length === 0 || pending}
          onClick={() => onSubmit(ordered)}
          title="Records this lens choice. It changes what TurboTab looks for; it never removes an option."
        >
          {ordered.length === 0
            ? "Record the lens"
            : `Record ${ordered.length === 1 ? "this lens" : `these ${ordered.length} lenses`}`}
        </button>
        <Keep onKeep={onKeep} />
        {ordered.length === 0 ? (
          <span className={c.reason}>
            Pick at least one. An empty answer would read the same as never having asked.
          </span>
        ) : null}
      </div>
    </>
  );
}

// ─── target ──────────────────────────────────────────────────────────────────
export function TargetAnswers({
  columns,
  summaries,
  current,
  onSubmit,
  pending,
  onKeep,
  labelId,
}: Common & {
  columns: ColumnInfo[];
  summaries?: Map<string, ColumnSummary>;
  current: string | null;
  onSubmit: (column: string) => void;
  labelId?: string;
}) {
  const [chosen, setChosen] = useState<string | null>(current);
  const changed = chosen !== null && chosen !== current;
  return (
    <>
      <ColumnPicker
        columns={columns}
        value={chosen}
        onChange={setChosen}
        onCommit={(name) => name !== current && onSubmit(name)}
        summaries={summaries}
        labelId={labelId}
      />
      <div className={c.actions}>
        <button
          type="button"
          className={c.primary}
          disabled={!changed || pending}
          onClick={() => chosen && onSubmit(chosen)}
          data-testid="record-target"
        >
          {chosen ? (
            <>
              Record <span className="num">{chosen}</span> as the outcome
            </>
          ) : (
            "Choose a column above"
          )}
        </button>
        <Keep onKeep={onKeep} />
      </div>
    </>
  );
}

// ─── task ────────────────────────────────────────────────────────────────────
export function TaskAnswers({
  info,
  current,
  onSubmit,
  pending,
  onKeep,
}: Common & { info: TargetInfoArtifact; current: Task | null; onSubmit: (t: Task) => void }) {
  const tasks: Task[] = ["regression", "binary", "multiclass"];
  return (
    <>
      <p className={c.detected}>
        TurboTab read <V>{info.column}</V> with <V>{info.confidence}</V> confidence.{" "}
        <Prose text={info.reason} />
        {info.confidence !== "high" ? " That is not certain enough to assume, so it is asked." : ""}
      </p>
      <div className={c.cards} role="group" aria-label="Task">
        {tasks.map((t) => (
          <button
            key={t}
            type="button"
            className={c.card}
            aria-pressed={current === t}
            disabled={pending}
            onClick={() => onSubmit(t)}
            data-testid={`task-${t}`}
          >
            <span className={c.cardTitle}>
              {TASK_TEXT[t].label}
              {t === info.detected_task ? <span className={c.tag}>detected</span> : null}
            </span>
            <span className={c.cardBody}>{TASK_TEXT[t].body}</span>
          </button>
        ))}
      </div>
      {onKeep ? (
        <div className={c.actions}>
          <Keep onKeep={onKeep} />
        </div>
      ) : null}
    </>
  );
}

// ─── purpose ─────────────────────────────────────────────────────────────────
export function PurposeAnswers({
  current,
  onSubmit,
  pending,
  onKeep,
}: Common & { current: Purpose | null; onSubmit: (p: Purpose) => void }) {
  const purposes: Purpose[] = ["prediction", "inference"];
  return (
    <>
      <div className={c.cards} role="group" aria-label="Purpose">
        {purposes.map((p) => (
          <button
            key={p}
            type="button"
            className={c.card}
            aria-pressed={current === p}
            disabled={pending}
            onClick={() => onSubmit(p)}
            data-testid={`purpose-${p}`}
          >
            <span className={c.cardTitle}>{PURPOSE_TEXT[p].label}</span>
            <span className={c.cardBody}>{PURPOSE_TEXT[p].body}</span>
          </button>
        ))}
      </div>
      {onKeep ? (
        <div className={c.actions}>
          <Keep onKeep={onKeep} />
        </div>
      ) : null}
    </>
  );
}

// ─── refusal ─────────────────────────────────────────────────────────────────
/** A refusal is an acknowledgment too: it says why, and offers the server's exits. */
export function RefusalNote({
  refusal,
  onExit,
  onDismiss,
}: {
  refusal: Refusal;
  onExit: (exit: Refusal["error"]["exits"][number]) => void;
  onDismiss: () => void;
}) {
  const id = useId();
  return (
    <div className={c.refusal} role="alert" aria-labelledby={id} data-testid="refusal">
      <p id={id} className={c.refusalText}>
        <Prose text={refusal.error.message} />
      </p>
      <div className={c.refusalExits}>
        {refusal.error.exits.map((exit) => (
          <button key={exit.label} type="button" className={c.ghost} onClick={() => onExit(exit)}>
            {exit.label}
          </button>
        ))}
        {refusal.error.exits.length === 0 ? (
          <button type="button" className={c.ghost} onClick={onDismiss}>
            Understood
          </button>
        ) : null}
      </div>
    </div>
  );
}
