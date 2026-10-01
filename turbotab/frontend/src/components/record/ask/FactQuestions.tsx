/**
 * The FACT questions about the table: the lens, the outcome, the task, the purpose.
 * Detection suggests beside an option; it never pre-selects one.
 */
import { useId, useMemo, useState, type ReactNode } from "react";
import type {
  ColumnInfo,
  ColumnSummary,
  Lens,
  LensHint,
  Purpose,
  TargetInfoArtifact,
  Task,
} from "../../../api/schema";
import { useStageFocus } from "../../../state/focus";
import { Prose, V } from "../../Prose";
import { ColumnPicker } from "../ColumnPicker";
import { Options, type OptionItem } from "../Options";
import { Question } from "../Question";
import { LENS_LABEL } from "../sentences";
import { Actions, Keep, RecordButton, taught, type AskProps } from "./common";
import c from "./ask.module.css";

const LENS_ORDER: Lens[] = ["dietary", "clinical", "metabolomics", "genomics", "survey"];

// ── lens ─────────────────────────────────────────────────────────────────────

export function LensAsk({
  current,
  hints,
  hintsNote,
  ...p
}: AskProps & { current: Lens[] | null; hints: LensHint[]; hintsNote?: ReactNode }) {
  // Reopening shows the user's own recorded answer; hints are never selected for them.
  const [chosen, setChosen] = useState<Lens[]>(current ?? []);
  const order = p.entry ? (p.entry.options.map((o) => o.value) as Lens[]) : LENS_ORDER;
  const ordered = order.filter((l) => chosen.includes(l));
  const toggle = (l: string) =>
    setChosen((cur) =>
      cur.includes(l as Lens) ? cur.filter((x) => x !== l) : [...cur, l as Lens],
    );
  const items: OptionItem[] = order.map((l) => {
    const t = taught(p.entry, l);
    const hint = hints.find((h) => h.lens === l);
    const withThis = order.filter((x) => x === l || chosen.includes(x));
    return {
      key: l,
      label: t?.label ?? LENS_LABEL[l],
      line: t?.consequence ?? "",
      decision: { kind: "set_lens", lenses: withThis },
      tags: hint ? [{ text: "suggested", tone: "suggested" }] : undefined,
      note: hint ? (
        <span className={c.hint}>
          because <Prose text={hint.because} />
        </span>
      ) : undefined,
    };
  });
  const record = () => ordered.length && p.record({ kind: "set_lens", lenses: ordered }, "record");
  const one = ordered.length === 1 ? (taught(p.entry, ordered[0]!)?.label ?? ordered[0]) : null;
  return (
    <Question {...p.shell} entry={p.entry}>
      <Options
        items={items}
        mode="multi"
        selected={new Set(chosen)}
        onToggle={toggle}
        onRecord={record}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Lenses — choose all that apply, then record"
        testId="options-lens"
      />
      {hintsNote}
      <Actions>
        <RecordButton
          disabled={ordered.length === 0 || p.pending}
          onClick={record}
          title="Records this lens choice. It changes what TurboTab looks for; it never removes an option."
          testId="record-lens"
        >
          {ordered.length === 0
            ? "Record the lens"
            : one
              ? `Record the ${one.toLowerCase()} lens`
              : `Record these ${ordered.length} lenses`}
        </RecordButton>
        <Keep keep={p.keep} />
        {ordered.length === 0 ? (
          <span className={c.reason}>
            Choose at least one. An empty answer would read the same as never having asked.
          </span>
        ) : null}
      </Actions>
      {p.answerAt?.key === "record" ? <div className={c.answer}>{p.answerAt.node}</div> : null}
    </Question>
  );
}

// ── target ───────────────────────────────────────────────────────────────────

export function TargetAsk({
  columns,
  summaries,
  current,
  ...p
}: AskProps & {
  columns: ColumnInfo[] | undefined;
  summaries?: Map<string, ColumnSummary>;
  current: string | null;
}) {
  const [chosen, setChosen] = useState<string | null>(current);
  const labelId = useId();
  const { setFocus } = useStageFocus();
  const choose = (name: string) => {
    setChosen(name);
    setFocus({
      kind: "option",
      decision: { kind: "set_target", column: name },
      label: `${name} as the outcome`,
    });
  };
  const changed = chosen !== null && chosen !== current;
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      title={<span id={labelId}>{p.entry?.question ?? "Which column is the outcome?"}</span>}
    >
      {columns ? (
        <ColumnPicker
          columns={columns}
          value={chosen}
          onChange={choose}
          onCommit={(name) => name !== current && p.record({ kind: "set_target", column: name })}
          summaries={summaries}
          labelId={labelId}
        />
      ) : (
        <p className={c.reason}>
          The file is still being read; its columns appear here when it is done.
        </p>
      )}
      <Actions>
        <RecordButton
          disabled={!changed || p.pending}
          onClick={() => chosen && p.record({ kind: "set_target", column: chosen })}
          title="Records this column as the outcome. Everything downstream is built around it."
          testId="record-target"
        >
          {chosen ? (
            <>
              Record <span className="num">{chosen}</span> as the outcome
            </>
          ) : (
            "Choose a column above"
          )}
        </RecordButton>
        <Keep keep={p.keep} />
      </Actions>
      {p.answerAt ? <div className={c.answer}>{p.answerAt.node}</div> : null}
    </Question>
  );
}

// ── task ─────────────────────────────────────────────────────────────────────

const TASKS: Task[] = ["regression", "binary", "multiclass"];

export function TaskAsk({
  info,
  current,
  ...p
}: AskProps & { info: TargetInfoArtifact; current: Task | null }) {
  const items: OptionItem[] = TASKS.map((task) => {
    const t = taught(p.entry, task);
    return {
      key: task,
      label: t?.label ?? task,
      line: t?.consequence ?? "",
      decision: { kind: "set_task", column: info.column, task },
      tags: task === info.detected_task ? [{ text: "detected", tone: "detected" }] : undefined,
    };
  });
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      title={
        <>
          What kind of outcome is <V>{info.column}</V>?
        </>
      }
      data={
        <>
          TurboTab read <V>{info.column}</V> with <V>{info.confidence}</V> confidence.{" "}
          <Prose text={info.reason} />
          {info.confidence !== "high"
            ? " That is not certain enough to assume, so it is asked."
            : ""}
        </>
      }
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={current}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Task — preview with the arrow keys, Enter to record"
        testId="options-task"
      />
      {p.keep ? (
        <Actions>
          <Keep keep={p.keep} label={current ? undefined : "Keep the detected task"} />
        </Actions>
      ) : null}
    </Question>
  );
}

// ── purpose ──────────────────────────────────────────────────────────────────

const PURPOSES: Purpose[] = ["prediction", "inference"];

export function PurposeAsk({ current, ...p }: AskProps & { current: Purpose | null }) {
  const items = useMemo<OptionItem[]>(
    () =>
      PURPOSES.map((purpose) => {
        const t = taught(p.entry, purpose);
        return {
          key: purpose,
          label: t?.label ?? purpose,
          line: t?.consequence ?? "",
          decision: { kind: "set_purpose", purpose },
        };
      }),
    [p.entry],
  );
  return (
    <Question {...p.shell} entry={p.entry}>
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={current}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Purpose — preview with the arrow keys, Enter to record"
        testId="options-purpose"
      />
      {p.keep ? (
        <Actions>
          <Keep keep={p.keep} />
        </Actions>
      ) : null}
    </Question>
  );
}
