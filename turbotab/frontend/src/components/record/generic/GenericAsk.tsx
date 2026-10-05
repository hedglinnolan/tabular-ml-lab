/**
 * The generic question: any open step without a bespoke component, never blank (compose.ts).
 *
 * It wears its grammar's silhouette (DESIGN_LANGUAGE §09: a FACT flat, a CHOICE a bordered card),
 * says the server's question and one line, the step's own data, then the options in the shared
 * option list: the arrow keys move and preview on the stage, Enter or a press records. An option
 * a choice shapes (the effect, the measure, the assumptions declared before any estimate) opens
 * its choices in place at the first press; the second records. A refusal answers at the option
 * with its exits, so the server finishes what this card cannot.
 */
import { useMemo, useState } from "react";
import { Prose, V } from "../../Prose";
import { Pending } from "../blocks";
import { Options, type OptionItem } from "../Options";
import { Question } from "../Question";
import { STAGE_LABEL } from "../../JobChips";
import { Actions, Keep, RecordButton, type AskProps } from "../ask/common";
import {
  decisionOf,
  fieldApplies,
  type FieldValues,
  type GenericField,
  type GenericMatrix,
  type GenericOption,
  type GenericQuestion,
  type TwoLabels,
} from "./compose";
import c from "../ask/ask.module.css";
import g from "./generic.module.css";

const CHIPS = 8;

export function GenericAsk({
  q,
  fallbackTitle,
  ...p
}: AskProps & { q: GenericQuestion; fallbackTitle?: string }) {
  const [values, setValues] = useState<FieldValues>(() =>
    Object.fromEntries(q.fields.map((f) => [f.name, f.initial])),
  );
  const [open, setOpen] = useState<string | null>(null);
  const fieldsFor = (key: string) =>
    q.fields.filter((f) => f.appliesTo !== undefined && fieldApplies(f, key) && f.choices.length);
  const shared = q.fields.filter((f) => f.appliesTo === undefined && f.choices.length);

  const items: OptionItem[] = q.options.map((o) => {
    const own = fieldsFor(o.key);
    return {
      key: o.key,
      label: o.label,
      line: o.line,
      decision: o.na ? null : decisionOf(o, values),
      na: o.na,
      tags: o.tags,
      data: o.chips?.length ? <Chips columns={o.chips} /> : undefined,
      note:
        o.labels || o.note ? (
          <>
            {o.labels ? <LabelPair labels={o.labels} /> : null}
            {o.note ? (
              <span className={g.reason}>
                <Prose text={o.note} />
              </span>
            ) : null}
          </>
        ) : undefined,
      extra:
        open === o.key && own.length ? (
          <span className={g.inlineFields} data-testid={`fields-${o.key}`}>
            {own.map((f) => (
              <FieldControl
                key={f.name}
                field={f}
                value={values[f.name] ?? []}
                onChange={(v) => setValues((cur) => ({ ...cur, [f.name]: v }))}
              />
            ))}
            <button
              type="button"
              className={c.small}
              disabled={p.pending}
              onClick={() => record(o)}
              data-testid={`record-${o.key}`}
            >
              Record {o.label}
            </button>
          </span>
        ) : undefined,
    };
  });

  const record = (o: GenericOption) => {
    const d = decisionOf(o, values);
    if (d) p.record(d, o.key);
  };
  const press = (item: OptionItem) => {
    const o = q.options.find((x) => x.key === item.key);
    if (!o || !o.build) return;
    if (fieldsFor(o.key).length && open !== o.key) {
      setOpen(o.key); // its choices open in place; the next press records
      return;
    }
    record(o);
  };

  return (
    <Question
      {...p.shell}
      entry={p.entry}
      // Without the teaching (an older server), the question is still named, never blank.
      title={p.entry ? undefined : fallbackTitle}
      data={
        q.data.length ? (
          <ul className={g.data} data-testid={`generic-data-${p.shell.qkey}`}>
            {q.data.map((line) => (
              <li key={line}>
                <Prose text={line} />
              </li>
            ))}
          </ul>
        ) : undefined
      }
    >
      <div data-testid={`generic-${p.shell.qkey}`} data-generic>
        {q.waiting ? (
          <Pending testId={`generic-waiting-${p.shell.qkey}`}>
            {STAGE_LABEL[q.waiting] ?? "Computing"}… Then this question's options are offered.
          </Pending>
        ) : null}
        {shared.length ? (
          <div className={g.fields}>
            {shared.map((f) => (
              <FieldControl
                key={f.name}
                field={f}
                value={values[f.name] ?? []}
                onChange={(v) => setValues((cur) => ({ ...cur, [f.name]: v }))}
              />
            ))}
          </div>
        ) : null}
        {items.length ? (
          <Options
            items={items}
            mode="single"
            onRecord={press}
            pending={p.pending}
            answerAt={p.answerAt}
            label={p.entry?.question ?? p.shell.qkey}
            testId={`options-${p.shell.qkey}`}
          />
        ) : null}
        {q.matrix ? (
          <Matrix
            matrix={q.matrix}
            pending={p.pending}
            onRecord={(d) => p.record(d, "matrix")}
          />
        ) : null}
        {p.answerAt?.key === "matrix" ? <div className={c.answer}>{p.answerAt.node}</div> : null}
        {q.note ? <p className={g.note}>{q.note}</p> : null}
        {p.keep ? (
          <Actions>
            <Keep keep={p.keep} />
          </Actions>
        ) : null}
      </div>
    </Question>
  );
}

/** North star 5's two labels, side by side: where the field uses it, and whether it is sound. */
export function LabelPair({ labels }: { labels: TwoLabels }) {
  return (
    <span className={g.labels} data-testid="two-labels">
      {labels.customary ? (
        <span className={g.label}>
          <span className={g.labelKey}>customary</span> <Prose text={labels.customary} />
          {labels.source ? <span className={g.source}> · {labels.source}</span> : null}
        </span>
      ) : null}
      {labels.sound ? (
        <span className={g.label}>
          <span className={g.labelKey}>{labels.verdict ?? "sound"}</span>{" "}
          <Prose text={labels.sound} />
        </span>
      ) : null}
    </span>
  );
}

function Chips({ columns }: { columns: string[] }) {
  const [all, setAll] = useState(false);
  const shown = all ? columns : columns.slice(0, CHIPS);
  return (
    <span className={g.chips}>
      {shown.map((col) => (
        <V key={col}>{col}</V>
      ))}
      {columns.length > CHIPS && !all ? (
        <button type="button" className={g.more} onClick={() => setAll(true)}>
          +{columns.length - CHIPS} more
        </button>
      ) : null}
    </span>
  );
}

function FieldControl({
  field,
  value,
  onChange,
}: {
  field: GenericField;
  value: string[];
  onChange: (v: string[]) => void;
}) {
  const many = field.kind === "many";
  const toggle = (v: string) =>
    many
      ? onChange(value.includes(v) ? value.filter((x) => x !== v) : [...value, v])
      : onChange([v]);
  return (
    <span
      className={g.field}
      role={many ? "group" : "radiogroup"}
      aria-label={field.label}
      data-testid={`field-${field.name}`}
    >
      <span className={g.fieldLabel}>{field.label}</span>
      <span className={many ? g.checks : g.segments}>
        {field.choices.map((ch) => (
          <button
            key={ch.value}
            type="button"
            role={many ? "checkbox" : "radio"}
            aria-checked={value.includes(ch.value)}
            className={many ? g.check : g.segment}
            onClick={() => toggle(ch.value)}
            title={ch.line}
            data-status={ch.status}
            data-testid={`choice-${field.name}-${ch.value || "default"}`}
          >
            <span className={g.choiceLabel}>{ch.label}</span>
            {many && ch.line ? (
              <span className={g.choiceLine}>
                <Prose text={ch.line} />
              </span>
            ) : null}
          </button>
        ))}
      </span>
    </span>
  );
}

function Matrix({
  matrix,
  pending,
  onRecord,
}: {
  matrix: GenericMatrix;
  pending: boolean;
  onRecord: (d: NonNullable<ReturnType<GenericMatrix["build"]>>) => void;
}) {
  const [answers, setAnswers] = useState<Record<string, Record<string, string>>>({});
  const decision = useMemo(() => matrix.build(answers), [matrix, answers]);
  const complete = matrix.rows.filter((r) => matrix.fields.every((f) => answers[r]?.[f.name]));
  return (
    <div className={g.matrix} data-testid="generic-matrix">
      <p className={g.matrixHead}>Asked of each, with no guess to lead:</p>
      <div className={g.matrixScroll}>
        <table className={g.table}>
          <thead>
            <tr>
              <th scope="col">Column</th>
              {matrix.fields.map((f) => (
                <th key={f.name} scope="col">
                  {f.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.rows.map((row) => (
              <tr key={row}>
                <th scope="row">
                  <V>{row}</V>
                </th>
                {matrix.fields.map((f) => (
                  <td key={f.name}>
                    <select
                      className={c.inlineSelect}
                      aria-label={`${row}: ${f.label}`}
                      value={answers[row]?.[f.name] ?? ""}
                      onChange={(e) =>
                        setAnswers((cur) => ({
                          ...cur,
                          [row]: { ...cur[row], [f.name]: e.target.value },
                        }))
                      }
                      data-testid={`matrix-${row}-${f.name}`}
                    >
                      <option value="">—</option>
                      {f.choices.map((ch) => (
                        <option key={ch.value} value={ch.value}>
                          {ch.label}
                        </option>
                      ))}
                    </select>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Actions>
        <RecordButton
          disabled={pending || !decision}
          onClick={() => decision && onRecord(decision)}
          title="Records the answers of every fully answered row; the others stay asked."
          testId="record-matrix"
        >
          {complete.length > 1
            ? `${matrix.recordLabel} (${complete.length})`
            : matrix.recordLabel}
        </RecordButton>
        <span className={c.reason}>
          {complete.length} of {matrix.rows.length} answered
        </span>
      </Actions>
    </div>
  );
}
