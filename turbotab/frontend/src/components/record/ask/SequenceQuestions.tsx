/**
 * The opening sequence's new questions (M2_CONTRACT §1, §10; OPENING_SEQUENCE §03), in the
 * Router's order: which way round the table is, the event, the grain, repeats or time points,
 * the unit of analysis, how rows are combined, and temporal prediction.
 *
 * Each reads what the server measured (the oriented table's shape reading, the structure stage's
 * grain and repeats readings, the outcome's levels) and puts it beside the option it suggests,
 * never selecting it. Every option previews on the stage before anything is recorded.
 */
import { useState } from "react";
import type {
  AggregationMethod,
  GrainAnswer,
  OrientedArtifact,
  StructureArtifact,
} from "../../../api/m2-types";
import { grainDecision } from "../../../api/m2-types";
import type { ProjectState, TargetInfoArtifact, Task } from "../../../api/schema";
import { V } from "../../Prose";
import { Options, type OptionItem } from "../Options";
import { Question } from "../Question";
import { Taught } from "../teach";
import { Actions, Keep, fmtCount, taught, type AskProps } from "./common";
import c from "./ask.module.css";

function KeepRow({ keep, label }: { keep?: (() => void) | undefined; label?: string }) {
  return keep ? (
    <Actions>
      <Keep keep={keep} label={label} />
    </Actions>
  ) : null;
}

const optionLine = (p: AskProps, value: string, fallback = "") =>
  taught(p.entry, value)?.consequence ?? fallback;
const optionLabel = (p: AskProps, value: string, fallback: string) =>
  taught(p.entry, value)?.label ?? fallback;

// ── 1.5 · which way round the table is ──────────────────────────────────────

export function OrientationAsk({
  oriented,
  current,
  ...p
}: AskProps & { oriented: OrientedArtifact | null; current: ProjectState["orientation"] }) {
  const reading = oriented?.reading;
  const turn = oriented?.turn;
  const items: OptionItem[] = (["sample_major", "feature_major"] as const).map((value) => ({
    key: value,
    label: optionLabel(
      p,
      value,
      value === "sample_major" ? "Rows are samples" : "Rows are features",
    ),
    line: optionLine(p, value),
    decision: { kind: "set_orientation", orientation: value },
    data:
      value === "feature_major" && turn && !turn.refusal ? (
        <>
          → {fmtCount(turn.n_samples)} rows × {fmtCount(turn.n_features)}
        </>
      ) : undefined,
    tags:
      reading && reading.reading === value
        ? [{ text: `read from the shape, ${reading.confidence}`, tone: "detected" }]
        : undefined,
    // The table cannot be turned (duplicate feature names, no name column): said in place.
    na: value === "feature_major" && turn?.refusal ? turn.refusal : undefined,
  }));
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={reading ? <Taught text={reading.sentence} /> : undefined}
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={current}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Orientation — preview with the arrow keys, Enter to record"
        testId="options-orientation"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── 2 · which level is the event ─────────────────────────────────────────────

export function EventAsk({
  info,
  current,
  ...p
}: AskProps & { info: TargetInfoArtifact; current: string | null }) {
  const classes = info.classes ?? [];
  const total = classes.reduce((n, k) => n + k.count, 0);
  const label = (v: unknown) => String(v);
  const items: OptionItem[] = classes.map((k) => {
    const level = label(k.value);
    const others = classes.filter((x) => x !== k).map((x) => `\`${label(x.value)}\``);
    return {
      key: level,
      label: (
        <>
          <span className="num">{level}</span> is the event
        </>
      ),
      previewLabel: `${level} as the event`,
      // Never guessed (OPENING_SEQUENCE §03.2): no tag, no order but the data's.
      line: `Models give the probability of \`${level}\`; ${others.join(", ")} ${others.length === 1 ? "is" : "are"} the reference.`,
      data: total ? (
        <>
          {fmtCount(k.count)} rows · {Math.round((k.count / total) * 100)}%
        </>
      ) : undefined,
      decision: { kind: "set_event", column: info.column, level },
    };
  });
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      title={
        <>
          Which level of <V>{info.column}</V> is the event?
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
        label="Event level — preview with the arrow keys, Enter to record"
        testId="options-event"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── 3 · can one unit appear in more than one row ────────────────────────────

const ID_LIKE =
  /(^|_)(id|seqn|code|key)$|^seqn$|participant|subject|person|patient|respondent|sample|household|record_?no/i;

/** "the gaps run…" → "The gaps run…": a reading's clause said as a sentence. */
const asSentence = (clause: string) =>
  `${clause.charAt(0).toUpperCase()}${clause.slice(1)}${/[.!?]$/.test(clause) ? "" : "."}`;

export function GrainAsk({
  structure,
  target,
  columns,
  current,
  ...p
}: AskProps & {
  structure: StructureArtifact | null;
  target: string | null;
  columns: string[];
  current: ProjectState["grain"];
}) {
  const reading = structure?.grain ?? null;
  const evidence = (reading?.evidence ?? []).filter((e) => e.column !== target && e.rows_per > 1);
  // The identifier choices: the reading's first suggestion, and any other that is named like an
  // identifier (an age or a date repeats too, but names no one); any column is one step away.
  const suggested = (reading?.suggested ?? []).filter((col) => col !== target);
  const choices = suggested.filter((col, i) => i === 0 || ID_LIKE.test(col)).slice(0, 3);
  const [idColumn, setIdColumn] = useState<string | null>(current?.id_column ?? choices[0] ?? null);
  const top = evidence.find((e) => e.column === (choices[0] ?? ""));
  const contradiction = reading?.if_one_row ?? null;
  const recorded = current
    ? (current.grain as GrainAnswer) === "repeated" && current.id_column !== idColumn
      ? null
      : current.grain
    : null;
  const others = columns.filter((col) => col !== target && !choices.includes(col));
  const repeatsExtra = (
    <span className={c.modifierInline} role="group" aria-label="Which column names the unit">
      <span className={c.modifierLabel}>named by</span>
      {choices.map((col) => {
        const ev = evidence.find((e) => e.column === col);
        return (
          <button
            key={col}
            type="button"
            className={c.toggle}
            aria-pressed={idColumn === col}
            onClick={() => setIdColumn(col)}
            data-testid={`grain-id-${col}`}
            title={
              ev ? `${fmtCount(ev.n_distinct)} values over ${fmtCount(ev.n_rows)} rows` : undefined
            }
          >
            <span className="num">{col}</span>
          </button>
        );
      })}
      {others.length ? (
        <select
          className={c.inlineSelect}
          aria-label="Another column"
          value={idColumn && !choices.includes(idColumn) ? idColumn : ""}
          onChange={(e) => setIdColumn(e.target.value || null)}
        >
          <option value="">another column…</option>
          {others.map((col) => (
            <option key={col} value={col}>
              {col}
            </option>
          ))}
        </select>
      ) : null}
    </span>
  );
  const order: GrainAnswer[] = ["one_row_per_unit", "repeated", "unknown"];
  const items: OptionItem[] = order.map((value) => {
    if (value === "unknown")
      return {
        key: value,
        label: optionLabel(p, value, "I don't know"),
        line: optionLine(
          p,
          value,
          "The seal is drawn by row and says so; its held-out scores are labeled exploratory.",
        ),
        decision: grainDecision("unknown", null),
      };
    if (value === "repeated")
      return {
        key: value,
        label: optionLabel(p, value, "People repeat"),
        line: optionLine(p, value),
        decision: idColumn ? grainDecision("repeated", idColumn) : null,
        previewLabel: idColumn ? `Repeats per ${idColumn}` : undefined,
        tags:
          top && top.regular_share >= 0.5 ? [{ text: "suggested", tone: "suggested" }] : undefined,
        extra: repeatsExtra,
      };
    return {
      key: value,
      label: optionLabel(p, value, "One row each"),
      line: optionLine(p, value),
      decision: grainDecision("one_row_per_unit", null),
      // The contradiction, in the coach's voice, beside the option it contradicts: what it would
      // cost (the counts are already the card's data line, so they are not said twice).
      note:
        contradiction && top ? (
          <span className={c.coachNote} data-testid="grain-contradiction">
            <Taught
              text={`\`${top.column}\` repeats: one person's rows could sit on both sides of the seal.`}
            />
          </span>
        ) : undefined,
    };
  });
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        top ? (
          <Taught
            text={`\`${top.column}\` has \`${fmtCount(top.n_distinct)}\` values across \`${fmtCount(top.n_rows)}\` rows, about \`${top.rows_per}\` each.`}
          />
        ) : reading ? (
          <Taught text="No column repeats the way an identifier would; only you know whether one person can appear twice." />
        ) : undefined
      }
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => {
          if (o.decision) p.record(o.decision, o.key);
        }}
        recordedKey={recorded}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Grain — preview with the arrow keys, Enter to record"
        testId="options-grain"
      />
      <KeepRow keep={p.keep} label={current ? undefined : "Keep the stated reading"} />
    </Question>
  );
}

// ── 4 · repeats or time points ──────────────────────────────────────────────

export function RepeatKindAsk({
  structure,
  current,
  ...p
}: AskProps & { structure: StructureArtifact | null; current: ProjectState["repeat_kind"] }) {
  const reading = structure?.repeats ?? null;
  const timeColumn = structure?.time_column ?? null;
  const items: OptionItem[] = (["repeats", "time_points"] as const).map((value) => ({
    key: value,
    label: optionLabel(p, value, value === "repeats" ? "Repeats" : "Time points"),
    line: optionLine(p, value),
    decision: {
      kind: "set_repeat_kind",
      repeat_kind: value,
      time_column: value === "time_points" ? timeColumn : null,
    },
    tags:
      reading?.reading === value
        ? [{ text: `read from the data, ${reading.confidence ?? "low"}`, tone: "detected" }]
        : undefined,
  }));
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        reading?.evidence.length ? <Taught text={asSentence(reading.evidence[0]!)} /> : undefined
      }
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={current?.repeat_kind ?? null}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Repeats or time points — preview with the arrow keys, Enter to record"
        testId="options-repeat_kind"
      />
      <KeepRow keep={p.keep} label={current ? undefined : "Keep the stated reading"} />
    </Question>
  );
}

// ── 5 · the unit of analysis ────────────────────────────────────────────────

export function UnitAsk({
  structure,
  nRows,
  current,
  ...p
}: AskProps & {
  structure: StructureArtifact | null;
  /** Rows of the table before any combining (the oriented table's). */
  nRows: number | null;
  current: ProjectState["unit"];
}) {
  const units = structure?.units ?? null;
  const items: OptionItem[] = (["unit", "row"] as const).map((value) => {
    const n = value === "unit" ? (units?.n_units ?? null) : nRows;
    return {
      key: value,
      label: optionLabel(p, value, value === "unit" ? "One row per person" : "One row per record"),
      line: optionLine(p, value),
      decision: { kind: "set_unit", unit: value },
      // No default (OPENING_SEQUENCE §03.5): the rows each answer leads to, side by side.
      data: n !== null ? <>{fmtCount(n)} rows</> : undefined,
    };
  });
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        units ? (
          <Taught
            text={`\`${units.column}\` names \`${fmtCount(units.n_units)}\` units, ${
              units.min_rows_per_unit === units.max_rows_per_unit
                ? `\`${units.max_rows_per_unit}\` rows each`
                : `\`${units.min_rows_per_unit}\` to \`${units.max_rows_per_unit}\` rows each`
            }.`}
          />
        ) : undefined
      }
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={current}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Unit of analysis — preview with the arrow keys, Enter to record"
        testId="options-unit"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── 6 · combining a unit's rows ─────────────────────────────────────────────

const OUTCOME_RULES = ["first", "last", "mean"] as const;
type OutcomeRule = (typeof OUTCOME_RULES)[number];

export function AggregationAsk({
  structure,
  task,
  current,
  ...p
}: AskProps & {
  structure: StructureArtifact | null;
  task: Task | null;
  current: ProjectState["aggregation"];
}) {
  const menu = structure?.aggregation ?? null;
  const outcome = structure?.outcome ?? null;
  const varies = !!outcome?.varies;
  const [rule, setRule] = useState<OutcomeRule | null>(current?.outcome ?? null);
  const rules = OUTCOME_RULES.filter(
    (r) => r !== "mean" || (outcome?.numeric && task !== "binary" && task !== "multiclass"),
  );
  const offered: AggregationMethod[] = menu?.options ?? ["mean", "first", "last", "change"];
  // The recommended option first, with its reason; the menu's own order for the rest.
  const ordered = [
    ...offered.filter((m) => m === menu?.recommended),
    ...offered.filter((m) => m !== menu?.recommended),
  ];
  const items: OptionItem[] = ordered.map((m) => ({
    key: m,
    label: optionLabel(p, m, m),
    line: optionLine(p, m),
    decision: { kind: "set_aggregation", method: m, outcome: varies ? rule : null },
    previewLabel: `${optionLabel(p, m, m)}${varies && rule ? `, the ${rule} outcome` : ""}`,
    tags: m === menu?.recommended ? [{ text: "recommended", tone: "usual" }] : undefined,
    note:
      m === menu?.recommended && menu?.reason ? (
        <span className={c.hint}>
          <Taught text={menu.reason} />
        </span>
      ) : undefined,
  }));
  const recordedKey =
    current && (!varies || (current.outcome ?? null) === rule) ? current.method : null;
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={menu && !menu.recommended && menu.reason ? <Taught text={menu.reason} /> : undefined}
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={recordedKey}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Combining rows — preview with the arrow keys, Enter to record"
        testId="options-aggregation"
      />
      {varies && outcome ? (
        // When the outcome itself changes within a unit, which value is the outcome is asked too.
        <div className={c.modifier} role="group" aria-label="Which outcome value is kept">
          <span className={c.modifierLabel}>
            <V>{outcome.column}</V> changes within {fmtCount(outcome.n_units_varying)} units; keep
            its
          </span>
          {rules.map((r) => (
            <button
              key={r}
              type="button"
              className={c.toggle}
              aria-pressed={rule === r}
              onClick={() => setRule(r)}
              data-testid={`outcome-${r}`}
            >
              {r === "mean" ? "average" : r}
            </button>
          ))}
        </div>
      ) : null}
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── 7 · temporal prediction ─────────────────────────────────────────────────

export function TemporalAsk({
  structure,
  current,
  ...p
}: AskProps & { structure: StructureArtifact | null; current: ProjectState["temporal"] }) {
  const timeColumn = structure?.time_column ?? null;
  const items: OptionItem[] = (["true", "false"] as const).map((value) => ({
    key: value,
    label: optionLabel(p, value, value === "true" ? "Yes, later from earlier" : "No"),
    line: optionLine(p, value),
    decision: {
      kind: "set_temporal",
      temporal: value === "true",
      time_column: value === "true" ? timeColumn : null,
    },
    data:
      value === "true" && timeColumn ? (
        <>
          by <span className="num">{timeColumn}</span>
        </>
      ) : undefined,
  }));
  return (
    <Question {...p.shell} entry={p.entry}>
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={current ? String(current.temporal) : null}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Temporal prediction — preview with the arrow keys, Enter to record"
        testId="options-temporal"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}
