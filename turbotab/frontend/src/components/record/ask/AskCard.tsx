/**
 * The ledger's one ask card, inside the open question whose answer feeds the consumer that reads
 * these readings (BLUEPRINT §14.2; turbotab/core/ask.py, served on the step as `ask`):
 *
 *   "Tell me about these columns": each reading's best guess, pre-filled, with its evidence, in
 *   the server's order (by consequence); a homogeneous family (twenty item columns read alike) is
 *   one line, confirmed as one block. A confirmation is `confirm_readings` listing exactly the
 *   readings of the line it confirms, each with the value the line shows — never a reading the
 *   card does not list (§14.1). Each line's value can be changed before it is confirmed. Once the
 *   user has confirmed guesses line by line, one block confirms every line shown (§11.4 rule 4:
 *   a shortcut unlocks with mastery, never with a setting).
 *
 *   The consumer's own ways forward, where its refusal names them (a value below a detection
 *   limit, an energy column's unit and days), are offered as the server words them.
 *
 *   "Read from your data" (§14.3): the readings the values settled, each with its evidence and
 *   the answers that change it; never a required tap.
 */
import { useId, useMemo, useState, type ReactNode } from "react";
import type { AskCard as Card, AskGroup, ReadingKind } from "../../../api/m3-types";
import type { Decision } from "../../../api/schema";
import { Prose, V } from "../../Prose";
import { fmtCount } from "./common";
import c from "./ask.module.css";
import g from "../generic/generic.module.css";

/** The kinds `confirm_reading(s)` records (readings.py CONFIRMABLE); others answer by their own
 *  decisions, offered as the server's exits. */
export const CONFIRMABLE: readonly ReadingKind[] = [
  "role",
  "cluster",
  "unit",
  "day_count",
  "code_or_count",
  "time_column",
  "nested_in",
  "sex_coding",
];

const ROLE_WORDS: Record<string, string> = {
  exposure: "an exposure",
  covariate: "a covariate",
  energy: "total energy intake",
  identifier: "the unit's identifier",
  cluster: "a cluster of units",
  design: "part of the survey design",
  time: "the time of each row",
  flag: "a flag on another column",
  excluded: "left out of the models",
};
const UNITS = [
  "kcal", "kj", "g", "kg", "lb", "cm", "m", "in", "years", "months", "weeks", "days", "pct_energy",
];

/** The values a line can be changed to, by kind (readings.py VALUES), in words. */
export function choicesFor(group: AskGroup): { value: string; label: string }[] {
  if (unitAndDays(group)) return [];
  switch (group.kind) {
    case "code_or_count":
      return [
        { value: "amount", label: "an amount" },
        { value: "code", label: "codes for categories" },
      ];
    case "cluster":
      return [
        { value: "yes", label: "rows belong together" },
        { value: "no", label: "groups nothing" },
      ];
    case "role":
      return Object.entries(ROLE_WORDS).map(([value, label]) => ({ value, label }));
    case "unit":
      return UNITS.map((u) => ({ value: u, label: u === "pct_energy" ? "% of energy" : u }));
    case "time_column":
      return [{ value: "orders", label: "orders the records" }];
    case "nested_in":
      return [
        ...(group.guess ? [{ value: group.guess, label: `part of ${group.guess}` }] : []),
        { value: "not_nested", label: "part of no total" },
      ];
    case "day_count":
      return [1, 2, 3, 4, 7].map((d) => ({ value: String(d), label: `${d} day${d === 1 ? "" : "s"}` }));
    default:
      return group.guess ? [{ value: group.guess, label: group.guess_words }] : [];
  }
}

/** Total energy's unit and days, asked as one line (`kcal:1`): answered by `set_column_unit`,
 *  the screens' own exit, never by a reading confirmation (ask.py `_energy_unit_lines`). */
const unitAndDays = (group: AskGroup) => group.kind === "unit" && (group.guess ?? "").includes(":");

/** The one decision that confirms a line: exactly its readings, each with `value`. */
export function confirmLine(group: AskGroup, value: string): Decision | null {
  if (!CONFIRMABLE.includes(group.kind as ReadingKind) || !value || unitAndDays(group)) return null;
  return {
    kind: "confirm_readings",
    items: group.columns.map((column) => ({
      reading: group.kind as ReadingKind,
      column,
      value,
    })),
  };
}

/** Every line the card shows, each with the value it shows: one block that lists them all. */
export function confirmAll(groups: AskGroup[], values: Record<number, string>): Decision | null {
  const items = groups.flatMap((grp, i) => {
    const value = values[i] ?? grp.guess ?? "";
    if (!confirmLine(grp, value)) return [];
    return grp.columns.map((column) => ({ reading: grp.kind as ReadingKind, column, value }));
  });
  const seen = new Set<string>();
  const unique = items.filter((it) => {
    const k = `${it.reading}:${it.column}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return unique.length ? { kind: "confirm_readings", items: unique } : null;
}

const isConfirmation = (d: Record<string, unknown> | null) =>
  d?.kind === "confirm_reading" || d?.kind === "confirm_readings";

export function AskCard({
  card,
  pending,
  record,
  answerAt,
  mastered = false,
}: {
  card: Card;
  pending: boolean;
  /** Record a decision; `at` names the line a refusal answers under. */
  record: (d: Decision, at: string) => void;
  answerAt: { key: string; node: ReactNode } | null;
  /** The user has confirmed guesses line by line before (BLUEPRINT §11.4 rule 4: a shortcut
   *  unlocks with mastery): the card then offers one block that lists every line it settles. */
  mastered?: boolean;
}) {
  const titleId = useId();
  const [values, setValues] = useState<Record<number, string>>({});
  const confirmable = card.groups.filter((grp, i) => confirmLine(grp, values[i] ?? grp.guess ?? ""));
  const all = useMemo(() => confirmAll(card.groups, values), [card.groups, values]);
  const nReadings = all?.kind === "confirm_readings" ? all.items.length : 0;
  // The consumer's own ways forward: what is not a confirmation (a repair, a unit and its days).
  const own = card.exits.filter((e) => e.decision && !isConfirmation(e.decision));
  const answer = (at: string) => (answerAt?.key === at ? answerAt.node : null);

  return (
    <section className={g.ask} aria-labelledby={titleId} data-testid="ask-card">
      <div className={g.askHead}>
        <h3 id={titleId} className={g.askTitle}>
          {card.groups.length === 1 ? "Tell me about this column" : "Tell me about these columns"}
        </h3>
        <span className={g.askConsumer}>
          <Prose text={card.consumer} /> reads {card.groups.length === 1 ? "it" : "them"}
        </span>
      </div>
      <ol className={g.askList}>
        {card.groups.map((grp, i) => {
          const value = values[i] ?? grp.guess ?? "";
          const choices = choicesFor(grp);
          const decision = confirmLine(grp, value);
          const changed = values[i] !== undefined && values[i] !== grp.guess;
          const words = changed ? (choices.find((ch) => ch.value === value)?.label ?? value) : grp.guess_words;
          return (
            <li key={`${grp.kind}:${grp.columns[0]}`} className={g.askRow} data-testid="ask-line" data-kind={grp.kind}>
              <span className={g.askWhat}>
                <Columns columns={grp.columns} />: {words || "what is it?"}
                {changed ? " (changed)" : grp.guess ? "?" : ""}
                {grp.evidence ? (
                  <span className={g.askEvidence}>
                    <Prose text={grp.evidence} />
                  </span>
                ) : null}
              </span>
              <span className={g.askControls}>
                {choices.length > 1 ? (
                  <select
                    className={g.askSelect}
                    aria-label={`Change what ${grp.columns.length > 1 ? "these columns are" : `${grp.columns[0]} is`}`}
                    value={value}
                    onChange={(e) => setValues((cur) => ({ ...cur, [i]: e.target.value }))}
                    data-testid="ask-change"
                  >
                    {!value ? <option value="">choose…</option> : null}
                    {choices.map((ch) => (
                      <option key={ch.value} value={ch.value}>
                        {ch.label}
                      </option>
                    ))}
                  </select>
                ) : null}
                {decision ? (
                  <button
                    type="button"
                    className={c.small}
                    disabled={pending}
                    onClick={() => record(decision, `ask:${i}`)}
                    data-testid="ask-confirm"
                    title="Records exactly the readings on this line, each with the value it shows."
                  >
                    {grp.columns.length > 1 ? `Confirm the ${fmtCount(grp.columns.length)}` : "Confirm"}
                  </button>
                ) : null}
              </span>
              {answer(`ask:${i}`) ? <div className={c.answer}>{answer(`ask:${i}`)}</div> : null}
            </li>
          );
        })}
      </ol>
      {mastered && confirmable.length > 1 && all ? (
        <div className={g.askAll}>
          <button
            type="button"
            className={c.small}
            disabled={pending}
            onClick={() => record(all, "ask:all")}
            data-testid="ask-confirm-all"
            title="One block that lists every line above, each with the value it shows; nothing else."
          >
            Confirm every line as shown ({fmtCount(nReadings)})
          </button>
          {answer("ask:all")}
        </div>
      ) : null}
      {own.length ? (
        <div className={g.exits} data-testid="ask-exits">
          {own.map((e) => (
            <button
              key={e.label}
              type="button"
              className={c.small}
              disabled={pending}
              onClick={() => record(e.decision as unknown as Decision, "ask:exit")}
            >
              <Prose text={e.label} />
            </button>
          ))}
          {answer("ask:exit")}
        </div>
      ) : null}
      <ReadFromData
        items={card.read_from_data}
        pending={pending}
        record={(d) => record(d, "ask:read")}
        answer={answer("ask:read")}
      />
    </section>
  );
}

function Columns({ columns }: { columns: string[] }) {
  const [all, setAll] = useState(false);
  if (columns.length === 1) return <V>{columns[0]}</V>;
  if (all)
    return (
      <span className={g.chips}>
        {columns.map((col) => (
          <V key={col}>{col}</V>
        ))}
      </span>
    );
  return (
    <>
      {fmtCount(columns.length)} columns like <V>{columns[0]}</V>{" "}
      <button
        type="button"
        className={g.more}
        onClick={() => setAll(true)}
        aria-label={`List all ${columns.length} columns`}
      >
        list them
      </button>
    </>
  );
}

/** §14.3: the readings the values settled for this consumer, each with the answers that change it. */
export function ReadFromData({
  items,
  pending,
  record,
  answer,
}: {
  items: Card["read_from_data"];
  pending: boolean;
  record: (d: Decision) => void;
  answer: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  if (!items.length) return null;
  return (
    <div className={g.read} data-testid="read-from-data">
      <button
        type="button"
        className={g.readToggle}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        Read from your data · {fmtCount(items.length)}
      </button>
      {open ? (
        <ul className={g.readList}>
          {items.map((it) => (
            <li key={`${it.kind}:${it.column}`} className={g.readItem}>
              <span className={g.readWords}>
                <V>{it.column}</V> <Prose text={it.words} />
              </span>
              <span className={g.askEvidence}>
                <Prose text={it.evidence} />
              </span>
              {it.change.map((ex) =>
                ex.decision ? (
                  <button
                    key={ex.label}
                    type="button"
                    className={c.small}
                    disabled={pending}
                    onClick={() => record(ex.decision as unknown as Decision)}
                    data-testid="read-change"
                  >
                    <Prose text={ex.label} />
                  </button>
                ) : null,
              )}
            </li>
          ))}
        </ul>
      ) : null}
      {answer}
    </div>
  );
}
