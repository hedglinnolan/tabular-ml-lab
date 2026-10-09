/**
 * A slot opened in place (BLUEPRINT §11.4, §14.2): the question the paragraph asks, set in the
 * document where its sentence will stand, with the server's best guess and its evidence, the
 * options in the order the server ranks them, and an outcome-labeled control for each. The rest
 * of the document is dimmed to a map while one is open.
 *
 * Each slot's own control records the scenario's answer, and the document moves to the moment the
 * server reached after it (walk.ts). The prototype captured one path, so a choice off it can still
 * be made, read and (where the server drew it) previewed on the canvas, but its record control
 * says why it waits instead of recording.
 *
 * Every question, guess, piece of evidence, label and consequence is the server's string.
 */
import { Fragment, type KeyboardEvent, type ReactNode } from "react";
import type { RoleProposal, ShelfFamily, TeachingEntry } from "../../api/m1-types";
import type { DecisionRecord } from "../../api/schema";
import { Rich } from "../../components/stage/text";
import { cx } from "../../util/format";
import type { Derivation } from "./fixture";
import {
  isPreview,
  type AdjustmentCard,
  type AskCard,
  type AskExit,
  type CapturedPreview,
  type EnergyReading,
  type EstimandCard,
  type ModelSequenceCard,
  type QuestionLabels,
} from "./fixture";
import { MASTERY, readingLines, type ConceptTeaching } from "./model";
import s from "./doc.module.css";

function Kicker({ children }: { children: ReactNode }) {
  return <span className={s.slotKicker}>{children}</span>;
}

function Badge({ status, source }: { status: string; source: string }) {
  return (
    <span className={s.badge} data-status={status} title={source}>
      {status}
    </span>
  );
}

/** The one sentence a question needs, and a "why?" that opens in place (§11 rule 5). */
function Ask({ entry, why }: { entry: TeachingEntry | null; why: boolean }) {
  if (!entry) return null;
  return (
    <div className={s.slotAsk}>
      <p className={s.slotQuestion}>{entry.question}</p>
      <p className={s.slotLine}>
        <Rich text={entry.one_liner} />
      </p>
      {why ? (
        <details className={s.why}>
          <summary>Why?</summary>
          <p>
            <Rich text={entry.why} />
          </p>
        </details>
      ) : null}
    </div>
  );
}

/** A slot's record control: one outcome-labeled press, or the reason it waits. */
function Record({
  label,
  onRecord,
  hold,
  note,
  testid,
}: {
  label: string;
  onRecord: () => void;
  hold: string | null;
  note: string;
  testid: string;
}) {
  return (
    <div className={s.slotActions}>
      <button
        type="button"
        className={s.primary}
        onClick={onRecord}
        disabled={!!hold}
        title={hold ? hold.replace(/`/g, "") : undefined}
        data-testid={testid}
      >
        <Rich text={label} />
      </button>
      <span className={s.actionNote} data-hold={hold ? true : undefined}>
        <Rich text={hold ?? note} />
      </span>
    </div>
  );
}

// ── the readings: "Tell me about these columns", folded into the Data section ──

export function ReadingsSlot({
  ask,
  proposals,
  singles,
  block,
  focusRow,
  onFocusRow,
  readFromData,
  next,
  onConfirm,
  onBlock,
  hold,
}: {
  ask: AskCard;
  proposals: RoleProposal[];
  singles: DecisionRecord[];
  block: AskExit | null;
  focusRow: string | null;
  onFocusRow: (key: string) => void;
  readFromData: number;
  /** The column the walk confirms on its own next (the scenario's path), if any. */
  next: string | null;
  onConfirm: (column: string) => void;
  /** Record the card's first exit: the unlocked block, or a lone reading's one answer (a unit). */
  onBlock: () => void;
  /** Why this card's controls wait (another slot comes first in the walk), if they do. */
  hold: string | null;
}) {
  const head = ask.text.split(":")[0] ?? ask.text;
  const roleAsk = ask.groups.some((g) => g.kind === "role");
  const offPath = "This prototype walks one captured path: the readings it confirms one at a time, then the block.";
  const exitsFor = (column: string) =>
    ask.exits.slice(1).filter((x) => {
      const d = x.decision as { column?: string } | null;
      return d?.column === column;
    });
  const items = block ? ((block.decision as { items: { column: string; value: string }[] }).items ?? []) : [];
  // A card with one reading and one answer (the screens' unit question) is answered in one press.
  const lone = !block && !roleAsk && ask.exits.length === 1 ? ask.exits[0]! : null;
  return (
    <div className={s.slot} data-slot="readings">
      <div className={s.slotAsk}>
        <p className={s.slotQuestion}>{head}</p>
        <p className={s.slotLine}>
          Their answers feed {ask.consumer}; each is confirmed with the value it shows.
        </p>
      </div>
      {block ? (
        <section className={s.unlock} aria-label="Confirm together" data-testid="block-unlock">
          <div className={s.unlockHead}>
            <Kicker>Unlocked</Kicker>
            <span className={s.unlockWhy}>after {singles.length} readings confirmed one at a time</span>
          </div>
          <p className={s.unlockLead}>
            The rest can now be confirmed together. The block settles exactly these {items.length}, each with the value
            listed, and nothing else.
          </p>
          <ul className={s.unlockList}>
            {items.map((i) => (
              <li key={i.column}>
                <code className="v">{i.column}</code>
                <span className={s.unlockValue}>{i.value}</span>
              </li>
            ))}
          </ul>
          <Record
            label={block.label}
            onRecord={onBlock}
            hold={hold}
            note="or keep confirming one at a time below"
            testid="block-confirm"
          />
        </section>
      ) : !roleAsk ? null : (
        <p className={s.mastery} data-testid="mastery-progress">
          <span className={s.masteryDots} aria-hidden="true">
            {Array.from({ length: MASTERY }, (_, i) => (
              <span key={i} data-on={i < singles.length || undefined} />
            ))}
          </span>
          <span>
            Tap a column to confirm it on its own. Confirmed so far: {Math.min(singles.length, MASTERY)} of {MASTERY};
            after {MASTERY}, the rest can be confirmed together, each listed.
          </span>
        </p>
      )}
      <ol className={s.readings}>
        {readingLines(ask, proposals).map((line) => {
          const confirmOf = (c: string) =>
            exitsFor(c).find((x) => (x.decision as { value?: string } | null)?.value === line.guess) ?? null;
          const others = line.columns.reduce((n, c) => n + exitsFor(c).length - (confirmOf(c) ? 1 : 0), 0);
          return (
            <li
              key={line.key}
              className={s.reading}
              data-focus={focusRow === line.key || undefined}
              data-family={line.family || undefined}
              onPointerEnter={() => onFocusRow(line.key)}
            >
              <div className={s.readingClaim}>
                {line.family ? (
                  <>
                    <span className={s.familyCount}>a family of {line.columns.length}</span>
                    {line.columns.map((c) => (
                      <code key={c} className="v">
                        {c}
                      </code>
                    ))}
                  </>
                ) : (
                  line.columns.map((c) => {
                    const x = confirmOf(c);
                    if (!x)
                      return (
                        <code key={c} className="v">
                          {c}
                        </code>
                      );
                    const live = c === next && !hold;
                    return (
                      <button
                        key={c}
                        type="button"
                        className={s.confirmChip}
                        aria-label={x.label.replace(/`/g, "")}
                        title={live ? x.label.replace(/`/g, "") : (hold ?? offPath)}
                        disabled={!live}
                        data-next={live || undefined}
                        data-testid={`confirm-${c}`}
                        onClick={() => onConfirm(c)}
                      >
                        {c}
                      </button>
                    );
                  })
                )}
                <span className={s.readingGuess}>{line.guessWords}?</span>
              </div>
              <p className={s.readingEvidence}>
                <Rich text={line.evidence} />
                {!line.family && others ? <span className={s.otherCount}> · other answers on each column</span> : null}
              </p>
              {line.family ? (
                <div className={s.readingActions}>
                  <button type="button" className={s.confirm} disabled title={offPath}>
                    Confirm these {line.columns.length} as {line.guess}s
                  </button>
                  <span className={s.otherCount}>a family: it settles exactly the {line.columns.length} listed</span>
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>
      {lone ? (
        <Record
          label={lone.label}
          onRecord={onBlock}
          hold={hold}
          note={`It records this one reading; ${ask.consumer} read it next.`}
          testid="unit-confirm"
        />
      ) : null}
      {readFromData ? (
        <p className={s.readFold}>
          {readFromData} more were settled by their values, no question asked (listed above the card, each with its
          evidence).
        </p>
      ) : null}
    </div>
  );
}

// ── the roles: the draft's first open slot ────────────────────────────────────

export function RolesSlot({
  entry,
  proposals,
  onRecord,
  hold,
}: {
  entry: TeachingEntry | null;
  proposals: RoleProposal[];
  onRecord: () => void;
  hold: string | null;
}) {
  const order = entry?.options ?? [];
  const groups = order
    .map((o) => ({ role: o.value, label: o.label, what: o.consequence, cols: proposals.filter((p) => p.proposed === o.value) }))
    .filter((g) => g.cols.length);
  const unsure = proposals.filter((p) => p.confidence !== "high").length;
  return (
    <div className={s.slot} data-slot="roles">
      <Ask entry={entry} why />
      <p className={s.slotCount}>
        The proposal for each of the {proposals.length} columns, from its name and its values; hover a column for its
        reason. {unsure} were proposed below high confidence: each comes back as a reading to confirm.
      </p>
      <ul className={s.roleList}>
        {groups.map((g) => (
          <li key={g.role} className={s.roleLine}>
            <span className={s.roleName}>
              {g.label} <span className={s.roleCount}>{g.cols.length}</span>
            </span>
            <span className={s.roleCols}>
              {g.cols.map((p) => (
                <code
                  key={p.column}
                  className="v"
                  title={p.reason}
                  data-unsure={p.confidence !== "high" || undefined}
                >
                  {p.column}
                </code>
              ))}
            </span>
          </li>
        ))}
      </ul>
      <Record
        label="Record these roles"
        onRecord={onRecord}
        hold={hold}
        note="Nothing is recorded until this is pressed."
        testid="record-roles"
      />
    </div>
  );
}

// ── a question's labeled alternatives (rows, missing data, the energy model) ──

export interface AltOption {
  key: string;
  label: string;
  verdict: string | null;
  verdictFor: string | null;
  what: string | null;
  refused: string | null;
  customary: { field: string; text: string } | null;
  sound: string | null;
  /** A short tag beside the label (in the draft, ranked first). */
  tag: string | null;
  /** A control beside the option (the rows: report a screen beside the primary). */
  aside?: ReactNode;
}

/** The labeled options of LabeledQuestion (proposals.labels), with the teaching's consequences. */
export function labeledOptions(
  q: QuestionLabels["energy_adjustment"],
  entry: TeachingEntry | null,
  previews: Record<string, CapturedPreview>,
): AltOption[] {
  return (q?.options ?? []).map((o) => {
    const p = previews[o.key];
    const refused = p && !isPreview(p.body) ? p.body.error.message : null;
    return {
      key: o.key,
      label: o.label,
      verdict: o.sound.verdict,
      verdictFor: o.sound.purpose,
      what: entry?.options.find((x) => x.value === o.key)?.consequence ?? null,
      refused,
      customary: o.customary,
      sound: o.sound.reason,
      tag: null,
    };
  });
}

export function Alternatives({
  kicker,
  line,
  options,
  hover,
  onHover,
  chosen,
  onChoose,
  current,
  hint,
  label,
  testid,
  children,
}: {
  kicker: string;
  line: string | null;
  options: AltOption[];
  hover: string | null;
  onHover: (key: string) => void;
  /** The option chosen and not yet recorded (a slot), or none (a recorded phrase, opened). */
  chosen?: string | null;
  onChoose?: (key: string) => void;
  /** The option the record holds now. */
  current?: string;
  hint: ReactNode;
  label: string;
  testid: string;
  children?: ReactNode;
}) {
  const choosable = !!onChoose;
  // What an option opens to (its whole consequence, customary and sound): the chosen one when the
  // list is a choice, so hovering (which plays the canvas) never moves a row under the pointer.
  const opened = (key: string) => (choosable ? chosen === key : hover === key);
  // ↑ ↓ move between the options; focusing one plays it on the canvas, as hovering does.
  const onKey = (e: KeyboardEvent<HTMLOListElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const picks = [...e.currentTarget.querySelectorAll<HTMLButtonElement>("button[data-alt]")];
    const at = picks.findIndex((b) => b === document.activeElement);
    const to = e.key === "ArrowDown" ? Math.min(picks.length - 1, at + 1) : Math.max(0, at - 1);
    picks[to]?.focus();
    e.preventDefault();
  };
  return (
    <div className={s.alts} data-testid={testid}>
      <div className={s.altsHead}>
        <Kicker>{kicker}</Kicker>
        {line ? (
          <p className={s.altsLine}>
            <Rich text={line} />
          </p>
        ) : null}
      </div>
      <ol className={s.altList} role={choosable ? "radiogroup" : undefined} aria-label={label} onKeyDown={onKey}>
        {options.map((o) => (
          <li
            key={o.key}
            className={s.alt}
            data-current={o.key === current || undefined}
            data-hover={hover === o.key || undefined}
            data-chosen={(choosable && chosen === o.key) || undefined}
            data-refused={o.refused ? true : undefined}
            onPointerEnter={() => onHover(o.key)}
          >
            <button
              type="button"
              className={s.altPick}
              role={choosable ? "radio" : undefined}
              aria-checked={choosable ? chosen === o.key : undefined}
              onFocus={() => onHover(o.key)}
              onClick={() => (onChoose ? onChoose(o.key) : onHover(o.key))}
              data-testid={`alt-${o.key}`}
              data-alt
            >
              <span className={s.altTop}>
                <span className={s.altLabel}>{o.label}</span>
                {o.key === current ? <span className={s.inDraft}>in the draft</span> : null}
                {o.tag ? <span className={s.guess}>{o.tag}</span> : null}
                {o.verdict ? (
                  <span className={s.verdict} data-verdict={o.verdict}>
                    {o.verdict} for {o.verdictFor}
                  </span>
                ) : null}
              </span>
              <span className={s.altWhat} data-clamp={!opened(o.key) || undefined}>
                {o.refused ? <Rich text={`Not available: ${o.refused}`} /> : (o.what ?? "")}
              </span>
              {opened(o.key) && !o.refused && (o.customary || o.sound) ? (
                <span className={s.altLabels}>
                  {o.customary ? (
                    <>
                      <span>Customary in {o.customary.field}:</span> <Rich text={o.customary.text} />{" "}
                    </>
                  ) : null}
                  {o.sound ? (
                    <>
                      <span>Sound:</span> <Rich text={o.sound} />
                    </>
                  ) : null}
                </span>
              ) : null}
            </button>
            {o.aside ? <div className={s.altAside}>{o.aside}</div> : null}
          </li>
        ))}
      </ol>
      {children ? <div className={s.altsFoot}>{children}</div> : null}
      <p className={s.altKeys}>{hint}</p>
    </div>
  );
}

const PLAYS = "Hover an option: the canvas plays what it does to your data. Nothing is recorded.";
const READS = "Choose one to read what it does; only recording it changes the record.";

/** The rows: who the study is about, and the screens reported beside the primary. */
export function ExclusionsSlot({
  entry,
  labels,
  previews,
  affected,
  chosen,
  beside,
  hover,
  onHover,
  onChoose,
  onBeside,
  onRecord,
  hold,
}: {
  entry: TeachingEntry | null;
  labels: QuestionLabels["exclusions"];
  previews: Record<string, CapturedPreview>;
  /** Rows each screen removes (the proposals' own count). */
  affected: Record<string, number>;
  chosen: string | null;
  beside: string[];
  hover: string | null;
  onHover: (key: string) => void;
  onChoose: (key: string) => void;
  onBeside: (key: string) => void;
  onRecord: () => void;
  hold: string | null;
}) {
  const plays = Object.keys(previews).length > 0;
  const options = labeledOptions(labels, entry, previews).map((o) => {
    const screen = o.key !== "keep_every_row" && !o.refused;
    const on = beside.includes(o.key);
    return {
      ...o,
      what: o.what ?? (affected[o.key] ? `Removes ${affected[o.key]!.toLocaleString("en-US")} rows.` : null),
      tag: affected[o.key] ? `− ${affected[o.key]!.toLocaleString("en-US")} rows` : null,
      aside: screen ? (
        <button
          type="button"
          className={s.beside}
          aria-pressed={on}
          onClick={() => onBeside(o.key)}
          data-testid={`beside-${o.key}`}
        >
          {on ? "Reported beside the primary" : "Report beside the primary"}
        </button>
      ) : null,
    };
  });
  const primary = options.find((o) => o.key === chosen);
  const label = primary
    ? `${primary.label}${beside.length ? `, with ${beside.length} ${beside.length === 1 ? "screen" : "screens"} beside it` : ""}`
    : "Choose the primary rows";
  return (
    <div className={s.slot} data-slot="exclusions">
      <Ask entry={entry} why={false} />
      <Alternatives
        kicker="The primary rows, and any screen beside them"
        line={labels?.tension ?? labels?.customary_first ?? null}
        options={options}
        hover={hover}
        onHover={onHover}
        chosen={chosen}
        onChoose={onChoose}
        hint={plays ? PLAYS : READS}
        label="Exclusions: the alternatives"
        testid="alternatives-exclusions"
      >
        <Record
          label={label}
          onRecord={onRecord}
          hold={hold}
          note="A screen beside the primary is the same model on its own rows."
          testid="record-exclusions"
        />
      </Alternatives>
    </div>
  );
}

/** A labeled question opened as a slot (missing data, the energy model): choose, then record. */
export function LabeledSlot({
  entry,
  labels,
  previews,
  ranking,
  chosen,
  hover,
  onHover,
  onChoose,
  onRecord,
  hold,
  slot,
}: {
  entry: TeachingEntry | null;
  labels: QuestionLabels["energy_adjustment"];
  previews: Record<string, CapturedPreview>;
  ranking: EnergyReading["ranking"] | null;
  chosen: string | null;
  hover: string | null;
  onHover: (key: string) => void;
  onChoose: (key: string) => void;
  onRecord: () => void;
  hold: string | null;
  slot: string;
}) {
  const plays = Object.keys(previews).length > 0;
  const first = ranking?.order[0] ?? null;
  const options = labeledOptions(labels, entry, previews).map((o) => ({
    ...o,
    tag: o.key === first ? "ranked first" : null,
  }));
  const pick = options.find((o) => o.key === chosen);
  return (
    <div className={s.slot} data-slot={slot}>
      <Ask entry={entry} why={false} />
      <Alternatives
        kicker={entry?.title ?? "The alternatives"}
        line={ranking?.line ?? labels?.tension ?? labels?.customary_first ?? null}
        options={options}
        hover={hover}
        onHover={onHover}
        chosen={chosen}
        onChoose={onChoose}
        hint={plays ? PLAYS : READS}
        label={`${entry?.title ?? slot}: the alternatives`}
        testid={`alternatives-${slot}`}
      >
        <Record
          label={pick ? `Record: ${pick.label}` : "Choose one to record"}
          onRecord={onRecord}
          hold={hold}
          note="Its sentence joins the methods where this slot stands."
          testid={`record-${slot}`}
        />
      </Alternatives>
    </div>
  );
}

/** A plain choice (the split, the model families): the options with their consequences. */
export interface Choice {
  key: string;
  label: string;
  what: string;
  tag: string | null;
}

export function ChoiceSlot({
  entry,
  options,
  chosen,
  multi,
  onChoose,
  onRecord,
  hold,
  slot,
  recordLabel,
}: {
  entry: TeachingEntry | null;
  options: Choice[];
  chosen: string[];
  multi: boolean;
  onChoose: (key: string) => void;
  onRecord: () => void;
  hold: string | null;
  slot: string;
  recordLabel: string;
}) {
  return (
    <div className={s.slot} data-slot={slot}>
      <Ask entry={entry} why />
      <ul className={s.options} role={multi ? "group" : "radiogroup"} aria-label={entry?.title ?? slot}>
        {options.map((o) => {
          const on = chosen.includes(o.key);
          return (
            <li key={o.key}>
              <button
                type="button"
                className={cx(s.option, s.pick)}
                role={multi ? "checkbox" : "radio"}
                aria-checked={on}
                data-on={on || undefined}
                onClick={() => onChoose(o.key)}
                data-testid={`choice-${o.key}`}
              >
                <span className={s.optionLabel}>
                  {o.label}
                  {o.tag ? <span className={s.guess}>{o.tag}</span> : null}
                </span>
                <span className={s.optionWhat}>
                  <Rich text={o.what} />
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <Record
        label={recordLabel}
        onRecord={onRecord}
        hold={hold}
        note="Its sentence joins the methods where this slot stands."
        testid={`record-${slot}`}
      />
    </div>
  );
}

/** The model families on the shelf, in the server's rank, as choices. */
export function shelfChoices(families: ShelfFamily[]): Choice[] {
  return [...families]
    .sort((a, b) => a.rank - b.rank)
    .map((f) => ({
      key: f.key,
      label: f.label,
      what: [f.inductive_bias, ...f.concerns].join(" "),
      tag: `${f.fit} fit`,
    }));
}

// ── the exposure and its effect ──────────────────────────────────────────────

export interface EstimandChoice {
  exposure: string;
  effect: string;
  contrast: string;
}

export function EstimandSlot({
  card,
  entry,
  choice,
  concept,
  onChoose,
  onPeek,
  onRecord,
  hold,
}: {
  card: EstimandCard;
  entry: TeachingEntry | null;
  choice: EstimandChoice;
  concept: ConceptTeaching | null;
  onChoose: (patch: Partial<EstimandChoice>) => void;
  /** An exposure under the pointer: the canvas shows it without choosing it. */
  onPeek: (column: string | null) => void;
  onRecord: () => void;
  /** Why recording waits, if it does (the readings first; or a choice the walk did not capture). */
  hold: string | null;
}) {
  const energy = card.exposures.filter((e) => e.energy_contrast);
  const others = card.exposures.filter((e) => !e.energy_contrast);
  const chosen = card.exposures.find((e) => e.column === choice.exposure);
  return (
    <div className={s.slot} data-slot="estimand">
      <Ask entry={entry} why={false} />
      <fieldset className={s.choiceSet}>
        <legend>Exposure</legend>
        <div className={s.chips} role="radiogroup" aria-label="Exposure">
          {energy.map((e) => (
            <button
              key={e.column}
              type="button"
              role="radio"
              aria-checked={e.column === choice.exposure}
              className={s.chip}
              onPointerEnter={() => onPeek(e.column)}
              onPointerLeave={() => onPeek(null)}
              onClick={() => onChoose({ exposure: e.column })}
              data-testid={`exposure-${e.column}`}
            >
              {e.column}
            </button>
          ))}
          {card.family ? (
            <button type="button" role="radio" aria-checked={false} className={cx(s.chip, s.chipWide)} disabled>
              all {card.family.n}, one by one
            </button>
          ) : null}
        </div>
        <details className={s.choiceNote}>
          <summary>
            The exposures from your roles, energy-bearing first; {others.length} other predictors can be the exposure too
          </summary>
          {others.map((e, i) => (
            <Fragment key={e.column}>
              {i ? ", " : ""}
              <code className="v">{e.column}</code>
            </Fragment>
          ))}
        </details>
      </fieldset>
      <fieldset className={s.choiceSet}>
        <legend>Effect</legend>
        <ul className={s.options} role="radiogroup" aria-label="Effect">
          {card.effects.map((e) => (
            <li key={e.effect}>
              <button
                type="button"
                role="radio"
                aria-checked={e.effect === choice.effect}
                className={cx(s.option, s.pick)}
                data-on={e.effect === choice.effect || undefined}
                onClick={() => onChoose({ effect: e.effect ?? "" })}
                data-testid={`effect-${e.effect}`}
              >
                <span className={s.optionLabel}>{e.label}</span>
                <span className={s.optionWhat}>{e.consequence}</span>
              </button>
            </li>
          ))}
        </ul>
      </fieldset>
      {chosen?.energy_contrast ? (
        <fieldset className={s.choiceSet} id="contrast">
          <legend>Which energy question</legend>
          <ul className={s.options} role="radiogroup" aria-label="Which energy question">
            {card.contrasts.map((c) => (
              <li key={c.contrast}>
                <button
                  type="button"
                  role="radio"
                  aria-checked={c.contrast === choice.contrast}
                  className={cx(s.option, s.pick)}
                  data-on={c.contrast === choice.contrast || undefined}
                  onClick={() => onChoose({ contrast: c.contrast ?? "" })}
                  data-testid={`contrast-${c.contrast}`}
                >
                  <span className={s.optionLabel}>{c.label}</span>
                  <span className={s.optionWhat}>{c.consequence}</span>
                </button>
              </li>
            ))}
          </ul>
          {concept ? <ConceptFull concept={concept} /> : null}
        </fieldset>
      ) : null}
      <fieldset className={s.choiceSet}>
        <legend>Measure</legend>
        <ul className={s.options}>
          {card.measures.map((m) => (
            <li key={m.measure} className={s.option} data-on={m.rank === 1 || undefined} data-off={!m.fitted || undefined}>
              <span className={s.optionLabel}>
                {m.label}
                {m.rank === 1 ? <span className={s.guess}>fitted here</span> : null}
              </span>
              <span className={s.optionWhat}>{m.reason}</span>
            </li>
          ))}
        </ul>
      </fieldset>
      <Record
        label="Record this estimand"
        onRecord={onRecord}
        hold={hold}
        note="No estimate is shown until it is recorded."
        testid="record-estimand"
      />
    </div>
  );
}

/** A pack citation as the engine's other reasons print it ("NUTRITION_PACK §04 · …"), not as a path. */
function packSource(source: string): string {
  return source.replace(/^research\/([A-Z_]+)\.md#(\w+)/, "$1 §$2");
}

/** A concept's first encounter: taught in full where it is first used (§11.4 rule 2). */
export function ConceptFull({ concept }: { concept: ConceptTeaching }) {
  return (
    <aside className={s.concept} data-testid="concept-full" aria-label={`New here: ${concept.term}`}>
      <div className={s.conceptHead}>
        <Kicker>New here</Kicker>
        <span className={s.conceptTitle}>{concept.heading}</span>
        {concept.evidence ? <Badge status={concept.evidence.status} source={concept.evidence.source} /> : null}
      </div>
      <p className={s.conceptBody}>
        <Rich text={concept.body} />
      </p>
      {concept.evidence ? <p className={s.conceptSource}>{packSource(concept.evidence.source)}</p> : null}
    </aside>
  );
}

// ── the adjustment set ──────────────────────────────────────────────────────

const ANSWER: Record<string, string> = { yes: "yes", no: "no", unknown: "don't know" };
const SHORT: Record<string, string> = { yes: "yes", no: "no", unknown: "?" };
export const ADJ_FIELDS = ["causes_exposure", "causes_outcome", "after_exposure"] as const;
export type AdjField = (typeof ADJ_FIELDS)[number];
export type AdjAnswers = Record<string, Partial<Record<AdjField, string>>>;

export function AdjustmentSlot({
  card,
  entry,
  focusGroup,
  onFocusGroup,
  concept,
  confirmed,
  onConfirmGroup,
  answers,
  onAnswer,
  derive,
  onRecord,
  hold,
}: {
  card: AdjustmentCard;
  entry: TeachingEntry | null;
  focusGroup: string | null;
  onFocusGroup: (key: string) => void;
  concept: ConceptTeaching | null;
  /** The guessed groups confirmed so far (nothing is recorded until the set is). */
  confirmed: string[];
  onConfirmGroup: (key: string) => void;
  answers: AdjAnswers;
  onAnswer: (column: string, field: AdjField, value: string) => void;
  /** The criterion's verdict for a column's three answers (estimand.derive, captured). */
  derive: (column: string) => Derivation | null;
  onRecord: () => void;
  hold: string | null;
}) {
  const q = card.questions as Record<string, string>;
  const total = card.groups.reduce((n, g) => n + g.columns.length, 0);
  const guessed = card.groups.filter((g) => g.guess);
  return (
    <div className={s.slot} data-slot="adjustment">
      <Ask entry={entry} why={false} />
      <p className={s.slotCount}>
        {total} covariates. {guessed.length} groups carry the pack&rsquo;s guess, confirmed in one tap each;{" "}
        {total - guessed.reduce((n, g) => n + g.columns.length, 0)} columns have none and are asked one by one.
      </p>
      <ol className={s.adjLegend}>
        {ADJ_FIELDS.map((f) => (
          <li key={f}>{q[f]}</li>
        ))}
      </ol>
      <div className={s.adjWrap}>
        <table className={s.adj}>
          <thead>
            <tr>
              <th scope="col">Covariates</th>
              {ADJ_FIELDS.map((f, i) => (
                <th key={f} scope="col" className={s.adjQ} title={q[f]}>
                  {i + 1}
                </th>
              ))}
              <th scope="col">Role</th>
            </tr>
          </thead>
          {card.groups.map((g) => {
            const guess = g.guess as Record<string, string> | null;
            const { text, source } = splitSource(g.reason);
            const on = confirmed.includes(g.key);
            const why = (
              <tr className={s.adjWhyRow}>
                <td colSpan={5}>
                  <span className={s.groupWhy}>
                    <Rich text={firstUpper(text)} />
                    {source ? <span className={s.source}> {source}</span> : null}
                  </span>
                  {guess ? (
                    <button
                      type="button"
                      className={s.confirm}
                      aria-pressed={on}
                      data-on={on || undefined}
                      onClick={() => onConfirmGroup(g.key)}
                      data-testid={`adj-confirm-${g.key}`}
                    >
                      {on ? "Confirmed: " : "Confirm "}
                      {g.columns.length === 2 ? "both" : `all ${g.columns.length}`} as{" "}
                      {g.derived_words?.endsWith("unknown") ? `of ${g.derived_words}` : `${g.derived_words}s`}
                    </button>
                  ) : null}
                </td>
              </tr>
            );
            if (!guess)
              return (
                <tbody
                  key={g.key}
                  className={s.adjGroup}
                  data-focus={focusGroup === g.key || undefined}
                  data-unguessed
                  onPointerEnter={() => onFocusGroup(g.key)}
                >
                  <tr>
                    <th scope="rowgroup" colSpan={5} className={s.adjName}>
                      <span className={s.groupLabel}>{g.label}: answer each</span>
                    </th>
                  </tr>
                  {g.columns.map((c) => {
                    const a = answers[c] ?? {};
                    const d = derive(c);
                    return (
                      <tr key={c} className={s.adjAsk}>
                        <th scope="row" className={s.adjName}>
                          <code className="v">{c}</code>
                        </th>
                        {ADJ_FIELDS.map((f, i) => (
                          <td key={f} className={s.adjPickCell}>
                            <span className={s.yn} role="radiogroup" aria-label={`${c}: ${q[f]}`}>
                              {(["yes", "no", "unknown"] as const).map((v) => (
                                <button
                                  key={v}
                                  type="button"
                                  role="radio"
                                  aria-checked={a[f] === v}
                                  aria-label={ANSWER[v]}
                                  title={`${q[f]} ${ANSWER[v]}`}
                                  data-on={a[f] === v || undefined}
                                  onClick={() => onAnswer(c, f, v)}
                                  data-testid={`adj-${c}-${i + 1}-${v}`}
                                >
                                  {SHORT[v]}
                                </button>
                              ))}
                            </span>
                          </td>
                        ))}
                        <td className={s.adjRole}>
                          {d ? (
                            <span className={s.derived} title={d.why}>
                              {d.words}
                            </span>
                          ) : (
                            <span className={s.derivedAsk}>asked</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                  {why}
                </tbody>
              );
            return (
              <tbody
                key={g.key}
                className={s.adjGroup}
                data-focus={focusGroup === g.key || undefined}
                onPointerEnter={() => onFocusGroup(g.key)}
              >
                <tr>
                  <th scope="rowgroup" className={s.adjName}>
                    <span className={s.groupLabel}>{g.label}</span>
                    <span className={s.groupCols}>
                      {g.columns.map((c) => (
                        <code key={c} className="v">
                          {c}
                        </code>
                      ))}
                    </span>
                  </th>
                  {ADJ_FIELDS.map((f) => (
                    <td key={f} className={s.adjA}>
                      {ANSWER[guess[f] ?? ""] ?? "—"}
                    </td>
                  ))}
                  <td className={s.adjRole}>
                    <span className={s.derived} data-on={on || undefined}>
                      {g.derived_words}
                    </span>
                  </td>
                </tr>
                {why}
              </tbody>
            );
          })}
        </table>
      </div>
      {concept ? <ConceptFull concept={concept} /> : null}
      <Record
        label="Record the adjustment set"
        onRecord={onRecord}
        hold={hold}
        note="Each group and each derived role joins the record as its own sentence."
        testid="record-adjustment"
      />
    </div>
  );
}

/** A pack reason ends with its citation, "(NUTRITION_PACK §08 (…))": set apart as the source line,
 *  in the data voice, so the reason reads as a sentence. */
function splitSource(reason: string): { text: string; source: string | null } {
  const at = reason.indexOf(" (NUTRITION_PACK");
  if (at < 0) return { text: reason, source: null };
  const source = reason.slice(at + 2).replace(/\)+\.?$/, "").replace(/ \(the nested model table.*$/, "");
  return { text: `${reason.slice(0, at).replace(/\.$/, "")}.`, source };
}

function firstUpper(t: string): string {
  return t.charAt(0).toUpperCase() + t.slice(1);
}

// ── the declared model sequence: Model 1 ────────────────────────────────────

export function ModelSequenceSlot({
  card,
  picked,
  onToggle,
  onGuess,
  onRecord,
  hold,
}: {
  card: ModelSequenceCard;
  picked: string[];
  onToggle: (column: string) => void;
  onGuess: () => void;
  onRecord: () => void;
  hold: string | null;
}) {
  const { text, source } = splitSource(card.reason);
  return (
    <div className={s.slot} data-slot="model_sequence">
      <div className={s.slotAsk}>
        <p className={s.slotQuestion}>Which columns does Model 1 adjust for?</p>
        <p className={s.slotLine}>
          <Rich text={text} />
          {source ? <span className={s.source}> {source}</span> : null}
        </p>
      </div>
      <div className={s.chips} role="group" aria-label="Model 1 adjusts for">
        {card.allowed.map((c) => {
          const on = picked.includes(c);
          const guessed = card.guess.includes(c);
          return (
            <button
              key={c}
              type="button"
              className={s.chip}
              aria-pressed={on}
              data-guess={guessed || undefined}
              onClick={() => onToggle(c)}
              data-testid={`model1-${c}`}
              title={guessed ? "the pack's guess" : undefined}
            >
              {c}
            </button>
          );
        })}
      </div>
      <p className={s.seqGuess}>
        <button type="button" className={s.confirm} onClick={onGuess} data-testid="model1-guess">
          Take the guess: {card.guess.join(", ")}
        </button>
        <span className={s.otherCount}>dotted: the pack&rsquo;s guess, never pre-selected</span>
      </p>
      <Record
        label={picked.length ? `Declare Model 1: ${picked.join(", ")}` : "Choose Model 1's columns"}
        onRecord={onRecord}
        hold={hold}
        note="Declared before any estimate is shown, as the sequence's first model."
        testid="record-sequence"
      />
    </div>
  );
}

// ── the lock: showing the estimates ─────────────────────────────────────────

export function LockSlot({ sentence, onLock, hold }: { sentence: string | null; onLock: () => void; hold: string | null }) {
  return (
    <div className={s.slot} data-slot="lock">
      <div className={s.slotAsk}>
        <p className={s.slotQuestion}>Show the estimates?</p>
        <p className={s.slotLine}>No estimate has been shown yet. Showing them locks the plan; the record will read:</p>
      </div>
      {sentence ? (
        <blockquote className={s.lockQuote}>
          <Rich text={sentence} />
        </blockquote>
      ) : null}
      <Record
        label="Show the estimates"
        onRecord={onLock}
        hold={hold}
        note="Table 2 and the declared alternatives follow."
        testid="lock-plan"
      />
    </div>
  );
}

// ── a stated phrase, opened: the alternatives (M5) ───────────────────────────

export function EnergyAlternatives({
  labels,
  energy,
  entry,
  previews,
  current,
  hover,
  onHover,
  keys,
  onKeep,
}: {
  labels: QuestionLabels["energy_adjustment"];
  energy: EnergyReading | null;
  entry: TeachingEntry | null;
  previews: Record<string, CapturedPreview>;
  current: string;
  hover: string | null;
  onHover: (key: string) => void;
  /** Keyboard movement is offered once the user has moved by pointer (§11.4 rule 4). */
  keys: boolean;
  onKeep: () => void;
}) {
  if (!labels) return null;
  const options = labeledOptions(labels, entry, previews);
  const currentLabel = options.find((o) => o.key === current)?.label ?? current;
  return (
    <Alternatives
      kicker="Change the energy model"
      line={energy?.ranking?.line ?? null}
      options={options}
      hover={hover}
      onHover={onHover}
      current={current}
      label="Energy adjustment: the alternatives"
      testid="alternatives"
      hint={
        keys ? (
          <>
            <kbd>↑</kbd> <kbd>↓</kbd> rifle through them · <kbd>Space</kbd> flips the canvas · <kbd>Esc</kbd> keeps the
            draft
          </>
        ) : (
          PLAYS
        )
      }
    >
      <div className={s.slotActions}>
        <button type="button" className={s.confirm} onClick={onKeep} data-testid="keep-energy">
          Keep the {currentLabel.toLowerCase()}
        </button>
        <span className={s.actionNote}>
          This prototype captured one path: recording another model is not on it, so each plays here and none is recorded.
        </span>
      </div>
    </Alternatives>
  );
}

/** Any other recorded phrase, opened: what else it could say, from the teaching. */
export function PhraseOptions({ entry, onKeep }: { entry: TeachingEntry | null; onKeep: () => void }) {
  if (!entry) return null;
  return (
    <div className={s.alts} data-testid="phrase-options">
      <div className={s.altsHead}>
        <Kicker>{entry.title}: what else it could say</Kicker>
        <p className={s.altsLine}>
          <Rich text={entry.one_liner} />
        </p>
      </div>
      <ol className={s.altList}>
        {entry.options.map((o) => (
          <li key={o.value} className={s.alt} data-static>
            <span className={s.altTop}>
              <span className={s.altLabel}>{o.label}</span>
            </span>
            <p className={s.altWhat}>{o.consequence}</p>
          </li>
        ))}
      </ol>
      <div className={s.altsFoot}>
        <div className={s.slotActions}>
          <button type="button" className={s.confirm} onClick={onKeep}>
            Keep it as written
          </button>
          <span className={s.actionNote}>
            Changing it records a new decision; this prototype captured one path, and plays alternatives for the energy
            model only.
          </span>
        </div>
      </div>
    </div>
  );
}

/** A concept met again: condensed to one phrase, expandable in place (§11.4 rule 2). */
export function ConceptCondensed({ concept, open }: { concept: ConceptTeaching; open: boolean }) {
  if (!open) return null;
  return (
    <span className={s.conceptLine} role="note" data-testid="concept-condensed">
      <span className={s.conceptTerm}>{concept.term}</span>
      <Rich text={concept.condensed} />
      <span className={s.conceptBack}>Taught at {concept.taughtAt === "estimand" ? "the exposure and its effect" : "the adjustment set"} ↑ · read in full</span>
    </span>
  );
}

/** A question slot with no bespoke panel: the teaching's question and options. */
export function QuestionSlot({ entry, hold }: { entry: TeachingEntry | null; hold: string | null }) {
  if (!entry) return null;
  return (
    <div className={s.slot}>
      <Ask entry={entry} why />
      <ul className={s.options}>
        {entry.options.map((o) => (
          <li key={o.value} className={s.option}>
            <span className={s.optionLabel}>{o.label}</span>
            <span className={s.optionWhat}>{o.consequence}</span>
          </li>
        ))}
      </ul>
      {hold ? <p className={s.actionNote}>{hold}</p> : null}
    </div>
  );
}
