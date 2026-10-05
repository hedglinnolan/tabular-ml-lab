/**
 * A slot opened in place (BLUEPRINT §11.4, §14.2): the question the paragraph asks, set in the
 * document where its sentence will stand, with the server's best guess and its evidence, the
 * options in the order the server ranks them, and an outcome-labeled control for each. The rest
 * of the document is dimmed to a map while one is open.
 *
 * Every question, guess, piece of evidence, label and consequence is the server's string.
 */
import { Fragment, type ReactNode } from "react";
import type { RoleProposal, TeachingEntry } from "../../api/m1-types";
import type { DecisionRecord } from "../../api/schema";
import { Rich } from "../../components/stage/text";
import { cx } from "../../util/format";
import {
  isPreview,
  type AdjustmentCard,
  type AskCard,
  type AskExit,
  type CapturedPreview,
  type EnergyReading,
  type EstimandCard,
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

// ── the readings: "Tell me about these columns", folded into the Data section ──

export function ReadingsSlot({
  ask,
  proposals,
  singles,
  block,
  focusRow,
  onFocusRow,
  readFromData,
}: {
  ask: AskCard;
  proposals: RoleProposal[];
  singles: DecisionRecord[];
  block: AskExit | null;
  focusRow: string | null;
  onFocusRow: (key: string) => void;
  readFromData: number;
}) {
  const head = ask.text.split(":")[0] ?? ask.text;
  const exitsFor = (column: string) =>
    ask.exits.slice(1).filter((x) => {
      const d = x.decision as { column?: string } | null;
      return d?.column === column;
    });
  const items = block ? ((block.decision as { items: { column: string; value: string }[] }).items ?? []) : [];
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
            <span className={s.unlockWhy}>
              after {singles.length} readings confirmed one at a time
            </span>
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
          <div className={s.slotActions}>
            <button type="button" className={s.primary}>
              <Rich text={block.label} />
            </button>
            <span className={s.actionNote}>or keep confirming one at a time below</span>
          </div>
        </section>
      ) : (
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
                    return (
                      <button
                        key={c}
                        type="button"
                        className={s.confirmChip}
                        aria-label={x?.label.replace(/`/g, "") ?? c}
                        title={x?.label.replace(/`/g, "") ?? c}
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
                  <button type="button" className={s.confirm}>
                    Confirm these {line.columns.length} as {line.guess}s
                  </button>
                  <span className={s.otherCount}>a family: it settles exactly the {line.columns.length} listed</span>
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>
      {readFromData ? (
        <p className={s.readFold}>
          {readFromData} more were settled by their values, no question asked (listed above the card, each with its
          evidence).
        </p>
      ) : null}
    </div>
  );
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
  onExposure,
}: {
  card: EstimandCard;
  entry: TeachingEntry | null;
  choice: EstimandChoice;
  concept: ConceptTeaching | null;
  onExposure: (column: string) => void;
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
              onPointerEnter={() => onExposure(e.column)}
              onClick={() => onExposure(e.column)}
            >
              {e.column}
            </button>
          ))}
          {card.family ? (
            <button type="button" role="radio" aria-checked={false} className={cx(s.chip, s.chipWide)}>
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
        <ul className={s.options}>
          {card.effects.map((e) => (
            <li key={e.effect} className={s.option} data-on={e.effect === choice.effect || undefined}>
              <span className={s.optionLabel}>{e.label}</span>
              <span className={s.optionWhat}>{e.consequence}</span>
            </li>
          ))}
        </ul>
      </fieldset>
      {chosen?.energy_contrast ? (
        <fieldset className={s.choiceSet} id="contrast">
          <legend>Which energy question</legend>
          <ul className={s.options}>
            {card.contrasts.map((c) => (
              <li key={c.contrast} className={s.option} data-on={c.contrast === choice.contrast || undefined}>
                <span className={s.optionLabel}>{c.label}</span>
                <span className={s.optionWhat}>{c.consequence}</span>
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
      <div className={s.slotActions}>
        <button type="button" className={s.primary}>
          Record this estimand
        </button>
        <span className={s.actionNote}>No estimate is shown until it is recorded.</span>
      </div>
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

export function AdjustmentSlot({
  card,
  entry,
  focusGroup,
  onFocusGroup,
  concept,
}: {
  card: AdjustmentCard;
  entry: TeachingEntry | null;
  focusGroup: string | null;
  onFocusGroup: (key: string) => void;
  concept: ConceptTeaching | null;
}) {
  const q = card.questions as Record<string, string>;
  const fields = ["causes_exposure", "causes_outcome", "after_exposure"] as const;
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
        {fields.map((f) => (
          <li key={f}>{q[f]}</li>
        ))}
      </ol>
      <div className={s.adjWrap}>
        <table className={s.adj}>
          <thead>
            <tr>
              <th scope="col">Covariates</th>
              {fields.map((f, i) => (
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
            return (
              <tbody
                key={g.key}
                className={s.adjGroup}
                data-focus={focusGroup === g.key || undefined}
                data-unguessed={!guess || undefined}
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
                  {fields.map((f) => (
                    <td key={f} className={guess ? s.adjA : s.adjBlank}>
                      {guess ? (ANSWER[guess[f] ?? ""] ?? "—") : "?"}
                    </td>
                  ))}
                  <td className={s.adjRole}>
                    {g.derived_words ? <span className={s.derived}>{g.derived_words}</span> : <span className={s.derivedAsk}>asked</span>}
                  </td>
                </tr>
                <tr className={s.adjWhyRow}>
                  <td colSpan={5}>
                    <span className={s.groupWhy}>
                      <Rich text={firstUpper(text)} />
                      {source ? <span className={s.source}> {source}</span> : null}
                    </span>
                    {guess ? (
                      <button type="button" className={s.confirm}>
                        Confirm {g.columns.length === 2 ? "both" : `all ${g.columns.length}`} as{" "}
                        {g.derived_words?.endsWith("unknown") ? `of ${g.derived_words}` : `${g.derived_words}s`}
                      </button>
                    ) : null}
                  </td>
                </tr>
              </tbody>
            );
          })}
        </table>
      </div>
      {concept ? <ConceptFull concept={concept} /> : null}
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
}) {
  if (!labels) return null;
  const consequence = (k: string) => entry?.options.find((o) => o.value === k)?.consequence ?? null;
  return (
    <div className={s.alts} role="listbox" aria-label="Energy adjustment: the alternatives" data-testid="alternatives">
      <div className={s.altsHead}>
        <Kicker>Change the energy model</Kicker>
        {energy?.ranking ? (
          <p className={s.altsLine}>
            <Rich text={energy.ranking.line ?? ""} />
          </p>
        ) : null}
      </div>
      <ol className={s.altList}>
        {labels.options.map((o) => {
          const p = previews[o.key];
          const refused = p && !isPreview(p.body) ? p.body.error.message : null;
          return (
            <li
              key={o.key}
              role="option"
              aria-selected={hover === o.key}
              className={s.alt}
              data-current={o.key === current || undefined}
              data-hover={hover === o.key || undefined}
              data-refused={refused ? true : undefined}
              onPointerEnter={() => onHover(o.key)}
            >
              <div className={s.altTop}>
                <span className={s.altLabel}>{o.label}</span>
                {o.key === current ? <span className={s.inDraft}>in the draft</span> : null}
                <span className={s.verdict} data-verdict={o.sound.verdict}>
                  {o.sound.verdict} for {o.sound.purpose}
                </span>
              </div>
              <p className={s.altWhat} data-clamp={hover !== o.key || undefined}>
                {refused ? <Rich text={`Not available: ${refused}`} /> : consequence(o.key)}
              </p>
              {hover === o.key && !refused ? (
                <p className={s.altLabels}>
                  <span>Customary in {o.customary.field}:</span> <Rich text={o.customary.text} />{" "}
                  <span>Sound:</span> <Rich text={o.sound.reason} />
                </p>
              ) : null}
            </li>
          );
        })}
      </ol>
      <p className={s.altKeys}>
        {keys ? (
          <>
            <kbd>↑</kbd> <kbd>↓</kbd> rifle through them · <kbd>Space</kbd> flips the canvas · <kbd>Enter</kbd> records
            · <kbd>Esc</kbd> keeps the draft
          </>
        ) : (
          "Hover an option: the canvas plays what it does to your data. Nothing is recorded."
        )}
      </p>
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
export function QuestionSlot({ entry }: { entry: TeachingEntry | null }) {
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
    </div>
  );
}
