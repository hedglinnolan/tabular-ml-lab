/**
 * The current objective as one focused card. Each card leads with the sentence it will write into
 * the methods section (its blanks dashed until filled), then the slot itself: the engine's guess,
 * its evidence, and the answers it offers. Hovering or choosing an answer plays it on the canvas;
 * the card's own control records it and the quest log moves on. Concepts are taught in full at
 * their first encounter and condensed to an expandable phrase afterwards (BLUEPRINT §11.4).
 *
 * The walk follows the shared scenario (methods-shared/SCENARIO.md): each record control takes
 * the scenario's answer; any other answer is previewed on the canvas, never recorded, and the card
 * says which answer the scenario records. A stated item opens the same card with its answer marked
 * recorded: the alternatives still play on the canvas.
 */
import { Fragment, useState, type ReactNode } from "react";
import { Rich } from "../../components/stage/text";
import { fmtInt, INF, previewOf, refusalOf, teaching, term, type Exit, type StepAsk } from "./data";
import { ADJUSTMENT_ANSWERS, ADJUSTMENT_SENTENCES, MASTERY, SCENARIO } from "./journey";
import { blockOf, readingSlots, VALUE_WORDS, type ReadingSlot } from "./readings";
import { phraseOf } from "./sections";
import s from "./questlog.module.css";

export type Mode = "open" | "stated";

// ── the frame every objective shares ─────────────────────────────────────────

export function Frame({
  section,
  refText,
  position,
  children,
  next,
}: {
  section: string;
  refText: string;
  position: string;
  children: ReactNode;
  next?: ReactNode;
}) {
  return (
    <>
      <div className={s.crumb}>
        <span className={s.kickerNow}>{section}</span>
        <span className={s.ref}>{refText}</span>
        <span className={s.hint}>· {position}</span>
      </div>
      <article className={s.card} data-testid="objective">
        {children}
      </article>
      {next ? <div className={s.next}>{next}</div> : null}
    </>
  );
}

function Blank({ children, filled = false }: { children: ReactNode; filled?: boolean }) {
  return <span className={filled ? s.blankFilled : s.blank}>{children}</span>;
}

/** The sentence a card writes: its changeable phrase a blank until the scenario's answer fills it;
 *  another answer shows in the blank, the rest of the sentence waiting for the record. */
function Draft({ kind, sentence, filled, chosen, placeholder }: { kind: string; sentence: string | null; filled: boolean; chosen: string | null; placeholder: string }) {
  const parts = sentence ? phraseOf(kind, sentence) : null;
  if (!sentence || !parts)
    return (
      <p className={s.draft}>
        <Blank filled={!!chosen}>{chosen ?? placeholder}</Blank>
      </p>
    );
  const [before, phrase, after] = parts;
  return (
    <p className={s.draft}>
      {/* An article alone ("A ") reads as a stray word before an open blank. */}
      {filled || before.trim().length > 2 ? <Rich text={before} /> : null}
      <Blank filled={filled || !!chosen}>{filled ? <Rich text={phrase} /> : (chosen ?? placeholder)}</Blank>
      {filled ? <Rich text={after} /> : "…"}
    </p>
  );
}

function Receipt({ text }: { text: string }) {
  return (
    <p className={s.receipt}>
      <Rich text={text} />
    </p>
  );
}

/** The record control: enabled for the scenario's answer; otherwise it names that answer. */
function RecordBar({ ok, label, note, scenario, onRecord }: { ok: boolean; label: string; note: ReactNode; scenario: string; onRecord: () => void }) {
  return (
    <div className={s.footer}>
      <span className={s.footNote}>
        {ok ? note : <Rich text={`This walk follows the shared scenario, which records ${scenario}; other answers play on the canvas but are not recorded.`} />}
      </span>
      <button type="button" className={s.btnPrimary} disabled={!ok} onClick={onRecord} data-testid="record">
        {label}
      </button>
    </div>
  );
}

/** A stated item's card: the recorded answer stays; the alternatives only play on the canvas. */
function KeepBar({ onBack }: { onBack: () => void }) {
  return (
    <div className={s.footer}>
      <span className={s.footNote}>
        Recorded. Changing it would re-derive everything after it; this walk previews the alternatives on the canvas and keeps the
        scenario's answer.
      </span>
      <button type="button" className={s.btn} onClick={onBack} data-testid="keep">
        Keep it
      </button>
    </div>
  );
}

// ── options: a radio list whose hover plays on the canvas ───────────────────

interface Opt {
  value: string;
  label: string;
  why?: ReactNode;
  /** The engine's own tag beside the option (its first, recommended, blocked…): never a pre-selection. */
  tag?: string;
  /** The engine's refusal of this option here (it cannot be chosen). */
  refused?: string | null;
}

function Options({
  label,
  options,
  chosen,
  recorded,
  onChoose,
  onPeek,
  multi = false,
  testPrefix = "option",
}: {
  label: string;
  options: Opt[];
  chosen: string[];
  recorded: string[];
  onChoose: (v: string) => void;
  onPeek?: (v: string) => void;
  multi?: boolean;
  /** The options' test ids, `<prefix>-<value>` (two lists on one card need two prefixes). */
  testPrefix?: string;
}) {
  return (
    <ul className={s.options} role={multi ? "group" : "radiogroup"} aria-label={label}>
      {options.map((o) => {
        const on = chosen.includes(o.value);
        const rec = recorded.includes(o.value);
        return (
          <li key={o.value}>
            <button
              type="button"
              className={s.option}
              role={multi ? "checkbox" : "radio"}
              aria-checked={on || rec}
              aria-disabled={o.refused ? true : undefined}
              data-focus={on || undefined}
              data-refused={o.refused ? true : undefined}
              data-testid={`${testPrefix}-${o.value}`}
              onPointerEnter={() => onPeek?.(o.value)}
              onFocus={() => onPeek?.(o.value)}
              onClick={() => {
                onPeek?.(o.value);
                if (!o.refused) onChoose(o.value);
              }}
            >
              <span className={multi ? s.check : s.radio} data-recorded={rec || undefined} data-on={on && !rec ? true : undefined} />
              <span className={s.optionLabel}>{o.label}</span>
              <span className={rec ? s.optionTagNow : o.tag ? s.optionTagGuess : s.optionTag}>
                {rec ? "recorded" : o.refused ? "not available here" : (o.tag ?? "")}
              </span>
              <span className={s.optionWhy}>{o.refused ? <Rich text={o.refused} /> : o.why}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

// ── teaching: first encounter in full, later encounters condensed ────────────

export function FirstEncounter({ name, more, source }: { name: string; more?: string; source?: string }) {
  const t = term(name);
  if (!t) return null;
  return (
    <aside className={s.teach} data-purpose="teach_first" data-testid="teach-first">
      <div className={s.teachHead}>
        <span className={s.teachNew}>New concept</span>
        <span className={s.teachTerm}>{t.term}</span>
      </div>
      <p className={s.teachBody}>
        <Rich text={t.definition} />
      </p>
      {more ? (
        <p className={s.teachMore}>
          <Rich text={more} />
        </p>
      ) : null}
      {source ? <span className={s.hint}>{source}</span> : null}
    </aside>
  );
}

export function Concepts({ seen, fresh, firstAt }: { seen: string[]; fresh: string[]; firstAt: Record<string, string> }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const shown = expanded ? term(expanded) : null;
  return (
    <div className={s.concepts} data-purpose="concepts" data-testid="concepts">
      <span>Concepts here:</span>
      {seen.map((c) => (
        <button key={c} type="button" className={s.concept} aria-expanded={expanded === c} onClick={() => setExpanded(expanded === c ? null : c)}>
          {c}
        </button>
      ))}
      {fresh.map((c) => (
        <span key={c}>
          <b style={{ color: "var(--accent)", fontWeight: 700 }}>{c}</b> (new)
        </span>
      ))}
      {shown ? (
        <div className={s.conceptCard} style={{ flexBasis: "100%" }} data-testid="concept-card">
          <Rich text={shown.definition} />
          <small>First taught at {firstAt[shown.term] ?? "an earlier slot"} · condensed since</small>
        </div>
      ) : null}
    </div>
  );
}

// ── the roles ────────────────────────────────────────────────────────────────

const ROLE_ORDER = ["energy", "exposure", "covariate", "identifier", "flag"];
const ROLE_NAME: Record<string, string> = {
  energy: "Energy",
  exposure: "Exposures",
  covariate: "Covariates",
  identifier: "Identifier",
  flag: "Flags",
};

export function RolesCard({ mode, sentence, onRecord, onBack }: { mode: Mode; sentence: string | null; onRecord: () => void; onBack: () => void }) {
  const t = teaching("roles")!;
  const cols = INF.roles.columns;
  const by = ROLE_ORDER.map((r) => ({ role: r, cols: cols.filter((c) => c.proposed === r) })).filter((g) => g.cols.length);
  const below = cols.filter((c) => c.confidence !== "high").length;
  const count = (r: string) => cols.filter((c) => c.proposed === r).length;
  return (
    <>
      {mode === "stated" && sentence ? (
        <Receipt text={sentence} />
      ) : (
        <p className={s.draft}>
          Column roles were set for <Blank>{cols.length}</Blank> columns: energy <Blank>kcal</Blank>; exposures <Blank>{count("exposure")}</Blank>;
          covariates <Blank>{count("covariate")}</Blank>; identifier <Blank>SEQN</Blank>; flags <Blank>{count("flag")}</Blank>.
        </p>
      )}
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      <div className={s.subhead}>
        <span className={s.subTitle}>The guesses, from your values</span>
        <span className={s.hint}>a dot marks a guess below high confidence</span>
      </div>
      <ul className={s.rows}>
        {by.map((g) => {
          const reasons = [...new Set(g.cols.map((c) => c.reason))];
          const highs = g.cols.filter((c) => c.confidence === "high").length;
          return (
            <li key={g.role} className={s.group}>
              <div className={s.groupHead}>
                <span className={s.groupName}>
                  {ROLE_NAME[g.role]} <span className={s.count}>{g.cols.length}</span>
                </span>
                <span className={s.hint}>{highs === g.cols.length ? "all high confidence" : `${g.cols.length - highs} below high`}</span>
              </div>
              <div className={s.chips}>
                {g.cols.map((c) => (
                  <code key={c.column} className={c.confidence === "high" ? "v" : `v ${s.attention}`} title={c.reason}>
                    {c.column}
                  </code>
                ))}
              </div>
              <span className={s.evidence}>
                <Rich text={reasons[0]!} />
                {reasons.length > 1 ? ` (+${reasons.length - 1} more reasons, on hover)` : ""}
              </span>
            </li>
          );
        })}
      </ul>
      {mode === "stated" ? (
        <KeepBar onBack={onBack} />
      ) : (
        <div className={s.footer}>
          <span className={s.footNote}>The {below} below high confidence then wait for their own confirmation under Column readings.</span>
          <button type="button" className={s.btnPrimary} onClick={onRecord} data-testid="record">
            Record these roles
          </button>
        </div>
      )}
    </>
  );
}

// ── the readings ─────────────────────────────────────────────────────────────

function answerWords(e: Exit): string {
  const d = (e.decision ?? {}) as Record<string, unknown>;
  if (d.kind === "set_column_unit") return Number(d.days) === 1 ? `${String(d.unit)}, one day's intake` : `${String(d.days)} days' total`;
  const v = String(d.value ?? "");
  return VALUE_WORDS[v] ?? v;
}

function ReadingRow({
  slot,
  focus,
  enabled,
  why,
  onFocus,
  onConfirm,
}: {
  slot: ReadingSlot;
  focus: boolean;
  enabled: boolean;
  why: string;
  onFocus: (id: string) => void;
  onConfirm: () => void;
}) {
  const family = slot.columns.length > 1;
  const [guess, ...alts] = slot.options;
  const question = family ? `${slot.columns.length} columns like \`${slot.columns[0]}\`: ${slot.guessWords}?` : `\`${slot.columns[0]}\`: ${slot.guessWords}?`;
  return (
    <li className={s.row} data-focus={focus || undefined} onPointerEnter={() => onFocus(slot.id)} data-testid="reading" data-column={slot.columns[0]}>
      <div className={s.rowMain}>
        <span className={s.rowQ}>
          <Rich text={question} />{" "}
          {slot.confidence ? (
            <span className={s.conf} data-level={slot.confidence}>
              {slot.confidence} confidence
            </span>
          ) : null}
        </span>
        <span className={`${s.evidence} ${s.clamp}`} title={slot.evidence.join(" ").replace(/`/g, "")}>
          <Rich text={slot.evidence[0]!} />
        </span>
        {family ? (
          <span className={s.meta}>
            Confirms exactly:{" "}
            {slot.columns.map((c) => (
              <code key={c} className="v">
                {c}
              </code>
            ))}
          </span>
        ) : null}
      </div>
      <div className={s.rowActions}>
        {guess ? (
          <button
            type="button"
            className={enabled ? s.btnPrimary : s.btn}
            disabled={!enabled}
            title={enabled ? guess.label.replace(/`/g, "") : why}
            onFocus={() => onFocus(slot.id)}
            onClick={onConfirm}
            data-testid={enabled ? "confirm-single" : undefined}
          >
            {family ? `Confirm the ${slot.columns.length}` : `Yes, ${answerWords(guess)}`}
          </button>
        ) : null}
        {alts.slice(0, 1).map((a) => (
          <button key={a.label} type="button" className={s.btnQuiet} disabled title={`${a.label.replace(/`/g, "")} (${why})`}>
            No: {answerWords(a)}
          </button>
        ))}
      </div>
    </li>
  );
}

export function ReadingsCard({
  ask,
  singles,
  next,
  focus,
  onFocus,
  onSingle,
  onBlock,
  stated,
  onBack,
}: {
  ask: StepAsk | null;
  /** The single confirmations recorded so far (their sentences). */
  singles: string[];
  /** The column the scenario confirms next on its own, if any. */
  next: string | null;
  focus: string | null;
  onFocus: (id: string) => void;
  onSingle: () => void;
  onBlock: () => void;
  /** The readings' recorded sentences, when nothing is open. */
  stated: string[];
  onBack: () => void;
}) {
  if (!ask)
    return (
      <>
        <p className={s.draft}>Every reading the open questions needed is settled.</p>
        <div className={s.rows}>
          {stated.map((t) => (
            <Receipt key={t} text={t} />
          ))}
        </div>
        <KeepBar onBack={onBack} />
      </>
    );
  const slots = readingSlots(ask);
  const roles = ask.consumer === "the exposure and its effect";
  const unlocked = singles.length >= MASTERY;
  const block = unlocked ? blockOf(ask) : null;
  const groups = new Map<string, string[]>();
  for (const it of block?.items ?? []) {
    const k = VALUE_WORDS[it.value] ?? it.value;
    groups.set(k, [...(groups.get(k) ?? []), it.column]);
  }
  const single = (r: ReadingSlot) =>
    r.kind === "unit" || (roles && !unlocked && r.columns.length === 1 && r.columns[0] === next);
  const why = roles
    ? unlocked
      ? "the rest are confirmed as one block"
      : `this walk confirms them one at a time in the card's order: \`${next ?? ""}\` next`
    : "the scenario confirms these as one block";
  return (
    <>
      <p className={s.draft}>Tell me about {slots.length === 1 && slots[0]!.columns.length === 1 ? "this column" : "these columns"}.</p>
      <p className={s.why}>
        <Rich text={`${ask.consumer[0]!.toUpperCase()}${ask.consumer.slice(1)} waits on ${slots.length === 1 ? "this reading" : "these readings"}: each is the engine's guess from your values, with its evidence on the canvas.`} />
      </p>
      {roles && singles.length ? (
        <div className={s.rows}>
          {singles.map((c) => (
            <Receipt key={c} text={c} />
          ))}
        </div>
      ) : null}
      {roles && !unlocked ? (
        <p className={s.hint} data-testid="mastery-progress">
          Confirm {MASTERY} one at a time ({singles.length} of {MASTERY} so far); then the rest can be confirmed as one block.
        </p>
      ) : null}
      {block ? (
        <section className={s.unlock} data-purpose="mastery_block" data-testid="unlock">
          <div className={s.unlockHead}>
            <span className={s.teachNew}>Shortcut unlocked</span>
            <span className={s.hint}>after {singles.length} single confirmations</span>
          </div>
          <span className={s.unlockTitle}>{roles ? "Confirm the rest as one block" : "Confirm these as one block"}</span>
          <dl className={s.settles}>
            {[...groups].map(([value, cols]) => (
              <Fragment key={value}>
                <dt>{value}</dt>
                <dd className={s.chips}>
                  {cols.map((c) => (
                    <code key={c} className="v">
                      {c}
                    </code>
                  ))}
                </dd>
              </Fragment>
            ))}
          </dl>
          <div className={s.footer}>
            <span className={s.footNote}>Settles exactly these {block.items.length} readings, each as shown; nothing else. One undo reverts it.</span>
            <button type="button" className={s.btnPrimary} onClick={onBlock} data-testid="block-confirm">
              {block.label}
            </button>
          </div>
        </section>
      ) : null}
      <div className={s.subhead}>
        <span className={s.subTitle}>Needed by {ask.consumer}</span>
        <span className={s.hint}>
          {slots.length} {slots.length === 1 ? "reading" : "readings"}
        </span>
      </div>
      <ul className={s.rows}>
        {slots.map((r) => (
          <ReadingRow
            key={r.id}
            slot={r}
            focus={focus === r.id}
            enabled={single(r)}
            why={why.replace(/`/g, "")}
            onFocus={onFocus}
            onConfirm={onSingle}
          />
        ))}
      </ul>
    </>
  );
}

// ── eligibility, and the screens reported beside it ─────────────────────────

const SCREEN_KEYS = INF.cards.exclusions.labels.options.filter((o) => o.key !== "keep_every_row").map((o) => o.key);
const offered = (key: string) => INF.cards.exclusions.offered.find((o) => o.key === key);
const screenLabel = (key: string) => INF.cards.exclusions.labels.options.find((o) => o.key === key)?.label ?? key;
const SCENARIO_SCREENS = SCREEN_KEYS.filter((k) => SCENARIO.screens.includes(screenLabel(k)));

export function ExclusionsCard({
  mode,
  sentences,
  onRecord,
  onBack,
  onPeek,
}: {
  mode: Mode;
  /** The two sentences the answer writes: the eligibility, then the declared screens. */
  sentences: [string | null, string | null];
  onRecord: () => void;
  onBack: () => void;
  onPeek: (previewKey: string) => void;
}) {
  const t = teaching("exclusions")!;
  const labels = INF.cards.exclusions.labels;
  const keep = labels.options.find((o) => o.key === "keep_every_row");
  const [primary, setPrimary] = useState<string | null>(null);
  const [screens, setScreens] = useState<string[]>([]);
  const stated = mode === "stated";
  const sound = (k: string) => labels.options.find((o) => o.key === k)?.sound?.reason;
  const primaryOpts: Opt[] = [
    { value: "none", label: keep?.label ?? "Keep every row", why: <Rich text={keep?.sound?.reason ?? ""} />, tag: "" },
    ...SCREEN_KEYS.map((k) => {
      const o = offered(k);
      return {
        value: k,
        label: screenLabel(k),
        why: <Rich text={`${o?.label ?? ""}${o ? ` · excludes \`${fmtInt(o.affected)}\` rows` : ""}. ${sound(k) ?? ""}`} />,
        tag: labels.customary_first === k ? "the field's habit" : "",
        refused: typeof o?.refused === "string" ? o.refused : null,
      };
    }),
  ];
  const screenOpts: Opt[] = SCREEN_KEYS.map((k) => {
    const o = offered(k);
    return {
      value: k,
      label: screenLabel(k),
      why: <Rich text={`The same model on the rows this screen keeps${o ? `: \`${fmtInt(o.affected)}\` fewer` : ""}.`} />,
      refused: typeof o?.refused === "string" ? o.refused : null,
    };
  });
  const peek = (k: string) => onPeek(`exclusions_${k}`);
  const same = (a: string[], b: string[]) => a.length === b.length && a.every((x) => b.includes(x));
  const ok = primary === "none" && same(screens, SCENARIO_SCREENS);
  const scenarioWords = `\`Keep every row\`, with ${SCENARIO_SCREENS.map((k) => `\`${screenLabel(k)}\``).join(" and ")} reported beside it`;
  return (
    <>
      {stated ? (
        sentences.filter((x): x is string => !!x).map((x) => <Receipt key={x} text={x} />)
      ) : (
        <Draft kind="set_exclusions" sentence={sentences[0]} filled={primary === "none"} chosen={primary ? (primaryOpts.find((o) => o.value === primary)?.label ?? null) : null} placeholder="who is excluded?" />
      )}
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      {labels.tension ? (
        <p className={s.evidence}>
          <Rich text={labels.tension} />
        </p>
      ) : null}
      <div className={s.subhead}>
        <span className={s.subTitle}>The primary analysis keeps</span>
        <span className={s.hint}>hover to see who each would exclude</span>
      </div>
      <Options
        label="Eligibility"
        options={primaryOpts}
        chosen={primary ? [primary] : []}
        recorded={stated ? ["none"] : []}
        onChoose={setPrimary}
        onPeek={peek}
        testPrefix="primary"
      />
      <div className={s.subhead}>
        <span className={s.subTitle}>Also report beside it</span>
        <span className={s.hint}>sensitivity analyses, declared before any estimate</span>
      </div>
      <Options
        label="Sensitivity analyses"
        options={screenOpts}
        chosen={screens}
        recorded={stated ? SCENARIO_SCREENS : []}
        multi
        onChoose={(k) => setScreens((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]))}
        onPeek={peek}
        testPrefix="screen"
      />
      {stated ? (
        <KeepBar onBack={onBack} />
      ) : (
        <RecordBar
          ok={ok}
          label={screens.length ? `Record: every row, ${screens.length} ${screens.length === 1 ? "screen" : "screens"} beside it` : "Record"}
          note="Each screen is the same model on its own rows, reported beside the primary."
          scenario={scenarioWords}
          onRecord={onRecord}
        />
      )}
    </>
  );
}

// ── missing values ────────────────────────────────────────────────────────────

const RUNG: Record<string, string> = { recommended: "recommended", available: "", block_and_record: "blocked under inference" };

export function MissingCard({ mode, sentence, onRecord, onBack, onPeek }: { mode: Mode; sentence: string | null; onRecord: () => void; onBack: () => void; onPeek: (decisionKey: string) => void }) {
  const t = teaching("missing")!;
  const card = INF.cards.missing.card as unknown as {
    columns?: { column: string; reason: string }[];
    methods: { key: string; label: string; rung: string; sound?: string; customary?: string; decision: Record<string, unknown> }[];
  };
  const [chosen, setChosen] = useState<string | null>(null);
  const stated = mode === "stated";
  const scenarioKey = card.methods.find((m) => m.decision.strategy === SCENARIO.missing && Object.keys(m.decision).length === 1)?.key ?? null;
  const opts: Opt[] = card.methods.map((m) => ({
    value: m.key,
    label: m.label,
    why: <Rich text={[m.customary, m.sound].filter(Boolean).join(" · ")} />,
    tag: RUNG[m.rung] ?? "",
    refused: refusalOf(previewOf({ kind: "set_missing", ...m.decision })),
  }));
  const label = (k: string | null) => card.methods.find((m) => m.key === k)?.label ?? null;
  return (
    <>
      {stated && sentence ? (
        <Receipt text={sentence} />
      ) : (
        <Draft kind="set_missing" sentence={sentence} filled={chosen === scenarioKey} chosen={label(chosen)} placeholder="how are missing values handled?" />
      )}
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      {(card.columns ?? []).map((c) => (
        <p key={c.column} className={s.evidence}>
          <code className="v">{c.column}</code> <Rich text={c.reason} />
        </p>
      ))}
      <Options
        label="Missing values"
        options={opts}
        chosen={chosen ? [chosen] : []}
        recorded={stated && scenarioKey ? [scenarioKey] : []}
        onChoose={setChosen}
        onPeek={(k) => onPeek(k)}
      />
      {stated ? (
        <KeepBar onBack={onBack} />
      ) : (
        <RecordBar ok={chosen === scenarioKey} label={chosen ? `Record: ${label(chosen)}` : "Record"} note="The rows this keeps are on the canvas." scenario={`\`${label(scenarioKey) ?? ""}\``} onRecord={onRecord} />
      )}
    </>
  );
}

/** The captured preview key of a missing-values method (its exact decision). */
export function missingPreviewKey(methodKey: string): string | null {
  const m = (INF.cards.missing.card.methods as { key: string; decision: Record<string, unknown> }[]).find((x) => x.key === methodKey);
  if (!m) return null;
  const p = previewOf({ kind: "set_missing", ...m.decision });
  return p ? (Object.entries(INF.previews).find(([, v]) => v === p)?.[0] ?? null) : null;
}

// ── the seal: held-out rows ───────────────────────────────────────────────────

export function splitPreviewKey(holdout: number): string | null {
  const p = previewOf({ kind: "set_split", holdout, seed: 0, folds: 5 });
  return p ? (Object.entries(INF.previews).find(([, v]) => v === p)?.[0] ?? null) : null;
}

export function SplitCard({ mode, sentence, onRecord, onBack, onPeek }: { mode: Mode; sentence: string | null; onRecord: () => void; onBack: () => void; onPeek: (holdout: number) => void }) {
  const t = teaching("split")!;
  const plan = INF.cards.seal_plan;
  const [chosen, setChosen] = useState<string | null>(null);
  const stated = mode === "stated";
  const scenario = SCENARIO.holdout === null ? null : String(SCENARIO.holdout);
  const opts: Opt[] = plan.options.map((o, i) => ({
    value: String(o.holdout),
    label: o.label,
    why: <Rich text={o.measures} />,
    tag: i === 0 && plan.cv_first ? "the engine's first under inference" : o.below_floor ? "below the floor" : "",
  }));
  const label = (v: string | null) => plan.options.find((o) => String(o.holdout) === v)?.label ?? null;
  return (
    <>
      {stated && sentence ? (
        <Receipt text={sentence} />
      ) : (
        <Draft kind="set_split" sentence={sentence} filled={chosen === scenario} chosen={label(chosen)} placeholder="how many rows are sealed?" />
      )}
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      <p className={s.evidence}>
        <Rich text={plan.reason} />
      </p>
      <Options
        label="Held-out rows"
        options={opts}
        chosen={chosen ? [chosen] : []}
        recorded={stated && scenario ? [scenario] : []}
        onChoose={setChosen}
        onPeek={(v) => onPeek(Number(v))}
      />
      {stated ? (
        <KeepBar onBack={onBack} />
      ) : (
        <RecordBar ok={chosen === scenario} label={chosen ? `Record: ${label(chosen)}` : "Record"} note="Five-fold cross-validation (seed 0) on the rows that train." scenario={`\`${label(scenario) ?? ""}\``} onRecord={onRecord} />
      )}
    </>
  );
}

// ── the exposure and its effect ───────────────────────────────────────────────

export function EstimandCard({ mode, sentence, onRecord, onBack, onPeek }: { mode: Mode; sentence: string | null; onRecord: () => void; onBack: () => void; onPeek: (exposure: string) => void }) {
  const card = INF.cards.estimand;
  const t = teaching("estimand")!;
  const want = SCENARIO.estimand;
  const stated = mode === "stated";
  const [exposure, setExposure] = useState<string | null>(stated ? (want?.exposure ?? null) : null);
  const [effect, setEffect] = useState<string | null>(stated ? (want?.effect ?? null) : null);
  const [contrast, setContrast] = useState<string | null>(stated ? (want?.contrast ?? null) : null);
  const energy = card.exposures.filter((e) => e.energy_contrast);
  const other = card.exposures.filter((e) => !e.energy_contrast);
  const measure = card.measures.find((m) => m.fitted && m.rank === 1) ?? card.measures[0];
  const coefficient = teaching("energy_adjustment")?.drawer?.sections.find((x) => x.heading === "What the coefficient means");
  const ok = !!want && exposure === want.exposure && effect === want.effect && contrast === want.contrast;
  const effectLabel = card.effects.find((e) => e.effect === effect)?.label.toLowerCase();
  const contrastLabel = card.contrasts.find((c) => c.contrast === contrast)?.label.toLowerCase();
  const chip = (col: string) => (
    <button
      key={col}
      type="button"
      className={s.pickChip}
      role="radio"
      aria-checked={exposure === col}
      aria-pressed={exposure === col}
      onPointerEnter={() => onPeek(col)}
      onFocus={() => onPeek(col)}
      onClick={() => {
        onPeek(col);
        if (!stated) setExposure(col);
      }}
      data-testid={`exposure-${col}`}
    >
      {col}
    </button>
  );
  return (
    <>
      {stated && sentence ? (
        <Receipt text={sentence} />
      ) : (
        <p className={s.draft}>
          The analysis estimates the <Blank filled={!!effect}>{effectLabel ?? "which effect?"}</Blank> of <Blank filled={!!exposure}>{exposure ?? "which exposure?"}</Blank> on{" "}
          <code className="v">glucose</code> (<Blank filled={!!contrast}>{contrastLabel ?? "which contrast?"}</Blank>), as a{" "}
          <Blank filled>{measure?.label ?? "difference in the mean outcome"}</Blank> per unit of the exposure.
        </p>
      )}
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      <div className={s.slotGrid}>
        <span className={s.slotName}>Exposure</span>
        <div>
          <div className={s.pickLabel}>Not guessed: the exposure is your question. Offered from your roles:</div>
          <div className={s.pick} role="radiogroup" aria-label="Exposure">
            {energy.map((e) => chip(e.column))}
            {other.map((e) => chip(e.column))}
          </div>
          {card.family ? (
            <div className={s.pickLabel}>
              Or the {card.family.n} nutrients as one family, each its own test (not in this walk).
            </div>
          ) : null}
        </div>
        <span className={s.slotName}>Effect</span>
        <div className={s.cards}>
          {card.effects.map((e, i) => (
            <button
              key={e.effect}
              type="button"
              className={s.optCard}
              aria-pressed={effect === e.effect}
              data-on={effect === e.effect || undefined}
              onClick={() => !stated && setEffect(e.effect)}
              data-testid={`effect-${e.effect}`}
            >
              <span className={s.optHead}>
                <span className={s.optionLabel}>{e.label}</span>
                <span className={s.optionTagGuess}>{i === 0 ? "the engine's first" : ""}</span>
              </span>
              <span className={s.optWhy}>{e.consequence}</span>
            </button>
          ))}
        </div>
        <span className={s.slotName}>Contrast</span>
        <div className={s.cards}>
          {card.contrasts.map((c, i) => (
            <button
              key={c.contrast}
              type="button"
              className={s.optCard}
              aria-pressed={contrast === c.contrast}
              data-on={contrast === c.contrast || undefined}
              onClick={() => !stated && setContrast(c.contrast)}
              data-testid={`contrast-${c.contrast}`}
            >
              <span className={s.optHead}>
                <span className={s.optionLabel}>{c.label}</span>
                <span className={s.optionTagGuess}>{i === 0 ? "the engine's first" : ""}</span>
              </span>
              <span className={s.optWhy}>{c.consequence}</span>
            </button>
          ))}
        </div>
        <span className={s.slotName}>Measure</span>
        <div className={s.cards}>
          {card.measures.slice(0, 3).map((m) => (
            <div key={m.measure} className={s.optCard} data-on={m === measure || undefined} data-refused={!m.fitted || undefined}>
              <span className={s.optHead}>
                <span className={s.optionLabel}>{m.label}</span>
                <span className={m.fitted ? s.optionTagGuess : s.optionTag}>{m.fitted ? "the only one fitted here" : "not fitted"}</span>
              </span>
              <span className={s.optWhy}>{m.reason}</span>
            </div>
          ))}
        </div>
        {stated ? null : (
          <>
            <span />
            <FirstEncounter
              name="substitution"
              more={coefficient?.body}
              source={coefficient ? `${coefficient.evidence.status} · ${coefficient.evidence.source}` : undefined}
            />
          </>
        )}
      </div>
      {stated ? (
        <>
          <Concepts seen={["substitution", "estimand"]} fresh={[]} firstAt={{ substitution: "Exposure and effect", estimand: "Exposure and effect" }} />
          <KeepBar onBack={onBack} />
        </>
      ) : (
        <RecordBar
          ok={ok}
          label={exposure ? `Record ${exposure} as the exposure` : "Choose an exposure"}
          note="No estimate is shown until this is recorded."
          scenario={want ? `the ${want.effect} effect of \`${want.exposure}\`, as a ${want.contrast}` : "its own answer"}
          onRecord={onRecord}
        />
      )}
    </>
  );
}

// ── the adjustment set ────────────────────────────────────────────────────────

const ANSWERS = ["yes", "no", "unknown"] as const;
const ANSWER_WORDS: Record<string, string> = { yes: "yes", no: "no", unknown: "don't know" };
const ANSWER_SHORT: Record<string, string> = { yes: "yes", no: "no", unknown: "?" };
const FIELDS = ["causes_exposure", "causes_outcome", "after_exposure"] as const;
type Field = (typeof FIELDS)[number];

function Seg({ value, label, onPick }: { value: string | null; label: string; onPick: (v: string) => void }) {
  return (
    <div className={s.seg} role="radiogroup" aria-label={label}>
      {ANSWERS.map((a) => (
        <button
          key={a}
          type="button"
          className={s.segBtn}
          role="radio"
          aria-checked={value === a}
          aria-pressed={value === a}
          title={ANSWER_WORDS[a]}
          aria-label={`${label}: ${ANSWER_WORDS[a]}`}
          onClick={() => onPick(a)}
          data-answer={a}
        >
          {ANSWER_SHORT[a]}
        </button>
      ))}
    </div>
  );
}

/** One of the unguessed group's answers: its columns, answered together as the scenario recorded them. */
function AskedRow({ index, done, short, onAnswer }: { index: number; done: boolean; short: Record<Field, string>; onAnswer: (i: number) => void }) {
  const a = ADJUSTMENT_ANSWERS[index]!;
  const truth = INF.cards.adjustment_answers[index]!.answers[a.columns[0]!]!;
  const [vals, setVals] = useState<Partial<Record<Field, string>>>({});
  const full = FIELDS.every((f) => vals[f]);
  const ok = FIELDS.every((f) => vals[f] === truth[f]);
  if (done) return <Receipt text={ADJUSTMENT_SENTENCES[index] ?? ""} />;
  const q = INF.cards.adjustment.questions;
  return (
    <div className={s.asked} data-testid="adjust-asked" data-columns={a.columns.join(",")}>
      <div className={s.chips}>
        {a.columns.map((c) => (
          <code key={c} className="v">
            {c}
          </code>
        ))}
        {a.columns.length > 1 ? <span className={s.hint}>answered together</span> : null}
      </div>
      <div className={s.colRows}>
        {FIELDS.map((f) => (
          <Fragment key={f}>
            <span className={s.colHead}>
              <Rich text={short[f]} />
            </span>
            <Seg value={vals[f] ?? null} label={`${a.columns.join(", ")}: ${q[f] ?? f}`} onPick={(v) => setVals((cur) => ({ ...cur, [f]: v }))} />
          </Fragment>
        ))}
      </div>
      <div className={s.groupFoot}>
        <span className={s.footNote}>
          {full && !ok
            ? `The shared scenario answers ${FIELDS.map((f) => ANSWER_WORDS[truth[f]]).join(" · ")} (the fixture's declared truth); other answers are not recorded here.`
            : "The role is derived from your three answers."}
        </span>
        <button type="button" className={full ? s.btnPrimary : s.btn} disabled={!ok} onClick={() => onAnswer(index)} data-testid="record-asked">
          Record {a.columns.length === 1 ? "it" : `the ${a.columns.length}`}
        </button>
      </div>
    </div>
  );
}

export function AdjustmentCard({
  mode,
  adjusted,
  focus,
  onFocus,
  onAnswer,
  onBack,
}: {
  mode: Mode;
  adjusted: number[];
  focus: string | null;
  onFocus: (previewKey: string) => void;
  onAnswer: (index: number) => void;
  onBack: () => void;
}) {
  const card = INF.cards.adjustment;
  const t = teaching("adjustment")!;
  const n = card.groups.reduce((m, g) => m + g.columns.length, 0);
  const stated = mode === "stated";
  const done = (i: number) => stated || adjusted.includes(i);
  const left = ADJUSTMENT_ANSWERS.filter((a) => !done(a.index)).reduce((m, a) => m + a.columns.length, 0);
  const short: Record<Field, string> = {
    causes_exposure: `Causes \`${card.exposure}\`?`,
    causes_outcome: "Causes `glucose`?",
    after_exposure: `Changed by \`${card.exposure}\`?`,
  };
  return (
    <>
      <p className={s.draft}>
        For the effect of <code className="v">{card.exposure}</code>, by the disjunctive cause criterion: <Blank filled={!left}>{n} covariates</Blank> are
        confounders, adjusted for, or left out.
      </p>
      {left && !stated ? (
        <p className={s.hint} data-testid="adjust-left">
          {n - left} of {n} answered; each group is recorded as you confirm it.
        </p>
      ) : null}
      <p className={s.why}>
        <Rich text={t.one_liner} />
      </p>
      <ul className={s.rows}>
        {card.groups.map((g) => {
          const answers = ADJUSTMENT_ANSWERS.filter((a) => a.group === g.key);
          const one = g.decision ? answers[0] : undefined;
          const key = one ? `adjust_${g.key}` : `adjust_answers_${answers.find((a) => !done(a.index))?.index ?? answers[0]?.index ?? 0}`;
          return (
            <li key={g.key} className={s.group} data-focus={focus === key || undefined} onPointerEnter={() => onFocus(key)} data-testid="adjust-group" data-group={g.key}>
              <div className={s.groupHead}>
                <span className={s.groupName}>
                  {g.label} <span className={s.count}>{g.columns.length}</span>
                </span>
                {g.derived_words ? <span className={s.derived}>→ {g.derived_words}</span> : <span className={s.hint}>each asked</span>}
              </div>
              {g.guess ? (
                one && done(one.index) ? (
                  <Receipt text={ADJUSTMENT_SENTENCES[one.index] ?? ""} />
                ) : (
                  <>
                    <div className={s.chips}>
                      {g.columns.map((c) => (
                        <code key={c} className="v">
                          {c}
                        </code>
                      ))}
                    </div>
                    <div className={s.answers}>
                      {FIELDS.map((f) => (
                        <span key={f} className={s.answer}>
                          <Rich text={short[f]} />
                          <b>{ANSWER_WORDS[g.guess![f] ?? ""] ?? "—"}</b>
                        </span>
                      ))}
                    </div>
                    <div className={s.groupFoot}>
                      <span className={`${s.evidence} ${s.clamp}`} title={g.reason}>
                        <Rich text={`Guessed: ${g.reason.replace(/ \(NUTRITION_PACK[^)]*\)\)?/, " (NUTRITION_PACK §08)")}`} />
                      </span>
                      <button
                        type="button"
                        className={focus === key ? s.btnPrimary : s.btn}
                        onFocus={() => onFocus(key)}
                        onClick={() => one && onAnswer(one.index)}
                        data-testid="confirm-group"
                      >
                        {g.derived === "confounder" ? `Confirm the ${g.columns.length} as confounders` : `Confirm the ${g.columns.length}: ${g.derived_words}`}
                      </button>
                    </div>
                  </>
                )
              ) : (
                <>
                  <span className={s.evidence}>{g.reason}</span>
                  {answers.map((a) => (
                    <AskedRow key={a.index} index={a.index} done={done(a.index)} short={short} onAnswer={onAnswer} />
                  ))}
                </>
              )}
            </li>
          );
        })}
      </ul>
      {stated ? <KeepBar onBack={onBack} /> : <span className={s.hint}>Derived by {card.source}.</span>}
    </>
  );
}

// ── energy adjustment: the phrase and its alternatives ─────────────────────

export function EnergyCard({
  mode,
  sentence,
  hovered,
  onPeek,
  onRecord,
  onBack,
}: {
  mode: Mode;
  sentence: string | null;
  hovered: string | null;
  onPeek: (method: string) => void;
  onRecord: () => void;
  onBack: () => void;
}) {
  const t = teaching("energy_adjustment")!;
  const card = INF.cards.energy;
  const stated = mode === "stated";
  const [chosen, setChosen] = useState<string | null>(null);
  const order = card.card.ranking.order;
  const labelOf = (v: string | null) => card.labels.options.find((o) => o.key === v)?.label ?? t.options.find((o) => o.value === v)?.label ?? null;
  const opts: Opt[] = order.map((v, i) => {
    const app = card.card.applicability[v];
    const sound = card.labels.options.find((o) => o.key === v)?.sound;
    const consequence = t.options.find((o) => o.value === v)?.consequence ?? "";
    const residual = hovered === v && v === "residual" ? term("residual method") : null;
    return {
      value: v,
      label: labelOf(v) ?? v,
      tag: i === 0 ? "ranked first here" : "",
      refused: refusalOf(INF.previews[`energy_${v}`]) ?? (app && !app.ok ? app.reason : null),
      why: (
        <>
          {consequence}
          {sound ? ` ${sound.verdict === "sound" ? "Sound" : sound.verdict === "unsound" ? "Unsound" : "Conditional"}: ${sound.reason}` : ""}
          {residual ? (
            <span style={{ display: "block", marginTop: 6 }}>
              <b style={{ color: "var(--accent)", fontFamily: "var(--sans)", fontSize: 11, letterSpacing: ".08em" }}>NEW · </b>
              <Rich text={residual.definition} /> Watch it on the canvas: fit on energy, then keep the residual.
            </span>
          ) : null}
        </>
      ),
    };
  });
  const parts = sentence ? phraseOf("set_energy_adjustment", sentence) : null;
  const filled = stated || chosen === SCENARIO.energy;
  return (
    <>
      {parts ? (
        <p className={s.draft}>
          <Rich text={parts[0]} />
          {filled ? (
            <span className={s.phraseOpen}>
              <Rich text={parts[1]} />
            </span>
          ) : (
            <Blank filled={!!chosen}>{labelOf(chosen) ?? "which method?"}</Blank>
          )}
          {filled ? <Rich text={parts[2]} /> : "…"}
        </p>
      ) : null}
      <p className={s.why}>
        <Rich text={t.one_liner} />
      </p>
      {card.card.ranking.line ? (
        <p className={s.evidence}>
          <Rich text={card.card.ranking.line} />
        </p>
      ) : null}
      <Concepts
        seen={["substitution", "estimand"]}
        fresh={hovered === "residual" ? ["residual method"] : []}
        firstAt={{ substitution: "Exposure and effect", estimand: "Exposure and effect" }}
      />
      <Options
        label="Energy adjustment"
        options={opts}
        chosen={chosen ? [chosen] : []}
        recorded={stated && SCENARIO.energy ? [SCENARIO.energy] : []}
        onChoose={(v) => !stated && setChosen(v)}
        onPeek={onPeek}
      />
      {stated ? (
        <KeepBar onBack={onBack} />
      ) : (
        <RecordBar
          ok={chosen === SCENARIO.energy}
          label={chosen ? `Record: ${labelOf(chosen)}` : "Record"}
          note="Space flips the canvas between your data now and with this method."
          scenario={`\`${labelOf(SCENARIO.energy) ?? ""}\``}
          onRecord={onRecord}
        />
      )}
    </>
  );
}

// ── the model sequence ────────────────────────────────────────────────────────

export function SequenceCard({ mode, sentence, onRecord, onBack }: { mode: Mode; sentence: string | null; onRecord: () => void; onBack: () => void }) {
  const card = INF.cards.model_sequence;
  const stated = mode === "stated";
  const [cols, setCols] = useState<string[]>(stated ? SCENARIO.model1 : []);
  const ok = cols.length === SCENARIO.model1.length && SCENARIO.model1.every((c) => cols.includes(c));
  const named = (xs: string[]) => {
    const l = xs.map((c) => `\`${c}\``);
    return l.length > 1 ? `${l.slice(0, -1).join(", ")} and ${l.at(-1)}` : (l[0] ?? "");
  };
  return (
    <>
      {stated && sentence ? (
        <Receipt text={sentence} />
      ) : (
        <Draft kind="set_model_sequence" sentence={sentence} filled={ok} chosen={cols.length ? `Model 1, adjusted for ${cols.join(", ")}` : null} placeholder="Model 1, adjusted for which columns?" />
      )}
      <p className={s.why}>
        <Rich text={card.reason} />
      </p>
      <div className={s.slotGrid}>
        <span className={s.slotName}>Unadjusted</span>
        <span className={s.optWhy}>The exposure alone.</span>
        <span className={s.slotName}>Model 1</span>
        <div>
          <div className={s.pick} role="group" aria-label="Model 1 adjusts for">
            {card.allowed.map((c) => (
              <button
                key={c}
                type="button"
                className={s.pickChip}
                aria-pressed={cols.includes(c)}
                onClick={() => !stated && setCols((cur) => (cur.includes(c) ? cur.filter((x) => x !== c) : [...cur, c]))}
                data-testid={`model1-${c}`}
              >
                {c}
                {card.guess.includes(c) ? <span className={s.guessDot} aria-label="the field's guess" /> : null}
              </button>
            ))}
          </div>
          <div className={s.pickLabel}>
            <Rich text={`A dot marks the field's Model 1: ${named(card.guess)}.`} />
          </div>
        </div>
        <span className={s.slotName}>Model 2</span>
        <span className={s.optWhy}>The primary: the adjustment set you declared.</span>
        <span className={s.slotName}>Model 3</span>
        <span className={s.optWhy}>Model 2 plus the covariates of unknown timing set beside it.</span>
      </div>
      {stated ? (
        <KeepBar onBack={onBack} />
      ) : (
        <RecordBar ok={ok} label="Record the sequence" note="Each model is fit once, with the primary, when the plan is locked." scenario={`Model 1 adjusted for ${named(SCENARIO.model1)}`} onRecord={onRecord} />
      )}
    </>
  );
}

// ── the model family ──────────────────────────────────────────────────────────

export function ModelsCard({
  mode,
  sentence,
  ready,
  onRecord,
  onBack,
  onChosen,
}: {
  mode: Mode;
  sentence: string | null;
  /** The family is asked now (the sequence and the code readings are recorded). */
  ready: boolean;
  onRecord: () => void;
  onBack: () => void;
  onChosen: (families: string[]) => void;
}) {
  const t = teaching("models")!;
  const shelf = INF.cards.shelf;
  const stated = mode === "stated";
  const [chosen, setChosen] = useState<string[]>([]);
  const ok = ready && chosen.length === SCENARIO.models.length && SCENARIO.models.every((m) => chosen.includes(m));
  const opts: Opt[] = (shelf?.families ?? []).map((f) => ({
    value: f.key,
    label: f.label,
    tag: f.rank === 1 ? "first on the shelf" : `${f.fit} fit`,
    why: <Rich text={[f.inductive_bias, ...f.concerns].join(" ")} />,
  }));
  const toggle = (k: string) => {
    if (stated) return;
    const next = chosen.includes(k) ? chosen.filter((x) => x !== k) : [...chosen, k];
    setChosen(next);
    onChosen(next);
  };
  const named = SCENARIO.models.map((m) => shelf?.families.find((f) => f.key === m)?.label ?? m).join(" and ");
  // The record folds what the engine read without asking into this sentence; the card leads with
  // the decision alone (the whole sentence is in the methods section).
  const short = sentence?.replace(/ Read from the values, no question asked:.*$/, "") ?? null;
  return (
    <>
      {stated && short ? (
        <Receipt text={short} />
      ) : (
        <Draft kind="select_models" sentence={short} filled={ok} chosen={chosen.length ? chosen.map((k) => shelf?.families.find((f) => f.key === k)?.label ?? k).join(", ") : null} placeholder="which model families?" />
      )}
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      <Options label="Model families" options={opts} chosen={chosen} recorded={stated ? SCENARIO.models : []} multi onChoose={toggle} />
      {stated ? (
        <KeepBar onBack={onBack} />
      ) : ready ? (
        <RecordBar ok={ok} label="Record the family" note="Nothing is fit until the plan is locked." scenario={`\`${named}\``} onRecord={onRecord} />
      ) : (
        <p className={s.footNote}>Asked once the model sequence and the code readings are recorded.</p>
      )}
    </>
  );
}

// ── the lock ──────────────────────────────────────────────────────────────────

export function LockCard({ sentence, declared, onRecord }: { sentence: string | null; declared: string[]; onRecord: () => void }) {
  return (
    <>
      <p className={`${s.draft} ${s.draftMuted}`}>
        <Rich text={sentence ? sentence.replace(/ \(SHA-256 `[0-9a-f]+`\)/, "") : "The analysis plan is declared before any estimate."} />
      </p>
      <p className={s.why}>Every question the estimate needs is answered. Fitting serves the first estimate, and the first estimate served locks the plan.</p>
      <div className={s.subhead}>
        <span className={s.subTitle}>What the lock declares</span>
        <span className={s.hint}>{declared.length} recorded sentences</span>
      </div>
      <div className={s.rows}>
        {declared.map((d) => (
          <Receipt key={d} text={d} />
        ))}
      </div>
      <div className={s.footer}>
        <span className={s.footNote}>Every later change is marked as made after the estimates were seen.</span>
        <button type="button" className={s.btnPrimary} onClick={onRecord} data-testid="record">
          Fit and lock the plan
        </button>
      </div>
    </>
  );
}

/** The plan is locked: the results are on the canvas. */
export function LockedCard({ sentence, onDoc }: { sentence: string | null; onDoc: () => void }) {
  return (
    <>
      {sentence ? (
        <div className={s.locked}>
          <span className={s.lockedTag}>Plan locked</span>
          <span>
            <Rich text={sentence} />
          </span>
        </div>
      ) : null}
      <p className={s.why}>Table 2 and the declared alternatives are on the canvas.</p>
      <div className={s.footer}>
        <span className={s.footNote}>The whole methods section, as recorded, is one press away.</span>
        <button type="button" className={s.btn} onClick={onDoc}>
          Read the whole methods section
        </button>
      </div>
    </>
  );
}

/** Any other item: its sentences, what it waits on, or what only the author can supply, with the
 *  way back to the open slot (no item is a dead end). */
export function ItemCard({
  tier,
  title,
  sentences,
  waitingOn,
  ask,
  onBack,
  backLabel,
}: {
  tier: string;
  title: string;
  sentences: string[];
  waitingOn: string[];
  ask?: string;
  onBack: (() => void) | null;
  backLabel: string;
}) {
  return (
    <>
      {tier === "stated" || tier === "engine" ? (
        sentences.map((t) => <Receipt key={t} text={t} />)
      ) : tier === "waiting" ? (
        <p className={s.draft}>
          <Blank>{title}</Blank> waits on {waitingOn.join(" and ") || "an earlier answer"}.
        </p>
      ) : tier === "author" ? (
        <p className={s.draft}>
          <Blank>{ask ?? title}</Blank>
        </p>
      ) : (
        <p className={s.draft}>{sentences.length ? <Rich text={sentences.join(" ")} /> : title}</p>
      )}
      <p className={s.why}>
        {tier === "engine"
          ? "Stated by the engine from your values; no question was asked."
          : tier === "author"
            ? "Only you can supply this; it is a blank in the export until you do."
            : tier === "silent"
              ? "This changes no number; it appears only in the export."
              : tier === "waiting"
                ? "It opens when what it waits on is recorded."
                : "Recorded."}
      </p>
      {onBack ? (
        <div className={s.footer}>
          <span className={s.footNote} />
          <button type="button" className={s.btnPrimary} onClick={onBack} data-testid="back-to-objective">
            {backLabel}
          </button>
        </div>
      ) : null}
    </>
  );
}
