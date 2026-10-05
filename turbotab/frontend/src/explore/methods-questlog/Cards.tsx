/**
 * The current objective as one focused card. Each card leads with the sentence it will write into
 * the methods section (its blanks dashed until recorded), then the slot itself: the engine's guess,
 * its evidence, and the answers it offers. Concepts are taught in full at their first encounter and
 * condensed to an expandable phrase afterwards (BLUEPRINT §11.4, the tutorial standard).
 */
import { Fragment, useState, type ReactNode } from "react";
import { Rich } from "../../components/stage/text";
import { INF, isPreview, teaching, term, type CapturedPreview } from "./data";
import { readingSlots, unlockedBlock, VALUE_WORDS, type ReadingSlot } from "./readings";
import s from "./questlog.module.css";

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

export function Concepts({
  seen,
  fresh,
  open,
  firstAt,
}: {
  seen: string[];
  fresh: string[];
  open?: string | null;
  firstAt: Record<string, string>;
}) {
  const [expanded, setExpanded] = useState<string | null>(open ?? null);
  const shown = expanded ? term(expanded) : null;
  return (
    <div className={s.concepts} data-purpose="concepts" data-testid="concepts">
      <span>Concepts here:</span>
      {seen.map((c) => (
        <button
          key={c}
          type="button"
          className={s.concept}
          aria-expanded={expanded === c}
          onClick={() => setExpanded(expanded === c ? null : c)}
        >
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

// ── M1: the roles ────────────────────────────────────────────────────────────

const ROLE_ORDER = ["energy", "exposure", "covariate", "identifier", "flag"];
const ROLE_NAME: Record<string, string> = {
  energy: "Energy",
  exposure: "Exposures",
  covariate: "Covariates",
  identifier: "Identifier",
  flag: "Flags",
};

export function RolesCard() {
  const t = teaching("roles")!;
  const cols = INF.roles.columns;
  const by = ROLE_ORDER.map((r) => ({ role: r, cols: cols.filter((c) => c.proposed === r) })).filter((g) => g.cols.length);
  const below = cols.filter((c) => c.confidence !== "high").length;
  const count = (r: string) => cols.filter((c) => c.proposed === r).length;
  return (
    <>
      <p className={s.draft}>
        Column roles were set for <Blank>{cols.length}</Blank> columns: energy <Blank>kcal</Blank>; exposures{" "}
        <Blank>{count("exposure")}</Blank>; covariates <Blank>{count("covariate")}</Blank>; identifier{" "}
        <Blank>SEQN</Blank>; flags <Blank>{count("flag")}</Blank>.
      </p>
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
                <span className={s.hint}>
                  {highs === g.cols.length ? "all high confidence" : `${g.cols.length - highs} below high`}
                </span>
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
      <div className={s.footer}>
        <span className={s.footNote}>
          The {below} below high confidence then wait for their own confirmation under Column readings.
        </span>
        <button type="button" className={s.btnPrimary}>
          Record these roles
        </button>
      </div>
    </>
  );
}

// ── M2 · M9: the readings ────────────────────────────────────────────────────

function answerWords(e: { decision: Record<string, unknown> | null }): string {
  const d = e.decision ?? {};
  if (d.kind === "set_column_unit") return Number(d.days) === 1 ? "one day's intake, kcal" : `${String(d.days)} days' total`;
  const v = String(d.value ?? "");
  return VALUE_WORDS[v] ?? (v === "covariate" ? "a covariate" : v);
}

function ReadingRow({ slot, focus, onFocus }: { slot: ReadingSlot; focus: boolean; onFocus: (id: string) => void }) {
  const family = slot.columns.length > 1;
  const [guess, ...alts] = slot.options;
  const question = family
    ? `${slot.columns.length} columns like \`${slot.columns[0]}\`: ${slot.guessWords}?`
    : `\`${slot.columns[0]}\`: ${slot.guessWords}?`;
  return (
    <li
      className={s.row}
      data-focus={focus || undefined}
      onPointerEnter={() => onFocus(slot.id)}
      onFocus={() => onFocus(slot.id)}
      data-testid="reading"
    >
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
          <button type="button" className={focus ? s.btnPrimary : s.btn} title={guess.label.replace(/`/g, "")}>
            {family ? `Confirm the ${slot.columns.length}` : `Yes, ${answerWords(guess)}`}
          </button>
        ) : null}
        {alts.slice(0, 1).map((a) => (
          <button key={a.label} type="button" className={s.btnQuiet} title={a.label.replace(/`/g, "")}>
            No: {answerWords(a)}
          </button>
        ))}
      </div>
    </li>
  );
}

export function ReadingsCard({
  focus,
  onFocus,
  confirmed = [],
  unlocked = false,
}: {
  focus: string | null;
  onFocus: (id: string) => void;
  confirmed?: string[];
  unlocked?: boolean;
}) {
  const all = readingSlots();
  const done = new Set(INF.singles.map((x) => String(x.decision.column)));
  const slots = all.filter((r) => !(confirmed.length && r.kind === "role" && r.columns.length === 1 && done.has(r.columns[0]!)));
  const byConsumer = new Map<string, ReadingSlot[]>();
  for (const r of slots) byConsumer.set(r.consumer, [...(byConsumer.get(r.consumer) ?? []), r]);
  const fitAsk = INF.asks.models_0.message;
  const why = fitAsk.slice(fitAsk.indexOf("The fit waits"), fitAsk.indexOf("Tell me about")).trim();
  const block = unlockedBlock();
  const groups = new Map<string, string[]>();
  for (const it of block.items) {
    const k = VALUE_WORDS[it.value] ?? it.value;
    groups.set(k, [...(groups.get(k) ?? []), it.column]);
  }
  return (
    <>
      <p className={s.draft}>Tell me about these columns.</p>
      <p className={s.why}>
        <Rich text={why} />
      </p>
      {confirmed.length ? (
        <div className={s.rows}>
          {confirmed.map((c) => (
            <p key={c} className={s.receipt}>
              <Rich text={c} />
            </p>
          ))}
        </div>
      ) : null}
      {unlocked ? (
        <section className={s.unlock} data-purpose="mastery_block" data-testid="unlock">
          <div className={s.unlockHead}>
            <span className={s.teachNew}>Shortcut unlocked</span>
            <span className={s.hint}>after {confirmed.length} single confirmations</span>
          </div>
          <span className={s.unlockTitle}>Confirm the rest as one block</span>
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
            <span className={s.footNote}>
              Settles exactly these {block.items.length} readings, each as shown; nothing else. One undo reverts it.
            </span>
            <button type="button" className={s.btnPrimary} data-testid="block-confirm">
              {block.label}
            </button>
          </div>
        </section>
      ) : null}
      {[...byConsumer].map(([consumer, rows]) => (
        <Fragment key={consumer}>
          <div className={s.subhead}>
            <span className={s.subTitle}>Needed by {consumer}</span>
            <span className={s.hint}>
              {rows.length} {rows.length === 1 ? "reading" : "readings"}
            </span>
          </div>
          <ul className={s.rows}>
            {rows.map((r) => (
              <ReadingRow key={r.id} slot={r} focus={focus === r.id} onFocus={onFocus} />
            ))}
          </ul>
        </Fragment>
      ))}
    </>
  );
}

// ── M3 · M8a: the exposure and its effect ────────────────────────────────────

export function EstimandCard({ hovered, teach }: { hovered: string; teach: boolean }) {
  const card = INF.estimand_card;
  const t = teaching("estimand")!;
  const energy = card.exposures.filter((e) => e.energy_contrast);
  const other = card.exposures.filter((e) => !e.energy_contrast);
  const unconfirmed = (INF.moments.m3.view.interview.find((x) => x.key === "estimand")?.ask?.groups ?? []).flatMap((g) => g.columns);
  const measure = card.measures;
  const coefficient = teaching("energy_adjustment")?.drawer?.sections.find((x) => x.heading === "What the coefficient means");
  return (
    <>
      <p className={s.draft}>
        The analysis estimates the <Blank filled>total effect</Blank> of <Blank>{hovered ? `${hovered}?` : "which exposure?"}</Blank> on{" "}
        <code className="v">glucose</code> (<Blank filled>a substitution</Blank>: in place of other energy sources at fixed total energy), as
        a <Blank filled>difference in the mean outcome</Blank> per unit of the exposure.
      </p>
      <p className={s.why}>
        <Rich text={`${t.question} ${t.one_liner}`} />
      </p>
      <div className={s.slotGrid}>
        <span className={s.slotName}>Exposure</span>
        <div>
          <div className={s.pickLabel}>
            Not guessed: the exposure is your question. Offered from your roles:
          </div>
          <div className={s.pick} role="radiogroup" aria-label="Exposure">
            {energy.map((e) => (
              <button key={e.column} type="button" className={s.pickChip} data-focus={hovered === e.column || undefined} role="radio" aria-checked={false}>
                {e.column}
              </button>
            ))}
            <button type="button" className={s.pickChip} role="radio" aria-checked={false} style={{ fontFamily: "var(--sans)" }}>
              all {card.family?.n ?? energy.length}, as a family
            </button>
            {other.map((e) => (
              <button key={e.column} type="button" className={s.pickChip} role="radio" aria-checked={false}>
                {e.column}
              </button>
            ))}
          </div>
          <div className={s.pickLabel}>
            <Rich
              text={`Not offered until its role is confirmed: ${unconfirmed
                .slice(0, 3)
                .map((c) => `\`${c}\``)
                .join(", ")} and ${unconfirmed.length - 3} more (Column readings).`}
            />
          </div>
        </div>
        <span className={s.slotName}>Measure</span>
        <div className={s.cards}>
          {measure.map((m) => (
            <div key={m.measure} className={s.optCard} data-on={m.rank === 1 || undefined} data-refused={!m.fitted || undefined}>
              <span className={s.optHead}>
                <span className={s.optionLabel}>{m.label}</span>
                <span className={m.fitted ? s.optionTagGuess : s.optionTag}>{m.fitted ? (m.rank === 1 ? "first" : "") : "not fitted"}</span>
              </span>
              <span className={s.optWhy}>{m.reason}</span>
            </div>
          ))}
        </div>
        <span className={s.slotName}>Effect</span>
        <div className={s.cards}>
          {card.effects.map((e, i) => (
            <div key={e.effect} className={s.optCard} data-on={i === 0 || undefined}>
              <span className={s.optHead}>
                <span className={s.optionLabel}>{e.label}</span>
                <span className={s.optionTagGuess}>{i === 0 ? "guess" : ""}</span>
              </span>
              <span className={s.optWhy}>{e.consequence}</span>
            </div>
          ))}
        </div>
        <span className={s.slotName}>Contrast</span>
        <div className={s.cards}>
          {card.contrasts.map((c, i) => (
            <div key={c.contrast} className={s.optCard} data-on={i === 0 || undefined}>
              <span className={s.optHead}>
                <span className={s.optionLabel}>{c.label}</span>
                <span className={s.optionTagGuess}>{i === 0 ? "guess" : ""}</span>
              </span>
              <span className={s.optWhy}>{c.consequence}</span>
            </div>
          ))}
        </div>
        {teach ? (
          <>
            <span />
            <FirstEncounter
              name="substitution"
              more={coefficient?.body}
              source={coefficient ? `${coefficient.evidence.status} · ${coefficient.evidence.source}` : undefined}
            />
          </>
        ) : null}
      </div>
      <div className={s.footer}>
        <span className={s.footNote}>No estimate is shown until this is recorded.</span>
        <button type="button" className={s.btnPrimary} disabled={!hovered}>
          {hovered ? `Record ${hovered} as the exposure` : "Choose an exposure"}
        </button>
      </div>
    </>
  );
}

// ── M4: the adjustment set ───────────────────────────────────────────────────

const ANSWERS = ["yes", "no", "unknown"] as const;
const ANSWER_WORDS: Record<string, string> = { yes: "yes", no: "no", unknown: "don't know" };
const ANSWER_SHORT: Record<string, string> = { yes: "yes", no: "no", unknown: "?" };
const FIELDS = ["causes_exposure", "causes_outcome", "after_exposure"] as const;

function Seg({ value, label }: { value: string | null; label: string }) {
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
          aria-label={ANSWER_WORDS[a]}
        >
          {ANSWER_SHORT[a]}
        </button>
      ))}
    </div>
  );
}

export function AdjustmentCard({ focus, onFocus }: { focus: string; onFocus: (k: string) => void }) {
  const card = INF.adjustment_card;
  const t = teaching("adjustment")!;
  const n = card.groups.reduce((m, g) => m + g.columns.length, 0);
  const short: Record<string, string> = {
    causes_exposure: "Causes `sugar`?",
    causes_outcome: "Causes `glucose`?",
    after_exposure: "Changed by `sugar`?",
  };
  return (
    <>
      <p className={s.draft}>
        For the effect of <code className="v">{card.exposure}</code>, by the disjunctive cause criterion: <Blank>{n} covariates</Blank> are
        confounders, adjusted for, or left out.
      </p>
      <p className={s.why}>
        <Rich text={t.one_liner} />
      </p>
      <ul className={s.rows}>
        {card.groups.map((g) => (
          <li
            key={g.key}
            className={s.group}
            data-focus={focus === g.key || undefined}
            onPointerEnter={() => onFocus(g.key)}
            data-testid="adjust-group"
          >
            <div className={s.groupHead}>
              <span className={s.groupName}>
                {g.label} <span className={s.count}>{g.columns.length}</span>
              </span>
              {g.derived_words ? <span className={s.derived}>→ {g.derived_words}</span> : <span className={s.hint}>each asked</span>}
            </div>
            {g.guess ? (
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
                      <Rich text={short[f]!} />
                      <b>{ANSWER_WORDS[g.guess![f] ?? ""] ?? "—"}</b>
                    </span>
                  ))}
                </div>
                <div className={s.groupFoot}>
                  <span className={`${s.evidence} ${s.clamp}`} title={g.reason}>
                    <Rich text={`Guessed: ${g.reason.replace(/ \(NUTRITION_PACK[^)]*\)\)?/, " (NUTRITION_PACK §08)")}`} />
                  </span>
                  <button type="button" className={focus === g.key ? s.btnPrimary : s.btn}>
                    {g.derived === "confounder" ? `Confirm the ${g.columns.length} as confounders` : `Confirm the ${g.columns.length}: ${g.derived_words}`}
                  </button>
                </div>
              </>
            ) : (
              <>
                <span className={s.evidence}>{g.reason}</span>
                <div className={s.colRows}>
                  <span />
                  {FIELDS.map((f) => (
                    <span key={f} className={s.colHead}>
                      <Rich text={short[f]!} />
                    </span>
                  ))}
                  {g.columns.map((c) => (
                    <Fragment key={c}>
                      <code className="v" style={{ justifySelf: "start" }}>
                        {c}
                      </code>
                      {FIELDS.map((f) => (
                        <Seg key={f} value={null} label={`${c}: ${card.questions[f]}`} />
                      ))}
                    </Fragment>
                  ))}
                </div>
              </>
            )}
          </li>
        ))}
      </ul>
      <span className={s.hint}>Derived by {card.source}.</span>
    </>
  );
}

// ── M5 · M8b: editing a stated phrase ────────────────────────────────────────

const ENERGY_PREVIEW: Record<string, string> = {
  standard: "energy_standard",
  residual: "energy_residual",
  residual_energy_dropped: "energy_residual_energy_dropped",
  density_multivariate: "energy_density_multivariate",
  density: "energy_density",
  partition: "energy_partition",
  all_components: "energy_all_components",
  none: "energy_none",
};

export function energyPreview(value: string): CapturedPreview | undefined {
  return INF.previews[ENERGY_PREVIEW[value] ?? ""];
}

export function PhraseCard({
  sentence,
  hovered,
  onHover,
  conceptOpen,
}: {
  sentence: string;
  hovered: string;
  onHover: (v: string) => void;
  conceptOpen: string | null;
}) {
  const t = teaching("energy_adjustment")!;
  const phrase = "the standard (multivariate) model";
  const at = sentence.indexOf(phrase);
  const order = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density", "partition", "all_components", "none"];
  const options = order.map((v) => t.options.find((o) => o.value === v)!).filter(Boolean);
  const residualTerm = term("residual method");
  return (
    <>
      <p className={s.draft}>
        <Rich text={sentence.slice(0, at)} />
        <span className={s.phraseOpen}>{phrase}</span>
        <Rich text={sentence.slice(at + phrase.length)} />
      </p>
      <p className={s.why}>
        <Rich text={t.one_liner} />
      </p>
      <Concepts
        seen={["substitution", "estimand"]}
        fresh={hovered === "residual" ? ["residual method"] : []}
        open={conceptOpen}
        firstAt={{ substitution: "Exposure and effect", estimand: "Exposure and effect" }}
      />
      <ul className={s.options} role="listbox" aria-label="Energy adjustment">
        {options.map((o) => {
          const p = energyPreview(o.value);
          const refused = p && !isPreview(p.body) ? p.body.error : null;
          const recorded = o.value === "standard";
          const on = hovered === o.value;
          return (
            <li key={o.value}>
              <div
                className={s.option}
                data-focus={on || undefined}
                data-refused={refused ? true : undefined}
                role="option"
                aria-selected={on}
                tabIndex={-1}
                onPointerEnter={() => onHover(o.value)}
              >
                <span className={s.radio} data-recorded={recorded || undefined} data-on={on && !recorded ? true : undefined} />
                <span className={s.optionLabel}>{o.label}</span>
                <span className={recorded ? s.optionTagNow : s.optionTag}>
                  {recorded ? "recorded" : refused ? "not available here" : on ? "on the canvas" : ""}
                </span>
                <span className={s.optionWhy}>
                  {refused ? <Rich text={refused.message} /> : o.consequence}
                  {on && o.value === "residual" && residualTerm ? (
                    <span style={{ display: "block", marginTop: 6 }}>
                      <b style={{ color: "var(--accent)", fontFamily: "var(--sans)", fontSize: 11, letterSpacing: ".08em" }}>
                        NEW ·{" "}
                      </b>
                      <Rich text={residualTerm.definition} /> Watch it on the canvas: fit on energy, then keep the residual.
                    </span>
                  ) : null}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
      <div className={s.footer}>
        <span className={s.footNote}>↑ ↓ move through the methods · Space flips the canvas · Enter records the change</span>
        <button type="button" className={s.btnPrimary}>
          Change to {t.options.find((o) => o.value === hovered)?.label ?? "…"}
        </button>
      </div>
    </>
  );
}
