/**
 * The node card: what clicking a node opens. An asked node is a slot — its guess with the evidence
 * for it, the options with their customary and sound labels — and a stated node is a phrase with
 * its alternatives. Hovering an option plays it on the canvas and redraws the map as it would be;
 * recording says, at the control, the sentence the engine wrote for it.
 */
import { Children, useState, type ReactNode } from "react";
import { Rich } from "../../components/stage/text";
import type { Scene } from "./Canvas";
import { FX, INF, PRED, fmtInt, type Answer3, type Preview } from "./fixture";
import {
  ADJ_FIELDS,
  EXPOSURE,
  NODE_TEACH,
  NODE_TERMS,
  NODE_TITLE,
  REGIONS,
  SINGLES_TO_UNLOCK,
  adjustmentPreview,
  answered,
  blockOffer,
  codesOffer,
  derive,
  estimandPreview,
  fitFor,
  formPreview,
  lockSentence,
  guessTriple,
  lockReady,
  model1Preview,
  previewFor,
  reduce,
  roleSingles,
  statedReason,
  truthTriple,
  type Action,
  type Answers,
  type NodeId,
  type Purpose,
  type Triple,
} from "./model";
import { teachLabel } from "./MapView";
import { Opts, type Opt } from "./Opts";
import { Ask, Concepts, Term } from "./Teach";
import c from "./screen.module.css";

export interface Ctx {
  purpose: Purpose;
  a: Answers;
  /** Record: the receipt is the engine's sentence, shown at the control. */
  record: (e: Action, node: NodeId) => void;
  /** Put an option on the canvas (and the map's preview); null clears it. */
  show: (key: string | null, scene: Scene | null, preview?: Answers | null) => void;
  shown: string | null;
  first: boolean;
  firstMet: Record<string, NodeId>;
  receipt: { node: NodeId; texts: string[] } | null;
  onPointerRecord: () => void;
  keysUnlocked: boolean;
  next: NodeId | null;
  goNext: () => void;
  focus: (n: NodeId) => void;
}

const regionOf = (n: NodeId, p: Purpose) => REGIONS[p].find((r) => r.nodes.includes(n) || (r.nodes.includes("source") && ["lens", "outcome", "purpose", "grain"].includes(n)));

export function sceneOf(group: string, key: string, label: string, p: Preview | null | undefined, recorded = false): Scene | null {
  if (!p) return null;
  if (p.refusal) return { kind: "refusal", group, key, label, message: p.refusal.message, exits: p.refusal.exits.map((x) => x.label) };
  const res = p.result!;
  if (!res.views.length) return { kind: "note", group, key, label, note: res.note ?? "", basis: res.basis };
  return { kind: "preview", group, key, label, result: res, recorded };
}

/** The canvas when no preview was taken on the answers' state: said so, never borrowed. */
export function notTaken(group: string, key: string, label: string): Scene {
  return {
    kind: "note",
    group,
    key,
    label,
    note: "This prototype took this preview on the scenario's own plan only; the map above draws the change.",
    aside: "nothing is recorded",
  };
}

function Shell({ ctx, node, children, tag }: { ctx: Ctx; node: NodeId; children: ReactNode; tag?: ReactNode }) {
  const region = regionOf(node, ctx.purpose);
  const terms = NODE_TERMS[node] ?? [];
  const receipt = ctx.receipt?.node === node ? ctx.receipt : null;
  return (
    <article className={c.card} data-testid={`card-${node}`} aria-labelledby={`card-title-${node}`}>
      <header className={c.cardHead}>
        <p className={c.cardKicker}>
          {region ? (
            <>
              <span>{region.title}</span>
              <span className={c.items}>{region.items}</span>
            </>
          ) : null}
        </p>
        <h1 id={`card-title-${node}`} className={c.cardTitle}>
          {NODE_TITLE[node]}
          {tag}
        </h1>
      </header>
      {receipt ? (
        <div className={c.receipt} role="status" data-testid="receipt">
          <span className={c.receiptKicker}>Recorded</span>
          {receipt.texts.map((t) => (
            <p key={t}>
              <Rich text={t} />
            </p>
          ))}
          {ctx.next && ctx.next !== node ? (
            <button type="button" className={c.nextInline} onClick={ctx.goNext} data-testid="receipt-next">
              Next asked: {NODE_TITLE[ctx.next]} →
            </button>
          ) : null}
        </div>
      ) : null}
      {/* the question first, then the concepts it uses (in full where first met), then the slot */}
      {Children.toArray(children)[0]}
      <Concepts node={node} terms={terms} firstMet={ctx.firstMet} />
      {Children.toArray(children).slice(1)}
      {ctx.keysUnlocked ? (
        <p className={c.keys} aria-hidden="true" data-testid="keys-hint">
          <kbd>↑</kbd>
          <kbd>↓</kbd> preview · <kbd>Space</kbd> flip · <kbd>Enter</kbd> record
        </p>
      ) : null}
    </article>
  );
}

function TierTag({ tier }: { tier: "asked" | "stated" | "recorded" | "waiting" | "silent" | "locked" }) {
  const word = {
    asked: "asked of you",
    stated: "written in · changeable",
    recorded: "recorded",
    waiting: "waiting",
    silent: "silent · export only",
    locked: "locked",
  }[tier];
  return (
    <span className={c.tierTag} data-tier={tier}>
      {word}
    </span>
  );
}

function Undo({ ctx, node }: { ctx: Ctx; node: NodeId }) {
  if (ctx.a.locked || !answered(ctx.a, node, ctx.purpose)) return null;
  return (
    <button type="button" className={c.undo} onClick={() => ctx.record({ type: "undo", node }, node)} data-testid={`undo-${node}`}>
      Undo this answer
    </button>
  );
}

export function NodeCard({ node, ctx }: { node: NodeId; ctx: Ctx }) {
  switch (node) {
    case "source":
      return <SourceCard ctx={ctx} />;
    case "readings":
      return <ReadingsCard ctx={ctx} />;
    case "exclusions":
      return ctx.purpose === "inference" ? <ExclusionsCard ctx={ctx} /> : <PredExclusionsCard ctx={ctx} />;
    case "exposure":
      return <ExposureCard ctx={ctx} />;
    case "adjustment":
      return <AdjustmentCard ctx={ctx} />;
    case "energy":
      return <EnergyCard ctx={ctx} />;
    case "form":
      return <FormCard ctx={ctx} />;
    case "missing":
      return <MissingCard ctx={ctx} />;
    case "model1":
      return <Model1Card ctx={ctx} />;
    case "lock":
    case "estimate":
    case "matter":
      return <ResultsCard ctx={ctx} node={node} />;
    case "seal":
      return <SealCard ctx={ctx} />;
    case "p_missing":
      return <PredMissingCard ctx={ctx} />;
    case "p_seal":
      return <PredSealCard ctx={ctx} />;
    case "p_energy":
    case "p_models":
    case "p_score":
      return <PredWaitingCard ctx={ctx} node={node} />;
    default:
      return <StatedCard ctx={ctx} node={node} />;
  }
}

// ── the table ───────────────────────────────────────────────────────────────

function SourceCard({ ctx }: { ctx: Ctx }) {
  const rd = INF.readings.read_from_data;
  return (
    <Shell ctx={ctx} node="source">
      <p className={c.lede}>
        <code className="v">{FX.meta.file}</code> · <span className="num">{fmtInt(FX.meta.rows)}</span> rows ×{" "}
        <span className="num">{FX.meta.cols}</span> columns
      </p>
      <h3 className={c.sub}>Read from your data, no question asked ({rd.length})</h3>
      <ul className={c.readList}>
        {rd.map((x) => (
          <li key={`${x.kind}:${x.column}`}>
            <code className="v">{x.column}</code> {x.words}
            <span className={c.evidence}>
              <Rich text={x.evidence} />
            </span>
          </li>
        ))}
      </ul>
    </Shell>
  );
}

// ── readings ────────────────────────────────────────────────────────────────

function ReadingsCard({ ctx }: { ctx: Ctx }) {
  const { a, purpose } = ctx;
  const roles = INF.readings.items.filter((i) => i.reading === "role");
  // the code-or-amount readings: under inference the fit's card asks them in a block of their
  // own; under prediction its drive asks them in the role readings' one block
  const inf = purpose === "inference";
  const codes = inf ? INF.readings.items.filter((i) => i.reading === "code_or_count") : PRED.readings.codes;
  const codesBy = inf ? INF.readings.codes.consumer : PRED.readings.consumer;
  const items = [...roles, ...codes];
  const unit = INF.readings.unit;
  const offer = blockOffer(a, purpose);
  const codeOffer = inf ? codesOffer(a) : null;
  const singles = roleSingles(a).length;
  const [showRead, setShowRead] = useState(false);
  const tier = answered(a, "readings", purpose) ? "recorded" : "asked";
  const valueWords = (i: (typeof items)[number]) => i.words ?? `a ${i.value}`;
  const one = (it: (typeof items)[number]) => {
    const done = a.readings[it.key];
    return (
      <li key={it.key} className={c.reading} data-done={done || undefined} data-testid={`reading-${it.key}`}>
        <p className={c.readingHead}>
          <code className="v">{it.column}</code>: {valueWords(it)}?
          {it.confidence ? <span className={c.conf}>{it.confidence}</span> : null}
        </p>
        <p className={c.evidence}>
          <Rich text={it.evidence} />
        </p>
        {done ? (
          <span className={c.done}>{done === "single" ? "confirmed on its own" : "confirmed in the block"}</span>
        ) : (
          <button
            type="button"
            className={c.confirmSmall}
            onClick={() => ctx.record({ type: "reading", key: it.key }, "readings")}
            data-testid={`confirm-${it.key}`}
          >
            Confirm
          </button>
        )}
      </li>
    );
  };
  return (
    <Shell ctx={ctx} node="readings" tag={<TierTag tier={tier} />}>
      <Ask teach="roles" first={ctx.first} question="Tell me about these columns" />
      <section className={c.group}>
        <h3 className={c.sub}>
          Read by {unit.consumer} <span className={c.subNote}>— confirmed on its own</span>
        </h3>
        <div className={c.reading} data-done={a.unit || undefined} data-testid="reading-unit">
          <p className={c.readingHead}>
            <code className="v">{unit.column}</code>: {unit.guess_words}?
          </p>
          <p className={c.evidence}>
            <Rich text={unit.evidence} />
          </p>
          {a.unit ? (
            <span className={c.done}>confirmed</span>
          ) : (
            <div className={c.readingActs}>
              <button type="button" className={c.confirm} onClick={() => ctx.record({ type: "unit" }, "readings")} data-testid="confirm-unit">
                <Rich text={unit.confirm} />
              </button>
              <span className={c.alts}>
                or{" "}
                {unit.alternatives.map((x, i) => (
                  <span key={x} className={c.alt} title="Not captured in this prototype: the screens' bounds would be read in it">
                    {i ? " · " : null}
                    <Rich text={x} />
                  </span>
                ))}
              </span>
            </div>
          )}
        </div>
      </section>
      <section className={c.group}>
        <h3 className={c.sub}>
          Proposed below high confidence <span className={c.subNote}>— each waits for you</span>
        </h3>
        {offer ? (
          <div className={c.block} data-testid="block-offer">
            <p className={c.blockHead}>
              <span className={c.unlock}>Unlocked</span> after {SINGLES_TO_UNLOCK} single confirmations: confirm the other{" "}
              {offer.keys.length} together. It settles exactly these, each as shown:
            </p>
            <ul className={c.blockList}>
              {offer.keys.map((k) => {
                const it = items.find((i) => i.key === k)!;
                return (
                  <li key={k}>
                    <code className="v">{it.column}</code> {valueWords(it)}
                  </li>
                );
              })}
            </ul>
            <button type="button" className={c.confirm} onClick={() => ctx.record({ type: "block", purpose }, "readings")} data-testid="confirm-block">
              Confirm these {offer.keys.length} as shown
            </button>
          </div>
        ) : null}
        <ul className={c.readings}>{roles.map(one)}</ul>
        {!offer && items.some((i) => !a.readings[i.key]) && singles < SINGLES_TO_UNLOCK ? (
          <p className={c.hint}>
            Confirm {SINGLES_TO_UNLOCK - singles} more on {SINGLES_TO_UNLOCK - singles === 1 ? "its" : "their"} own to unlock a block
            confirm for the rest.
          </p>
        ) : null}
      </section>
      {codes.length ? (
        <section className={c.group}>
          <h3 className={c.sub}>
            Read by {codesBy} <span className={c.subNote}>— amounts or codes</span>
          </h3>
          {codeOffer ? (
            <div className={c.block} data-testid="codes-offer">
              <p className={c.blockHead}>It settles exactly these, each as shown:</p>
              <ul className={c.blockList}>
                {codeOffer.keys.map((k) => {
                  const it = items.find((i) => i.key === k)!;
                  return (
                    <li key={k}>
                      <code className="v">{it.column}</code> {valueWords(it)}
                    </li>
                  );
                })}
              </ul>
              <button type="button" className={c.confirm} onClick={() => ctx.record({ type: "codes" }, "readings")} data-testid="confirm-codes">
                Confirm these {codeOffer.keys.length} as shown
              </button>
            </div>
          ) : null}
          <ul className={c.readings}>{codes.map(one)}</ul>
        </section>
      ) : null}
      <button type="button" className={c.disclose} aria-expanded={showRead} onClick={() => setShowRead((v) => !v)}>
        {showRead ? "Hide" : "Show"} what was read from your data ({INF.readings.read_from_data.length})
      </button>
      {showRead ? (
        <ul className={c.readList}>
          {INF.readings.read_from_data.map((x) => (
            <li key={`${x.kind}:${x.column}`}>
              <code className="v">{x.column}</code> {x.words}
              <span className={c.evidence}>
                <Rich text={x.evidence} />
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      <Undo ctx={ctx} node="readings" />
    </Shell>
  );
}

// ── exclusions (inference) ──────────────────────────────────────────────────

function ExclusionsCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const ex = INF.exclusions;
  const teach = FX.teaching.exclusions!;
  const order = ["none", "willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000", "sex_neutral_500_3500", "goldberg_schofield"];
  const options: Opt[] = order.map((k) => {
    const lab = ex.labels.options[k === "none" ? "keep_every_row" : k];
    const off = ex.offered.find((o) => o.key === k);
    const tOpt = teach.options.find((o) => o.value === k);
    const captured = k === "none" || k === "willett_2013_by_sex";
    const refused = ex.previews[k]?.refusal;
    return {
      key: k,
      label: lab?.label ?? tOpt?.label ?? k,
      line: tOpt?.consequence ?? null,
      data: off ? `−${fmtInt(off.affected)} of ${fmtInt(ex.n_base)}` : null,
      customary: lab?.customary,
      sound: lab?.sound,
      verdict: lab?.verdict,
      tags: k === ex.labels.customary_first ? [{ text: "the field's usual", tone: "usual" as const }] : [],
      blocked: refused
        ? refused.message
        : captured || !ex.sentences[k]
          ? null
          : INF.sensitivity.keys.includes(k)
            ? "This prototype captured the fits for keeping every row and for Willett 2013 only; this screen is previewed, and can be reported beside the primary."
            : "This prototype captured the fits for keeping every row and for Willett 2013 only; this screen is previewed only.",
    };
  });
  const recordedKey = a.exclusions;
  const show = (k: string) => {
    const lab = options.find((o) => o.key === k)!.label;
    ctx.show(k, sceneOf("exclusions", k, lab, previewFor("exclusions", k, a)), reduce(a, { type: "exclusions", key: k }));
  };
  const rec = (k: string, via: "pointer" | "key") => {
    if (via === "pointer") ctx.onPointerRecord();
    const opt = options.find((o) => o.key === k)!;
    if (opt.blocked || ex.previews[k]?.refusal || !ex.sentences[k]) return show(k);
    ctx.record({ type: "exclusions", key: k }, "exclusions");
  };
  return (
    <Shell ctx={ctx} node="exclusions" tag={<TierTag tier={recordedKey ? "recorded" : "asked"} />}>
      <Ask teach="exclusions" first={ctx.first} />
      {ex.labels.tension ? (
        <p className={c.tension}>
          <Rich text={ex.labels.tension} />
        </p>
      ) : null}
      <Opts options={options} shown={ctx.shown} recorded={recordedKey} onShow={show} onRecord={rec} label="Exclusions" />
      {recordedKey ? (
        <section className={c.group}>
          <h3 className={c.sub}>Report beside the primary</h3>
          <ul className={c.checks}>
            {INF.sensitivity.keys
              .filter((k) => k !== recordedKey)
              .map((k) => (
                <li key={k}>
                  <label className={c.checkRow}>
                    <input
                      type="checkbox"
                      checked={a.sensitivity.includes(k)}
                      disabled={!!a.locked}
                      onChange={() => ctx.record({ type: "sensitivity", key: k }, "exclusions")}
                      data-testid={`beside-${k}`}
                    />
                    {ex.labels.options[k]!.label}
                  </label>
                </li>
              ))}
          </ul>
        </section>
      ) : null}
      <Undo ctx={ctx} node="exclusions" />
    </Shell>
  );
}

// ── the exposure and its estimand ───────────────────────────────────────────

function ExposureCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const E = INF.estimand;
  const [exp, setExp] = useState("sugar");
  const [effect, setEffect] = useState("total");
  const [contrast, setContrast] = useState<string | null>(a.exposure ? "substitution" : null);
  const captured = exp === "sugar" && effect === "total" && contrast === "substitution";
  const measure = E.measures.find((m) => m.fitted)!;
  const guessed = E.exposures.find((x) => x.column === EXPOSURE)!;
  const nutrients = E.exposures.filter((x) => x.energy_contrast);
  const others = E.exposures.filter((x) => !x.energy_contrast);
  const label = "The exposure and its estimand";
  const note = () =>
    ctx.show("estimand", sceneOf("estimand", "estimand", label, estimandPreview(a), a.exposure) ?? notTaken("estimand", "estimand", label), reduce(a, { type: "exposure" }));
  return (
    <Shell ctx={ctx} node="exposure" tag={<TierTag tier={a.exposure ? "recorded" : "asked"} />}>
      <Ask teach="estimand" first={ctx.first} />
      <section className={c.group}>
        <h3 className={c.sub}>Exposure</h3>
        <div className={c.chips} role="radiogroup" aria-label="Exposure">
          {nutrients.map((x) => (
            <button
              key={x.column}
              type="button"
              role="radio"
              aria-checked={exp === x.column}
              className={c.chip}
              data-guess={x.column === "sugar" || undefined}
              onClick={() => setExp(x.column)}
              onFocus={note}
              disabled={!!a.locked}
            >
              <code>{x.column}</code>
              {x.column === "sugar" ? <span className={c.guessMark}>guess</span> : null}
            </button>
          ))}
        </div>
        <p className={c.evidence}>
          Guess <code className="v">{EXPOSURE}</code>, from your roles: <Rich text={guessed.evidence ?? ""} />
        </p>
        <p className={c.subNote}>
          {others.length} more columns could be the exposure ({others.map((o) => o.column).slice(0, 3).join(", ")}, …).
        </p>
      </section>
      <section className={c.group}>
        <h3 className={c.sub}>Which effect</h3>
        <div className={c.pair}>
          {E.effects.map((x) => (
            <button key={x.effect} type="button" className={c.pairOpt} aria-pressed={effect === x.effect} onClick={() => setEffect(x.effect)} disabled={!!a.locked}>
              <span className={c.pairLabel}>{x.label}</span>
              <span className={c.pairLine}>
                <Rich text={x.consequence} />
              </span>
            </button>
          ))}
        </div>
      </section>
      <section className={c.group}>
        <h3 className={c.sub}>
          <Term teach="energy_adjustment" term="substitution">
            Substitution
          </Term>{" "}
          or addition
        </h3>
        <p className={c.evidence}>
          <Rich text={E.which_contrast} />
        </p>
        <div className={c.pair}>
          {E.contrasts.map((x) => (
            <button key={x.contrast} type="button" className={c.pairOpt} aria-pressed={contrast === x.contrast} onClick={() => setContrast(x.contrast)} disabled={!!a.locked} data-testid={`contrast-${x.contrast}`}>
              <span className={c.pairLabel}>{x.label}</span>
              <span className={c.pairLine}>
                <Rich text={x.consequence} />
              </span>
            </button>
          ))}
        </div>
      </section>
      <p className={c.measure}>
        Measure: <b>{measure.label}</b> — <Rich text={measure.reason} />
      </p>
      {!a.exposure ? (
        <div className={c.actions}>
          <button
            type="button"
            className={c.primary}
            disabled={!contrast || !captured}
            onClick={() => ctx.record({ type: "exposure" }, "exposure")}
            data-testid="record-exposure"
          >
            Record the exposure and estimand
          </button>
          {!contrast ? <span className={c.actNote}>Say substitution or addition first: the engine asks.</span> : null}
          {contrast && !captured ? (
            <span className={c.actNote}>This prototype follows the total effect of `sugar` as a substitution; the full app records this too.</span>
          ) : null}
        </div>
      ) : null}
      <Undo ctx={ctx} node="exposure" />
    </Shell>
  );
}

// ── the adjustment set ──────────────────────────────────────────────────────

const ANS: Answer3[] = ["yes", "no", "unknown"];

function AdjustmentCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const A = INF.adjustment;
  const [open, setOpen] = useState<string | null>(() => A.groups.find((g) => !a.adjustment[g.key])?.key ?? null);
  const [draft, setDraft] = useState<Record<string, Triple>>({});
  if (!a.exposure) {
    return (
      <Shell ctx={ctx} node="adjustment" tag={<TierTag tier="waiting" />}>
        <Ask teach="adjustment" first={ctx.first} />
        <p className={c.waitNote}>
          The adjustment set is asked for one exposure: it waits on{" "}
          <button type="button" className={c.link} onClick={() => ctx.focus("exposure")}>
            the exposure and its estimand
          </button>
          .
        </p>
      </Shell>
    );
  }
  const triple = (g: string, col: string): Triple => draft[col] ?? a.adjustment[g]?.[col] ?? guessTriple(g) ?? (["unknown", "unknown", "unknown"] as Triple);
  const groupAnswers = (g: string) => Object.fromEntries(A.groups.find((x) => x.key === g)!.columns.map((col) => [col, triple(g, col)]));
  const previewWith = (g: string, answers: Record<string, Triple>) => {
    const label = `The adjustment set · ${A.groups.find((x) => x.key === g)!.label}`;
    ctx.show(`adj:${g}`, sceneOf("adjustment", g, label, adjustmentPreview(a, g, answers)) ?? notTaken("adjustment", g, label), reduce(a, { type: "adjust", group: g, answers }));
  };
  const setAnswer = (g: string, col: string, i: number, v: Answer3) => {
    const t = [...triple(g, col)] as Triple;
    t[i] = v;
    const next = { ...draft, [col]: t };
    setDraft(next);
    previewWith(g, { ...groupAnswers(g), [col]: t });
  };
  const n = A.groups.reduce((s, g) => s + g.columns.length, 0);
  return (
    <Shell ctx={ctx} node="adjustment" tag={<TierTag tier={answered(a, "adjustment") ? "recorded" : "asked"} />}>
      <Ask teach="adjustment" first={ctx.first} />
      <p className={c.lede}>
        {n} covariates in {A.groups.length} groups, asked by the disjunctive cause criterion ({A.source}). A group the pack
        guesses alike is one tap.
      </p>
      {A.groups.map((g) => {
        const rec = a.adjustment[g.key];
        const isOpen = open === g.key;
        const guess = guessTriple(g.key);
        return (
          <section key={g.key} className={c.adjGroup} data-open={isOpen || undefined} data-done={rec ? true : undefined} data-testid={`adj-${g.key}`}>
            <button type="button" className={c.adjHead} aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : g.key)}>
              <span className={c.adjLabel}>{g.label}</span>
              <span className={c.adjCount}>{g.columns.length}</span>
              {rec ? <span className={c.done}>recorded</span> : guess ? <span className={c.guessMark}>guess: {g.derived_words}</span> : <span className={c.conf}>no guess · each asked</span>}
            </button>
            {isOpen ? (
              <div className={c.adjBody}>
                <p className={c.evidence}>
                  <Rich text={g.reason} />
                </p>
                <table className={c.adjTable}>
                  <colgroup>
                    <col style={{ width: 100 }} />
                    <col style={{ width: 58 }} />
                    <col style={{ width: 58 }} />
                    <col style={{ width: 58 }} />
                    <col />
                  </colgroup>
                  <thead>
                    <tr>
                      <th />
                      {ADJ_FIELDS.map((f) => (
                        <th key={f} scope="col" title={A.questions[f]}>
                          {f === "causes_exposure" ? "causes sugar?" : f === "causes_outcome" ? "causes glucose?" : "after sugar?"}
                        </th>
                      ))}
                      <th scope="col">derives</th>
                    </tr>
                  </thead>
                  <tbody>
                    {g.columns.map((col) => {
                      const t = triple(g.key, col);
                      const d = derive(t);
                      return (
                        <tr key={col}>
                          <th scope="row">
                            <code className="v">{col}</code>
                          </th>
                          {ADJ_FIELDS.map((f, i) => (
                            <td key={f}>
                              <span
                                className={c.seg3}
                                role="radiogroup"
                                aria-label={`${col}: ${A.questions[f]}`}
                                data-unanswered={!rec && !guess && !draft[col] ? true : undefined}
                              >
                                {ANS.map((v) => (
                                  <button
                                    key={v}
                                    type="button"
                                    role="radio"
                                    aria-checked={t[i] === v}
                                    aria-label={v}
                                    title={v}
                                    disabled={!!a.locked}
                                    onClick={() => setAnswer(g.key, col, i, v)}
                                  >
                                    {v === "yes" ? "Y" : v === "no" ? "N" : "?"}
                                  </button>
                                ))}
                              </span>
                            </td>
                          ))}
                          <td className={c.derived} data-role={d.role} title={d.why}>
                            {d.words}
                            {d.adjusted ? "" : d.secondary ? " · beside" : " · out"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className={c.questions}>
                  {ADJ_FIELDS.map((f) => (
                    <span key={f}>{A.questions[f]} </span>
                  ))}
                </p>
                {g.key === "unguessed" ? <MediatorKeep ctx={ctx} /> : null}
                {!a.locked ? (
                  <div className={c.actions}>
                    <button
                      type="button"
                      className={c.primary}
                      disabled={!guess && g.columns.some((col) => !draft[col] && !rec)}
                      onClick={() => {
                        const answers = groupAnswers(g.key);
                        ctx.record({ type: "adjust", group: g.key, answers }, "adjustment");
                        setDraft({});
                        const nextOpen = A.groups.find((x) => x.key !== g.key && !a.adjustment[x.key])?.key ?? null;
                        setOpen(nextOpen);
                      }}
                      data-testid={`record-adj-${g.key}`}
                    >
                      {guess && g.columns.every((col) => triple(g.key, col).join() === guess.join()) ? `Confirm the guess for ${g.columns.length}` : `Record these ${g.columns.length} answers`}
                    </button>
                    {!guess && g.columns.some((col) => !draft[col] && !rec) ? (
                      <button
                        type="button"
                        className={c.secondaryBtn}
                        title="Fill each column with the fixture's declared causal truth (truths.py), as its author answers"
                        onClick={() => {
                          const filled = Object.fromEntries(g.columns.map((col) => [col, truthTriple(col)!]));
                          setDraft((d) => ({ ...d, ...filled }));
                          previewWith(g.key, filled);
                        }}
                        data-testid="adj-truth"
                      >
                        Answer as the fixture's author
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        );
      })}
      <Undo ctx={ctx} node="adjustment" />
    </Shell>
  );
}

/** The leash: a mediator kept in a total-effect set is blocked and recorded (the engine's refusal). */
function MediatorKeep({ ctx }: { ctx: Ctx }) {
  const m = INF.adjustment.mediator_kept;
  return (
    <button
      type="button"
      className={c.linkSmall}
      onClick={() =>
        ctx.show("keep-hdl", { kind: "refusal", group: "adjustment", key: "keep-hdl", label: "Keep `hdl` in the primary model", message: m.message, exits: m.exits })
      }
      data-testid="keep-mediator"
    >
      Keep <code className="v">hdl</code> in the primary model anyway?
    </button>
  );
}

// ── energy ──────────────────────────────────────────────────────────────────

function EnergyCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const E = INF.energy;
  const teach = FX.teaching.energy_adjustment!;
  const options: Opt[] = E.ranking.order.map((k) => {
    const lab = E.labels.options[k]!;
    const ap = E.applicability[k]!;
    const refusal = E.refusals[k];
    return {
      key: k,
      label: teachLabel("energy_adjustment", k),
      line: teach.options.find((o) => o.value === k)?.consequence ?? null,
      customary: lab.customary,
      sound: lab.sound,
      verdict: lab.verdict,
      tags: [
        ...(k === E.ranking.order[0] ? [{ text: "ranked first", tone: "usual" as const }] : []),
        ...(k === E.usual ? [{ text: "the field's usual", tone: "badge" as const }] : []),
      ],
      blocked: refusal ? refusal.message : ap.ok ? null : ap.reason,
    };
  });
  const show = (k: string) => {
    const label = teachLabel("energy_adjustment", k);
    ctx.show(k, sceneOf("energy", k, label, previewFor("energy", k, a), k === a.energy), E.refusals[k] ? null : reduce(a, { type: "energy", method: k }));
  };
  const rec = (k: string, via: "pointer" | "key") => {
    if (via === "pointer") ctx.onPointerRecord();
    if (E.refusals[k]) return show(k);
    ctx.record({ type: "energy", method: k }, "energy");
  };
  return (
    <Shell ctx={ctx} node="energy" tag={<TierTag tier="stated" />}>
      <Ask teach="energy_adjustment" first={ctx.first} />
      <p className={c.statedNow}>
        <span className={c.statedKey}>written in</span>
        <Rich text={INF.energy.sentences[a.locked?.energy ?? a.energy] ?? ""} />
      </p>
      <p className={c.tension}>
        <Rich text={E.ranking.line} />
      </p>
      {a.locked ? <p className={c.afterNote}>The plan is locked: a change now is recorded as made after the estimates were seen.</p> : null}
      <Opts options={options} shown={ctx.shown} recorded={a.energy} onShow={show} onRecord={rec} label="Energy adjustment" />
    </Shell>
  );
}

// ── form ────────────────────────────────────────────────────────────────────

function FormCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const F = INF.form;
  const declared = a.locked ? a.locked.form : a.form;
  const options: Opt[] = F.options.map((o) => ({
    key: o.value,
    label: o.label,
    line: o.consequence,
    customary: o.customary,
    sound: o.sound,
    tags: [
      ...(o.value === F.options[0]!.value ? [{ text: "ranked first", tone: "usual" as const }] : []),
      ...(a.form === null && o.value === "linear" ? [{ text: "fitted now", tone: "badge" as const }] : []),
    ],
    blocked: o.value === "quintiles" ? "This prototype captured the straight line and the spline; the quintile table is previewed only." : null,
  }));
  const show = (k: string) =>
    ctx.show(k, sceneOf("form", k, F.options.find((o) => o.value === k)!.label, formPreview(a, k), k === a.form) ?? notTaken("form", k, F.options.find((o) => o.value === k)!.label), reduce(a, { type: "form", form: k }));
  const rec = (k: string, via: "pointer" | "key") => {
    if (via === "pointer") ctx.onPointerRecord();
    if (k === "quintiles") return show(k);
    ctx.record({ type: "form", form: k }, "form");
  };
  return (
    <Shell ctx={ctx} node="form" tag={<TierTag tier={a.form === null ? "stated" : "recorded"} />}>
      <Ask teach={undefined} first={ctx.first} question="In what form does the exposure enter the models?" />
      {declared !== null ? (
        <p className={c.statedNow}>
          <span className={c.statedKey}>{a.locked ? "locked" : "recorded"}</span>
          <Rich text={F.sentences[declared] ?? ""} />
        </p>
      ) : (
        <p className={c.statedNow} data-testid="form-unrecorded">
          <span className={c.statedKey}>not in the record</span>
          No form is recorded, so the models fit a straight line; choosing one records it.
        </p>
      )}
      {a.locked ? <p className={c.afterNote}>The plan is locked: a change now is recorded as made after the estimates were seen.</p> : null}
      <Opts options={options} shown={ctx.shown} recorded={a.form} onShow={show} onRecord={rec} label="Form" />
    </Shell>
  );
}

// ── missing data ────────────────────────────────────────────────────────────

function MissingCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const M = INF.missing;
  const first = M.labels.customary_first;
  const keys = [...(first ? [first] : []), ...Object.keys(M.labels.options).filter((k) => k !== first)];
  const options: Opt[] = keys.map((k) => {
    const lab = M.labels.options[k]!;
    return {
      key: k,
      label: lab.label,
      customary: lab.customary,
      sound: lab.sound,
      verdict: lab.verdict,
      tags: k === first ? [{ text: "the field's usual", tone: "usual" as const }] : [],
      blocked: k === "complete_case" ? null : M.previews[k] ? "Previewed only in this prototype." : "Not captured in this prototype.",
    };
  });
  const show = (k: string) =>
    ctx.show(k, sceneOf("missing", k, M.labels.options[k]!.label, previewFor("missing", k, a), k === "complete_case" && a.missing), k === "complete_case" ? reduce(a, { type: "missing" }) : null);
  const rec = (k: string, via: "pointer" | "key") => {
    if (via === "pointer") ctx.onPointerRecord();
    if (k !== "complete_case") return show(k);
    ctx.record({ type: "missing" }, "missing");
  };
  return (
    <Shell ctx={ctx} node="missing" tag={<TierTag tier={a.missing ? "recorded" : "asked"} />}>
      <Ask teach="missing" first={ctx.first} />
      {M.labels.tension ? (
        <p className={c.tension}>
          <Rich text={M.labels.tension} />
        </p>
      ) : null}
      <Opts options={options} shown={ctx.shown} recorded={a.missing ? "complete_case" : null} onShow={show} onRecord={rec} label="Missing data" />
      <Undo ctx={ctx} node="missing" />
    </Shell>
  );
}

// ── Model 1 ─────────────────────────────────────────────────────────────────

function Model1Card({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const M = INF.model_1;
  const label = (v: "guess" | "empty") => (v === "guess" ? `Model 1: ${M.guess.join(", ")}` : "No Model 1");
  const play = (v: "guess" | "empty") =>
    ctx.show(v, sceneOf("model1", v, label(v), model1Preview(a, v), a.model1 === v) ?? notTaken("model1", v, label(v)), reduce(a, { type: "model1", value: v }));
  return (
    <Shell ctx={ctx} node="model1" tag={<TierTag tier={a.model1 ? "recorded" : "asked"} />}>
      <Ask teach={undefined} first={ctx.first} question="Which columns does Model 1 adjust for?" />
      <p className={c.evidence}>
        <Rich text={M.reason} />
      </p>
      <div className={c.chips}>
        {M.guess.map((col) => (
          <span key={col} className={c.chipStatic}>
            <code>{col}</code>
          </span>
        ))}
        <span className={c.guessMark}>guess</span>
      </div>
      {!a.locked ? (
        <div className={c.actions}>
          <button
            type="button"
            className={c.primary}
            onPointerEnter={() => play("guess")}
            onFocus={() => play("guess")}
            onClick={() => ctx.record({ type: "model1", value: "guess" }, "model1")}
            data-testid="model1-guess"
          >
            Confirm Model 1: {M.guess.join(", ")}
          </button>
          <button
            type="button"
            className={c.secondaryBtn}
            onPointerEnter={() => play("empty")}
            onFocus={() => play("empty")}
            onClick={() => ctx.record({ type: "model1", value: "empty" }, "model1")}
            data-testid="model1-none"
          >
            No Model 1
          </button>
        </div>
      ) : null}
      <Undo ctx={ctx} node="model1" />
    </Shell>
  );
}

// ── the seal (stated under inference) ───────────────────────────────────────

function SealCard({ ctx }: { ctx: Ctx }) {
  const S = INF.seal;
  const teach = FX.teaching.split!;
  const options: Opt[] = [...S.options]
    .sort((p, q) => p.holdout - q.holdout)
    .map((o) => ({
      key: String(o.holdout),
      label: o.label,
      line: o.measures,
      tags: o.holdout === 0 ? [{ text: "comes first", tone: "usual" as const }] : [],
      blocked: o.holdout === 0 ? null : "Not captured in this prototype: under inference a holdout changes no estimate.",
    }));
  const show = (k: string) => {
    const opt = options.find((o) => o.key === k)!;
    ctx.show(k, sceneOf("seal", k, opt.label, previewFor("split", k === "0" ? "0.0" : k, ctx.a), k === "0"));
  };
  return (
    <Shell ctx={ctx} node="seal" tag={<TierTag tier="stated" />}>
      <Ask teach="split" first={ctx.first} question={teach.question} />
      <p className={c.statedNow}>
        <span className={c.statedKey}>written in</span>
        <Rich text={S.sentence} />
      </p>
      <p className={c.tension}>
        <Rich text={S.reason} />
      </p>
      <Opts options={options} shown={ctx.shown} recorded={"0"} onShow={show} onRecord={(k) => show(k)} label="Held-out rows" />
    </Shell>
  );
}

// ── results ─────────────────────────────────────────────────────────────────

function ResultsCard({ ctx, node }: { ctx: Ctx; node: NodeId }) {
  const { a } = ctx;
  const ready = lockReady(a);
  if (!a.locked) {
    return (
      <Shell ctx={ctx} node={node} tag={<TierTag tier={ready.ready ? "asked" : "waiting"} />}>
        <Ask teach={undefined} first={ctx.first} question="Lock the plan, then see the estimates" />
        <p className={c.lede}>
          Locking records the plan with its SHA-256 as the first estimate is shown. A change after it is kept and marked, never
          folded in.
        </p>
        {ready.ready ? null : (
          <>
            <p className={c.sub}>Still asked</p>
            <ul className={c.objectiveList}>
              {ready.waiting.map((n) => (
                <li key={n}>
                  <button type="button" className={c.link} onClick={() => ctx.focus(n)}>
                    {NODE_TITLE[n]}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
        <div className={c.actions}>
          <button type="button" className={c.primary} disabled={!ready.ready} onClick={() => ctx.record({ type: "lock" }, "lock")} data-testid="lock">
            Lock the plan and show the estimates
          </button>
        </div>
      </Shell>
    );
  }
  return (
    <Shell ctx={ctx} node={node} tag={<TierTag tier="locked" />}>
      <div className={c.lockShown}>
        <button type="button" className={c.tab} aria-pressed={node === "estimate" || node === "lock"} onClick={() => ctx.focus("estimate")}>
          Table 2
        </button>
        <button type="button" className={c.tab} aria-pressed={node === "matter"} onClick={() => ctx.focus("matter")} data-testid="tab-matter">
          Which decisions mattered?
        </button>
      </div>
      {/* the lock's sentence, unless the receipt just above says it */}
      {lockSentence(ctx.a) && !(ctx.receipt?.node === node && ctx.receipt.texts.includes(lockSentence(ctx.a)!)) ? (
        <p className={c.statedNow}>
          <span className={c.statedKey}>the lock</span>
          <Rich text={lockSentence(ctx.a)!} />
        </p>
      ) : null}
      {(() => {
        const pick = fitFor(ctx.a);
        return pick.kind === "fit" && pick.fit.sensitivity.concerns.length ? (
          <p className={c.tension}>
            <Rich text={pick.fit.sensitivity.concerns.join(" ")} />
          </p>
        ) : null;
      })()}
      <p className={c.sub}>Edit after the lock</p>
      <p className={c.subNote}>
        The written-in phrases stay editable; the record keeps the plan as declared and marks each change.{" "}
        <button type="button" className={c.link} onClick={() => ctx.focus("energy")}>
          Energy adjustment
        </button>{" "}
        ·{" "}
        <button type="button" className={c.link} onClick={() => ctx.focus("form")}>
          Exposure's form
        </button>
      </p>
    </Shell>
  );
}

// ── stated phrases ──────────────────────────────────────────────────────────

function StatedCard({ ctx, node }: { ctx: Ctx; node: NodeId }) {
  const teachKey = NODE_TEACH[node];
  const teach = teachKey ? FX.teaching[teachKey] : undefined;
  const st = ctx.purpose === "inference" ? INF.stated : PRED.stated;
  const sentence =
    node === "lens"
      ? st.lens
      : node === "outcome"
        ? st.target
        : node === "purpose"
          ? st.purpose
          : node === "grain"
            ? statedReason("grain", ctx.purpose)
            : node === "clusters"
              ? statedReason("clusters", ctx.purpose)
              : node === "family"
                ? INF.models_sentence.split(". ")[0] + "."
                : null;
  const current: Record<string, string> = { lens: "dietary", purpose: ctx.purpose, grain: "one_row_per_unit", clusters: "none", family: "linear" };
  return (
    <Shell ctx={ctx} node={node} tag={<TierTag tier="stated" />}>
      <Ask teach={teachKey} first={ctx.first} />
      {sentence ? (
        <p className={c.statedNow}>
          <span className={c.statedKey}>written in</span>
          <Rich text={sentence} />
        </p>
      ) : null}
      {node === "family" ? (
        <ul className={c.shelf}>
          {INF.shelf.map((f) => (
            <li key={f.key} data-on={f.key === "linear" || undefined}>
              <span className={c.shelfRank}>{f.rank}</span> {f.label}
              {f.key !== "linear" ? <span className={c.conf}>not captured</span> : <span className={c.done}>recorded</span>}
            </li>
          ))}
        </ul>
      ) : teach && teach.options.length ? (
        <ul className={c.altList}>
          {teach.options.map((o) => (
            <li key={o.value} data-on={o.value === current[node] || undefined}>
              <span className={c.altLabel}>{o.label}</span> <Rich text={o.consequence} />
              {node === "purpose" && o.value !== ctx.purpose ? (
                <span className={c.altNote}> — the header's toggle shows this version</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
      {node !== "purpose" && node !== "family" ? <p className={c.subNote}>Its alternatives are listed; this prototype records only the scenario's answer here.</p> : null}
    </Shell>
  );
}

// ── prediction ──────────────────────────────────────────────────────────────

function PredExclusionsCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const P = PRED.exclusions;
  const keys = ["keep_every_row", "willett_2013_by_sex"];
  const options: Opt[] = keys.map((k) => ({
    key: k,
    label: P.labels.options[k]!.label,
    customary: P.labels.options[k]!.customary,
    sound: P.labels.options[k]!.sound,
    verdict: P.labels.options[k]!.verdict,
    tags: k === "keep_every_row" ? [{ text: "guess", tone: "usual" as const }] : [],
    blocked: k === "keep_every_row" ? null : "Previewed only under prediction in this prototype.",
  }));
  const show = (k: string) => ctx.show(k, sceneOf("p-excl", k, P.labels.options[k]!.label, P.previews[k === "keep_every_row" ? "none" : k]));
  return (
    <Shell ctx={ctx} node="exclusions" tag={<TierTag tier={a.pExclusions ? "recorded" : "asked"} />}>
      <Ask teach="exclusions" first={ctx.first} />
      <p className={c.tension}>
        <Rich text={P.labels.tension ?? ""} />
      </p>
      <Opts
        options={options}
        shown={ctx.shown}
        recorded={a.pExclusions ? "keep_every_row" : null}
        onShow={show}
        onRecord={(k) => (k === "keep_every_row" ? ctx.record({ type: "p_exclusions" }, "exclusions") : show(k))}
        label="Exclusions"
      />
    </Shell>
  );
}

function PredMissingCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const P = PRED.missing;
  const keys = ["impute", "indicators", "complete_case", "missing_category", "multiple_imputation"];
  const options: Opt[] = keys.map((k) => ({
    key: k,
    label: P.labels.options[k]!.label,
    customary: P.labels.options[k]!.customary,
    sound: P.labels.options[k]!.sound,
    verdict: P.labels.options[k]!.verdict,
    tags: k === "impute" ? [{ text: "guess", tone: "usual" as const }] : [],
    blocked: k === "impute" ? null : P.previews[k] ? "Previewed only in this prototype." : "Not captured in this prototype.",
  }));
  const show = (k: string) => ctx.show(k, sceneOf("p-missing", k, P.labels.options[k]!.label, P.previews[k]));
  return (
    <Shell ctx={ctx} node="p_missing" tag={<TierTag tier={a.pMissing ? "recorded" : "asked"} />}>
      <Ask teach="missing" first={ctx.first} />
      <p className={c.tension}>
        <Rich text={P.labels.tension ?? ""} />
      </p>
      <Opts
        options={options}
        shown={ctx.shown}
        recorded={a.pMissing ? "impute" : null}
        onShow={show}
        onRecord={(k) => (k === "impute" ? ctx.record({ type: "p_missing" }, "p_missing") : show(k))}
        label="Missing data"
      />
    </Shell>
  );
}

function PredSealCard({ ctx }: { ctx: Ctx }) {
  const { a } = ctx;
  const S = PRED.seal;
  const options: Opt[] = S.options.map((o) => ({
    key: String(o.holdout),
    label: o.label,
    line: o.measures,
    tags: o === S.options[0] ? [{ text: "guess", tone: "usual" as const }] : [],
    blocked: S.sentences[o.holdout === 0 ? "0.0" : String(o.holdout)] ? null : "Previewed only in this prototype.",
  }));
  const keyOf = (k: string) => (k === "0" ? "0.0" : k);
  const show = (k: string) => ctx.show(k, sceneOf("p-seal", k, options.find((o) => o.key === k)!.label, S.previews[keyOf(k)]));
  return (
    <Shell ctx={ctx} node="p_seal" tag={<TierTag tier={a.pSeal ? "recorded" : "asked"} />}>
      <Ask teach="split" first={ctx.first} />
      <p className={c.tension}>
        <Rich text={S.reason} />
      </p>
      <Opts
        options={options}
        shown={ctx.shown}
        recorded={a.pSeal === "0.0" ? "0" : a.pSeal}
        onShow={show}
        onRecord={(k) => (S.sentences[keyOf(k)] ? ctx.record({ type: "p_seal", holdout: keyOf(k) }, "p_seal") : show(k))}
        label="Held-out rows"
      />
    </Shell>
  );
}

function PredWaitingCard({ ctx, node }: { ctx: Ctx; node: NodeId }) {
  return (
    <Shell ctx={ctx} node={node} tag={<TierTag tier="waiting" />}>
      <Ask teach={NODE_TEACH[node]} first={ctx.first} />
      {node === "p_energy" ? (
        <p className={c.tension}>
          <Rich text={PRED.energy.ranking.line} />
        </p>
      ) : null}
      {node === "p_models" ? (
        <ul className={c.shelf}>
          {PRED.shelf.map((f) => (
            <li key={f.key}>
              <span className={c.shelfRank}>{f.rank}</span> {f.label}
              {f.estimate ? <span className={c.conf}>{f.estimate}</span> : null}
              {f.inductive_bias ? (
                <span className={c.evidence}>
                  <Rich text={f.inductive_bias} />
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
      <p className={c.waitNote}>
        Under prediction this prototype stops at the seal: what follows waits on it, and the prediction journey is captured up
        to there.
      </p>
    </Shell>
  );
}
