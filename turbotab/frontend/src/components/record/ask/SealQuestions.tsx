/**
 * The seal (M2_CONTRACT §3, §10; ROADMAP lockbox constitution §01–§05):
 *
 *   SealAsk       the split question: it states the seal's basis (grouped by a column, repetition
 *                 found but grouping abandoned, or undetermined — never drawn as a clean lock),
 *                 and each held-out size states what a holdout that size can measure. Below the
 *                 floor, cross-validation alone comes first, with its reason; nothing is refused.
 *   OpenSealCard  the Router's last step: a CONSEQUENCE (DESIGN_LANGUAGE §09) — declarative, then
 *                 first person. The held-out scores exist and are withheld until it is pressed;
 *                 then they are fixed in the record, and any later change is marked post-seal.
 */
import { useId, useMemo, useState } from "react";
import type {
  FitArtifact,
  RolesArtifact,
  SplitArtifact,
  TeachingEntry,
} from "../../../api/m1-types";
import type { SealBasis, SealPlan } from "../../../api/m2-types";
import type { Decision, ProjectState } from "../../../api/schema";
import { Options, type OptionItem } from "../Options";
import { Question } from "../Question";
import { SealGlyph } from "../seal/SealGlyph";
import { Taught, TermsProvider } from "../teach";
import { Actions, Keep, fmtCount, taught, type AskProps } from "./common";
import c from "./ask.module.css";
import k from "./seal.module.css";

const HOLDOUTS = ["0", "0.1", "0.2", "0.3"];

const splitDecision = (holdout: number): Decision => ({
  kind: "set_split",
  holdout,
  seed: 0,
  folds: 5,
});

/** The seal's basis, said once: the glyph, its label, and the sentence that says how it was drawn. */
export function SealBasisLine({
  basis,
  chronology,
  recorded = false,
}: {
  basis: SealBasis;
  chronology?: SealPlan["chronology"];
  recorded?: boolean;
}) {
  const sentence = chronology?.drawn ? chronology.sentence : basis.sentence;
  return (
    <span className={c.basis} data-testid="seal-basis" data-basis={basis.state}>
      <SealGlyph state={basis.state} recorded={recorded} size={20} />
      <span className={c.basisText}>
        <span className={c.basisLabel}>The seal</span>
        <Taught text={`${basis.label[0]!.toUpperCase()}${basis.label.slice(1)}.`} />{" "}
        <Taught text={sentence} />
        {basis.exploratory ? <span className={c.exploratory}>exploratory</span> : null}
      </span>
    </span>
  );
}

export function SealAsk({
  plan,
  roles,
  current,
  ...p
}: AskProps & {
  plan: SealPlan | null;
  roles: RolesArtifact | undefined;
  current: ProjectState["split"];
}) {
  // The intervals' basis defines itself where the widths are stated ("known to about ±0.36").
  const note = plan?.precision_note ?? null;
  const entry = useMemo<TeachingEntry | undefined>(
    () =>
      p.entry && note
        ? { ...p.entry, terms: [...p.entry.terms, { term: "known to about", definition: note }] }
        : p.entry,
    [p.entry, note],
  );
  let items: OptionItem[];
  if (plan) {
    const cv = plan.options.filter((o) => o.holdout === 0);
    const held = plan.options.filter((o) => o.holdout > 0);
    const ordered = plan.cv_first ? [...cv, ...held] : [...held, ...cv];
    items = ordered.map((o) => ({
      key: String(o.holdout),
      label: o.label,
      line: o.measures,
      decision: splitDecision(o.holdout),
      // Judgment is order and its reason, never absence: the first option says why it is first.
      note:
        plan.cv_first && o.holdout === 0 ? (
          <span className={c.hint}>
            <Taught text={plan.reason} />
          </span>
        ) : undefined,
    }));
  } else {
    const values = p.entry?.options.map((o) => o.value) ?? HOLDOUTS;
    items = values.map((v) => ({
      key: v,
      label: taught(p.entry, v)?.label ?? v,
      line: taught(p.entry, v)?.consequence ?? "",
      decision: splitDecision(Number(v)),
    }));
  }
  const recordedKey = current
    ? (items.find((o) => Number(o.key) === current.holdout)?.key ?? null)
    : null;
  const repeats = roles?.repeats;
  return (
    <Question
      {...p.shell}
      entry={entry}
      data={
        plan ? (
          <>
            <SealBasisLine basis={plan.basis} chronology={plan.chronology} />
            {plan.refusal ? (
              <span className={c.dataNote}>
                <Taught text={plan.refusal} />
              </span>
            ) : null}
          </>
        ) : repeats ? (
          <Taught
            text={`\`${repeats.column}\` repeats, so each person's rows stay on one side of the split.`}
          />
        ) : undefined
      }
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={recordedKey}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Held-out rows — preview with the arrow keys, Enter to record"
        testId="options-split"
      />
      {p.keep ? (
        <Actions>
          <Keep keep={p.keep} />
        </Actions>
      ) : null}
    </Question>
  );
}

// ── the Router's last step: open the seal ───────────────────────────────────

const NUMBER = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const count = (n: number) => NUMBER[n] ?? fmtCount(n);

export function OpenSealCard({
  entry,
  fit,
  split,
  pending,
  record,
  answerAt,
  now,
}: {
  entry: TeachingEntry | undefined;
  fit: FitArtifact | null;
  split: SplitArtifact | null;
  pending: boolean;
  record: (d: Decision, at?: string) => void;
  answerAt: AskProps["answerAt"];
  now: boolean;
}) {
  const titleId = useId();
  const whyId = useId();
  const [why, setWhy] = useState(false);
  const n = fit?.models.length ?? 0;
  const held = split?.n_holdout ?? fit?.n_holdout ?? null;
  const metric = fit ? (fit.metric_labels[fit.primary_metric] ?? fit.primary_metric) : "score";
  const basis = split?.basis ?? null;
  const models = n === 1 ? "the model" : `the ${count(n)} models`;
  return (
    <TermsProvider terms={entry?.terms}>
      <section
        className={k.consequence}
        aria-labelledby={titleId}
        data-testid="open-seal-card"
        data-now={now || undefined}
      >
        <div className={k.rule} aria-hidden="true" />
        <div className={k.head}>
          <SealGlyph state={basis?.state ?? "grouped"} recorded size={24} />
          <span className={k.signal}>Opened once</span>
        </div>
        <h2 id={titleId} className={k.title} tabIndex={-1}>
          Opening the seal scores {models}
          {held !== null ? <> on {fmtCount(held)} held-out rows</> : null} no choice has seen.
        </h2>
        <p className={k.body}>
          It happens once. The held-out {metric} is then fixed in the record; any later change still
          refits, and is marked post-seal in the Results and the manuscript.
          {basis?.exploratory
            ? " The seal's basis is not verified, so these scores are labeled exploratory."
            : ""}{" "}
          {entry ? (
            <button
              type="button"
              className={k.why}
              aria-expanded={why}
              aria-controls={whyId}
              onClick={() => setWhy((w) => !w)}
              data-testid="why-open_seal"
            >
              why?
            </button>
          ) : null}
        </p>
        {why && entry ? (
          <p id={whyId} className={k.whyText}>
            <Taught text={entry.why} />
          </p>
        ) : null}
        <div className={k.exits}>
          <button
            type="button"
            className={k.attest}
            disabled={pending || !fit}
            onClick={() => record({ kind: "open_seal" }, "open")}
            data-testid="open-seal"
          >
            {pending ? "Opening…" : "I'm done choosing: open the seal"}
          </button>
          <span className={k.or}>or keep choosing; nothing is opened until you press it.</span>
        </div>
        {answerAt?.key === "open" ? <div className={c.answer}>{answerAt.node}</div> : null}
      </section>
    </TermsProvider>
  );
}
