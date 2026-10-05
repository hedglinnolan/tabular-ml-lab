/**
 * The seal (M2_CONTRACT §3, §10; ROADMAP lockbox constitution §01–§05):
 *
 *   SealAsk       the split question: it states the seal's basis (grouped by a column, repetition
 *                 found but grouping abandoned, or undetermined — never drawn as a clean lock),
 *                 and each held-out size states what a holdout that size can measure. Below the
 *                 floor, cross-validation alone comes first, with its reason; nothing is refused.
 *   OpenSealStep  the Router's last step: it says what opening the seal does and goes to the
 *                 CONSEQUENCE card under the Results (stage/seal/OpenSeal.tsx), the one place the
 *                 seal is opened. The held-out scores are withheld until it is pressed; then they
 *                 are fixed in the record, and any later change is marked post-seal.
 */
import { useId, useMemo, useState } from "react";
import { useStageFocus } from "../../../state/focus";
import type { RolesArtifact, SplitArtifact, TeachingEntry } from "../../../api/m1-types";
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
  // The server's defaults (decisions.py SplitSpec); the validation choice is not drawn yet (WP9).
  validation: "kfold",
  repeats: 10,
  n_boot: 500,
  cluster: null,
  nested_cv: false,
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
            {/* No basis until there is a grain: the refusal below says the grain comes first. */}
            {plan.basis ? <SealBasisLine basis={plan.basis} chronology={plan.chronology} /> : null}
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

/**
 * The Router's last step, in the Record's flow. The CONSEQUENCE card that opens the seal lives
 * once, under the Results on the stage (M2_CONTRACT §3, §10): this step says what it does and
 * takes the user there, rather than drawing the same card a second time (BLUEPRINT §11.2).
 */
export function OpenSealStep({
  entry,
  split,
  now,
}: {
  entry: TeachingEntry | undefined;
  split: SplitArtifact | null;
  now: boolean;
}) {
  const titleId = useId();
  const whyId = useId();
  const [why, setWhy] = useState(false);
  const { reset } = useStageFocus();
  const held = split?.n_holdout ?? null;
  const go = () => {
    reset(); // the stage's live view, which is the Results once fitted
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        const card = document.querySelector<HTMLElement>('[data-purpose="open_seal"]');
        card?.scrollIntoView({ block: "center" });
        card?.querySelector<HTMLButtonElement>("button")?.focus({ preventScroll: true });
      }),
    );
  };
  return (
    <TermsProvider terms={entry?.terms}>
      <section
        className={k.step}
        aria-labelledby={titleId}
        data-testid="open-seal-step"
        data-now={now || undefined}
      >
        <span className={k.stepKicker}>Opening the seal</span>
        <p id={titleId} className={k.stepText}>
          The last step is under the Results: the models are scored once on
          {held !== null ? <> the {fmtCount(held)}</> : " the"} held-out rows, and the scores are
          fixed in the record.{" "}
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
        <button type="button" className={k.go} onClick={go} data-testid="go-open-seal">
          Show the seal under the Results
        </button>
      </section>
    </TermsProvider>
  );
}
