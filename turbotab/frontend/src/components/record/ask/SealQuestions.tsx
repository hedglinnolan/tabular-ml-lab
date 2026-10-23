/**
 * The seal (M2_CONTRACT §3, §10; lockbox constitution §01–§05):
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
import { useId, useMemo, useRef, useState, type FormEvent } from "react";
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
const CUT = "cut"; // the cut-point's option (teaching content's SPLIT, decisions.py CutPoint)

type SetSplit = Extract<Decision, { kind: "set_split" }>;
export type FoldOrder = NonNullable<SetSplit["fold_order"]>;

/**
 * The draw, with the validation scheme as it stands. `set_split` replaces the whole split, so
 * the scheme `set_validation` set is sent back with it (ruling 10 of W1_PART2_PLAN: the scheme is
 * `set_validation`'s). With no split recorded yet there is no scheme to keep, and the server's
 * defaults (decisions.py SplitSpec) stand. A share holds no cut-point, and its folds' order is
 * the draw's; a cut-point holds no share (`cutDecision`).
 */
const splitDecision = (holdout: number, current: ProjectState["split"]): SetSplit => ({
  kind: "set_split",
  holdout,
  seed: 0,
  folds: current?.folds ?? 5,
  validation: current?.validation ?? "kfold",
  repeats: current?.repeats ?? 10,
  n_boot: current?.n_boot ?? 500,
  cluster: current?.cluster ?? null,
  nested_cv: current?.nested_cv ?? false,
  cut_point: null,
  fold_order: null,
});

/**
 * A temporal seal by cut-point (SWARM_PLAN §8): every row at or after `at` on `column` held out,
 * no share beside it, the scheme kept, and the folds' order its own answer, random or time. A
 * numeral on a numeric column is sent as the number; anything else as written, which the server
 * reads as the column reads (core/seal.py `cut_value`). Null until the column, the value and the
 * folds' order are all given.
 */
export function cutDecision(
  column: string,
  at: string,
  order: FoldOrder | null,
  numeric: boolean,
  current: ProjectState["split"],
): SetSplit | null {
  const text = at.trim();
  if (!column || !text || !order) return null;
  const n = Number(text);
  const value = numeric && Number.isFinite(n) ? n : text;
  return { ...splitDecision(0, current), cut_point: { column, at: value }, fold_order: order };
}

/**
 * A cut-point's value as the server shows and compares it (decisions.py `cut_shown`): a whole
 * number without its decimals, another number to ten significant digits, text as written.
 */
export function cutShown(at: number | string): string {
  if (typeof at === "number")
    return Number.isInteger(at) ? String(at) : String(+at.toPrecision(10));
  return at.trim();
}

/**
 * The cut-point option's line, which says a unit is kept whole only where the seal keeps it so:
 * under a grouped basis the draw groups rows by its column, and a unit with the outcome recorded
 * on both sides of the cut is held out whole (core/seal.py `seal_inputs`, whose draw reads only
 * the rows with the outcome recorded, and `cut_holdout`; claim seal.cut_point_units); under any
 * other basis the rows are held out one by one (seal.cut_point); with no basis yet it says
 * neither. The teaching entry's line says only what holds on every basis, so it is not read (the
 * package 4 verifiers' V7 and W1).
 */
export function cutLine(basis: SealBasis | null | undefined): string {
  const lead = "Rows at or after a time value you name are sealed";
  if (basis?.state === "grouped" && basis.column)
    return `${lead}; a \`${basis.column}\` with the outcome recorded on both sides is sealed whole.`;
  return basis ? `${lead}, row by row.` : `${lead}.`;
}

/**
 * The chronology the seal's basis line states. The seal plan's is a usual share's draw, which a
 * recorded cut-point replaces (the temporal answer then draws nothing): under one, the line states
 * the split's own sentence once the split is drawn for that cut-point, else the basis alone.
 */
export function basisChronology(
  plan: SealPlan,
  current: ProjectState["split"],
  split: SplitArtifact | undefined,
): SealPlan["chronology"] {
  const cut = current?.cut_point;
  if (!cut) return plan.chronology;
  const drawn = split?.chronology;
  return drawn?.drawn && drawn.time_column === cut.column && drawn.cut_point === cutShown(cut.at)
    ? drawn
    : null;
}

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
  split,
  timeColumns = [],
  ...p
}: AskProps & {
  plan: SealPlan | null;
  roles: RolesArtifact | undefined;
  current: ProjectState["split"];
  /** The split as drawn: under a recorded cut-point, its sentence is the basis line's. */
  split?: SplitArtifact;
  /** The columns a cut-point can be read on, numbers or dates, never the outcome. */
  timeColumns?: { name: string; numeric: boolean }[];
}) {
  const [chosenColumn, setCutColumn] = useState(current?.cut_point?.column ?? "");
  // The column the select shows: the one chosen, else the first offered (columns can arrive late).
  const cutColumn = chosenColumn || timeColumns[0]?.name || "";
  const [cutAt, setCutAt] = useState(current?.cut_point ? String(current.cut_point.at) : "");
  const [order, setOrder] = useState<FoldOrder | null>(current?.fold_order ?? null);
  const atRef = useRef<HTMLInputElement>(null);
  const numeric = timeColumns.find((t) => t.name === cutColumn)?.numeric ?? false;
  const cut = cutDecision(cutColumn, cutAt, order, numeric, current);
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
      decision: splitDecision(o.holdout, current),
      // Judgment is order and its reason, never absence: the first option says why it is first.
      note:
        plan.cv_first && o.holdout === 0 ? (
          <span className={c.hint}>
            <Taught text={plan.reason} />
          </span>
        ) : undefined,
    }));
  } else {
    const values = (p.entry?.options.map((o) => o.value) ?? HOLDOUTS).filter((v) => v !== CUT);
    items = values.map((v) => ({
      key: v,
      label: taught(p.entry, v)?.label ?? v,
      line: taught(p.entry, v)?.consequence ?? "",
      decision: splitDecision(Number(v), current),
    }));
  }
  items.push({
    key: CUT,
    label: taught(p.entry, CUT)?.label ?? "Hold out by cut-point",
    line: cutLine(plan?.basis),
    decision: cut,
    previewLabel: cut ? `${cutColumn} from ${cutAt.trim()}` : undefined,
    extra: (
      <form
        className={c.range}
        onSubmit={(e: FormEvent) => {
          e.preventDefault();
          if (cut) p.record(cut, CUT);
        }}
      >
        <label className={c.field}>
          <span>column</span>
          <select value={cutColumn} onChange={(e) => setCutColumn(e.target.value)}>
            {timeColumns.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        <label className={c.field}>
          <span>hold out from</span>
          <input
            ref={atRef}
            value={cutAt}
            onChange={(e) => setCutAt(e.target.value)}
            placeholder="a value or a date"
            aria-label="The first value held out"
          />
        </label>
        <label className={c.field}>
          <span>folds</span>
          <select
            value={order ?? ""}
            onChange={(e) => setOrder((e.target.value || null) as FoldOrder | null)}
            aria-label="How the training folds are ordered"
          >
            <option value="">choose…</option>
            <option value="random">random</option>
            <option value="time">in time order</option>
          </select>
        </label>
        <button type="submit" className={c.small} disabled={!cut || p.pending}>
          Seal from here
        </button>
      </form>
    ),
  });
  const recordedKey = current
    ? current.cut_point
      ? CUT
      : (items.find((o) => o.key !== CUT && Number(o.key) === current.holdout)?.key ?? null)
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
            {plan.basis ? (
              <SealBasisLine
                basis={plan.basis}
                chronology={basisChronology(plan, current, split)}
              />
            ) : null}
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
        onRecord={(o) => {
          if (o.decision) p.record(o.decision, o.key);
          else if (o.key === CUT) atRef.current?.focus();
        }}
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
