/**
 * The CHOICE questions — modeling decisions, each a bordered card (DESIGN_LANGUAGE §09):
 * exclusions, missing values, the split, energy adjustment, the model families, and the
 * substitution (whose pair navigator is the stage's; the Record keeps its question and its
 * recorded sentence). The pack's usual choice is tagged and ordered first, never pre-selected;
 * an option that does not apply stays on the shelf and says why.
 */
import { useRef, useState, type FormEvent } from "react";
import type {
  EnergyMethod,
  EnergyReading,
  ExclusionRule,
  MissingColumn,
  ProposalsArtifact,
  ShelfArtifact,
  ShelfFamily,
} from "../../../api/m1-types";
import type { Decision, ProjectState } from "../../../api/schema";
import { listJoin } from "../../../util/format";
import { Prose, V } from "../../Prose";
import { Options, type OptionItem } from "../Options";
import { Question } from "../Question";
import { Taught } from "../teach";
import { Actions, Keep, RecordButton, fmtCount, taught, type AskProps } from "./common";
import c from "./ask.module.css";

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const pct = (share: number) => `${Math.round(share * 100)}%`;

function KeepRow({ keep }: { keep?: (() => void) | undefined }) {
  return keep ? (
    <Actions>
      <Keep keep={keep} />
    </Actions>
  ) : null;
}

// ── exclusions ───────────────────────────────────────────────────────────────

export function ExclusionsAsk({
  proposals,
  current,
  numericColumns,
  ...p
}: AskProps & {
  proposals: ProposalsArtifact | undefined;
  current: ExclusionRule[] | null;
  numericColumns: string[];
}) {
  const offered = proposals?.exclusions ?? [];
  const energy = proposals?.energy?.energy_column ?? null;
  const [column, setColumn] = useState<string>(energy ?? numericColumns[0] ?? "");
  const [low, setLow] = useState("");
  const [high, setHigh] = useState("");
  const lowRef = useRef<HTMLInputElement>(null);
  const lo = low.trim() === "" ? null : Number(low);
  const hi = high.trim() === "" ? null : Number(high);
  const valid =
    column !== "" &&
    (lo !== null || hi !== null) &&
    (lo === null || Number.isFinite(lo)) &&
    (hi === null || Number.isFinite(hi));
  const customRule: ExclusionRule | null = valid
    ? {
        kind: "range",
        column,
        low: lo,
        high: hi,
        by: null,
        reason: "implausible values (your own range)",
      }
    : null;

  const matches = (rules: ExclusionRule[]) =>
    current !== null && current.length === rules.length && same(current, rules);
  let recordedKey: string | null = null;
  if (current !== null) {
    if (current.length === 0) recordedKey = "none";
    else recordedKey = offered.find((o) => matches([o.rule]))?.key ?? "custom";
  }

  const items: OptionItem[] = [
    {
      key: "none",
      label: taught(p.entry, "none")?.label ?? "Keep every row",
      line: taught(p.entry, "none")?.consequence ?? "No row is excluded.",
      decision: { kind: "set_exclusions", rules: [] },
    },
    ...offered.map<OptionItem>((o) => ({
      key: o.key,
      label: taught(p.entry, o.key)?.label ?? o.label,
      line: taught(p.entry, o.key)?.consequence ?? o.label,
      decision: { kind: "set_exclusions", rules: [o.rule] },
      // The denominator is named: proposals count every row with the outcome recorded, before
      // the split; the stage's preview counts the rows outside the held-out set.
      data: (
        <>
          −{fmtCount(o.affected)} of {fmtCount(proposals?.n_base ?? 0)}
        </>
      ),
      tags: [{ text: o.evidence.status.toUpperCase(), tone: "badge" }],
    })),
    {
      key: "custom",
      label: taught(p.entry, "custom")?.label ?? "Your own range",
      line:
        taught(p.entry, "custom")?.consequence ??
        "Rows outside a range you set on any numeric column are excluded.",
      decision: customRule ? { kind: "set_exclusions", rules: [customRule] } : null,
      previewLabel: customRule ? `${column} outside ${lo ?? "−∞"}–${hi ?? "∞"}` : undefined,
      extra: (
        <form
          className={c.range}
          onSubmit={(e: FormEvent) => {
            e.preventDefault();
            if (customRule) p.record({ kind: "set_exclusions", rules: [customRule] }, "custom");
          }}
        >
          <label className={c.field}>
            <span>column</span>
            <select value={column} onChange={(e) => setColumn(e.target.value)}>
              {numericColumns.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className={c.field}>
            <span>keep from</span>
            <input
              ref={lowRef}
              inputMode="decimal"
              value={low}
              onChange={(e) => setLow(e.target.value)}
              placeholder="no lower bound"
              aria-label="Lowest value kept"
            />
          </label>
          <label className={c.field}>
            <span>to</span>
            <input
              inputMode="decimal"
              value={high}
              onChange={(e) => setHigh(e.target.value)}
              placeholder="no upper bound"
              aria-label="Highest value kept"
            />
          </label>
          <button type="submit" className={c.small} disabled={!customRule || p.pending}>
            Exclude the rest
          </button>
        </form>
      ),
    },
  ];

  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        offered.length && energy ? (
          <Taught
            text={`Each screen is counted on your table: the rows whose \`${energy}\` falls outside it, of the ${fmtCount(proposals?.n_base ?? 0)} with the outcome recorded.`}
          />
        ) : undefined
      }
    >
      <Options
        items={items}
        mode="single"
        onRecord={(o) => {
          if (o.decision) p.record(o.decision, o.key);
          else lowRef.current?.focus();
        }}
        recordedKey={recordedKey}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Exclusions — preview with the arrow keys, Enter to record"
        testId="options-exclusions"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── missing values, by mechanism ─────────────────────────────────────────────

/** Whether a column's blanks can be a level of their own: categorical, or two values at most
 *  (turbotab/core/models/pipeline.py `takes_level`). */
export function takesLevel(info: { dtype: string; n_unique: number } | undefined): boolean {
  if (!info) return false;
  return ["boolean", "categorical", "text"].includes(info.dtype) || info.n_unique <= 2;
}

type MissingDecision = Extract<Decision, { kind: "set_missing" }>;

/** The option an answer on record reads back as. */
export function missingKey(current: ProjectState["missing"]): string | null {
  if (current === null) return null;
  if ((current.drop_columns ?? []).length) return "leave_out";
  if (current.strategy === "complete_case" && current.categorical === "missing_category")
    return "missing_level";
  return current.strategy;
}

export function MissingAsk({
  proposals,
  columns,
  purpose,
  current,
  ...p
}: AskProps & {
  proposals: ProposalsArtifact | undefined;
  /** The working table's columns (dtype and distinct values): which blanks can be a level. */
  columns: ReadonlyMap<string, { dtype: string; n_unique: number }>;
  purpose: ProjectState["purpose"];
  current: ProjectState["missing"];
}) {
  const blanks: MissingColumn[] = proposals?.missing?.columns ?? [];
  const notAsked = blanks.filter((b) => b.likely_not_asked);
  // Missingness by mechanism (constitution §07): a categorical or yes/no blank can be a level.
  const levelable = blanks.filter((b) => takesLevel(columns.get(b.column)));
  const numeric = blanks.filter((b) => !takesLevel(columns.get(b.column)));
  const levelNotAsked = levelable.filter((b) => b.likely_not_asked);
  // The server's offer names the columns and the share of rows blank in any of them.
  const offer = proposals?.missing?.leave_out ?? null;
  const drop = offer?.columns ?? notAsked.map((b) => b.column);
  const [levels, setLevels] = useState(current?.categorical === "missing_category");
  const [indicators, setIndicators] = useState(current?.indicators ?? false);
  const shares = notAsked.map((b) => Math.round(b.share * 100));
  const lo = shares.length ? Math.min(...shares) : 0;
  const hi = shares.length ? Math.max(...shares) : 0;
  const blankOn = offer ? pct(offer.share) : lo === hi ? `${lo}%` : `${lo}–${hi}%`;
  const ticked = (cols: { column: string }[]) => listJoin(cols.map((b) => "`" + b.column + "`"));
  const spec = (patch: Partial<MissingDecision>): MissingDecision => ({
    kind: "set_missing",
    strategy: "complete_case",
    drop_columns: [],
    categorical: "impute",
    indicators: false,
    ...patch,
  });

  const missingLevel: OptionItem | null = levelable.length
    ? {
        key: "missing_level",
        label: "Blanks as a level",
        line: `${ticked(levelable.slice(0, 3))} keep their rows: a blank becomes the level \`Missing\`.${numeric.length ? " Other blanks drop their row." : ""}`,
        decision: spec({ categorical: "missing_category" }),
        previewLabel: `Blanks in ${listJoin(levelable.map((b) => b.column))} as a Missing level`,
        // Recommended with its reason when the blanks read as "not asked" (§10).
        tags: levelNotAsked.length ? [{ text: "recommended", tone: "usual" }] : undefined,
        note: levelNotAsked.length ? (
          <span className={c.hint}>
            <Taught text={levelNotAsked[0]!.reason} />
          </span>
        ) : undefined,
      }
    : null;
  const leaveOut: OptionItem | null = drop.length
    ? {
        key: "leave_out",
        label: `Leave ${drop.length === 1 ? "it" : "them"} out first`,
        line: `Leave out ${listJoin(drop.map((d) => "`" + d + "`"))} (blank on ${blankOn}), then complete cases.`,
        decision: spec({ drop_columns: drop }),
        previewLabel: `Leave out ${listJoin(drop)}, then complete cases`,
        // Where the not-asked blanks cannot be a level, leaving them out is the honest offer.
        tags: missingLevel ? undefined : [{ text: "likely not asked", tone: "suggested" }],
      }
    : null;
  const completeCase: OptionItem = {
    key: "complete_case",
    label: taught(p.entry, "complete_case")?.label ?? "Complete cases",
    line: taught(p.entry, "complete_case")?.consequence ?? "",
    decision: spec({}),
  };
  const imputeDecision = spec({
    strategy: "impute",
    categorical: levels && levelable.length ? "missing_category" : "impute",
    indicators: indicators && numeric.length > 0,
  });
  const impute: OptionItem = {
    key: "impute",
    label: taught(p.entry, "impute")?.label ?? "Impute",
    line: taught(p.entry, "impute")?.consequence ?? "",
    decision: imputeDecision,
    previewLabel: `Impute${imputeDecision.categorical === "missing_category" ? ", blanks as a level" : ""}${imputeDecision.indicators ? ", with indicators" : ""}`,
    extra:
      levelable.length || numeric.length ? (
        <span className={c.modifierInline} role="group" aria-label="How imputation treats blanks">
          {levelable.length ? (
            <label className={c.check}>
              <input
                type="checkbox"
                checked={levels}
                onChange={(e) => setLevels(e.target.checked)}
                data-testid="missing-levels"
              />
              <Taught text={`blanks in ${ticked(levelable.slice(0, 2))} as \`Missing\``} />
            </label>
          ) : null}
          {numeric.length ? (
            <label className={c.check}>
              <input
                type="checkbox"
                checked={indicators}
                onChange={(e) => setIndicators(e.target.checked)}
                data-testid="missing-indicators"
              />
              a was-missing indicator for each filled number
            </label>
          ) : null}
          {indicators && numeric.length && purpose === "inference" ? (
            <span className={c.coachNote}>Under inference, an indicator biases the estimates.</span>
          ) : null}
        </span>
      ) : undefined,
  };
  const items = [missingLevel, leaveOut, completeCase, impute].filter(
    (x): x is OptionItem => x !== null,
  );
  const recordedKey = (() => {
    const key = missingKey(current);
    if (key !== "impute" || !current) return key;
    const same =
      (current.categorical === "missing_category") === (levels && levelable.length > 0) &&
      current.indicators === (indicators && numeric.length > 0);
    return same ? key : null;
  })();
  const listed = blanks.slice(0, 4).map((b) => "`" + b.column + "` " + pct(b.share));
  const data =
    proposals?.missing === undefined || proposals?.missing === null
      ? undefined
      : blanks.length === 0
        ? "No predictor has a blank cell."
        : `Blank among the predictors: ${listJoin(listed)}${blanks.length > 4 ? `, and ${blanks.length - 4} more` : ""}.`;
  return (
    <Question {...p.shell} entry={p.entry} data={data ? <Taught text={data} /> : undefined}>
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={recordedKey}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Missing values — preview with the arrow keys, Enter to record"
        testId="options-missing"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── energy adjustment ────────────────────────────────────────────────────────

const METHODS: EnergyMethod[] = [
  "none",
  "standard",
  "residual",
  "density_multivariate",
  "density",
  "partition",
];
const STRATIFIABLE: EnergyMethod[] = ["residual", "density", "density_multivariate"];

/** "Left as they are: `a` and `b` (already a share of energy); `c` (carries no energy)." */
function notAdjustedText(entries: { column: string; reason: string }[]): string {
  const byReason = new Map<string, string[]>();
  for (const e of entries) byReason.set(e.reason, [...(byReason.get(e.reason) ?? []), e.column]);
  const parts = [...byReason.entries()].map(
    ([reason, cols]) => `${listJoin(cols.map((col) => `\`${col}\``))} (${reason})`,
  );
  return `Left as they are: ${parts.join("; ")}.`;
}

/** The specific cause of a refusal reason: its last sentence (the first states the method's need). */
function cause(reason: string): string {
  const sentences = reason.match(/[^.!?]+[.!?]+(?:\s|$)/g)?.map((x) => x.trim()) ?? [reason];
  return sentences[sentences.length - 1] ?? reason;
}

export function EnergyAsk({
  reading,
  current,
  leftOut = [],
  ...p
}: AskProps & {
  reading: EnergyReading | null | undefined;
  current: ProjectState["energy_adjustment"];
  /** Columns the missing-values answer left out: never offered as strata. */
  leftOut?: string[];
}) {
  const [strata, setStrata] = useState<string | null>(current?.strata ?? null);
  const strataCandidates = (reading?.strata_candidates ?? []).filter((c) => !leftOut.includes(c));
  const usual = reading?.usual ?? null;
  const ok = (m: EnergyMethod) => m === "none" || (reading?.applicability[m]?.ok ?? false);
  const values = (p.entry?.options.map((o) => o.value) as EnergyMethod[] | undefined) ?? METHODS;
  // Judgment is order and emphasis, never absence (§11.9): the usual first, the inapplicable
  // still on the shelf, and no adjustment last.
  const ordered = [
    ...values.filter((m) => m === usual),
    ...values.filter((m) => m !== usual && m !== "none" && ok(m)),
    ...values.filter((m) => m !== "none" && !ok(m)),
    ...values.filter((m) => m === "none"),
  ];
  const decisionFor = (m: EnergyMethod): Decision =>
    m === "none"
      ? {
          kind: "set_energy_adjustment",
          method: "none",
          energy_column: null,
          nutrients: [],
          log_transform: false,
          strata: null,
        }
      : {
          kind: "set_energy_adjustment",
          method: m,
          energy_column: reading?.energy_column ?? null,
          nutrients: reading?.nutrients ?? [],
          log_transform: false,
          strata: STRATIFIABLE.includes(m) ? strata : null,
        };
  const items: OptionItem[] = ordered.map((m) => {
    const verdict = reading?.applicability[m];
    const t = taught(p.entry, m);
    return {
      key: m,
      label: t?.label ?? m,
      line: t?.consequence ?? "",
      decision: decisionFor(m),
      previewLabel:
        (t?.label ?? m) + (strata && STRATIFIABLE.includes(m) ? `, within ${strata}` : ""),
      tags: m === usual ? [{ text: "usual", tone: "usual" }] : undefined,
      na: m !== "none" && verdict && !verdict.ok ? cause(verdict.reason) : undefined,
    };
  });
  const r = Object.entries(reading?.r_with_energy ?? {}).sort((a, b) => b[1] - a[1])[0];
  const energy = reading?.energy_column;
  const recordedKey =
    current && (current.method === "none" || (current.strata ?? null) === strata)
      ? current.method
      : null;
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        r && energy ? (
          <>
            <Taught
              // The correlation itself is the stage's: it reads only rows outside the held-out
              // set, while the proposals were counted before the split existed.
              text={`\`${r[0]}\` tracks \`${energy}\` most closely${
                reading && reading.nutrients.length > 1
                  ? `; ${listJoin(reading.nutrients.map((n) => `\`${n}\``))} are adjusted together`
                  : ""
              }.`}
            />
            {reading?.notes.length ? (
              <span className={c.dataNote} data-testid="energy-notes">
                <Taught text={reading.notes.join(" ")} />
              </span>
            ) : null}
            {reading?.not_adjusted.length ? (
              // Nothing is left out silently: each exposure the adjustment leaves alone, and why.
              <span className={c.dataNote} data-testid="energy-not-adjusted">
                <Taught text={notAdjustedText(reading.not_adjusted)} />
              </span>
            ) : null}
          </>
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
        label="Energy adjustment methods — preview with the arrow keys, Enter to record"
        testId="options-energy_adjustment"
      />
      {strataCandidates.length > 0 ? (
        <div className={c.modifier} role="group" aria-label="Fit within levels of">
          <span className={c.modifierLabel}>Fit the residual or density within each level of</span>
          {[null, ...strataCandidates].map((sc) => (
            <button
              key={sc ?? "none"}
              type="button"
              className={c.toggle}
              aria-pressed={strata === sc}
              onClick={() => setStrata(sc)}
              data-testid={`strata-${sc ?? "none"}`}
            >
              {sc === null ? "no grouping" : <span className="num">{sc}</span>}
            </button>
          ))}
        </div>
      ) : null}
      <KeepRow keep={p.keep} />
    </Question>
  );
}

// ── model families ───────────────────────────────────────────────────────────

/** Below this a fit's length answers no question the user has (the server's
 * `cost.NOTEWORTHY_SECONDS`): the models card says nothing about it. */
export const NOTEWORTHY_SECONDS = 10;

export function ModelsAsk({
  shelf,
  current,
  ...p
}: AskProps & {
  shelf: ShelfArtifact;
  current: string[] | null;
}) {
  // Nothing is chosen for the user: the shelf's order is its judgment (§0).
  const [chosen, setChosen] = useState<string[]>(current ?? []);
  const families = [...shelf.families].sort((a, b) => a.rank - b.rank);
  const ordered = families.map((f) => f.key).filter((k) => chosen.includes(k));
  const fitLabel = (keys: string[]) => {
    const one = keys.length === 1 ? families.find((f) => f.key === keys[0])?.label : null;
    return one ? `Fit the ${one.toLowerCase()}` : `Fit these ${keys.length} families`;
  };
  // Enter and the stage's record button do the same thing: record the chosen families, or the
  // shown one when none is chosen yet.
  const recordFor = (key: string): string[] => (ordered.length ? ordered : [key]);
  // Honest cost at scale (M2_CONTRACT §12.6): a family's measured fit time, in the server's words
  // ("about 5 minutes at `20,004` columns"), shown whenever it is long enough to weigh.
  const cost = (f: ShelfFamily) => {
    const sec = f.estimate_seconds;
    if (sec === null || sec < NOTEWORTHY_SECONDS || !f.estimate) return undefined;
    return (
      <span data-testid={`cost-${f.key}`}>
        <Prose text={f.estimate} />
      </span>
    );
  };
  const items: OptionItem[] = families.map((f) => ({
    key: f.key,
    label: f.label,
    line: taught(p.entry, f.key)?.consequence ?? f.inductive_bias,
    data: cost(f),
    decision: { kind: "select_models", models: [f.key] },
    record: { kind: "select_models", models: recordFor(f.key) },
    recordLabel: fitLabel(recordFor(f.key)),
    tags: [{ text: `${f.fit} fit`, tone: "fit" }],
    note: f.concerns.length ? (
      <span className={c.concerns}>
        {f.concerns.map((concern) => (
          <span key={concern} className={c.concern}>
            <Taught text={concern} />
          </span>
        ))}
      </span>
    ) : undefined,
  }));
  const record = () =>
    ordered.length && p.record({ kind: "select_models", models: ordered }, "record");
  return (
    <Question {...p.shell} entry={p.entry}>
      <Options
        items={items}
        mode="multi"
        selected={new Set(chosen)}
        onToggle={(k) =>
          setChosen((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]))
        }
        onRecord={(item) =>
          p.record({ kind: "select_models", models: recordFor(item.key) }, "record")
        }
        pending={p.pending}
        answerAt={p.answerAt}
        label="Model families — choose any, then fit"
        testId="options-models"
      />
      <Actions>
        <RecordButton
          disabled={ordered.length === 0 || p.pending}
          onClick={record}
          title="Records the families; each is fit on the training rows and compared by cross-validation."
          testId="record-models"
        >
          {ordered.length === 0 ? "Choose a family to fit" : fitLabel(ordered)}
        </RecordButton>
        <Keep keep={p.keep} />
      </Actions>
      {p.answerAt?.key === "record" ? <div className={c.answer}>{p.answerAt.node}</div> : null}
    </Question>
  );
}

// ── substitution ─────────────────────────────────────────────────────────────

export function SubstitutionAsk({ pairs, ...p }: AskProps & { pairs: number | null }) {
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        <>
          Choose the <strong>donor</strong> and the <strong>recipient</strong> in the matrix on the
          stage
          {pairs ? (
            <>
              : <V>{fmtCount(pairs)}</V> pairs are possible among your energy-bearing nutrients
            </>
          ) : null}
          .
        </>
      }
    >
      {p.answerAt ? <div className={c.answer}>{p.answerAt.node}</div> : null}
      <KeepRow keep={p.keep} />
    </Question>
  );
}
