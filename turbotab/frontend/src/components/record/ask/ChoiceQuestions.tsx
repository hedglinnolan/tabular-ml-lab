/**
 * The CHOICE questions — modeling decisions, each a bordered card (DESIGN_LANGUAGE §09):
 * exclusions, missing values, the split, energy adjustment, the model families, and the
 * substitution (whose pair navigator is the stage's; the Record keeps its question and its
 * recorded sentence). The pack's usual choice is tagged and ordered first, never pre-selected;
 * an option that does not apply stays on the shelf and says why.
 */
import { createContext, useContext, useRef, useState, type FormEvent, type RefObject } from "react";
import type {
  EnergyMethod,
  EnergyReading,
  EligibilityRule,
  ExclusionRule,
  LevelRule,
  MissingColumn,
  MissingReading,
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

/** Levels as typed, comma-separated: each trimmed, blanks dropped. */
export const parseLevels = (text: string): string[] =>
  text
    .split(",")
    .map((v) => v.trim())
    .filter((v) => v !== "");

/** One spelling per level, as the server compares them (stages/rows.py `_level_key`): `2`,
 *  `2.0` and `"2"` are one level. */
const levelKey = (v: string): string => {
  const n = Number(v);
  return v !== "" && Number.isFinite(n) && Number.isInteger(n) ? String(n) : v;
};

/**
 * The categorical eligibility rule a card's fields make (decisions.py LevelRule; SWARM_PLAN §5):
 * the levels it excludes and the codes that record a refused or a don't-know answer, each its own
 * list. Null while it excludes no level or names one level twice, which the server refuses too.
 * A refused or don't-know answer, like a blank, cannot confirm a row and leaves on a line of its
 * own (the server's default).
 */
export function levelRule(
  column: string,
  exclude: string,
  refused: string,
  dontKnow: string,
): LevelRule | null {
  const lists = [parseLevels(exclude), parseLevels(refused), parseLevels(dontKnow)] as const;
  const keys = lists.flat().map(levelKey);
  if (!column || lists[0].length === 0 || new Set(keys).size !== keys.length) return null;
  return {
    kind: "levels",
    column,
    exclude: lists[0],
    refused: lists[1],
    dont_know: lists[2],
    reason: "not in the study population (your own levels)",
    missing: "exclude",
  };
}

/** Why a range that says who the study is about leaves its rows, as the participant flow and its
 *  figure print it ("Excluded: …"): not the plausible range's "implausible values". The teaching
 *  content holds the same words (core/teaching/content.py `ELIGIBILITY_RANGE_REASON`), and the
 *  swarm driver records them from there. */
export const ELIGIBILITY_REASON = "outside the study population (your own range)";

/** Why a range of plausible values of one's own leaves its rows, as the flow prints it. */
const PLAUSIBLE_REASON = "implausible values (your own range)";

/** A range of one's own as a card's fields hold it, typed. */
interface RangeFields {
  column: string;
  setColumn: (v: string) => void;
  low: string;
  setLow: (v: string) => void;
  high: string;
  setHigh: (v: string) => void;
}

function useRangeFields(initialColumn: string): RangeFields {
  const [column, setColumn] = useState<string>(initialColumn);
  const [low, setLow] = useState("");
  const [high, setHigh] = useState("");
  return { column, setColumn, low, setLow, high, setHigh };
}

/**
 * The range rule a card's fields make (decisions.py ExclusionRule) for `reason`: the rows outside
 * low–high on the column excluded, either bound open. Null without a column or a bound, or with a
 * bound that is not a number. A row with no value cannot be confirmed eligible and leaves (the
 * server's default).
 */
function rangeRule(f: RangeFields, reason: string): ExclusionRule | null {
  const lo = f.low.trim() === "" ? null : Number(f.low);
  const hi = f.high.trim() === "" ? null : Number(f.high);
  const valid =
    f.column !== "" &&
    (lo !== null || hi !== null) &&
    (lo === null || Number.isFinite(lo)) &&
    (hi === null || Number.isFinite(hi));
  return valid
    ? { kind: "range", column: f.column, low: lo, high: hi, by: null, reason, missing: "exclude" }
    : null;
}

function RangeForm({
  f,
  lowRef,
  columns,
  rule,
  pending,
  onSubmit,
}: {
  f: RangeFields;
  /** The lower bound's input, focused when the option is chosen before it is set. */
  lowRef: RefObject<HTMLInputElement | null>;
  columns: string[];
  rule: ExclusionRule | null;
  pending: boolean;
  onSubmit: () => void;
}) {
  return (
    <form
      className={c.range}
      onSubmit={(e: FormEvent) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <label className={c.field}>
        <span>column</span>
        <select value={f.column} onChange={(e) => f.setColumn(e.target.value)}>
          {columns.map((n) => (
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
          value={f.low}
          onChange={(e) => f.setLow(e.target.value)}
          placeholder="no lower bound"
          aria-label="Lowest value kept"
        />
      </label>
      <label className={c.field}>
        <span>to</span>
        <input
          inputMode="decimal"
          value={f.high}
          onChange={(e) => f.setHigh(e.target.value)}
          placeholder="no upper bound"
          aria-label="Highest value kept"
        />
      </label>
      <button type="submit" className={c.small} disabled={!rule || pending}>
        Exclude the rest
      </button>
    </form>
  );
}

const rangeShown = (rule: ExclusionRule) =>
  `${rule.column} outside ${rule.low ?? "−∞"}–${rule.high ?? "∞"}`;

export function ExclusionsAsk({
  proposals,
  current,
  numericColumns,
  levelColumns = [],
  ...p
}: AskProps & {
  proposals: ProposalsArtifact | undefined;
  current: EligibilityRule[] | null;
  numericColumns: string[];
  /** The columns a level rule can read: every column but the outcome. */
  levelColumns?: string[];
}) {
  const offered = proposals?.exclusions ?? [];
  const energy = proposals?.energy?.energy_column ?? null;
  // Two ranges of one's own, each with its reason: plausible values, and who the study is about.
  const plausible = useRangeFields(energy ?? numericColumns[0] ?? "");
  const eligible = useRangeFields(numericColumns[0] ?? "");
  const plausibleLow = useRef<HTMLInputElement>(null);
  const eligibleLow = useRef<HTMLInputElement>(null);
  const customRule = rangeRule(plausible, PLAUSIBLE_REASON);
  const eligibilityRule = rangeRule(eligible, ELIGIBILITY_REASON);

  const [levelColumn, setLevelColumn] = useState("");
  const shownLevelColumn = levelColumn || levelColumns[0] || "";
  const [excluded, setExcluded] = useState("");
  const [refusedCodes, setRefusedCodes] = useState("");
  const [unknownCodes, setUnknownCodes] = useState("");
  const levelsRef = useRef<HTMLInputElement>(null);
  const byLevel = levelRule(shownLevelColumn, excluded, refusedCodes, unknownCodes);

  // The option an answer on record reads back as. Each option records one rule, so an answer of
  // several (the swarm driver's age range beside its level rules) is marked on none: no option's
  // line says what it holds. A range reads back as one of one's own only by that option's reason.
  let recordedKey: string | null = null;
  if (current?.length === 0) recordedKey = "none";
  else if (current?.length === 1) {
    const rule = current[0]!;
    recordedKey =
      offered.find((o) => same([rule], [o.rule]))?.key ??
      (rule.kind === "levels"
        ? "levels"
        : rule.kind === "range" && rule.reason === ELIGIBILITY_REASON
          ? "eligibility"
          : rule.kind === "range" && rule.reason === PLAUSIBLE_REASON
            ? "custom"
            : null);
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
        "Rows outside a plausible range you set on a numeric column, never the outcome, are excluded.",
      decision: customRule ? { kind: "set_exclusions", rules: [customRule] } : null,
      previewLabel: customRule ? rangeShown(customRule) : undefined,
      extra: (
        <RangeForm
          f={plausible}
          lowRef={plausibleLow}
          columns={numericColumns}
          rule={customRule}
          pending={p.pending}
          onSubmit={() => {
            if (customRule) p.record({ kind: "set_exclusions", rules: [customRule] }, "custom");
          }}
        />
      ),
    },
    {
      key: "eligibility",
      label: taught(p.entry, "eligibility")?.label ?? "Your own eligibility range",
      line:
        taught(p.entry, "eligibility")?.consequence ??
        "Rows outside a range you set, such as an age range, leave the study population.",
      decision: eligibilityRule ? { kind: "set_exclusions", rules: [eligibilityRule] } : null,
      previewLabel: eligibilityRule ? rangeShown(eligibilityRule) : undefined,
      extra: (
        <RangeForm
          f={eligible}
          lowRef={eligibleLow}
          columns={numericColumns}
          rule={eligibilityRule}
          pending={p.pending}
          onSubmit={() => {
            if (eligibilityRule)
              p.record({ kind: "set_exclusions", rules: [eligibilityRule] }, "eligibility");
          }}
        />
      ),
    },
    {
      key: "levels",
      label: taught(p.entry, "levels")?.label ?? "By level",
      line:
        taught(p.entry, "levels")?.consequence ??
        "Rows at levels you name are excluded, as are blanks and refused or don't-know codes named.",
      decision: byLevel ? { kind: "set_exclusions", rules: [byLevel] } : null,
      previewLabel: byLevel ? `${shownLevelColumn} at ${byLevel.exclude.join(", ")}` : undefined,
      extra: (
        <form
          className={c.range}
          onSubmit={(e: FormEvent) => {
            e.preventDefault();
            if (byLevel) p.record({ kind: "set_exclusions", rules: [byLevel] }, "levels");
          }}
        >
          <label className={c.field}>
            <span>column</span>
            <select value={shownLevelColumn} onChange={(e) => setLevelColumn(e.target.value)}>
              {levelColumns.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className={c.field}>
            <span>exclude</span>
            <input
              ref={levelsRef}
              value={excluded}
              onChange={(e) => setExcluded(e.target.value)}
              placeholder="levels, by commas"
              aria-label="Levels excluded"
            />
          </label>
          <label className={c.field}>
            <span>refused codes</span>
            <input
              value={refusedCodes}
              onChange={(e) => setRefusedCodes(e.target.value)}
              placeholder="none"
              aria-label="Codes that record a refused answer"
            />
          </label>
          <label className={c.field}>
            <span>don't know codes</span>
            <input
              value={unknownCodes}
              onChange={(e) => setUnknownCodes(e.target.value)}
              placeholder="none"
              aria-label="Codes that record a don't-know answer"
            />
          </label>
          <button type="submit" className={c.small} disabled={!byLevel || p.pending}>
            Exclude these
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
          else if (o.key === "levels") levelsRef.current?.focus();
          else if (o.key === "eligibility") eligibleLow.current?.focus();
          else plausibleLow.current?.focus();
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
type MissingOffer = MissingReading["methods"][number];
type BelowDetection = NonNullable<MissingDecision["below_detection"]>;
type ImputationModel = NonNullable<MissingDecision["imputation_model"]>;

/** m, the imputations multiple imputation draws at least, as the contract bounds it (decisions.py
 *  SetMissing `m`; missing.test.tsx holds the card to openapi.json's). The fit draws one per
 *  percent of the rows with a blank to impute where that is more (methods/imputation.py
 *  `m_rule`). */
const M_IMPUTATIONS = { default: 20, min: 20, max: 200 } as const;

/** The m a typed value asks for: a whole number within the contract's bounds, else null. */
function parseM(text: string): number | null {
  const t = text.trim();
  if (!/^\d+$/.test(t)) return null;
  const n = Number(t);
  return n >= M_IMPUTATIONS.min && n <= M_IMPUTATIONS.max ? n : null;
}

/** The answers the server offers for the declared purpose: every one it lists but those it
 *  refuses (methods/missing.py `methods_for`, `imputation_model_options`,
 *  `below_detection_options`). */
const offered = (list: MissingOffer[] | undefined): MissingOffer[] =>
  (list ?? []).filter((o) => o.rung !== "refused");

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
    m: 20,
    below_detection: null,
    censored_columns: [],
    acknowledged: false,
    reason: null,
    imputation_model: "compatible",
    imputation_levels: "clustered",
    ...patch,
  });

  // Values below a detection limit (audit ME-08), where the record names their columns: the server
  // names them when it refuses a fill that reads them as any other blank, and its ways forward
  // record them. Before that this card cannot know them (it is given no findings), so the first
  // answer meets the server's refusal. The fills offered are the server's for the purpose.
  const censored = current?.censored_columns ?? [];
  const fillOnRecord = current?.below_detection ?? null;
  const fillOffers = offered(proposals?.missing?.below_detection);
  const askFill = censored.length > 0 && fillOnRecord !== null && fillOffers.length > 0;
  // The fill on record until one is chosen here, so the choice always shows what a press posts
  // (an exit taken at this card's refusal records one while the card stays open).
  const [chosenFill, setFill] = useState<BelowDetection | null>(null);
  const fill = chosenFill ?? fillOnRecord;
  const belowDetection: Partial<MissingDecision> = askFill
    ? { below_detection: fill, censored_columns: censored }
    : {};
  const fillChoice = (where: string) =>
    askFill ? (
      <>
        <span className={c.modifierLabel}>values below detection</span>
        <select
          className={c.inlineSelect}
          aria-label="How values below detection are filled"
          value={fill ?? ""}
          onChange={(e) => setFill(e.target.value as BelowDetection)}
          data-testid={`missing-fill-${where}`}
        >
          {fillOffers.map((o) => (
            <option key={o.key} value={o.key} title={o.sound}>
              {o.label}
            </option>
          ))}
          {fillOffers.some((o) => o.key === fillOnRecord) ? null : (
            <option value={fillOnRecord ?? ""}>as recorded</option>
          )}
        </select>
      </>
    ) : null;

  // Multiple imputation where the server offers it (inference's; refused under prediction, so not
  // offered there), with its m and the imputation model as the server serves them (MS1–MS2). The
  // compatible, clustered imputation is the engine's reading of an answer that names neither.
  const miOffer = offered(proposals?.missing?.methods).find((o) => o.key === "multiple_imputation");
  const subOffers = offered(proposals?.missing?.imputation_models);
  const modelOffers = subOffers.filter((o) => typeof o.decision.imputation_model === "string");
  const singleOffer = subOffers.find((o) => o.decision.imputation_levels === "single_level");
  const miOnRecord = current?.strategy === "multiple_imputation" ? current : null;
  const [mText, setMText] = useState(String(miOnRecord?.m ?? M_IMPUTATIONS.default));
  const [model, setModel] = useState<ImputationModel>(
    miOnRecord?.imputation_model === "passive" ? "passive" : "compatible",
  );
  const [single, setSingle] = useState(miOnRecord?.imputation_levels === "single_level");
  const m = parseM(mText);
  const multipleImputation: OptionItem | null = miOffer
    ? {
        key: "multiple_imputation",
        label: taught(p.entry, "multiple_imputation")?.label ?? "Multiple imputation",
        // Not the teaching's line, which says 20 times: m is the user's, and the fit may draw more.
        line: "Each blank imputed several times with the outcome; results pooled by Rubin's rules.",
        decision:
          m === null
            ? null
            : spec({
                strategy: "multiple_imputation",
                m,
                imputation_model: model,
                imputation_levels: single ? "single_level" : "clustered",
                ...belowDetection,
              }),
        extra: (
          <span
            className={c.modifierInline}
            role="group"
            aria-label="How multiple imputation draws"
          >
            <label className={c.check}>
              at least
              <input
                inputMode="numeric"
                size={3}
                className={c.inlineSelect}
                value={mText}
                onChange={(e) => setMText(e.target.value)}
                aria-label="Imputations, at least"
                data-testid="missing-m"
              />
              imputations
            </label>
            {m === null ? (
              <span className={c.coachNote} data-testid="missing-m-bounds">
                A whole number from {M_IMPUTATIONS.min} to {M_IMPUTATIONS.max}.
              </span>
            ) : (
              <span className={c.hint}>
                more where over {m}% of the rows have a blank to impute
              </span>
            )}
            {modelOffers.length ? (
              <select
                className={c.inlineSelect}
                aria-label="Imputation model"
                value={model}
                onChange={(e) => setModel(e.target.value as ImputationModel)}
                data-testid="missing-imputation-model"
              >
                {modelOffers.map((o) => (
                  <option key={o.key} value={String(o.decision.imputation_model)} title={o.sound}>
                    {o.label}
                  </option>
                ))}
              </select>
            ) : null}
            {singleOffer ? (
              <label className={c.check} title={singleOffer.sound}>
                <input
                  type="checkbox"
                  checked={single}
                  onChange={(e) => setSingle(e.target.checked)}
                  data-testid="missing-single-level"
                />
                {singleOffer.label}
              </label>
            ) : null}
            {fillChoice("mi")}
          </span>
        ),
      }
    : null;

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
    ...belowDetection,
  });
  const impute: OptionItem = {
    key: "impute",
    label: taught(p.entry, "impute")?.label ?? "Impute",
    line: taught(p.entry, "impute")?.consequence ?? "",
    decision: imputeDecision,
    previewLabel: `Impute${imputeDecision.categorical === "missing_category" ? ", blanks as a level" : ""}${imputeDecision.indicators ? ", with indicators" : ""}`,
    extra:
      levelable.length || numeric.length || askFill ? (
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
          {fillChoice("impute")}
        </span>
      ) : undefined,
  };
  const items = [missingLevel, leaveOut, multipleImputation, completeCase, impute].filter(
    (x): x is OptionItem => x !== null,
  );
  // An option reads as the answer on record only while pressing it would record that answer.
  const recordedKey = (() => {
    const key = missingKey(current);
    if (!current) return key;
    const sameFill = !askFill || fill === fillOnRecord;
    if (key === "impute") {
      const same =
        (current.categorical === "missing_category") === (levels && levelable.length > 0) &&
        current.indicators === (indicators && numeric.length > 0) &&
        sameFill;
      return same ? key : null;
    }
    if (key === "multiple_imputation") {
      const same =
        m === current.m &&
        (current.imputation_model === "passive") === (model === "passive") &&
        (current.imputation_levels === "single_level") === single &&
        sameFill;
      return same ? key : null;
    }
    return key;
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

/**
 * The models card's words for recording the chosen families: `one` is the family's name when one is
 * chosen (lower case), else null, and `n` their count. Today's screen fits the families as it
 * records them, so it says Fit; a screen where Fit is pressed elsewhere provides its own words.
 */
export type FamiliesLabel = (one: string | null, n: number) => string;
const FIT_FAMILIES: FamiliesLabel = (one, n) =>
  one ? `Fit the ${one}` : `Fit these ${n} families`;
/** The words ModelsAsk uses where its `label` prop is not given (the quest shell provides them
 *  around today's card, src/quest/cards/CardFor.tsx); today's Fit words by default. */
export const FamiliesLabelContext = createContext<FamiliesLabel>(FIT_FAMILIES);

/** The families chosen on the models card and not yet recorded, with the record they were chosen
 *  against (`base`, the recorded families as JSON). */
export interface HeldFamilies {
  base: string;
  chosen: string[];
}

/**
 * Where a screen keeps the families chosen and not yet recorded, so they survive the models card
 * leaving the screen and coming back (finding 12 of the 2026-10-17 walkthrough): in the quest shell
 * the card's line waits while the shelf is computed again, which a reading confirmed on the card
 * sets off, and the card is drawn anew when the line opens. A choice held against a record that has
 * changed since is not restored. Absent (today's screen), the card holds its own.
 */
export const HeldFamiliesContext = createContext<{
  get: () => HeldFamilies | null;
  set: (held: HeldFamilies) => void;
} | null>(null);

export function ModelsAsk({
  shelf,
  current,
  label,
  ...p
}: AskProps & {
  shelf: ShelfArtifact;
  current: string[] | null;
  /** The record button's words for the chosen families; FamiliesLabelContext's when absent. */
  label?: FamiliesLabel;
}) {
  // Nothing is chosen for the user: the shelf's order is its judgment (§0). A choice the screen held
  // against this same record comes back as it was.
  const keeper = useContext(HeldFamiliesContext);
  const base = JSON.stringify(current ?? null);
  const [chosen, setChosenHere] = useState<string[]>(() => {
    const held = keeper?.get();
    return held && held.base === base ? held.chosen : (current ?? []);
  });
  const setChosen = (next: string[]) => {
    setChosenHere(next);
    keeper?.set({ base, chosen: next });
  };
  const provided = useContext(FamiliesLabelContext);
  const words = label ?? provided;
  const families = [...shelf.families].sort((a, b) => a.rank - b.rank);
  const ordered = families.map((f) => f.key).filter((k) => chosen.includes(k));
  const fitLabel = (keys: string[]) => {
    const one = keys.length === 1 ? families.find((f) => f.key === keys[0])?.label : null;
    return words(one ? one.toLowerCase() : null, keys.length);
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
          setChosen(chosen.includes(k) ? chosen.filter((x) => x !== k) : [...chosen, k])
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

// ── the survey answer (audit §5 WP10) ────────────────────────────────────────

/** Whose estimate it is under a survey design: the server offers one "surveyed population"
 *  option per weight it recognized, and "these participants"; each carries its decision. */
export function SurveyAsk({
  proposal,
  current,
  ...p
}: AskProps & {
  proposal: ProposalsArtifact["survey"] | undefined;
  current: ProjectState["survey"];
}) {
  const items: OptionItem[] = (proposal?.options ?? []).map((o) => ({
    key: o.key,
    label: o.label,
    line: o.consequence,
    decision: o.decision as unknown as Decision,
  }));
  const recorded = current
    ? current.estimand === "sample"
      ? "sample"
      : `population:${current.weight ?? ""}`
    : null;
  return (
    <Question {...p.shell} entry={p.entry}>
      <Options
        items={items}
        mode="single"
        onRecord={(o) => o.decision && p.record(o.decision, o.key)}
        recordedKey={recorded}
        pending={p.pending}
        answerAt={p.answerAt}
        label="Survey design — preview with the arrow keys, Enter to record"
        testId="options-survey"
      />
      <KeepRow keep={p.keep} />
    </Question>
  );
}
