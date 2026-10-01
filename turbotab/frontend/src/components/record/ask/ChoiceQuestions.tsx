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
  RolesArtifact,
  SetMissingM1,
  ShelfArtifact,
} from "../../../api/m1-types";
import type { Decision, ProjectState } from "../../../api/schema";
import { listJoin } from "../../../util/format";
import { V } from "../../Prose";
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
      data: (
        <>
          −{fmtCount(o.affected)} {o.affected === 1 ? "row" : "rows"}
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
            text={`Each screen is counted on your table: the rows whose \`${energy}\` falls outside it.`}
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

// ── missing values ───────────────────────────────────────────────────────────

export function MissingAsk({
  proposals,
  current,
  currentDecision,
  ...p
}: AskProps & {
  proposals: ProposalsArtifact | undefined;
  current: ProjectState["missing"];
  currentDecision: Decision | null;
}) {
  const blanks: MissingColumn[] = proposals?.missing?.columns ?? [];
  const notAsked = blanks.filter((b) => b.likely_not_asked);
  const drop = notAsked.map((b) => b.column);
  const recordedDrop =
    currentDecision?.kind === "set_missing"
      ? ((currentDecision as SetMissingM1).drop_columns ?? [])
      : [];
  let recordedKey: string | null = null;
  if (current !== null) {
    recordedKey = recordedDrop.length ? "leave_out" : (current as string);
  }
  const shares = notAsked.map((b) => Math.round(b.share * 100));
  const lo = shares.length ? Math.min(...shares) : 0;
  const hi = shares.length ? Math.max(...shares) : 0;
  const blankOn = lo === hi ? `${lo}%` : `${lo}–${hi}%`;
  const option = (strategy: "complete_case" | "impute"): OptionItem => ({
    key: strategy,
    label: taught(p.entry, strategy)?.label ?? strategy,
    line: taught(p.entry, strategy)?.consequence ?? "",
    decision: { kind: "set_missing", strategy },
  });
  const leaveOut: OptionItem | null = notAsked.length
    ? {
        key: "leave_out",
        label: `Leave ${notAsked.length === 1 ? "it" : "them"} out first`,
        line: `Leave out ${listJoin(drop.map((d) => `\`${d}\``))} (blank on ${blankOn}), then complete cases.`,
        decision: {
          kind: "set_missing",
          strategy: "complete_case",
          drop_columns: drop,
        } as SetMissingM1 as Decision,
        previewLabel: `Leave out ${listJoin(drop)}, then complete cases`,
        tags: [{ text: "likely not asked", tone: "suggested" }],
      }
    : null;
  const items = [leaveOut, option("complete_case"), option("impute")].filter(
    (x): x is OptionItem => x !== null,
  );
  const listed = blanks.slice(0, 4).map((b) => `\`${b.column}\` ${pct(b.share)}`);
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

// ── the split ────────────────────────────────────────────────────────────────

const HOLDOUTS = ["0", "0.1", "0.2", "0.3"];

export function SplitAsk({
  roles,
  current,
  ...p
}: AskProps & { roles: RolesArtifact | undefined; current: ProjectState["split"] }) {
  const values = p.entry?.options.map((o) => o.value) ?? HOLDOUTS;
  const items: OptionItem[] = values.map((v) => ({
    key: v,
    label: taught(p.entry, v)?.label ?? v,
    line: taught(p.entry, v)?.consequence ?? "",
    decision: { kind: "set_split", holdout: Number(v), seed: 0, folds: 5 },
  }));
  const recordedKey = current ? (values.find((v) => Number(v) === current.holdout) ?? null) : null;
  const repeats = roles?.repeats;
  return (
    <Question
      {...p.shell}
      entry={p.entry}
      data={
        repeats ? (
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

/** The specific cause of a refusal reason: its last sentence (the first states the method's need). */
function cause(reason: string): string {
  const sentences = reason.match(/[^.!?]+[.!?]+(?:\s|$)/g)?.map((x) => x.trim()) ?? [reason];
  return sentences[sentences.length - 1] ?? reason;
}

export function EnergyAsk({
  reading,
  current,
  ...p
}: AskProps & {
  reading: EnergyReading | null | undefined;
  current: ProjectState["energy_adjustment"];
}) {
  const [strata, setStrata] = useState<string | null>(current?.strata ?? null);
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
              text={`\`${r[0]}\` tracks \`${energy}\` at r ${r[1].toFixed(2)} in this table${
                reading && reading.nutrients.length > 1
                  ? `; ${reading.nutrients.length} nutrients are adjusted together`
                  : ""
              }.`}
            />
            {reading?.notes.length ? (
              <span className={c.dataNote} data-testid="energy-notes">
                <Taught text={reading.notes.join(" ")} />
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
      {reading && reading.strata_candidates.length > 0 ? (
        <div className={c.modifier} role="group" aria-label="Fit within levels of">
          <span className={c.modifierLabel}>Fit the residual or density within each level of</span>
          {[null, ...reading.strata_candidates].map((sc) => (
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

export function ModelsAsk({
  shelf,
  current,
  ...p
}: AskProps & { shelf: ShelfArtifact; current: string[] | null }) {
  // Nothing is chosen for the user: the shelf's order is its judgment (§0).
  const [chosen, setChosen] = useState<string[]>(current ?? []);
  const families = [...shelf.families].sort((a, b) => a.rank - b.rank);
  const ordered = families.map((f) => f.key).filter((k) => chosen.includes(k));
  const items: OptionItem[] = families.map((f) => ({
    key: f.key,
    label: f.label,
    line: taught(p.entry, f.key)?.consequence ?? f.inductive_bias,
    decision: { kind: "select_models", models: [f.key] },
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
  const one = ordered.length === 1 ? families.find((f) => f.key === ordered[0])?.label : null;
  return (
    <Question {...p.shell} entry={p.entry}>
      <Options
        items={items}
        mode="multi"
        selected={new Set(chosen)}
        onToggle={(k) =>
          setChosen((cur) => (cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]))
        }
        onRecord={record}
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
          {ordered.length === 0
            ? "Choose a family to fit"
            : one
              ? `Fit the ${one.toLowerCase()}`
              : `Fit these ${ordered.length} families`}
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
