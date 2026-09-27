/**
 * S1 — the energy-adjustment question on the real NHANES export. Six methods, every one on the
 * shelf (partition is refused on these nutrients and stays, with its reason and a way forward).
 * Each card is its option's consequence at sparkline scale: `fat_total` against `kcal` after the
 * method, so the six tilts compare at a glance. The stage enlarges the focused one: the same 800
 * training rows move to where the method puts them.
 */
import { useMemo, useState } from "react";
import { LayoutGroup } from "motion/react";
import { DecisionSentence } from "../../components/record/blocks";
import { Prose } from "../../components/Prose";
import { NumberTween } from "../../motion/NumberTween";
import { ENERGY, FIXTURE, type EnergyOption, type ScatterState } from "./fixture";
import { fmtCell, fmtInt, fmtR } from "./format";
import { tracksOf, touched } from "./lineage";
import { ColumnsSection, Panel, RowsSection, type FlowStep, type PanelStatus } from "./Pipeline";
import { Question } from "./Question";
import { every } from "./util";
import { Histogram } from "./views/Histogram";
import { Scatter } from "./views/Scatter";
import type { TableFocusView } from "./types";
import s from "./inline.module.css";

const SPARK = every(ENERGY.xs.length, 220);
const SPARK_W = 112;
const SPARK_H = 46;

/** Pack §04: the standard and residual models give the same nutrient coefficient. */
const EQUIVALENT = new Set(["residual", "standard"]);

export function EnergyScenario() {
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [pinKey, setPinKey] = useState<string | null>(null);
  const [variantOn, setVariantOn] = useState(false);
  const [recorded, setRecorded] = useState<EnergyOption | null>(null);

  const options = ENERGY.options;
  const focus = options.find((o) => o.key === focusKey) ?? null;
  /** What the focused card actually previews (the refused method's accepted subset, when asked). */
  const shown = focus?.refused ? (variantOn ? focus.variant : null) : focus;

  const onFocus = (key: string | null) => {
    if (key !== focusKey) setVariantOn(false);
    setFocusKey(key);
  };

  const status: PanelStatus = recorded ? "recorded" : shown ? "preview" : "idle";
  const panelOption = recorded ?? shown;

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <LayoutGroup id="energy">
          {recorded ? (
            <DecisionSentence
              layoutId="q-energy"
              subject="the energy adjustment"
              onChange={() => setRecorded(null)}
            >
              <Prose text={recorded.sentence} />
            </DecisionSentence>
          ) : (
            <Question
              layoutId="q-energy"
              testId="q-energy"
              kicker="Energy adjustment"
              question={ENERGY.question}
              why={ENERGY.why}
              options={options}
              focusKey={focusKey}
              pinKey={pinKey}
              onFocus={onFocus}
              onPin={setPinKey}
              onChoose={(key) => {
                const o = options.find((x) => x.key === key);
                const eff = o?.refused ? (variantOn ? o.variant : null) : o;
                if (eff) setRecorded(eff);
              }}
              canChoose={(o) => !o.refused || (o.key === focusKey && variantOn)}
              chooseLabel={(o) => (o.refused && variantOn && o.variant ? o.variant.choose : null)}
              spark={(o) => <EnergySpark o={o} />}
              stat={(o) =>
                o.refused ? (
                  <span className={s.statMuted}>not applicable</span>
                ) : (
                  <>
                    <span className={s.statKey}>r</span> {fmtR(o.scatter?.r)}
                  </>
                )
              }
              stageHeight={372}
              stage={(f, p) => (
                <EnergyStage
                  focus={f}
                  pinned={p}
                  variantOn={variantOn}
                  onVariant={() => setVariantOn(true)}
                />
              )}
            />
          )}
        </LayoutGroup>
      </div>
      <EnergyPanel option={panelOption} status={status} />
    </div>
  );
}

function EnergySpark({ o }: { o: EnergyOption }) {
  if (o.refused || !o.scatter) {
    return (
      <svg width={SPARK_W} height={SPARK_H} className={s.refusedSpark} aria-hidden="true">
        <rect x={0.5} y={0.5} width={SPARK_W - 1} height={SPARK_H - 1} rx={5} />
        <line x1={8} y1={SPARK_H - 8} x2={SPARK_W - 8} y2={8} />
      </svg>
    );
  }
  return (
    <Scatter
      xs={ENERGY.xs}
      state={o.scatter}
      width={SPARK_W}
      height={SPARK_H}
      variant="spark"
      sample={SPARK}
      title={`${o.label}: ${o.scatter.yLabel} against ${ENERGY.xLabel}, r ${fmtR(o.scatter.r)}`}
    />
  );
}

// ── the stage ─────────────────────────────────────────────────────────────────

/** The chart's head: the y column (it changes with the option) and its correlation with energy. */
function RReadout({ state, tone = "c1" }: { state: ScatterState; tone?: "c1" | "c2" }) {
  const idle = state === ENERGY.base;
  return (
    <div className={s.rRead} data-tone={tone}>
      <span className={s.yName}>↑ {state.yLabel}</span>
      <span className={s.rKey}>r with {ENERGY.energyColumn}</span>
      {idle ? null : (
        <>
          <span className={s.rFrom}>{fmtR(ENERGY.base.r)}</span>
          <span className={s.rArrow} aria-hidden="true">
            →
          </span>
        </>
      )}
      <NumberTween value={state.r ?? 0} format={fmtR} className={s.rTo} />
    </div>
  );
}

function EnergyStage({
  focus,
  pinned,
  variantOn,
  onVariant,
}: {
  focus: EnergyOption | null;
  pinned: EnergyOption | null;
  variantOn: boolean;
  onVariant: () => void;
}) {
  const eff = focus?.refused ? (variantOn ? focus.variant : null) : focus;
  const pinEff = pinned && !pinned.refused ? pinned : null;
  const compare = !!(eff && pinEff);
  const state = eff?.scatter ?? ENERGY.base;

  if (focus?.refused && !variantOn) {
    return (
      <div className={s.refusal}>
        <div className={s.refusalHead}>
          <span className={s.stageLabel}>{focus.label}</span>
          <span className={s.refusedTag}>not applicable here</span>
        </div>
        <p className={s.refusalWhy}>
          <Prose text={focus.refused} />
        </p>
        {focus.variant && focus.variantLever ? (
          <button type="button" className={s.lever} onClick={onVariant}>
            <Prose text={focus.variantLever} />
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div className={s.stageInner}>
      <div className={s.stageHead}>
        {eff ? (
          <>
            <span className={s.stageLabel}>
              {compare ? (
                <>
                  <span className={s.swatch} data-tone="c2" />
                  {pinEff!.label}
                  <span className={s.vs}>vs</span>
                  <span className={s.swatch} data-tone="c1" />
                  {eff.label}
                </>
              ) : (
                eff.label
              )}
            </span>
            {!compare ? (
              <span className={s.stageLine}>
                <Prose text={eff.consequence} />
              </span>
            ) : null}
          </>
        ) : (
          <>
            <span className={s.stageKicker}>Your data now</span>
            <span className={s.stageLine}>
              Each dot is one of {fmtInt(ENERGY.nShown)} training rows. Preview an option to watch
              them move.
            </span>
          </>
        )}
      </div>

      <div className={s.stageBody} data-compare={compare}>
        {compare ? (
          <div className={s.compareCell}>
            <RReadout state={pinEff!.scatter!} tone="c2" />
            <Scatter
              key="pin"
              xs={ENERGY.xs}
              state={pinEff!.scatter!}
              ghost={ENERGY.base}
              width={392}
              height={196}
              variant="stage"
              tone="c2"
              xLabel={ENERGY.xLabel}
              title={pinEff!.rel?.caption ?? ""}
            />
          </div>
        ) : null}
        <div className={compare ? s.compareCell : s.primary}>
          <RReadout state={state} />
          <Scatter
            key="focus"
            xs={ENERGY.xs}
            state={state}
            ghost={eff ? ENERGY.base : null}
            width={compare ? 392 : 492}
            height={compare ? 196 : 226}
            variant="stage"
            xLabel={ENERGY.xLabel}
            title={eff?.rel?.caption ?? `${ENERGY.focus} against ${ENERGY.energyColumn}`}
          />
          {eff && !compare ? (
            <div className={s.legend}>
              <span className={s.ghostKey}>
                <span className={s.ghostDot} />
                <span className={s.mono}>{ENERGY.focus}</span> as recorded
              </span>
            </div>
          ) : null}
        </div>
        {!compare ? <RowsTable option={eff} /> : null}
      </div>

      {compare ? (
        <Differences a={pinEff!} b={eff!} />
      ) : (
        <div className={s.stageFoot}>
          {eff ? (
            <>
              <span className={s.estimand}>{eff.estimandKind}</span>
              {eff.standing ? (
                <span className={s.badge} title={eff.evidence?.source}>
                  {eff.standing}
                </span>
              ) : null}
              <details className={s.why}>
                <summary>Why?</summary>
                <p>{eff.why}</p>
              </details>
            </>
          ) : null}
          <span className={s.basis}>
            {fmtInt(ENERGY.nTrain)} training rows · {fmtInt(ENERGY.nShown)} shown
          </span>
        </div>
      )}
    </div>
  );
}

/** The working table, narrowed to the column the scatter follows, on real training rows. */
function RowsTable({ option }: { option: EnergyOption | null }) {
  const base = ENERGY.baseTable;
  const view: TableFocusView = option?.table ?? base;
  const col = ENERGY.focus;
  const afterCol = option?.scatter ? option.scatter.yLabel : col;
  const rows = useMemo(() => {
    const sorted = [...base.rows].sort(
      (a, b) => Number(a.before[ENERGY.energyColumn]) - Number(b.before[ENERGY.energyColumn]),
    );
    return every(sorted.length, 5).map((i) => sorted[i]!);
  }, [base]);
  const byId = new Map(view.rows.map((r) => [r.row_id, r]));
  const changed = new Set(view.changed.map(([r, c]) => `${r}:${c}`));
  const dist = option?.dist ?? null;
  const baseDist = ENERGY.options[0]?.dist ?? null;

  return (
    <div className={s.rowsTable}>
      <div className={s.rtKicker}>
        <span className="kicker">Your rows</span>
      </div>
      <table className={s.rt}>
        <thead>
          <tr>
            <th>{ENERGY.energyColumn}</th>
            <th>{col}</th>
            <th className={s.rtAfter}>{option ? afterCol : ""}</th>
          </tr>
          <tr className={s.rtSparks} aria-hidden="true">
            <th />
            <th>
              {baseDist ? (
                <Histogram
                  hist={baseDist.before}
                  width={70}
                  height={18}
                  variant="spark"
                  tone="faint"
                  title=""
                />
              ) : null}
            </th>
            <th>
              {option && dist ? (
                <Histogram
                  key={option.key}
                  hist={dist.after}
                  width={86}
                  height={18}
                  variant="spark"
                  title={`${dist.after_label} distribution`}
                />
              ) : null}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const after = byId.get(r.row_id);
            const v = after?.after[afterCol] ?? after?.before[afterCol];
            const isChanged = changed.has(`${r.row_id}:${afterCol}`);
            return (
              <tr key={r.row_id}>
                <td>{fmtCell(r.before[ENERGY.energyColumn])}</td>
                <td>{fmtCell(r.before[col])}</td>
                <td className={s.rtAfter} data-changed={isChanged}>
                  {option ? (typeof v === "number" ? fmtCell(v) : "·") : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Differences({ a, b }: { a: EnergyOption; b: EnergyOption }) {
  const rows: { label: string; va: string; vb: string }[] = [
    { label: `r with ${ENERGY.energyColumn}`, va: fmtR(a.scatter?.r), vb: fmtR(b.scatter?.r) },
    {
      label: `${ENERGY.energyColumn} in the model`,
      va: a.kcalInModel ? "yes" : "no",
      vb: b.kcalInModel ? "yes" : "no",
    },
    {
      label: "model columns",
      va: String(a.matrixColumns.length),
      vb: String(b.matrixColumns.length),
    },
    { label: "estimand", va: a.estimandKind, vb: b.estimandKind },
  ];
  const equivalent = EQUIVALENT.has(a.method) && EQUIVALENT.has(b.method) && a.method !== b.method;
  return (
    <div className={s.diff}>
      <ul className={s.diffRow} aria-label="Differences">
        {rows.map((r) => {
          const same = r.va === r.vb;
          return (
            <li key={r.label} className={s.diffItem} data-same={same}>
              <span className={s.diffLabel}>{r.label}</span>
              <span className={s.diffVals}>
                {same ? (
                  <span>same · {r.va}</span>
                ) : (
                  <>
                    <span data-tone="c2">{r.va}</span>
                    <span className={s.diffSep}>/</span>
                    <span data-tone="c1">{r.vb}</span>
                  </>
                )}
              </span>
            </li>
          );
        })}
      </ul>
      {equivalent ? (
        <p className={s.diffNote}>
          Different pictures, one answer: both give the same nutrient coefficient.{" "}
          <span className={s.badge}>SETTLED</span>
        </p>
      ) : null}
    </div>
  );
}

// ── the panel ─────────────────────────────────────────────────────────────────

const setup = FIXTURE.scenario_a.setup;

function stepLabel(key: string, rule?: string): string {
  if (key === "loaded") return "Rows loaded";
  if (key === "outcome_measured") return `\`${setup.outcome}\` measured`;
  if (key === "complete_cases") return "Complete cases";
  if (rule) {
    const m = /^(\w+) (\d+)–(\d+)$/.exec(rule);
    if (m) return `\`${m[1]}\` ${fmtInt(Number(m[2]))}–${fmtInt(Number(m[3]))}`;
    return rule;
  }
  return key;
}

const BASE_TRACKS = tracksOf(ENERGY.baseLineage ?? { nodes: [], links: [], collapsed: false });

function EnergyPanel({ option, status }: { option: EnergyOption | null; status: PanelStatus }) {
  const steps: FlowStep[] = setup.cohort_steps.map((st, i, all) => ({
    key: st.key,
    label: stepLabel(st.key, st.rule),
    n: st.n,
    dropped: i > 0 ? all[i - 1]!.n - st.n : 0,
  }));
  const lineage = option?.lineage?.after ?? ENERGY.baseLineage;
  const tracks = lineage ? tracksOf(lineage) : BASE_TRACKS;
  const hit = option ? touched(tracks, BASE_TRACKS) : new Set<string>();
  const energyTrack = tracks.find((t) => t.raw === ENERGY.energyColumn);
  const leaves = !!energyTrack && energyTrack.outs.length === 0;
  // A column that leaves is said once ("kcal leaves"), not counted among the changed.
  const changedN = [...hit].filter((id) => !(leaves && id === energyTrack?.id)).length;
  const summary = option
    ? [
        changedN ? `${changedN} change` : "nothing changes",
        leaves ? `${ENERGY.energyColumn} leaves` : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : `${tracks.length} predictors`;
  // Only the residual method fits anything (a regression of each nutrient on energy); density
  // and partition are row-local arithmetic.
  const fits = option?.method === "residual";

  return (
    <Panel status={status}>
      <RowsSection
        status={status}
        steps={steps}
        split={{ train: setup.n_train, holdout: setup.n_holdout, fitHere: !!option && fits }}
      />
      <ColumnsSection tracks={tracks} touchedIds={hit} status={status} summary={summary} />
    </Panel>
  );
}
