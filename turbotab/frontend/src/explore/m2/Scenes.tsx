/**
 * The stage for each prototype scene, built from the production stage's parts (StageBar, the
 * transform player, its card styles) around M2's new views.
 */
import { useEffect, useMemo, type ReactNode } from "react";
import { PlayerControls, StageBar } from "../../components/stage/StageBar";
import { Rich } from "../../components/stage/text";
import type { ReadoutItem, Storyboard } from "../../components/stage/tracks";
import { usePlayerStore } from "../../components/stage/usePlayer";
import st from "../../components/stage/Stage.module.css";
import { FX, fmtInt, fmtNum, pct } from "./data";
import type { FlowStep, MethodFixture, ReshapeFixture, SealVariant } from "./types";
import { ColumnRank } from "./views/ColumnRank";
import { ColumnStrip } from "./views/ColumnStrip";
import { EnergySpread } from "./views/EnergySpread";
import { ReshapeFlow } from "./views/ReshapeFlow";
import { ReshapeTable } from "./views/ReshapeTable";
import { RowStrip } from "./views/RowStrip";
import { SealedResults, type SealPhase } from "./views/SealedResults";
import { basisLabel, SealFork } from "./views/SealFork";
import { phasesOf } from "./seal";
import { SpreadStrips } from "./views/SpreadStrips";
import { TurnTable } from "./views/TurnTable";
import m from "./m2.module.css";

const plain = (t: string) => t.replace(/`/g, "");

/** Tell the player a new option is on the stage: hold the side, land on this option's state. */
function useOptionStory(key: string, last: number) {
  const store = usePlayerStore();
  useEffect(() => {
    if (store.get().last !== last) store.dispatch({ type: "options", last });
  }, [store, key, last]);
}

function Card({
  title,
  aside,
  sub,
  caption,
  basis,
  primary,
  children,
  testId,
}: {
  title: string;
  aside?: ReactNode;
  /** A line under the title (the column strip). */
  sub?: ReactNode;
  caption?: string;
  basis?: string;
  primary?: boolean;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <section className={primary ? st.primary : st.thumb} data-card={testId} style={{ minHeight: 0, flex: "0 0 auto" }}>
      <header className={st.cardHead}>
        <h3 className={primary ? st.cardTitle : m.thumbTitle}>
          <Rich text={title} />
        </h3>
        {aside ? <span className={m.cardAside}>{aside}</span> : null}
      </header>
      {sub ? <div className={m.cardSub}>{sub}</div> : null}
      <div className={st.cardBody}>{children}</div>
      {caption ? (
        <footer className={st.caption}>
          <p className={st.captionText}>
            <Rich text={caption} />
          </p>
          {basis ? <p className={st.basis}>{basis}</p> : null}
        </footer>
      ) : null}
    </section>
  );
}

function RecordButton({ recorded, onRecord }: { recorded: boolean; onRecord: () => void }) {
  return (
    <button type="button" className={st.record} onClick={onRecord} disabled={recorded} data-testid="record">
      {recorded ? "Recorded" : "Record this choice"}
    </button>
  );
}

// ── reshape ──────────────────────────────────────────────────────────────────

const RESHAPE_TITLES: Record<string, string> = {
  mean: "Each {unit}'s rows, combined by their mean",
  first: "Each {unit}'s first {noun}, kept whole",
  last: "Each {unit}'s last {noun}, kept whole",
  change: "Each {unit}'s change from their first {noun}",
};

function reshapeCaption(fx: ReshapeFixture, method: MethodFixture): string {
  const n = fx.dataset.rows;
  const k = fx.per_unit;
  const u = fx.unit_noun;
  const nn = fx.noun;
  switch (method.key) {
    case "mean":
      return `Each ${u}'s ${k} ${nn}s become one row: ${fmtInt(n)} rows become ${fmtInt(method.n_after)}.`;
    case "first":
      return `Each ${u}'s earliest ${nn} stays whole; ${fmtInt(n - method.n_after)} ${nn}s leave.`;
    case "last":
      return `Each ${u}'s latest ${nn} stays whole; ${fmtInt(n - method.n_after)} ${nn}s leave.`;
    default:
      return `Each ${u}'s row holds last minus first; constant columns pass through.`;
  }
}

export function reshapeStory(fx: ReshapeFixture, method: MethodFixture): Storyboard {
  return {
    last: 3,
    labels: [
      "Your data now",
      plain(method.steps[0]!),
      plain(method.steps[1]!),
      `One row per ${fx.unit_noun}: ${fmtInt(method.n_after)} rows`,
    ],
  };
}

export function ReshapeStage({
  fx,
  method,
  recorded,
  onRecord,
  wide,
}: {
  fx: ReshapeFixture;
  method: MethodFixture;
  recorded: boolean;
  onRecord: () => void;
  wide?: boolean;
}) {
  const story = useMemo(() => reshapeStory(fx, method), [fx, method]);
  useOptionStory(method.key, story.last);
  const readout: ReadoutItem[] = [
    { key: "n", name: "n", before: fmtInt(fx.dataset.rows), after: fmtInt(method.n_after) },
    { key: "c", name: "columns", before: fmtInt(fx.dataset.cols), after: fmtInt(method.columns_after) },
  ];
  const title = RESHAPE_TITLES[method.key]!.replace("{unit}", fx.unit_noun).replace("{noun}", fx.noun);
  return (
    <div className={st.stage} data-testid="stage" data-scene="reshape">
      <StageBar
        pill="Preview"
        label={method.label}
        loading={false}
        aside={recorded ? "recorded" : "nothing is recorded"}
        action={<RecordButton recorded={recorded} onRecord={onRecord} />}
      >
        <PlayerControls story={story} readout={readout} still={false} />
      </StageBar>
      <div className={st.body}>
        <div className={m.scene}>
          <Card
            primary
            testId="reshape"
            title={title}
            sub={<ColumnStrip fx={fx} method={method} last={story.last} />}
            caption={reshapeCaption(fx, method)}
            basis={`All ${fmtInt(fx.dataset.rows)} rows, before the seal.`}
          >
            <div className={m.tableRow}>
              <RowStrip fx={fx} method={method} last={story.last} height={wide ? 236 : 216} />
              <ReshapeTable fx={fx} method={method} last={story.last} />
            </div>
          </Card>
          <div className={m.pair}>
            <Card title="Rows through the flow" testId="flow">
              <ReshapeFlow steps={method.flow} last={story.last} emphasis={method.flow[1]?.key} />
            </Card>
            {wide && fx.rank ? (
              <Card title="How much combining changes each column" testId="rank">
                <ColumnRank rank={fx.rank} total={fx.columns.n_changed} />
              </Card>
            ) : null}
            {!wide && method.energy ? (
              <Card title="`energy_kcal` per row, before and after" testId="spread">
                <EnergySpread view={method.energy} notes={method.coach} last={story.last} methodKey={method.key} />
              </Card>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── orientation ──────────────────────────────────────────────────────────────

export function OrientationStage({ option, recorded, onRecord }: { option: string; recorded: boolean; onRecord: () => void }) {
  const fx = FX.orientation;
  const turns = option === "features";
  const story: Storyboard = turns
    ? {
        last: 3,
        labels: [
          "Your data now",
          `Read ${fx.label_column} as the names`,
          "Turn: each sample column becomes a row",
          `${fx.after.rows} rows × ${fx.after.cols} columns`,
        ],
      }
    : { last: 1, labels: ["Your data now", "Your data now"] };
  useOptionStory(option, story.last);
  const readout: ReadoutItem[] = turns
    ? [
        { key: "r", name: "rows", before: fmtInt(fx.before.rows), after: fmtInt(fx.after.rows) },
        { key: "c", name: "columns", before: fmtInt(fx.before.cols), after: fmtInt(fx.after.cols) },
        { key: "q", name: "ratio", before: fmtNum(fx.before.ratio, 2), after: fmtNum(fx.after.ratio, 2) },
      ]
    : [{ key: "r", name: "rows", before: fmtInt(fx.before.rows), after: fmtInt(fx.before.rows) }];
  return (
    <div className={st.stage} data-testid="stage" data-scene="orientation">
      <StageBar
        pill="Preview"
        label={turns ? "Rows are features" : "Rows are samples"}
        loading={false}
        aside={recorded ? "recorded" : "nothing is recorded"}
        action={<RecordButton recorded={recorded} onRecord={onRecord} />}
      >
        <PlayerControls
          story={story}
          readout={readout}
          still={!turns}
          stillLabel="Nothing changes; the record says the table was checked"
        />
      </StageBar>
      <div className={st.body}>
        <div className={m.scene}>
          <Card
            primary
            testId="turn"
            title={turns ? "The table, turned around" : "The table as loaded"}
            caption={
              turns
                ? `Each sample column becomes a row before any diagnosis runs: ${fmtInt(fx.after.rows)} samples, ${fmtInt(fx.after.cols - 1)} measurements.`
                : "Recorded as one row per sample; nothing is transposed."
            }
            basis={`All ${fmtInt(fx.before.rows)} rows and ${fmtInt(fx.before.cols)} columns.`}
          >
            <TurnTable fx={fx} last={turns ? story.last : 1} />
          </Card>
          <Card title="Why the shape reads feature-major" testId="spread">
            <SpreadStrips fx={fx} last={turns ? story.last : 1} />
          </Card>
        </div>
      </div>
    </div>
  );
}

// ── the seal ─────────────────────────────────────────────────────────────────

export function variantAt(v: SealVariant, fraction: number): SealVariant {
  const d = v.draws[String(fraction)];
  return d ? { ...v, ...d } : v;
}

function sealCaption(v: SealVariant): string {
  switch (v.key) {
    case "grouped":
      return `Whole ${v.unit_noun} are held out: ${v.straddle} of ${fmtInt(v.n_units ?? 0)} are on both sides.`;
    case "chronological":
      return `The ${v.n_hold_units} ${v.unit_noun} seen last are held out whole, so the model is scored on later people.`;
    case "abandoned":
      return `${v.n_units} ${v.unit_noun} are too few to hold any out whole: ${v.n_hold_rows} ${v.row_noun} drawn, ${v.straddle} ${v.unit_noun} on both sides.`;
    default:
      return `${fmtInt(v.n_hold_rows)} ${v.row_noun} drawn at random; whether one person is on both sides is not known.`;
  }
}

function exploratoryText(v: SealVariant): string | null {
  if (!v.exploratory) return null;
  if (v.key === "abandoned")
    return `Held-out scores will read better than the model is: ${v.straddle} ${v.unit_noun} train and are scored.`;
  const e = v.evidence;
  return e
    ? `\`${e.column}\` repeats, and ${e.both_sides} of its ${e.held_values} held-out values also train. Treat scores as exploratory.`
    : "Treat held-out scores as exploratory.";
}

export function sealStory(v: SealVariant): Storyboard {
  const { labels } = phasesOf(v);
  return {
    last: labels.length - 1,
    labels: labels.map((l, i) =>
      i === 0 ? "Your data now" : i === labels.length - 1 ? `Sealed: ${plain(basisLabel(v))}` : plain(l),
    ),
  };
}

function SealFlow({ v, last }: { v: SealVariant; last: number }) {
  const steps: FlowStep[] = [
    { key: "loaded", label: `${v.row_noun[0]!.toUpperCase()}${v.row_noun.slice(1)} in the table`, n: v.n_rows, combined: 0, dropped: 0 },
    { key: "train", label: "Training rows", n: v.n_train_rows, combined: 0, dropped: v.n_hold_rows },
    { key: "held", label: "Held-out rows · sealed", n: v.n_hold_rows, combined: 0, dropped: 0 },
  ];
  return <ReshapeFlow steps={steps} last={last} emphasis="train" />;
}

export function SealStage({
  v,
  fraction,
  recorded,
  onRecord,
}: {
  v: SealVariant;
  fraction: number;
  recorded: boolean;
  onRecord: () => void;
}) {
  const cvOnly = fraction === 0;
  const at = variantAt(v, fraction || 0.2);
  const story = useMemo(() => (cvOnly ? { last: 1, labels: ["Your data now", "Your data now"] } : sealStory(at)), [cvOnly, at]);
  useOptionStory(`${v.key}-${fraction}`, story.last);
  const readout: ReadoutItem[] = cvOnly
    ? [{ key: "t", name: "train", before: fmtInt(v.n_rows), after: fmtInt(v.n_rows) }]
    : [
        { key: "t", name: "train", before: fmtInt(v.n_rows), after: fmtInt(at.n_train_rows) },
        { key: "h", name: "held out", before: "0", after: fmtInt(at.n_hold_rows) },
      ];
  const ex = exploratoryText(at);
  return (
    <div className={st.stage} data-testid="stage" data-scene="seal">
      <StageBar
        pill={recorded ? null : "Preview"}
        label={cvOnly ? "Cross-validation only" : `Hold out ${pct(fraction)}`}
        loading={false}
        aside={recorded ? "sealed and recorded" : "nothing is recorded"}
        action={<RecordButton recorded={recorded} onRecord={onRecord} />}
      >
        <PlayerControls
          story={story}
          readout={readout}
          still={cvOnly}
          stillLabel="No rows are sealed; every score comes from cross-validation"
        />
      </StageBar>
      <div className={st.body}>
        <div className={m.scene}>
          <Card
            primary
            testId="fork"
            title="Where the held-out rows come from"
            aside={<span className={m.dataset}>{at.dataset}</span>}
            caption={cvOnly ? "Every row trains and is scored by cross-validation; there is no seal to open." : sealCaption(at)}
            basis={`Drawn by row identity, seed ${at.seed}; no outcome values read.`}
          >
            {cvOnly ? (
              <p className={m.cvOnly}>
                {fmtInt(v.n_rows)} {v.row_noun} · nothing held out
              </p>
            ) : (
              <SealFork v={at} recorded={recorded} />
            )}
            {ex && !cvOnly ? (
              <p className={m.exploratory} data-testid="exploratory">
                <span className={m.exploratoryKicker}>Exploratory</span> <Rich text={ex} />
              </p>
            ) : null}
          </Card>
          <Card title="Rows through the flow" testId="flow">
            <SealFlow v={cvOnly ? { ...at, n_train_rows: v.n_rows, n_hold_rows: 0 } : at} last={story.last} />
          </Card>
        </div>
      </div>
    </div>
  );
}

// ── results ──────────────────────────────────────────────────────────────────

export function ResultsStage({ phase, onOpen }: { phase: SealPhase; onOpen: () => void }) {
  return (
    <div className={st.stage} data-testid="stage" data-scene="results">
      <StageBar pill={null} label="Results" loading={false} aside="what the recorded pipeline fitted" />
      <div className={st.body}>
        <div className={m.scene}>
          <SealedResults fx={FX.results} phase={phase} onOpen={onOpen} openedAs={11} changedAs={12} />
        </div>
      </div>
    </div>
  );
}
