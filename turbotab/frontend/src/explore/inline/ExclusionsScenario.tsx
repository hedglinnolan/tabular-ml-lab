/**
 * S2 — implausible-intake exclusions. Three options, each a small `kcal` histogram in which the
 * rows it would remove are drawn in clay; the bins are shared, so moving between options regrows
 * the same bars. The stage adds the cut values and the count lost by level (women and men lose
 * different rows under the sex-specific rule); the panel's participant flow retallies.
 */
import { useState } from "react";
import { LayoutGroup } from "motion/react";
import { DecisionSentence } from "../../components/record/blocks";
import { Prose } from "../../components/Prose";
import { NumberTween } from "../../motion/NumberTween";
import { ENERGY, EXCLUSIONS, FIXTURE, type ExclusionOption } from "./fixture";
import { fmtInt, pct } from "./format";
import { tracksOf } from "./lineage";
import { ColumnsSection, Panel, RowsSection, type FlowStep, type PanelStatus } from "./Pipeline";
import { Question } from "./Question";
import { Histogram } from "./views/Histogram";
import s from "./inline.module.css";

const X_MAX = 6300; // bins starting past this are counted in the overflow label, not drawn
const LEVEL = (g: string) => (g === "female" ? "women" : g === "male" ? "men" : g);
const ALL = must(EXCLUSIONS.options[0]).dist.before;
const Y_MAX = Math.max(...ALL.counts);

function must<T>(v: T | undefined): T {
  if (v === undefined) throw new Error("fixture: exclusions missing");
  return v;
}

export function ExclusionsScenario() {
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [pinKey, setPinKey] = useState<string | null>(null);
  const [recorded, setRecorded] = useState<ExclusionOption | null>(null);
  const options = EXCLUSIONS.options;
  const focus = options.find((o) => o.key === focusKey) ?? null;
  const status: PanelStatus = recorded ? "recorded" : focus ? "preview" : "idle";

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <LayoutGroup id="exclusions">
          {recorded ? (
            <DecisionSentence
              layoutId="q-excl"
              subject="the exclusions"
              onChange={() => setRecorded(null)}
            >
              <Prose text={recorded.sentence} />
            </DecisionSentence>
          ) : (
            <Question
              layoutId="q-excl"
              testId="q-exclusions"
              kicker="Exclusions"
              question={EXCLUSIONS.question}
              why={EXCLUSIONS.why}
              options={options}
              focusKey={focusKey}
              pinKey={pinKey}
              onFocus={setFocusKey}
              onPin={setPinKey}
              onChoose={(k) => setRecorded(options.find((o) => o.key === k) ?? null)}
              spark={(o) => (
                <Histogram
                  hist={o.dist.before}
                  kept={o.dist.after}
                  marks={o.dist.marks}
                  width={246}
                  height={48}
                  variant="spark"
                  xMax={X_MAX}
                  yMax={Y_MAX}
                  title={`${o.label}: ${fmtInt(o.excluded)} rows excluded`}
                />
              )}
              stat={(o) => (
                <>
                  {o.excluded ? (
                    <span className={s.statDrop}>−{fmtInt(o.excluded)}</span>
                  ) : (
                    <span>0</span>
                  )}
                  <span className={s.statMuted}> rows excluded</span>
                  {o.evidence ? <span className={s.badgeSm}>{o.evidence.status}</span> : null}
                </>
              )}
              stageHeight={336}
              stage={(f, p) => <ExclusionStage focus={f} pinned={p} />}
            />
          )}
        </LayoutGroup>
      </div>
      <ExclusionPanel option={recorded ?? focus} status={status} />
    </div>
  );
}

/** Rows kept and lost. With no option in view it shows the data now, so the count tweens from it. */
function Flow({ o, tone = "c1" }: { o: ExclusionOption | null; tone?: "c1" | "c2" }) {
  const total = EXCLUSIONS.loaded;
  const kept = o?.kept ?? total;
  const excluded = o?.excluded ?? 0;
  return (
    <div className={s.flowBox} data-tone={tone}>
      <div className={s.flowNums}>
        {o ? (
          <>
            <span className={s.flowFrom}>{fmtInt(total)}</span>
            <span className={s.rArrow} aria-hidden="true">
              →
            </span>
          </>
        ) : null}
        <NumberTween value={kept} format={fmtInt} className={s.flowTo} />
        {o ? null : <span className={s.rKey}>recalls, before any rule</span>}
      </div>
      <div className={s.flowBar} aria-hidden="true">
        <span style={{ width: `${(100 * kept) / total}%` }} className={s.flowKept} />
        <span style={{ width: `${(100 * excluded) / total}%` }} className={s.flowGone} />
      </div>
      {!o ? null : (
        <div className={s.flowLost}>
          −<NumberTween value={excluded} format={fmtInt} /> · {pct(excluded, total)}
        </div>
      )}
      {!o ? null : o.byLevel.length ? (
        <table className={s.levelTable}>
          <tbody>
            {o.byLevel.map((l) => (
              <tr key={l.level}>
                <th>{l.level === "all" ? "everyone" : LEVEL(l.level)}</th>
                <td>
                  <span className={s.statDrop}>{fmtInt(l.below)}</span> &lt; {fmtInt(l.low)}
                </td>
                <td>
                  <span className={s.statDrop}>{fmtInt(l.above)}</span> &gt; {fmtInt(l.high)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className={s.flowNone}>Every recall stays in the analysis.</p>
      )}
    </div>
  );
}

function ExclusionStage({
  focus,
  pinned,
}: {
  focus: ExclusionOption | null;
  pinned: ExclusionOption | null;
}) {
  const compare = !!(focus && pinned);
  const hist = (o: ExclusionOption | null, w: number, h: number, tone: "c1" | "c2" = "c1") => (
    <Histogram
      key={tone}
      hist={ALL}
      kept={o ? o.dist.after : null}
      marks={o ? o.dist.marks : []}
      width={w}
      height={h}
      variant="stage"
      xMax={X_MAX}
      yMax={Y_MAX}
      tone={tone}
      groupLabel={LEVEL}
      ticks={[0, 1000, 2000, 3000, 4000, 5000, 6000]}
      title={o ? o.dist.caption : `kcal on all ${fmtInt(EXCLUSIONS.loaded)} loaded rows`}
    />
  );
  return (
    <div className={s.stageInner}>
      <div className={s.stageHead}>
        {focus ? (
          <>
            <span className={s.stageLabel}>
              {compare ? (
                <>
                  <span className={s.swatch} data-tone="c2" />
                  {pinned!.label}
                  <span className={s.vs}>vs</span>
                  <span className={s.swatch} data-tone="c1" />
                  {focus.label}
                </>
              ) : (
                focus.label
              )}
            </span>
            {!compare ? (
              <span className={s.stageLine}>
                <Prose text={focus.consequence} />
              </span>
            ) : null}
          </>
        ) : (
          <>
            <span className={s.stageKicker}>Your data now</span>
            <span className={s.stageLine}>
              <Prose text={`\`kcal\` on all \`${fmtInt(EXCLUSIONS.loaded)}\` loaded rows.`} />
            </span>
          </>
        )}
      </div>
      {compare ? (
        <div className={s.stageBody} data-compare="true">
          <div className={s.compareCell}>
            {hist(pinned, 392, 128, "c2")}
            <Flow o={pinned!} tone="c2" />
          </div>
          <div className={s.compareCell}>
            {hist(focus, 392, 128)}
            <Flow o={focus!} />
          </div>
        </div>
      ) : (
        <div className={s.stageBody}>
          <div className={s.primary}>
            {hist(focus, 500, 206)}
            <div className={s.axisY}>
              <span className={s.mono}>kcal / day</span>
              {focus && focus.excluded > 0 ? (
                <span className={s.ghostKey}>
                  <span className={s.goneDot} /> excluded
                </span>
              ) : null}
            </div>
          </div>
          <Flow o={focus} />
        </div>
      )}
      {!compare ? (
        <div className={s.stageFoot}>
          {focus?.evidence ? (
            <span className={s.badge} title={focus.evidence.source}>
              {focus.evidence.status}
            </span>
          ) : null}
          {focus?.evidence?.quote ? (
            <details className={s.why}>
              <summary>Why?</summary>
              <p>{focus.evidence.quote}.</p>
            </details>
          ) : null}
          <span className={s.basis}>
            all {fmtInt(EXCLUSIONS.loaded)} loaded rows · before the split
          </span>
        </div>
      ) : null}
    </div>
  );
}

const setup = FIXTURE.scenario_a.setup;
const TRACKS = tracksOf(ENERGY.baseLineage ?? { nodes: [], links: [], collapsed: false });

function ExclusionPanel({
  option,
  status,
}: {
  option: ExclusionOption | null;
  status: PanelStatus;
}) {
  const flow = option?.flow.after ?? must(EXCLUSIONS.options[0]).flow.before;
  const steps: FlowStep[] = [];
  const byKey = new Map(flow.map((st) => [st.key, st]));
  const loaded = byKey.get("loaded");
  const outcome = byKey.get("outcome_measured");
  const cut = byKey.get("exclude_kcal");
  const cc = byKey.get("complete_cases");
  if (loaded) steps.push({ key: "loaded", label: "Rows loaded", n: loaded.n });
  if (outcome) steps.push({ key: "outcome", label: `\`${setup.outcome}\` measured`, n: outcome.n });
  steps.push(
    cut
      ? {
          key: "exclusion",
          label:
            option?.byLevel.length === 1
              ? `\`kcal\` ${option.label.replace(" kcal", "")}`
              : "`kcal` by sex",
          n: cut.n,
          dropped: cut.dropped,
          mark: option ? "touched" : "none",
        }
      : {
          key: "exclusion",
          label: option ? "No exclusion rule" : "Exclusions",
          n: option ? (outcome?.n ?? null) : null,
          mark: option ? "touched" : "open",
          note: option ? undefined : "this question",
        },
  );
  if (cc) steps.push({ key: "complete", label: "Complete cases", n: cc.n });
  return (
    <Panel status={status}>
      <RowsSection status={status} steps={steps} split={{ waiting: true }} />
      <ColumnsSection
        tracks={TRACKS}
        touchedIds={new Set()}
        status={status}
        summary={`${TRACKS.length} predictors`}
        untouchedNote="No column changes: an exclusion removes rows only."
      />
    </Panel>
  );
}
