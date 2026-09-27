/**
 * S4 — a transform that touches 495 columns of a 60 × 500 genomics table. The view stays
 * readable because it never tries to show the table: the stage shows one column's distribution
 * (the one whose shape changed most), a heat grid of the 12 most-changed columns over 8 rows, and
 * a barcode with one tick per affected column so the reader knows the twelve are twelve of 495.
 * The lineage collapses the counts into one group track.
 */
import { useState } from "react";
import { LayoutGroup } from "motion/react";
import { DecisionSentence } from "../../components/record/blocks";
import { Prose } from "../../components/Prose";
import { NumberTween } from "../../motion/NumberTween";
import { WIDE, type WideOption } from "./fixture";
import { fmtInt } from "./format";
import { tracksOf, touched } from "./lineage";
import { ColumnsSection, Panel, RowsSection, type PanelStatus } from "./Pipeline";
import { Question } from "./Question";
import { Histogram } from "./views/Histogram";
import { Barcode, HeatGrid } from "./views/Wide";
import s from "./inline.module.css";

const SHOWN = WIDE.table.columns_before;
const fmtSkew = (v: number) => v.toFixed(1);

export function WideScenario() {
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [pinKey, setPinKey] = useState<string | null>(null);
  const [recorded, setRecorded] = useState<WideOption | null>(null);
  const options = WIDE.options;
  const focus = options.find((o) => o.key === focusKey) ?? null;
  const status: PanelStatus = recorded ? "recorded" : focus ? "preview" : "idle";

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <LayoutGroup id="wide">
          {recorded ? (
            <DecisionSentence
              layoutId="q-wide"
              subject="the count transform"
              onChange={() => setRecorded(null)}
            >
              <Prose text={recorded.sentence} />
            </DecisionSentence>
          ) : (
            <Question
              layoutId="q-wide"
              testId="q-wide"
              kicker="Transform · genomics sample"
              question={WIDE.question}
              why={WIDE.why}
              options={options}
              focusKey={focusKey}
              pinKey={pinKey}
              onFocus={setFocusKey}
              onPin={setPinKey}
              onChoose={(k) => setRecorded(options.find((o) => o.key === k) ?? null)}
              spark={(o) => (
                <span className={s.wideSpark}>
                  <Histogram
                    hist={o.transformed ? WIDE.dist.after : WIDE.dist.before}
                    width={360}
                    height={44}
                    variant="spark"
                    title={`${WIDE.dist.column}, ${o.transformed ? WIDE.dist.after_label : WIDE.dist.before_label}`}
                  />
                  <Barcode
                    n={WIDE.nCounts}
                    shown={SHOWN}
                    on={o.transformed}
                    width={360}
                    height={12}
                    title={`${o.transformed ? WIDE.nCounts : 0} of ${WIDE.nCounts} count columns change`}
                  />
                </span>
              )}
              stat={(o) => (
                <>
                  <span className={s.statKey}>skew</span> {o.skew?.toFixed(1)}
                  <span className={s.statMuted}>
                    {" "}
                    · {o.transformed ? fmtInt(WIDE.nCounts) : "0"} columns change
                  </span>
                </>
              )}
              stageHeight={352}
              stage={(f, p) => <WideStage focus={f} pinned={p} />}
            />
          )}
        </LayoutGroup>
      </div>
      <WidePanel option={recorded ?? focus} status={status} />
    </div>
  );
}

function WideStage({ focus, pinned }: { focus: WideOption | null; pinned: WideOption | null }) {
  const on = !!focus?.transformed;
  const compare = !!(focus && pinned);
  const skewNow = focus?.skew ?? WIDE.options[0]!.skew ?? 0;
  const dist = (o: WideOption | null, w: number, h: number, tone: "c1" | "c2" = "c1") => (
    <Histogram
      key={`${o?.transformed ? "after" : "before"}-${tone}`}
      hist={o?.transformed ? WIDE.dist.after : WIDE.dist.before}
      width={w}
      height={h}
      variant="stage"
      tone={tone}
      title={o?.transformed ? WIDE.dist.after_label : WIDE.dist.before_label}
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
              <Prose
                text={`\`${WIDE.dataset.n_rows}\` samples × \`${fmtInt(WIDE.nCounts)}\` count columns.`}
              />
            </span>
          </>
        )}
      </div>
      <div className={s.stageBody} data-compare={compare}>
        {compare ? (
          <div className={s.compareCell}>
            <div className={s.distHead}>
              <span className={s.mono}>{WIDE.dist.column}</span>
              <span className={s.statKey}>skew</span>
              <span className={s.num2}>{pinned!.skew?.toFixed(1)}</span>
            </div>
            {dist(pinned, 250, 150, "c2")}
          </div>
        ) : null}
        <div className={compare ? s.compareCell : s.primaryNarrow}>
          <div className={s.distHead}>
            <span className={s.mono}>{WIDE.dist.column}</span>
            <span className={s.statKey}>skew</span>
            <NumberTween value={skewNow} format={fmtSkew} className={s.num2} />
            <span className={s.distSub}>{on ? WIDE.dist.after_label : "counts"}</span>
          </div>
          {dist(focus, compare ? 250 : 290, compare ? 150 : 190)}
        </div>
        {!compare ? (
          <div className={s.gridCol}>
            <div className={s.gridHead}>
              <span className="kicker">Most-changed columns</span>
              <span className={s.distSub}>gene_ · 8 of {WIDE.dataset.n_rows} rows</span>
            </div>
            <HeatGrid
              view={WIDE.table}
              transformed={on}
              width={420}
              cell={{ w: 34, h: 19 }}
              title={WIDE.table.caption}
            />
            <div className={s.barcodeRow}>
              <Barcode
                n={WIDE.nCounts}
                shown={SHOWN}
                on={on}
                width={408}
                height={16}
                title={`${SHOWN.length} shown of ${WIDE.nCounts} count columns`}
              />
              <span className={s.barcodeKey}>
                <span className="num">{SHOWN.length}</span> of{" "}
                <span className="num">{WIDE.nCounts}</span> shown
                {on ? (
                  <>
                    {" "}
                    · all <span className="num">{WIDE.nCounts}</span> change
                  </>
                ) : null}
              </span>
            </div>
          </div>
        ) : null}
      </div>
      <div className={s.stageFoot}>
        <span className={s.estimand}>not an M1 decision kind</span>
        <details className={s.why}>
          <summary>Why these 12?</summary>
          <p>{WIDE.changeMetric}</p>
        </details>
        <span className={s.basis}>all {WIDE.dataset.n_rows} rows · row-local, nothing is fit</span>
      </div>
    </div>
  );
}

const BASE = tracksOf(WIDE.lineage.before ?? WIDE.lineage.after);
const AFTER = tracksOf(WIDE.lineage.after);

function WidePanel({ option, status }: { option: WideOption | null; status: PanelStatus }) {
  const tracks = option?.transformed ? AFTER : BASE;
  const hit = option ? touched(tracks, BASE) : new Set<string>();
  return (
    <Panel status={status}>
      <RowsSection
        status={status}
        steps={[{ key: "loaded", label: "Samples loaded", n: WIDE.dataset.n_rows }]}
        split={null}
      />
      <ColumnsSection
        tracks={tracks}
        touchedIds={hit}
        status={status}
        summary={
          option
            ? option.transformed
              ? `${fmtInt(WIDE.nCounts)} change`
              : "nothing changes"
            : `${fmtInt(WIDE.dataset.n_cols)} columns`
        }
      />
    </Panel>
  );
}
