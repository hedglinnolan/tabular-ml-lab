/**
 * The pipeline banner (M1_CONTRACT §10; BLUEPRINT §11.1): "once over the world" — the
 * pipeline as it stands, always visible above the working window. Rows through the flow,
 * the column path to the model matrix, the models, the primary result.
 *
 * It is also the map of where you are: the segment the open question acts on wears the
 * accent ("now"). Each segment is a button that puts its full view on the stage; pressing it
 * again goes back to live. When an answer changes, the segments it reaches veil in order
 * (propagate) and un-veil as their fresh results arrive; numbers tween to their new values.
 */
import { Fragment } from "react";
import type { ProjectView } from "../../api/schema";
import { NumberTween } from "../../motion/NumberTween";
import { StaleVeil } from "../../motion/StaleVeil";
import { useStageFocus, type BannerSegment } from "../../state/focus";
import { useStage } from "../../state/stages";
import { cx } from "../../util/format";
import {
  deriveBanner,
  formatMetric,
  type BannerModel,
  type ColumnsSegment,
  type ModelsSegment,
  type ResultSegment,
  type RowsSegment,
  type Segment,
} from "./derive";
import s from "./Banner.module.css";

export function Banner({ pid, view }: { pid: string; view: ProjectView }) {
  const ingest = useStage(pid, view, "ingest");
  const cohort = useStage(pid, view, "cohort");
  const split = useStage(pid, view, "split");
  const design = useStage(pid, view, "design");
  const fit = useStage(pid, view, "fit");
  const shelf = useStage(pid, view, "shelf");
  const model = deriveBanner({ view, ingest, cohort, split, design, fit, shelf });
  return <BannerView model={model} />;
}

export function BannerView({ model }: { model: BannerModel }) {
  const { focus, toggle } = useStageFocus();
  const pressed = focus.kind === "banner" ? focus.segment : null;
  return (
    <nav className={s.banner} aria-label="The pipeline" data-testid="banner">
      <ol className={s.track}>
        {model.segments.map((seg, i) => {
          const now = model.now === seg.key;
          return (
            <Fragment key={seg.key}>
              {i > 0 ? (
                <li className={s.join} aria-hidden="true">
                  ›
                </li>
              ) : null}
              <li className={s.item}>
                <button
                  type="button"
                  className={cx(s.segment, now && s.now)}
                  aria-pressed={pressed === seg.key}
                  aria-current={now ? "step" : undefined}
                  aria-label={`${seg.summary}${now ? ` Now: ${model.nowLabel}.` : ""} ${
                    pressed === seg.key ? "Shown on the stage." : "Show it on the stage."
                  }`}
                  onClick={() => toggle({ kind: "banner", segment: seg.key as BannerSegment })}
                  data-segment={seg.key}
                  data-now={now || undefined}
                  data-testid={`banner-${seg.key}`}
                >
                  <span className={s.kicker}>
                    {seg.label}
                    {now ? (
                      <span className={s.nowTag}>
                        {model.nowWaiting ? "next" : "now"} · {model.nowLabel}
                      </span>
                    ) : null}
                  </span>
                  <StaleVeil state={seg.veil} order={seg.order} compact className={s.veil}>
                    <span className={s.values} data-veil-body={seg.veil}>
                      <SegmentValues seg={seg} />
                    </span>
                  </StaleVeil>
                </button>
              </li>
            </Fragment>
          );
        })}
      </ol>
    </nav>
  );
}

function SegmentValues({ seg }: { seg: Segment }) {
  if (seg.waiting) return <span className={s.waiting}>{seg.waiting}</span>;
  switch (seg.key) {
    case "rows":
      return <Rows seg={seg} />;
    case "columns":
      return <Columns seg={seg} />;
    case "models":
      return <Models seg={seg} />;
    case "result":
      return <Result seg={seg} />;
  }
}

const Arrow = () => (
  <span className={s.arrow} aria-hidden="true">
    →
  </span>
);

function Rows({ seg }: { seg: RowsSegment }) {
  return (
    <>
      {seg.flow.map((f, i) => (
        <Fragment key={`${i}-${f.label}`}>
          {i > 0 ? <Arrow /> : null}
          <span className={s.count} title={f.label}>
            <NumberTween value={f.n} data-testid={`banner-rows-${i}`} />
          </span>
        </Fragment>
      ))}
      {seg.train !== null ? (
        <>
          <span className={s.fork} aria-hidden="true">
            ▸
          </span>
          {seg.cvOnly ? (
            <span className={s.word}>cross-validation only</span>
          ) : (
            <span className={s.split}>
              <span className={s.word}>train</span>
              <NumberTween value={seg.train} data-testid="banner-train" />
              <span className={s.bar} aria-hidden="true">
                |
              </span>
              <span className={s.word}>held out</span>
              <NumberTween value={seg.holdout ?? 0} data-testid="banner-holdout" />
            </span>
          )}
        </>
      ) : null}
    </>
  );
}

function Columns({ seg }: { seg: ColumnsSegment }) {
  if (seg.from === null) return null;
  return (
    <>
      <NumberTween value={seg.from} data-testid="banner-columns-from" />
      {seg.to !== null ? (
        <>
          <Arrow />
          <NumberTween value={seg.to} data-testid="banner-columns-to" />
          {seg.method ? <span className={s.method}>{seg.method}</span> : null}
        </>
      ) : (
        <span className={s.word}>{seg.unit}</span>
      )}
    </>
  );
}

function Models({ seg }: { seg: ModelsSegment }) {
  return (
    <>
      <NumberTween value={seg.labels.length} />
      <span className={s.word} title={seg.labels.join(", ")}>
        {seg.labels.join(" · ")}
      </span>
    </>
  );
}

function Result({ seg }: { seg: ResultSegment }) {
  if (seg.value === null) return null;
  return (
    <>
      <span className={s.metric}>{seg.metric}</span>
      <NumberTween value={seg.value} format={formatMetric} data-testid="banner-result-value" />
      <span className={s.basis}>{seg.basis}</span>
      {seg.family ? <span className={s.word}>{seg.family}</span> : null}
    </>
  );
}
