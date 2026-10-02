/**
 * The pipeline banner with M2's two additions (prototype; production is
 * components/banner/Banner.tsx, whose styles this reuses):
 *   - a combination step in the row flow is its own arrow, carrying its method ("600 →mean 300"),
 *     so a reshape reads differently from an exclusion;
 *   - the held-out count carries the seal glyph in its basis state, and the result says whether
 *     the held-out scores are sealed, opened once, or post-seal.
 */
import { Fragment } from "react";
import { cx } from "../../util/format";
import b from "../../components/banner/Banner.module.css";
import { fmtInt, fmtMetric } from "./data";
import { SealGlyph, type SealState } from "./views/SealGlyph";
import s from "./m2.module.css";

export interface BannerModel {
  now: "rows" | "columns" | "models" | "result";
  nowLabel: string;
  flow: { n: number; via?: string }[];
  split: { train: number; held: number; seal: SealState; recorded: boolean } | null;
  columns: { from: number; to?: number; method?: string; unit?: string };
  models: string[];
  result: { metric: string; value: number; basis: string; family: string; held?: number | null; post?: boolean } | null;
}

export function M2Banner({ model }: { model: BannerModel }) {
  const segs = ["rows", "columns", "models", "result"] as const;
  return (
    <nav className={b.banner} aria-label="The pipeline" data-testid="banner">
      <ol className={b.track}>
        {segs.map((key, i) => {
          const now = model.now === key;
          return (
            <Fragment key={key}>
              {i > 0 ? (
                <li className={b.join} aria-hidden="true">
                  ›
                </li>
              ) : null}
              <li className={b.item}>
                <span className={cx(b.segment, now && b.now)} data-segment={key}>
                  <span className={b.kicker}>
                    {key}
                    {now ? <span className={b.nowTag}>now · {model.nowLabel}</span> : null}
                  </span>
                  <span className={b.values}>
                    {key === "rows" ? <Rows model={model} /> : null}
                    {key === "columns" ? (
                      <>
                        {fmtInt(model.columns.from)}
                        {model.columns.to !== undefined ? (
                          <>
                            <span className={b.arrow}>→</span>
                            {fmtInt(model.columns.to)}
                          </>
                        ) : null}
                        {model.columns.method ? <span className={b.method}>{model.columns.method}</span> : null}
                        {model.columns.unit ? <span className={b.word}>{model.columns.unit}</span> : null}
                      </>
                    ) : null}
                    {key === "models" ? (
                      model.models.length ? (
                        <>
                          {model.models.length} <span className={b.word}>{model.models.join(" · ")}</span>
                        </>
                      ) : (
                        <span className={b.waiting}>not chosen yet</span>
                      )
                    ) : null}
                    {key === "result" ? (
                      model.result ? (
                        <>
                          <span className={b.metric}>{model.result.metric}</span>
                          {fmtMetric(model.result.value)}
                          <span className={b.basis}>{model.result.basis}</span>
                          <span className={b.word}>{model.result.family}</span>
                          {model.result.held === null ? (
                            <span className={s.bannerSealed}>
                              <SealGlyph state="grouped" recorded size={12} /> held out sealed
                            </span>
                          ) : model.result.held !== undefined ? (
                            <span className={model.result.post ? s.bannerPost : s.bannerHeld}>
                              held out {fmtMetric(model.result.held)}
                              {model.result.post ? " · post-seal" : ""}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className={b.waiting}>after the fit</span>
                      )
                    ) : null}
                  </span>
                </span>
              </li>
            </Fragment>
          );
        })}
      </ol>
    </nav>
  );
}

function Rows({ model }: { model: BannerModel }) {
  return (
    <>
      {model.flow.map((f, i) => (
        <Fragment key={i}>
          {i > 0 ? (
            <span className={b.arrow} aria-hidden="true">
              →{f.via ? <span className={s.via}>{f.via}</span> : null}
            </span>
          ) : null}
          <span className={b.count}>{fmtInt(f.n)}</span>
        </Fragment>
      ))}
      {model.split ? (
        <>
          <span className={b.fork} aria-hidden="true">
            ▸
          </span>
          <span className={b.split}>
            <span className={b.word}>train</span>
            {fmtInt(model.split.train)}
            <span className={b.bar} aria-hidden="true">
              |
            </span>
            <span className={b.word}>held out</span>
            {fmtInt(model.split.held)}
            <SealGlyph state={model.split.seal} recorded={model.split.recorded} size={14} className={s.bannerGlyph} />
          </span>
        </>
      ) : null}
    </>
  );
}
