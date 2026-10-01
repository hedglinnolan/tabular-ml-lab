/**
 * /lab/m2 — the M2 design prototype (M2_CONTRACT §8, "design"): two new moments on real data.
 *
 *   S1 reshape      dietary_recalls.csv: "combine each person's two recalls" — the working table
 *                   under a reshape (gather → combine → settle), 600 → 300, with the coach's notes
 *   S2 wide         the same storyboard at 20,002 columns: the affected columns only
 *   S3 orientation  a feature-major metabolomics table turning around
 *   S4 the seal     the row flow forking with its basis named, in four draws (grouped,
 *                   chronological grouped, grouping abandoned, undetermined)
 *   S5 results      held-out scores sealed, the open-once CONSEQUENCE, and a post-seal change
 *
 * Every number is from fixture.json (docs/turbotab-next/m2/explore/capture.py). URL parameters
 * pick a state for review captures: ?s=reshape&o=first&side=with · ?s=seal&v=abandoned&f=0.2&rec=1
 * · ?s=results&phase=post.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useState, type ReactNode } from "react";
import { LayoutGroup } from "motion/react";
import { Header } from "../../components/Header";
import { DecisionSentence, History, Pending, QuestionBlock, SkipRow } from "../../components/record/blocks";
import { initial } from "../../components/stage/player";
import { Rich } from "../../components/stage/text";
import { PlayerContext, usePlayerStoreInstance } from "../../components/stage/usePlayer";
import { useMotionPrefs } from "../../motion/prefs";
import { M2Banner, type BannerModel } from "./Banner";
import { FX, fmtInt, pct } from "./data";
import { OptionList, type Opt } from "./OptionList";
import { OrientationStage, ReshapeStage, ResultsStage, SealStage, variantAt } from "./Scenes";
import type { ReshapeFixture, SealVariant } from "./types";
import { SealGlyph, sealStateOf } from "./views/SealGlyph";
import type { SealPhase } from "./views/SealedResults";
import m from "./m2.module.css";

type Scene = "reshape" | "wide" | "orientation" | "seal" | "results";
type VariantKey = SealVariant["key"];

const SCENES: { id: Scene; label: string }[] = [
  { id: "reshape", label: "Reshape" },
  { id: "wide", label: "Wide" },
  { id: "orientation", label: "Orientation" },
  { id: "seal", label: "The seal" },
  { id: "results", label: "Results" },
];
const VARIANTS: { id: VariantKey; label: string }[] = [
  { id: "grouped", label: "grouped" },
  { id: "chronological", label: "chronological" },
  { id: "abandoned", label: "abandoned" },
  { id: "undetermined", label: "undetermined" },
];

function param(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

function setParams(p: Record<string, string | null>) {
  const url = new URL(window.location.href);
  for (const [k, v] of Object.entries(p)) {
    if (v === null) url.searchParams.delete(k);
    else url.searchParams.set(k, v);
  }
  window.history.replaceState(null, "", url);
}

const METHOD_ORDER = ["mean", "first", "last", "change"];

function reshapeOptions(fx: ReshapeFixture): Opt[] {
  const n = fx.dataset.rows;
  const u = fx.unit_noun;
  const nn = fx.noun;
  return METHOD_ORDER.filter((k) => fx.methods[k]).map((k) => {
    const meth = fx.methods[k]!;
    const line =
      k === "mean"
        ? `One row per ${u}: each value the average of their ${fx.per_unit} ${nn}s.`
        : k === "first"
          ? `Each ${u}'s earliest ${nn} stays whole; the other ${fmtInt(n - meth.n_after)} leave.`
          : k === "last"
            ? `Each ${u}'s latest ${nn} stays whole; the other ${fmtInt(n - meth.n_after)} leave.`
            : `Last minus first for each measure; constant columns pass through.`;
    return { key: k, label: meth.label, line, tag: meth.recommended ? "usual" : undefined };
  });
}

function splitOptions(v: SealVariant): Opt[] {
  const floor = v.n_rows < 100;
  const fracs = [0.1, 0.2, 0.3];
  const held = fracs.map((f): Opt => {
    const d = variantAt(v, f);
    const units = d.n_hold_units !== null && v.group_column ? ` (${d.n_hold_units} ${v.unit_noun})` : "";
    const rmse = FX.seal.sizes.find((s) => s.fraction === f);
    const line =
      v.key === "grouped" && rmse?.rmse_pm
        ? `${d.n_hold_rows} ${v.unit_noun} sealed: their RMSE known to about ±${Math.round(rmse.rmse_pm * 100)}%.`
        : v.key === "chronological"
          ? `The ${d.n_hold_units} ${v.unit_noun} seen last sealed${d.boundary ? `, last visit after ${d.boundary}` : ""}.`
          : `${d.n_hold_rows} ${v.row_noun}${units} sealed for one final score.`;
    return { key: String(f), label: `Hold out ${pct(f)}`, line };
  });
  const cv: Opt = {
    key: "0",
    label: "Cross-validation only",
    line: floor
      ? `${v.n_rows} rows: a held-out set this small measures little; cross-validation uses every row.`
      : "No rows are sealed; every score comes from cross-validation on all rows.",
    tag: floor ? `under 100 rows` : undefined,
  };
  return floor ? [cv, ...held] : [...held, cv];
}

function earlier(text: string) {
  return (
    <p className={m.earlier}>
      <Rich text={text} />
    </p>
  );
}

export function M2Screen() {
  const [scene, setScene] = useState<Scene>(() => {
    const s = param("s");
    return SCENES.some((x) => x.id === s) ? (s as Scene) : "reshape";
  });
  const [method, setMethod] = useState<string>(() => param("o") ?? "mean");
  const [orient, setOrient] = useState<string>(() => param("o") ?? "features");
  const [variant, setVariant] = useState<VariantKey>(() => {
    const v = param("v");
    return VARIANTS.some((x) => x.id === v) ? (v as VariantKey) : "grouped";
  });
  const [fraction, setFraction] = useState<number>(() => Number(param("f") ?? "0.2"));
  const [recorded, setRecorded] = useState<Partial<Record<string, string>>>(() =>
    param("rec") === "1" ? { [param("s") ?? ""]: "1" } : {},
  );
  const [phase, setPhase] = useState<SealPhase>(() => {
    const p = param("phase");
    return p === "opened" || p === "post" ? p : "sealed";
  });

  const store = usePlayerStoreInstance();
  const { reduced } = useMotionPrefs();
  useEffect(() => store.setReduced(reduced), [store, reduced]);
  // Review captures step the clock by hand (the production player's own hook for it).
  useEffect(() => {
    (window as unknown as { __m2?: unknown }).__m2 = { store };
  }, [store]);
  // A new scene starts on the side the URL names (default: your data now).
  useLayoutEffect(() => {
    store.reset(initial(1, param("side") === "with" ? "with" : "now"));
  }, [store, scene, variant]);
  // Space is the stage's flip (frontend CLAUDE.md).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " ") return;
      const t = e.target as HTMLElement | null;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
      e.preventDefault();
      store.dispatch({ type: "flip" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [store]);

  const go = useCallback((s: Scene) => {
    setScene(s);
    setParams({ s, o: null, v: null, f: null, rec: null, phase: null, side: null });
  }, []);

  const record = (key: string) => {
    setRecorded((r) => ({ ...r, [scene === "seal" ? `seal-${variant}` : scene]: key }));
    // The seal moment: recording draws the seal (plays to "with this choice" when it is not there).
    store.dispatch({ type: "show", side: "with" });
  };
  const rec = (k: string) => recorded[k] ?? null;

  const reshapeFx = scene === "wide" ? FX.wide : FX.reshape;
  const v = FX.seal.variants[variant];
  const sealKey = `seal-${variant}`;
  const sealRecorded = rec(sealKey) !== null || recorded["seal"] === "1";

  const banner = useMemo((): BannerModel => {
    const r = FX.results;
    const best = [...r.fits.residual.models].sort((a, b) => b.cv.mean - a.cv.mean)[0]!;
    const post = phase === "post";
    const nowFit = post ? r.fits.density : r.fits.residual;
    const nowBest = nowFit.models.find((x) => x.family === best.family)!;
    switch (scene) {
      case "reshape":
      case "wide":
        return {
          now: "rows",
          nowLabel: "combining rows",
          flow: [{ n: reshapeFx.dataset.rows }],
          split: null,
          columns: { from: reshapeFx.dataset.cols, unit: "as loaded" },
          models: [],
          result: null,
        };
      case "orientation":
        return {
          now: "rows",
          nowLabel: "which way round",
          flow: [{ n: FX.orientation.before.rows }],
          split: null,
          columns: { from: FX.orientation.before.cols, unit: "as loaded" },
          models: [],
          result: null,
        };
      case "seal": {
        const at = variantAt(v, fraction || 0.2);
        const flow =
          variant === "grouped" ? [{ n: FX.reshape.dataset.rows }, { n: v.n_rows, via: "mean" }] : [{ n: v.n_rows }];
        return {
          now: "rows",
          nowLabel: "held-out rows",
          flow,
          split:
            sealRecorded && fraction > 0
              ? { train: at.n_train_rows, held: at.n_hold_rows, seal: sealStateOf(v.basis), recorded: true }
              : null,
          columns: { from: v.n_cols, unit: "as loaded" },
          models: [],
          result: null,
        };
      }
      default:
        return {
          now: "result",
          nowLabel: phase === "sealed" ? "open the seal" : "the result",
          flow: [{ n: FX.reshape.dataset.rows }, { n: r.n_train + r.n_holdout, via: "mean" }],
          split: { train: r.n_train, held: r.n_holdout, seal: "grouped", recorded: true },
          columns: { from: 9, to: 9, method: post ? "density" : "residual" },
          models: r.fits.residual.models.map((x) => x.label.toLowerCase()),
          result: {
            metric: r.label,
            value: nowBest.cv.mean,
            basis: "CV",
            family: best.label.toLowerCase(),
            held: phase === "sealed" ? null : nowBest.holdout,
            post,
          },
        };
    }
  }, [scene, reshapeFx, v, variant, fraction, sealRecorded, phase]);

  // ── the Record ──
  let recordBody: ReactNode;
  if (scene === "reshape" || scene === "wide") {
    const fx = reshapeFx;
    const sp = fx.repeats.spacing;
    const recKey = rec(scene);
    const shown = fx.methods[method] ? method : "mean";
    recordBody = (
      <>
        {earlier(scene === "reshape" ? "#1–#3 · lens dietary · outcome `hba1c` · prediction" : "#1–#3 · lens metabolomics · outcome `responder` · prediction")}
        <DecisionSentence layoutId="grain" subject="the grain" meta="#4">
          <Rich
            text={`${fx.unit_noun === "person" ? "People" : "Samples"} repeat: \`${fx.id_column}\` names the ${fx.unit_noun} (\`${fmtInt(fx.n_units)}\` ${fx.unit_noun === "person" ? "people" : `${fx.unit_noun}s`}, \`${fx.per_unit}\` rows each).`}
          />
        </DecisionSentence>
        <SkipRow layoutId="repeats" onAsk={() => {}}>
          <Rich
            text={
              sp
                ? `Not asked: repeats, not time points — \`${sp.column}\` gaps run ${sp.min_days} to ${sp.max_days} days (median ${sp.median_days}).`
                : `Not asked: repeats — \`${fx.repeats.replicate_index}\` numbers each ${fx.unit_noun}'s rows and there is no date column.`
            }
          />
        </SkipRow>
        <DecisionSentence layoutId="unit" subject="the unit" meta="#5">
          One row of the analysis is a {fx.unit_noun}.
        </DecisionSentence>
        {recKey ? (
          <DecisionSentence layoutId="agg" subject="the combination" meta="#6" onChange={() => setRecorded((r) => ({ ...r, [scene]: undefined }))}>
            <Rich
              text={`${fx.methods[recKey]!.sentence} \`${fmtInt(fx.dataset.rows)}\` rows became \`${fmtInt(fx.methods[recKey]!.n_after)}\`, one per \`${fx.id_column}\`.`}
            />
          </DecisionSentence>
        ) : (
          <QuestionBlock
            layoutId="agg"
            kicker="Combining rows"
            title={`How should each ${fx.unit_noun}'s rows be combined?`}
            why={
              <Rich
                text={
                  scene === "reshape"
                    ? `You have ${fx.per_unit} recalls per person. Their mean reduces the day-to-day error that weakens diet–outcome associations.`
                    : `You have ${fx.per_unit} assays per sample. Their mean reduces measurement error; it does not lose information.`
                }
              />
            }
          >
            <OptionList
              options={reshapeOptions(fx)}
              shown={shown}
              recorded={null}
              onShow={(k) => {
                setMethod(k);
                setParams({ o: k });
              }}
              onRecord={record}
            />
          </QuestionBlock>
        )}
        <p className={m.then}>
          <span className={m.thenKicker}>Then</span> Eligibility · Held-out rows · Energy adjustment · Model families
        </p>
      </>
    );
  } else if (scene === "orientation") {
    const fx = FX.orientation;
    const recKey = rec("orientation");
    recordBody = (
      <>
        <DecisionSentence layoutId="lens" subject="the lens" meta="#1">
          The lens is metabolomics.
        </DecisionSentence>
        {recKey ? (
          <DecisionSentence layoutId="orient" subject="the orientation" meta="#2" onChange={() => setRecorded((r) => ({ ...r, orientation: undefined }))}>
            <Rich text={recKey === "features" ? fx.methods_sentence : fx.kept_sentence} />
          </DecisionSentence>
        ) : (
          <QuestionBlock
            layoutId="orient"
            kicker="Orientation"
            title="Which way round is this table?"
            why={
              <Rich
                text={`Across ${fmtInt(fx.before.rows)} rows, row means differ by orders of magnitude and column means barely differ: in an assay table, features in rows.`}
              />
            }
          >
            <OptionList
              options={[
                { key: "samples", label: "Rows are samples", line: "Nothing changes; the record states the table was checked." },
                {
                  key: "features",
                  label: "Rows are features",
                  line: `The table turns around before any diagnosis: ${fmtInt(fx.after.rows)} rows, one per sample.`,
                },
              ]}
              shown={orient}
              recorded={null}
              onShow={(k) => {
                setOrient(k);
                setParams({ o: k });
              }}
              onRecord={record}
            />
          </QuestionBlock>
        )}
        <Pending>
          <Rich text="What are you predicting? Withheld until this is answered: on a turned-around table the column list is a list of samples." />
        </Pending>
      </>
    );
  } else if (scene === "seal") {
    const at = variantAt(v, fraction || 0.2);
    const sentences: Record<VariantKey, ReactNode> = {
      grouped: (
        <>
          {earlier("#1–#5 · lens dietary · outcome `hba1c` · people repeat · one row per person")}
          <DecisionSentence layoutId="agg" subject="the combination" meta="#6">
            <Rich text={`Each person's \`2\` recalls were averaged: \`600\` rows became \`300\`, one per \`participant_id\`.`} />
          </DecisionSentence>
          <DecisionSentence layoutId="excl" subject="eligibility" meta="#7">
            No exclusion criteria were applied; the study is about everyone here.
          </DecisionSentence>
        </>
      ),
      chronological: (
        <>
          {earlier("#1–#3 · lens clinical · outcome `progressed`, event `1` · prediction")}
          <DecisionSentence layoutId="grain" subject="the grain" meta="#4">
            <Rich text={`Subjects repeat: \`subject_id\` names the person (\`200\` subjects, \`3\` visits each).`} />
          </DecisionSentence>
          <SkipRow layoutId="repeats" onAsk={() => {}}>
            <Rich text="Not asked: time points — `visit_date` is 90 days apart at the median (80 to 100)." />
          </SkipRow>
          <DecisionSentence layoutId="unit" subject="the unit" meta="#5">
            One row of the analysis is a visit.
          </DecisionSentence>
          <DecisionSentence layoutId="temporal" subject="temporal prediction" meta="#6">
            Yes: a later outcome is predicted from earlier visits.
          </DecisionSentence>
        </>
      ),
      abandoned: (
        <>
          {earlier("#1–#3 · lens clinical · outcome `progressed`, event `1` · prediction")}
          <DecisionSentence layoutId="grain" subject="the grain" meta="#4">
            <Rich text={`Subjects repeat: \`subject_id\` names the person (\`${v.n_units}\` subjects, \`3\` visits each).`} />
          </DecisionSentence>
          <DecisionSentence layoutId="unit" subject="the unit" meta="#5">
            One row of the analysis is a visit.
          </DecisionSentence>
          <DecisionSentence layoutId="temporal" subject="temporal prediction" meta="#6">
            No: outcomes and measurements are from the same visit.
          </DecisionSentence>
        </>
      ),
      undetermined: (
        <>
          {earlier("#1–#3 · lens dietary · outcome `hba1c` · prediction")}
          <DecisionSentence layoutId="grain" subject="the grain" meta="#4">
            Not sure whether one person can appear in more than one row.
          </DecisionSentence>
        </>
      ),
    };
    const done = sealRecorded;
    recordBody = (
      <LayoutGroup id={`seal-${variant}`}>
        {sentences[variant]}
        {done ? (
          <DecisionSentence
            layoutId="split"
            subject="the held-out rows"
            meta={
              <span className={m.metaSeal}>
                <SealGlyph state={sealStateOf(v.basis)} recorded size={13} /> #8
              </span>
            }
            onChange={() => setRecorded((r) => ({ ...r, [sealKey]: undefined, seal: undefined }))}
            note={at.exploratory ? <Rich text="Held-out scores will carry an exploratory label." /> : undefined}
          >
            <Rich
              text={
                fraction === 0
                  ? "No rows were sealed; every score comes from cross-validation."
                  : v.key === "grouped"
                    ? `\`${at.n_hold_rows}\` of \`${v.n_rows}\` participants were sealed as held-out rows, grouped by \`participant_id\` (seed \`0\`).`
                    : v.key === "chronological"
                      ? `The \`${at.n_hold_units}\` subjects seen last (\`${at.n_hold_rows}\` visits) were sealed: chronological, grouped by \`subject_id\`.`
                      : v.key === "abandoned"
                        ? `\`${at.n_hold_rows}\` of \`${v.n_rows}\` visits were sealed by row: ${v.n_units} subjects are too few to hold out whole.`
                        : `\`${at.n_hold_rows}\` of \`${v.n_rows}\` recalls were sealed by row; the basis is undetermined.`
              }
            />
          </DecisionSentence>
        ) : (
          <QuestionBlock
            layoutId="split"
            kicker="Held-out rows · the seal"
            title="How many rows should be held out for one final, untouched score?"
            why="Held-out rows are sealed now, before any modeling choice, and opened once at the end."
          >
            <OptionList
              options={splitOptions(v)}
              shown={String(fraction)}
              recorded={null}
              onShow={(k) => {
                setFraction(Number(k));
                setParams({ f: k });
              }}
              onRecord={record}
            />
          </QuestionBlock>
        )}
        <p className={m.then}>
          <span className={m.thenKicker}>Then</span> {variant === "grouped" ? "Energy adjustment · " : ""}Model families · Results
        </p>
      </LayoutGroup>
    );
  } else {
    const r = FX.results;
    const post = phase === "post";
    recordBody = (
      <>
        {earlier("#1–#7 · dietary · `hba1c` · prediction · people repeat · one row per person · mean of 2 recalls · everyone")}
        <DecisionSentence
          layoutId="split"
          subject="the held-out rows"
          meta={
            <span className={m.metaSeal}>
              <SealGlyph state="grouped" recorded size={13} /> #8
            </span>
          }
        >
          <Rich text={`\`${r.n_holdout}\` of \`${r.n_train + r.n_holdout}\` participants were sealed as held-out rows, grouped by \`participant_id\` (seed \`0\`).`} />
        </DecisionSentence>
        <DecisionSentence
          layoutId="energy"
          subject="energy adjustment"
          meta={post ? "#12" : "#9"}
          onChange={phase === "opened" ? () => {
            setPhase("post");
            setParams({ phase: "post" });
          } : undefined}
          note={
            post ? (
              <>
                <span className={m.postNote}>
                  <span className={m.postTag}>post-seal</span> changed after the seal was opened (#11)
                </span>
                <History
                  items={[
                    {
                      id: "e9",
                      seq: 9,
                      when: "before opening",
                      sentence: "Energy was adjusted by the residual method.",
                    },
                  ]}
                />
              </>
            ) : undefined
          }
        >
          <Rich
            text={
              post
                ? "Energy was adjusted by nutrient density on training rows: `protein_g`, `fat_g`, `carbohydrate_g` per `energy_kcal`."
                : "Energy was adjusted by the residual method on training rows: `protein_g`, `fat_g`, `carbohydrate_g` on `energy_kcal`."
            }
          />
        </DecisionSentence>
        <DecisionSentence layoutId="models" subject="the model families" meta="#10">
          Three model families were chosen: linear, elastic net and boosted trees.
        </DecisionSentence>
        {phase !== "sealed" ? (
          <DecisionSentence layoutId="open" subject="opening the seal" meta="#11">
            <Rich text={`The seal was opened once: held-out ${r.label} for \`${r.n_holdout}\` participants is fixed in the record.`} />
          </DecisionSentence>
        ) : (
          <Pending>
            <Rich text="Held-out rows: sealed. Opening them is at the end of the Results, once." />
          </Pending>
        )}
      </>
    );
  }

  // ── the stage ──
  let stage: ReactNode;
  if (scene === "reshape" || scene === "wide") {
    const meth = reshapeFx.methods[method] ?? reshapeFx.methods.mean!;
    stage = (
      <ReshapeStage
        key={scene}
        fx={reshapeFx}
        method={meth}
        recorded={rec(scene) !== null}
        onRecord={() => record(meth.key)}
        wide={scene === "wide"}
      />
    );
  } else if (scene === "orientation") {
    stage = <OrientationStage option={orient} recorded={rec("orientation") !== null} onRecord={() => record(orient)} />;
  } else if (scene === "seal") {
    stage = <SealStage key={variant} v={v} fraction={fraction} recorded={sealRecorded} onRecord={() => record(String(fraction))} />;
  } else {
    stage = (
      <ResultsStage
        phase={phase}
        onOpen={() => {
          setPhase("opened");
          setParams({ phase: "opened" });
        }}
      />
    );
  }

  const ds =
    scene === "orientation"
      ? FX.orientation.dataset
      : scene === "seal"
        ? { name: v.dataset, rows: v.n_rows, cols: null }
        : scene === "results"
          ? { name: "dietary_recalls.csv, combined", rows: FX.results.n_train + FX.results.n_holdout, cols: null }
          : reshapeFx.dataset;

  return (
    <div className={m.screen}>
      <Header>
        <span className={m.dsName}>{ds.name}</span>
        <span className={m.dsSize}>
          {fmtInt(ds.rows)} rows{ds.cols ? ` × ${fmtInt(ds.cols)} columns` : ""}
        </span>
        <nav className={m.scenes} aria-label="Prototype scenes">
          {SCENES.map((x, i) => (
            <button
              key={x.id}
              type="button"
              className={m.sceneBtn}
              aria-current={scene === x.id ? "page" : undefined}
              onClick={() => go(x.id)}
              data-scene={x.id}
            >
              <span className={m.sceneNum}>S{i + 1}</span> {x.label}
            </button>
          ))}
        </nav>
        {scene === "seal" ? (
          <nav className={m.scenes} aria-label="Seal basis">
            {VARIANTS.map((x) => (
              <button
                key={x.id}
                type="button"
                className={m.sceneBtn}
                aria-current={variant === x.id ? "page" : undefined}
                onClick={() => {
                  setVariant(x.id);
                  setParams({ v: x.id, rec: null });
                }}
                data-variant={x.id}
              >
                {x.label}
              </button>
            ))}
          </nav>
        ) : null}
      </Header>
      <M2Banner model={banner} />
      <PlayerContext.Provider value={store}>
        <div className={m.layout}>
          <main className={m.record} key={`${scene}-${variant}`}>
            {recordBody}
          </main>
          <aside className={m.stageCol} aria-label="The stage">
            {stage}
          </aside>
        </div>
      </PlayerContext.Provider>
    </div>
  );
}
