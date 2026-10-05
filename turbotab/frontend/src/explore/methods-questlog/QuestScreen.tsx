/**
 * /lab/methods-questlog — design prototype (angle B, "quest log") of the living methods section
 * (BLUEPRINT §11.4). Left: the methods section as an objective list, sections in the reporting
 * guideline's order, each with its open slots counted and finished ones collapsed to their
 * sentence. Center: the current objective as one focused card. Right: the canvas (the production
 * stage's pieces). The whole document is one press away (M). Above it all, the pipeline banner.
 *
 * Every sentence, guess, piece of evidence and number is the real server's, captured on the
 * NHANES export (fixture.json; capture_drive.py, trim.py). `?m=` picks the moment under review:
 * 1–9, with 8a and 8b for the two encounters of one concept.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { FitArtifact, PreviewResult, ShelfArtifact, SplitArtifact } from "../../api/m1-stage-types";
import type { ProjectView } from "../../api/schema";
import { BannerView } from "../../components/banner/Banner";
import { deriveBanner, type BannerInput } from "../../components/banner/derive";
import { Header } from "../../components/Header";
import { ColumnsContext } from "../../components/stage/text";
import { StageFocusProvider } from "../../state/focus";
import { adjustmentLineage, ComparisonCanvas, exposureLineage, PreviewCanvas, ResultsCanvas } from "./Canvas";
import { AdjustmentCard, EstimandCard, energyPreview, Frame, PhraseCard, ReadingsCard, RolesCard } from "./Cards";
import { FX, INF, isPreview, type MethodsLine, type Moment } from "./data";
import { MethodsDoc } from "./Doc";
import { Rail } from "./Rail";
import { readingSlots } from "./readings";
import { sectionsOf, type Extras, type Guideline, type Section } from "./sections";
import s from "./questlog.module.css";

export const MOMENTS = ["1", "2", "3", "4", "5", "6", "7", "8a", "8b", "9"] as const;
export type MomentKey = (typeof MOMENTS)[number];

interface Config {
  guideline: Guideline;
  moment: Moment;
  extras: Extras;
  current: { section: string; item: string } | null;
  doc: boolean;
}

const SLOTS = readingSlots();
const ROLE_AND_CODE = SLOTS.filter((r) => r.kind !== "unit").length;

/** M9's state: the M2 state with the three readings the drive confirmed one at a time. */
function withSingles(m: Moment): Moment {
  const extra: MethodsLine[] = INF.singles.map((x, i) => ({
    record_id: `single-${i}`,
    seq: x.record.seq,
    kind: "confirm_reading",
    sentence: x.record.sentence,
    in_force: true,
    post_seal: false,
    after_estimates: false,
  }));
  return { ...m, methods: { ...m.methods, lines: [...m.methods.lines, ...extra] } };
}

function analyzedOf(m: Moment): number | null {
  const missing = m.view.interview.find((x) => x.key === "missing");
  const cohort = m.stages.cohort?.artifact as { n_final?: number } | undefined;
  return missing?.status === "answered" && cohort?.n_final ? cohort.n_final : null;
}

function configOf(key: MomentKey): Config {
  const M = INF.moments;
  const ex = (m: Moment, open: number, waiting = false): Extras => ({ readings: { open, waiting }, analyzed: analyzedOf(m) });
  switch (key) {
    case "1":
      return { guideline: "STROBE-nut", moment: M.m1, extras: ex(M.m1, 0, true), current: { section: "data", item: "roles" }, doc: false };
    case "2":
      return { guideline: "STROBE-nut", moment: M.m2, extras: ex(M.m2, SLOTS.length), current: { section: "data", item: "readings" }, doc: false };
    case "9": {
      const m = withSingles(M.m2);
      return { guideline: "STROBE-nut", moment: m, extras: ex(m, SLOTS.length - INF.singles.length), current: { section: "data", item: "readings" }, doc: false };
    }
    case "3":
    case "8a":
      return { guideline: "STROBE-nut", moment: M.m3, extras: ex(M.m3, ROLE_AND_CODE), current: { section: "variables", item: "estimand" }, doc: false };
    case "4":
      return { guideline: "STROBE-nut", moment: M.m4, extras: ex(M.m4, ROLE_AND_CODE), current: { section: "variables", item: "adjustment" }, doc: false };
    case "5":
    case "8b":
      return {
        guideline: "STROBE-nut",
        moment: M.m5,
        extras: ex(M.m5, ROLE_AND_CODE),
        current: { section: "quantitative", item: "energy_adjustment" },
        doc: false,
      };
    case "6":
      return { guideline: "STROBE-nut", moment: M.m6, extras: ex(M.m6, 0), current: null, doc: true };
    case "7": {
      const m = FX.prediction.moment;
      return { guideline: "TRIPOD+AI", moment: m, extras: ex(m, 0), current: null, doc: true };
    }
  }
}

function readParams() {
  const q = new URLSearchParams(window.location.search);
  const m = q.get("m");
  return {
    key: (MOMENTS as readonly string[]).includes(m ?? "") ? (m as MomentKey) : ("1" as MomentKey),
    shot: q.has("shot"),
    appendix: q.has("appendix"),
    rest: q.get("rest"),
  };
}

function banner(m: Moment) {
  const st = m.stages as unknown as Record<string, BannerInput["ingest"]>;
  const model = deriveBanner({
    view: m.view as unknown as ProjectView,
    ingest: st.ingest,
    oriented: st.oriented as BannerInput["oriented"],
    working: st.working as BannerInput["working"],
    cohort: st.cohort as BannerInput["cohort"],
    split: st.split as BannerInput["split"],
    design: st.design as BannerInput["design"],
    fit: st.fit as BannerInput["fit"],
    shelf: st.shelf as BannerInput["shelf"],
  });
  // The production banner states a cross-validated MSE under inference, where MODELING_SEQUENCE
  // §1 row 11 shows none: under inference the result is the exposure's primary estimate.
  const result = model.segments[3];
  if (m.view.state.purpose === "inference" && result.value !== null) {
    const primary = INF.effects.families[0]!.sequence.find((f) => f.key === "model_2")!.effects[0]!;
    model.segments[3] = {
      ...result,
      metric: "sugar",
      value: primary.estimate,
      basis: "Model 2",
      family: "per unit",
      summary: `Result: the primary estimate for sugar, ${primary.estimate.toFixed(4)} per unit.`,
    };
  }
  return model;
}

function findSection(sections: Section[], key: string | undefined) {
  return sections.find((x) => x.key === key);
}

export function QuestScreen() {
  const [params] = useState(readParams);
  const key = params.key;
  const cfg = useMemo(() => configOf(key), [key]);
  const sections = useMemo(() => sectionsOf(cfg.moment, cfg.extras, cfg.guideline), [cfg]);
  const [current, setCurrent] = useState(cfg.current);
  const [doc, setDoc] = useState(cfg.doc);
  const [appendix, setAppendix] = useState(params.appendix);
  const [readingFocus, setReadingFocus] = useState<string | null>(
    key === "9"
      ? (SLOTS.find((r) => r.columns.length > 1)?.id ?? null)
      : (SLOTS.find((r) => r.kind === "role" && r.columns[0] === "cycle_begin_year")?.id ?? null),
  );
  const [adjFocus, setAdjFocus] = useState("body");
  const [energyHover, setEnergyHover] = useState("residual");

  const locked = INF.moments.m6.methods.lines.find((l) => l.kind === "lock_plan")?.sentence ?? null;
  const isLocked = key === "6";

  const next = useCallback(() => {
    const flat = sections.flatMap((sec) => sec.items.filter((i) => i.tier === "asked" && !i.optional).map((i) => ({ section: sec.key, item: i.key })));
    if (!flat.length) return;
    const at = flat.findIndex((f) => f.section === current?.section && f.item === current?.item);
    setCurrent(flat[(at + 1) % flat.length]!);
    setDoc(false);
  }, [sections, current]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "m" || e.key === "M") setDoc((d) => !d);
      if (e.key === "n" || e.key === "N") next();
      if (key === "5" || key === "8b") {
        const order = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density", "partition", "all_components", "none"];
        const i = order.indexOf(energyHover);
        const step = e.key === "ArrowDown" ? 1 : e.key === "ArrowUp" ? -1 : 0;
        if (step) {
          e.preventDefault();
          setEnergyHover(order[Math.min(order.length - 1, Math.max(0, i + step))]!);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, key, energyHover]);

  const known = useMemo(() => new Set(Object.keys(cfg.moment.view.state.roles ?? {}).concat(["glucose"])), [cfg]);
  const sec = findSection(sections, current?.section);
  const item = sec?.items.find((i) => i.key === current?.item);
  const position = sec
    ? `objective ${Math.max(1, sec.items.findIndex((i) => i.key === item?.key) + 1)} of ${sec.items.length}`
    : "";
  const nextAsked = sections
    .flatMap((x) => x.items.filter((i) => i.tier === "asked" && !i.optional).map((i) => ({ sec: x, i })))
    .find((x) => !(x.sec.key === current?.section && x.i.key === current?.item));
  const nextLine = nextAsked ? (
    <>
      <span className={s.kicker}>Then</span>
      <span>
        <b>{nextAsked.i.title}</b> · {nextAsked.sec.title}
        {nextAsked.i.count > 1 ? ` · ${nextAsked.i.count} open` : ""}
      </span>
    </>
  ) : null;

  // ── the center ──
  let center: ReactNode = null;
  let canvas: ReactNode = null;
  const lines = cfg.moment.methods.lines;
  if (doc) {
    const pred = cfg.guideline === "TRIPOD+AI";
    center = (
      <MethodsDoc
        guideline={cfg.guideline}
        sections={sections}
        lines={lines}
        title={pred ? "Methods, as drafted: predicting `glucose`" : "Methods, as drafted: `sugar` and `glucose`"}
        locked={isLocked ? locked : null}
      />
    );
  } else if (sec && item) {
    const frame = (body: ReactNode) => (
      <Frame section={sec.title} refText={sec.ref} position={position} next={nextLine}>
        {body}
      </Frame>
    );
    if (item.key === "roles" && key === "1") center = frame(<RolesCard />);
    else if (item.key === "readings")
      center = frame(
        <ReadingsCard
          focus={readingFocus}
          onFocus={setReadingFocus}
          confirmed={key === "9" ? INF.singles.map((x) => x.record.sentence) : []}
          unlocked={key === "9"}
        />,
      );
    else if (item.key === "estimand" && (key === "3" || key === "8a")) center = frame(<EstimandCard hovered="sugar" teach />);
    else if (item.key === "adjustment" && key === "4") center = frame(<AdjustmentCard focus={adjFocus} onFocus={setAdjFocus} />);
    else if (item.key === "energy_adjustment" && (key === "5" || key === "8b")) {
      const sentence = lines.find((l) => l.kind === "set_energy_adjustment")!.sentence;
      center = (
        <Frame section={sec.title} refText={sec.ref} position="editing a stated phrase" next={nextLine}>
          <PhraseCard sentence={sentence} hovered={energyHover} onHover={setEnergyHover} conceptOpen={key === "8b" ? "substitution" : null} />
        </Frame>
      );
    } else
      center = frame(
        <>
          <p className={s.draft}>
            {item.sentences.length ? item.sentences.join(" ") : item.tier === "waiting" ? `${item.title} waits on ${item.waitingOn.join(" and ")}.` : item.title}
          </p>
          <p className={s.why}>This objective's card is not drawn in the prototype; the moments under review are M1 to M9.</p>
        </>,
      );
  }

  // ── the canvas ──
  const composedNote = (
    <p className={s.composed}>
      Composed by the prototype from your recorded roles; for this question the engine says: “Nothing about this choice can be shown on your
      data yet.”
    </p>
  );
  const rest = params.rest === null ? undefined : params.rest === "now" || params.rest === "with" ? params.rest : Number(params.rest);
  if (key === "1") {
    canvas = (
      <PreviewCanvas
        result={INF.previews.roles!.body as PreviewResult}
        pill="Preview"
        label="Record the roles as proposed"
        aside="nothing is recorded"
      />
    );
  } else if (key === "2" || key === "9") {
    const ev = key === "9" ? INF.evidence["voice__flag__imputed_bmi"] : INF.evidence["voice__pooled_cycles__cycle_begin_year"];
    const label =
      key === "9"
        ? "`imputed_bmi` flags `306` imputed values of `bmi`; it describes the data, not the participant."
        : "`cycle_begin_year` pools `9` survey cycles, `2001` to `2017`; methods may differ across them.";
    canvas = ev ? (
      <PreviewCanvas result={ev} pill="Evidence" label={label} aside="your data as loaded" promote={key === "2" ? "distribution" : undefined} />
    ) : null;
  } else if (key === "3" || key === "8a") {
    canvas = <PreviewCanvas result={exposureLineage("sugar")} pill="Preview" label="`sugar` as the exposure" aside="nothing is recorded" note={composedNote} />;
  } else if (key === "4") {
    const g = INF.adjustment_card.groups.find((x) => x.key === adjFocus);
    const leave = g && g.derived && g.derived !== "confounder" ? g.columns : [];
    canvas = (
      <PreviewCanvas
        result={adjustmentLineage(leave)}
        pill="Preview"
        label={g ? `${g.label} as ${g.derived_words ?? "asked"}` : "The adjustment set"}
        aside="nothing is recorded"
        note={composedNote}
        rest={rest}
      />
    );
  } else if (key === "5" || key === "8b") {
    const p = energyPreview(energyHover);
    const t = FX.teaching.find((x) => x.key === "energy_adjustment")!;
    const label = t.options.find((o) => o.value === energyHover)?.label ?? energyHover;
    canvas =
      p && isPreview(p.body) ? (
        <PreviewCanvas result={p.body} pill="Preview" label={label} aside="nothing is recorded" rest={rest} />
      ) : null;
  } else if (key === "6") {
    canvas = <ResultsCanvas appendix={appendix} onAppendix={() => setAppendix((a) => !a)} />;
  } else if (key === "7") {
    const st = FX.prediction.moment.stages;
    canvas = (
      <ComparisonCanvas
        fit={FX.prediction.fit as unknown as FitArtifact}
        shelf={(st.shelf?.artifact ?? null) as unknown as ShelfArtifact | null}
        split={(st.split?.artifact ?? null) as unknown as SplitArtifact | null}
      />
    );
  }

  const summary = cfg.moment.view.summary;
  return (
    <StageFocusProvider>
      <ColumnsContext.Provider value={known}>
        <div className={s.screen} data-moment={key}>
          <Header>
            <span style={{ fontFamily: "var(--serif)", fontSize: 15, fontWeight: 700 }}>{summary.name}</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)" }}>
              {summary.n_rows?.toLocaleString("en-US")} rows × {summary.n_cols} columns
            </span>
          </Header>
          <BannerView model={banner(cfg.moment)} />
          <div className={s.window} data-doc={doc || undefined}>
            <Rail
              guideline={cfg.guideline}
              sections={sections}
              current={doc ? null : current}
              locked={isLocked}
              docOpen={doc}
              onPick={(section, it) => {
                setCurrent({ section, item: it });
                setDoc(false);
              }}
              onNext={next}
              onDoc={() => setDoc((d) => !d)}
            />
            <main className={s.center} aria-label={doc ? "The methods section" : "The current objective"} data-testid="center">
              {center}
            </main>
            <aside className={s.canvas} aria-label="The canvas">
              {canvas}
            </aside>
          </div>
          {params.shot ? null : (
            <nav className={s.switcher} aria-label="Moments under review">
              {MOMENTS.map((m) => (
                <a key={m} href={`?m=${m}`} aria-current={m === key ? "page" : undefined}>
                  M{m}
                </a>
              ))}
            </nav>
          )}
        </div>
      </ColumnsContext.Provider>
    </StageFocusProvider>
  );
}
