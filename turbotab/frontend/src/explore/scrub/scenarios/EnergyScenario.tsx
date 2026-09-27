/**
 * S1 — the energy-adjustment question on the real NHANES export. Six methods, one refused (with
 * its reason and its exit), each previewed as a scrub from the user's data to the choice.
 */
import { useMemo, useState } from "react";
import { Prose } from "../../../components/Prose";
import { ENERGY_OPTIONS, ENERGY_ORDER, ENERGY_Q } from "../copy";
import { ENERGY, fmtR, view } from "../data";
import { useScrub } from "../engine/scrub";
import { Question, type ShelfOption } from "../Question";
import { PipelineStrip, RecordBar, ScrubBar, StageFrame, StageSection } from "../Stage";
import { ScrubHistogram } from "../views/Histogram";
import { ScrubLineage } from "../views/Lineage";
import { MorphTable } from "../views/MorphTable";
import { ScrubScatter } from "../views/Scatter";
import s from "../Scenario.module.css";
import {
  ENERGY_BASIS,
  energyDistribution,
  energyLineage,
  energyScatter,
  energyStrip,
  energyTable,
} from "./model";

type ViewTab = "relationship" | "distribution" | "lineage";

function shelf(): ShelfOption[] {
  return ENERGY_ORDER.map((key) => {
    const o = ENERGY[key]!;
    const copy = ENERGY_OPTIONS[key]!;
    const rel = o.preview ? view(o.preview.views, "relationship") : undefined;
    const kcalIn = o.matrix_columns ? o.matrix_columns.includes("kcal") : null;
    if (!o.applicable.ok) {
      return {
        key,
        label: copy.label,
        cols: ["—", "—"],
        consequence: copy.consequence,
        tag: "not applicable",
        alias: ["partition3"],
        disabled: {
          reason: copy.consequence,
          exit: { key: "partition3", label: "Preview it on `protein`, `carb`, `fat_total`" },
        },
      };
    }
    return {
      key,
      label: copy.label,
      cols: [fmtR(rel?.r_after), kcalIn ? "in" : "out"],
      consequence: copy.consequence,
      tag: key === "residual" ? "usual" : undefined,
      badge: o.method_card.standing ?? undefined,
      why: o.estimand,
      caveats: o.method_card.caveats,
    };
  });
}

export function EnergyScenario() {
  const { active, focus } = useScrub();
  const [tab, setTab] = useState<ViewTab>("relationship");
  const [recorded, setRecorded] = useState<string | null>(null);
  const [hotRow, setHotRow] = useState<number | null>(null);
  const options = useMemo(() => shelf(), []);
  const scatter = useMemo(() => energyScatter(), []);
  const table = useMemo(() => energyTable(), []);
  const strip = useMemo(() => energyStrip(), []);
  const dist = useMemo(() => energyDistribution(), []);
  const lineage = useMemo(() => energyLineage(), []);

  const refused = active === "partition";
  const unchanged = active === "none" || active === "standard";
  const afterLabel = refused
    ? "Energy partition cannot run here"
    : active
      ? `With ${ENERGY_OPTIONS[active]?.short ?? active}`
      : null;
  const recordable = (k: string) => k !== "partition" && k in ENERGY_OPTIONS;

  return (
    <div className={s.layout}>
      <div className={s.record}>
        <Question
          id="energy"
          kicker={ENERGY_Q.kicker}
          question={ENERGY_Q.question}
          why={ENERGY_Q.why}
          columns={["r with `kcal`", "`kcal`"]}
          options={options}
          recorded={recorded ? { key: recorded, sentence: ENERGY_OPTIONS[recorded]!.sentence } : null}
          onRecord={(k) => setRecorded(k)}
          onChange={() => setRecorded(null)}
          recordable={recordable}
        />
      </div>
      <StageFrame label="Consequence preview">
        <PipelineStrip states={strip} />
        <ScrubBar afterLabel={afterLabel} recorded={!!recorded} unchanged={unchanged} refused={refused} />
        <StageSection
          kicker={
            <span className={s.tabs} role="tablist" aria-label="View">
              {(["relationship", "distribution", "lineage"] as ViewTab[]).map((t) => (
                <button
                  key={t}
                  type="button"
                  role="tab"
                  aria-selected={tab === t}
                  className={s.tab}
                  onClick={() => setTab(t)}
                >
                  {t}
                </button>
              ))}
            </span>
          }
        >
          <div className={s.viewBox} style={{ height: 292 }}>
            {tab === "relationship" ? (
              <ScrubScatter xs={scatter.xs} xLabel={scatter.xLabel} states={scatter.states} height={292} span={[0, 0.8]} />
            ) : tab === "distribution" ? (
              <ScrubHistogram states={dist} height={250} span={[0, 0.8]} hotRow={hotRow} label="fat_total, before and after" />
            ) : (
              <ScrubLineage states={lineage.states} sources={lineage.sources} rowH={23} span={[0, 0.8]} />
            )}
            {refused ? (
              <div className={s.refusal} role="status">
                <p className={s.refusalTitle}>
                  Energy partition cannot run on these 7 nutrients.
                </p>
                <p className={s.refusalWhy}>
                  <Prose text="`sugar` carries no energy: no Atwater factor is known for it." />
                </p>
                <button type="button" className={s.exitButton} onClick={() => focus("partition3")}>
                  <Prose text="Preview it on `protein`, `carb`, `fat_total`" />
                </button>
              </div>
            ) : null}
          </div>
        </StageSection>
        <StageSection kicker="Working table · 5 training rows" aside={<Prose text="`age` `gender` `bmi` pass through" />}>
          <MorphTable
            slots={table.slots}
            rowIds={table.rowIds}
            states={table.states}
            span={[0.2, 1]}
            label="The adjusted columns on five training rows"
            fontPx={11.5}
            hotRow={hotRow}
            onRowHover={setHotRow}
          />
        </StageSection>
        <RecordBar
          basis={ENERGY_BASIS}
          action={active && recordable(active) ? `Record: ${ENERGY_OPTIONS[active]!.label}` : null}
          onRecord={() => active && setRecorded(active)}
          recorded={!!recorded}
          onChange={() => setRecorded(null)}
        />
      </StageFrame>
    </div>
  );
}
