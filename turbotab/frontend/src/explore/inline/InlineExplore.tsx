/**
 * /lab/explore/inline — "Consequences in place", a design prototype for BLUEPRINT §11.
 *
 * Every option card carries its own consequence at sparkline scale, so all options are visible and
 * comparable at once; focusing one enlarges its picture in a stage right under the strip, and a
 * pin compares two. The pipeline panel shows structure only (rows, lineage) and marks what the
 * focused option would touch. Four scenarios, all on the real-data fixture.
 */
import { useState } from "react";
import { Header } from "../../components/Header";
import { useMotionPrefs } from "../../motion/prefs";
import { EnergyScenario } from "./EnergyScenario";
import { ExclusionsScenario } from "./ExclusionsScenario";
import { FindingsScenario } from "./FindingsScenario";
import { FIXTURE, type Route } from "./fixture";
import { fmtInt } from "./format";
import { WideScenario } from "./WideScenario";
import s from "./inline.module.css";

type Scenario = "energy" | "exclusions" | "findings" | "wide";

const SCENARIOS: { key: Scenario; label: string }[] = [
  { key: "energy", label: "Energy" },
  { key: "exclusions", label: "Exclusions" },
  { key: "findings", label: "Findings" },
  { key: "wide", label: "Wide data" },
];

function initial(): Scenario {
  const v = new URLSearchParams(window.location.search).get("s");
  return SCENARIOS.some((x) => x.key === v) ? (v as Scenario) : "energy";
}

export function InlineExplore() {
  const [scenario, setScenario] = useState<Scenario>(initial);
  const { reduced, setOverride } = useMotionPrefs();

  const go = (next: Scenario) => {
    setScenario(next);
    const url = new URL(window.location.href);
    url.searchParams.set("s", next);
    window.history.replaceState(null, "", url);
  };

  const route = (r: Route) => {
    if (r === "energy") go("energy");
    else if (r === "exclusions") go("exclusions");
  };

  const data =
    scenario === "wide"
      ? `genomics · ${FIXTURE.scenario_b.dataset.n_rows} × ${FIXTURE.scenario_b.dataset.n_cols}`
      : `NHANES · ${fmtInt(FIXTURE.scenario_a.dataset.n_rows)} rows`;

  return (
    <>
      <Header>
        <span className={s.project}>{data}</span>
      </Header>
      <main className={s.page}>
        <div className={s.labBar}>
          <div className={s.tabs} role="tablist" aria-label="Scenario">
            {SCENARIOS.map((x) => (
              <button
                key={x.key}
                type="button"
                role="tab"
                aria-selected={scenario === x.key}
                className={s.tab}
                onClick={() => go(x.key)}
              >
                {x.label}
              </button>
            ))}
          </div>
          <label className={s.motionToggle}>
            <input
              type="checkbox"
              checked={reduced}
              onChange={(e) => setOverride(e.target.checked)}
            />
            Reduce motion
          </label>
        </div>
        <div role="tabpanel" aria-label={scenario}>
          {scenario === "energy" ? <EnergyScenario key="energy" /> : null}
          {scenario === "exclusions" ? <ExclusionsScenario key="exclusions" /> : null}
          {scenario === "findings" ? <FindingsScenario key="findings" onRoute={route} /> : null}
          {scenario === "wide" ? <WideScenario key="wide" /> : null}
        </div>
      </main>
    </>
  );
}
