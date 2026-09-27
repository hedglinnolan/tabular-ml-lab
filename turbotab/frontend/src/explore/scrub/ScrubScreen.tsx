/**
 * /lab/explore/scrub — "Before and after, scrubbed": a design prototype for BLUEPRINT §11.
 *
 * Every choice is shown as a transformation the user can scrub, on the real fixtures
 * (docs/turbotab-next/m1/explore/fixtures.json): drag or arrow between "your data now" and
 * "with this choice"; the picture, the working table and the pipeline counts move together, with
 * sidenotes beside exactly what changes. Four scenarios: ?s=energy | exclusions | findings | wide.
 * ?focus=<option> opens with that option previewed.
 */
import { useSyncExternalStore } from "react";
import { Header } from "../../components/Header";
import { ScrubProvider } from "./engine/scrub";
import { EnergyScenario } from "./scenarios/EnergyScenario";
import { ExclusionScenario } from "./scenarios/ExclusionScenario";
import { FindingsScenario } from "./scenarios/FindingsScenario";
import { WideScenario } from "./scenarios/WideScenario";
import s from "./ScrubScreen.module.css";

const SCENARIOS = [
  { key: "energy", label: "Energy adjustment", el: <EnergyScenario /> },
  { key: "exclusions", label: "Exclusions", el: <ExclusionScenario /> },
  { key: "findings", label: "Findings", el: <FindingsScenario /> },
  { key: "wide", label: "Wide table", el: <WideScenario /> },
] as const;

const subscribe = (cb: () => void) => {
  window.addEventListener("popstate", cb);
  window.addEventListener("turbotab:navigate", cb);
  return () => {
    window.removeEventListener("popstate", cb);
    window.removeEventListener("turbotab:navigate", cb);
  };
};

function useQuery(): URLSearchParams {
  const search = useSyncExternalStore(subscribe, () => window.location.search, () => "");
  return new URLSearchParams(search);
}

export default function ScrubScreen() {
  const q = useQuery();
  const current = SCENARIOS.find((x) => x.key === q.get("s")) ?? SCENARIOS[0];
  const focus = q.get("focus");
  return (
    <>
      <Header>
        <span className={s.title}>Explore · scrub</span>
        <nav className={s.scenarios} aria-label="Scenario">
          {SCENARIOS.map((x) => (
            <a
              key={x.key}
              href={`?s=${x.key}`}
              className={s.scenario}
              aria-current={x.key === current.key ? "page" : undefined}
              onClick={(e) => {
                e.preventDefault();
                window.history.pushState(null, "", `/lab/explore/scrub?s=${x.key}`);
                window.dispatchEvent(new Event("turbotab:navigate"));
              }}
            >
              {x.label}
            </a>
          ))}
        </nav>
      </Header>
      <main className={s.main} data-scenario={current.key}>
        <ScrubProvider key={`${current.key}:${focus ?? ""}`} initial={focus}>
          {current.el}
        </ScrubProvider>
      </main>
    </>
  );
}
