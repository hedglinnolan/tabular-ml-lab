/**
 * The map (calm.html#/map; dev /lab/calm/map): the pipeline map is the interface (prototype C's
 * idea, drawn calm). In place of the one-line stage bar, the analysis is drawn as a compact
 * lineage in one band of at most 120 px (./LineageMap.tsx): the stages as regions, the decisions
 * as nodes, solid once stated and open while asked. Clicking a node opens its card. Below it the
 * page is the kit's, at the kit's widths: the manuscript rail at the far left, the card, and the
 * canvas with all the remaining width.
 *
 * A newcomer answers the card in front of them and continues; the map fills in behind them, one
 * solid node per answer, and says how much is left. An expert reads the map and clicks any node
 * the walk can reach (or arrows along it) to change an answer, or opens a phrase in the
 * manuscript. After the lock, the Results region's two nodes hold Table 2 and what mattered.
 *
 * At 900 px and narrower the kit hides the rail, so the manuscript takes the kit's column below
 * the canvas instead: a phone keeps the record. Every part, word and number is the kit's.
 */
import { useEffect, useState } from "react";
import { Canvas, Card, Shell, useWalk } from "../calm-kit";
import { LineageMap } from "./LineageMap";

const NARROW = "(max-width: 900px)";

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(NARROW).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(NARROW);
    if (!mq) return;
    const on = () => setNarrow(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return narrow;
}

/** The chooser: calm.html's hash route, or the dev app's /lab/calm. */
const home = () => (typeof window !== "undefined" && window.location.pathname.startsWith("/lab/calm") ? "/lab/calm" : "#/");

export function Screen() {
  const walk = useWalk();
  const narrow = useNarrow();
  return (
    <Shell
      walk={walk}
      name="The map"
      home={home()}
      manuscript={narrow ? "column" : "rail"}
      chain={<LineageMap walk={walk} />}
      card={<Card walk={walk} />}
      canvas={<Canvas walk={walk} />}
    />
  );
}
