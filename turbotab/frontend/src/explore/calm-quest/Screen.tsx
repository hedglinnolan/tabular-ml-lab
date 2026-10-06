/**
 * The quest log (calm.html#/quest; dev /lab/calm/quest): the open questions as a list of
 * objectives, walked one at a time. Where the kit puts its chain, the objective line lists the
 * methods section by its sections, each with how many of its questions are answered; choosing a
 * section discloses its objectives, each named by its question, and choosing one opens it on the
 * card. The card keeps the kit's stage label ("Exposure · step 2 of 3"), as in the other three
 * structures. The card and the canvas are the kit's, at the kit's widths; the manuscript opens
 * from the kit's rail, as in the Q&A card (at 900 px and narrower, the kit's column below the
 * canvas). A newcomer answers the card in front of them and continues, watching the counts fill;
 * an expert jumps to any answered objective from the line or the manuscript and comes back with
 * "Next objective". Every part, word and number is the kit's (../calm-kit).
 *
 * The line stays on top rather than becoming a vertical rail: at 1440 px a 220 px rail beside the
 * card would leave the canvas 48% of the screen, short of the 60% FOUNDATION §3 gives it.
 */
import { useEffect, useState } from "react";
import { Canvas, Card, Shell, useWalk } from "../calm-kit";
import { ObjectiveLine } from "./ObjectiveLine";

const NARROW = "(max-width: 900px)";

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && !!window.matchMedia?.(NARROW).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia?.(NARROW);
    if (!mq) return;
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return narrow;
}

/** The chooser: calm.html's hash route, or the dev app's /lab/calm. */
const home = () =>
  typeof window !== "undefined" && window.location.pathname.startsWith("/lab/calm")
    ? "/lab/calm"
    : "#/";

export function Screen() {
  const walk = useWalk();
  const narrow = useNarrow();
  return (
    <Shell
      walk={walk}
      name="The quest log"
      home={home()}
      manuscript={narrow ? "column" : "rail"}
      chain={<ObjectiveLine walk={walk} />}
      card={<Card walk={walk} />}
      canvas={<Canvas walk={walk} />}
    />
  );
}
