/**
 * The questions (calm.html#/qa; dev /lab/calm/qa): the baseline structure, FOUNDATION §3 as drawn.
 * The question card is the interface: the stage bar across the top, one question on the card at
 * the left with its options, "Why does this matter?" and Continue, the canvas on the right with
 * all the remaining width, and the manuscript in the slim rail at the far left (opened, it lies over
 * the card column; at 1680 px and wider it can be pinned as a column). A newcomer answers the
 * question in front of them and continues; an expert goes back through the stage bar or the
 * manuscript's phrases and blanks. Every part, word and number is the kit's (../calm-kit).
 *
 * At 900 px and narrower the kit hides the rail, so the manuscript takes the kit's stacked column
 * below the canvas instead: a phone keeps the record.
 */
import { useEffect, useState } from "react";
import { Canvas, Card, Shell, useWalk } from "../calm-kit";

const NARROW = "(max-width: 900px)";

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(NARROW).matches);
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
const home = () => (typeof window !== "undefined" && window.location.pathname.startsWith("/lab/calm") ? "/lab/calm" : "#/");

export function Screen() {
  const walk = useWalk();
  const narrow = useNarrow();
  return (
    <Shell
      walk={walk}
      name="The questions"
      home={home()}
      manuscript={narrow ? "column" : "rail"}
      card={<Card walk={walk} />}
      canvas={<Canvas walk={walk} />}
    />
  );
}
