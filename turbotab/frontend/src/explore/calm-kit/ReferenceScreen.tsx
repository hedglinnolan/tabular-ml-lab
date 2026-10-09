/**
 * The kit's reference wiring: the shell's zones, the card, the canvas and the manuscript rail on
 * the one walk. The four structures' placeholders show it until each structure replaces its
 * Screen.tsx; it is the walker contract's baseline (e2e/calm-protos/), not a design.
 */
import { Canvas, Card, Shell, useWalk } from "./index";

export function ReferenceScreen({ name }: { name: string }) {
  const walk = useWalk();
  return <Shell walk={walk} name={name} card={<Card walk={walk} />} canvas={<Canvas walk={walk} />} />;
}
