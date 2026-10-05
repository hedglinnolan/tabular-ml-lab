/**
 * The static build of the calm structures (calm.html; `npm run build:calm`): the chooser, the four
 * structures and the kit demo, on the kit's captured fixture alone — no server, no mock worker,
 * nothing fetched. Routes are hashes (#/, #/qa, #/paper, #/quest, #/map, #/kit), so the folder runs
 * from any static host or straight from disk.
 */
import { StrictMode, useEffect, useSyncExternalStore, type ComponentType } from "react";
import { createRoot } from "react-dom/client";
import { CalmChooser, STRUCTURES, type StructureId } from "./calm-kit/Chooser";
import { KitDemo } from "./calm-kit/KitDemo";
import * as mapMod from "./calm-map/Screen";
import * as paperMod from "./calm-paper/Screen";
import * as qaMod from "./calm-qa/Screen";
import * as questMod from "./calm-quest/Screen";

const MapScreen = mapMod.Screen;
const PaperScreen = paperMod.Screen;
const QaScreen = qaMod.Screen;
const QuestScreen = questMod.Screen;
/** A structure is built once its agent has replaced the placeholder (which exports PLACEHOLDER). */
const BUILT: Record<StructureId, boolean> = {
  qa: !("PLACEHOLDER" in qaMod),
  paper: !("PLACEHOLDER" in paperMod),
  quest: !("PLACEHOLDER" in questMod),
  map: !("PLACEHOLDER" in mapMod),
};

type RouteId = StructureId | "kit";

const SCREENS: Record<RouteId, ComponentType> = {
  qa: QaScreen,
  paper: PaperScreen,
  quest: QuestScreen,
  map: MapScreen,
  kit: KitDemo,
};

const HREFS: Record<RouteId, string> = { qa: "#/qa", paper: "#/paper", quest: "#/quest", map: "#/map", kit: "#/kit" };

function subscribe(cb: () => void): () => void {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
}

function useHashRoute(): RouteId | null {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash, () => "");
  const id = hash.replace(/^#\/?/, "").split(/[?/]/)[0] ?? "";
  return id in SCREENS ? (id as RouteId) : null;
}

function Calm() {
  const id = useHashRoute();
  useEffect(() => {
    window.scrollTo(0, 0);
    const title = id === "kit" ? "The calm kit" : STRUCTURES.find((s) => s.id === id)?.title;
    document.title = title ? `${title} · Calm structures` : "Calm structures";
  }, [id]);
  if (!id) return <CalmChooser hrefs={HREFS} built={BUILT} />;
  const Screen = SCREENS[id];
  return <Screen key={id} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Calm />
  </StrictMode>,
);
