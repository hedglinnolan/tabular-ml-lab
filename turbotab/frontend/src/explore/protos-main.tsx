/**
 * The static build of the living-methods prototypes (protos.html; `npm run build:protos`): the
 * chooser and the three prototypes with their captured fixtures, and nothing else — no server, no
 * mock worker. Routes are hashes (#/, #/document, #/questlog, #/map), so the folder runs from any
 * static host or straight from disk.
 */
import { StrictMode, useEffect, useState, useSyncExternalStore, type ComponentType } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import "../styles/tokens.css";
import "../styles/base.css";
import { makeQueryClient } from "../api/queries";
import { MotionPrefsProvider } from "../motion/prefs";
import { setNavigateOverride } from "../router";
import { ThemeProvider } from "../theme";
import { MethodsDocScreen } from "./methods-document/MethodsDocScreen";
import { MethodsMapScreen } from "./methods-map/MethodsMapScreen";
import { QuestScreen } from "./methods-questlog/QuestScreen";
import { MethodsChooser, type ProtoId } from "./methods-shared/Chooser";

const HASH_HREFS: Record<ProtoId, string> = {
  document: "#/document",
  questlog: "#/questlog",
  map: "#/map",
};

const SCREENS: Record<ProtoId, ComponentType> = {
  document: MethodsDocScreen,
  questlog: QuestScreen,
  map: MethodsMapScreen,
};

const TITLES: Record<ProtoId, string> = {
  document: "The paper",
  questlog: "The quest log",
  map: "The map",
};

// Every in-app link (the header's TurboTab mark) goes back to the chooser.
setNavigateOverride(() => {
  window.location.hash = "#/";
  return true;
});

function subscribe(cb: () => void): () => void {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
}

function useHashRoute(): ProtoId | null {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash, () => "");
  const id = hash.replace(/^#\/?/, "").split(/[?/]/)[0] ?? "";
  return id in SCREENS ? (id as ProtoId) : null;
}

function Protos() {
  const id = useHashRoute();
  useEffect(() => {
    window.scrollTo(0, 0);
    document.title = id ? `${TITLES[id]} · Living methods prototypes` : "Living methods prototypes";
  }, [id]);
  if (!id) return <MethodsChooser hrefs={HASH_HREFS} />;
  const Screen = SCREENS[id];
  return <Screen key={id} />;
}

function Root() {
  const [client] = useState(makeQueryClient);
  return (
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <MotionPrefsProvider>
          <Protos />
        </MotionPrefsProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
