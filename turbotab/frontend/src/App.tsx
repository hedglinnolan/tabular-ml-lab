import { lazy, Suspense, useState, type ComponentType, type LazyExoticComponent } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { makeQueryClient } from "./api/queries";
import { Header } from "./components/Header";
import { MotionPrefsProvider } from "./motion/prefs";
import { Link, useRoute, type Route } from "./router";
import { ProjectScreen } from "./screens/ProjectScreen";
import { StartScreen } from "./screens/StartScreen";
import { ThemeProvider } from "./theme";

type LabRoute = Exclude<Route["name"], "start" | "project" | "missing">;

// The review surfaces under /lab load on demand, and exist only in npm run dev:mock: the
// condition is the literal env flag, so a production build drops each import, its chunk and its
// fixtures (INBOX 123, 162).
const LAB_SCREENS: Partial<Record<LabRoute, LazyExoticComponent<ComponentType>>> =
  import.meta.env.VITE_MOCK === "1"
    ? {
        lab: lazy(() => import("./screens/LabScreen").then((m) => ({ default: m.LabScreen }))),
        "explore-stage": lazy(() =>
          import("./explore/stage/StageScreen").then((m) => ({ default: m.StageScreen })),
        ),
        "stage-lab": lazy(() =>
          import("./screens/StageLabScreen").then((m) => ({ default: m.StageLabScreen })),
        ),
        "stage-lab-m2": lazy(() =>
          import("./screens/StageLabM2Screen").then((m) => ({ default: m.StageLabM2Screen })),
        ),
        "m2-lab": lazy(() => import("./explore/m2/M2Screen").then((m) => ({ default: m.M2Screen }))),
        "m3-lab": lazy(() =>
          import("./screens/M3LabScreen").then((m) => ({ default: m.M3LabScreen })),
        ),
        "methods-protos": lazy(() =>
          import("./explore/methods-shared/Chooser").then((m) => ({ default: m.MethodsChooser })),
        ),
        "methods-document": lazy(() =>
          import("./explore/methods-document/MethodsDocScreen").then((m) => ({
            default: m.MethodsDocScreen,
          })),
        ),
        "methods-questlog": lazy(() =>
          import("./explore/methods-questlog/QuestScreen").then((m) => ({ default: m.QuestScreen })),
        ),
        "methods-map": lazy(() =>
          import("./explore/methods-map/MethodsMapScreen").then((m) => ({
            default: m.MethodsMapScreen,
          })),
        ),
      }
    : {};

function Missing({ path }: { path: string }) {
  return (
    <>
      <Header />
      <main style={{ maxWidth: 640, margin: "60px auto", padding: "0 16px" }}>
        <p style={{ fontFamily: "var(--serif)", fontSize: 17 }}>
          There is nothing at <code className="v">{path}</code>.
        </p>
        <Link href="/">Back to the start</Link>
      </main>
    </>
  );
}

function Routes() {
  const route = useRoute();
  switch (route.name) {
    case "start":
      return <StartScreen />;
    case "project":
      return <ProjectScreen key={route.pid} pid={route.pid} />;
    case "missing":
      return <Missing path={route.path} />;
    default: {
      const Screen = LAB_SCREENS[route.name];
      if (!Screen) return <Missing path={window.location.pathname} />;
      return (
        <Suspense fallback={<Header />}>
          <Screen />
        </Suspense>
      );
    }
  }
}

export function App() {
  const [client] = useState(makeQueryClient);
  return (
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <MotionPrefsProvider>
          <Routes />
        </MotionPrefsProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
