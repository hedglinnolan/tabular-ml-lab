import { lazy, Suspense, useState } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { makeQueryClient } from "./api/queries";
import { Header } from "./components/Header";
import { MotionPrefsProvider } from "./motion/prefs";
import { Link, useRoute } from "./router";
import { ProjectScreen } from "./screens/ProjectScreen";
import { StartScreen } from "./screens/StartScreen";
import { ThemeProvider } from "./theme";

// The motion lab is a review surface, not part of the analysis: load it on demand.
const LabScreen = lazy(() => import("./screens/LabScreen").then((m) => ({ default: m.LabScreen })));
const StageScreen = lazy(() =>
  import("./explore/stage/StageScreen").then((m) => ({ default: m.StageScreen })),
);
const StageLabScreen = lazy(() =>
  import("./screens/StageLabScreen").then((m) => ({ default: m.StageLabScreen })),
);
const StageLabM2Screen = lazy(() =>
  import("./screens/StageLabM2Screen").then((m) => ({ default: m.StageLabM2Screen })),
);
const M2Screen = lazy(() => import("./explore/m2/M2Screen").then((m) => ({ default: m.M2Screen })));

function Routes() {
  const route = useRoute();
  switch (route.name) {
    case "start":
      return <StartScreen />;
    case "project":
      return <ProjectScreen key={route.pid} pid={route.pid} />;
    case "lab":
      return (
        <Suspense fallback={<Header />}>
          <LabScreen />
        </Suspense>
      );
    case "explore-stage":
      return (
        <Suspense fallback={<Header />}>
          <StageScreen />
        </Suspense>
      );
    case "stage-lab":
      return (
        <Suspense fallback={<Header />}>
          <StageLabScreen />
        </Suspense>
      );
    case "stage-lab-m2":
      return (
        <Suspense fallback={<Header />}>
          <StageLabM2Screen />
        </Suspense>
      );
    case "m2-lab":
      return (
        <Suspense fallback={<Header />}>
          <M2Screen />
        </Suspense>
      );
    case "missing":
      return (
        <>
          <Header />
          <main style={{ maxWidth: 640, margin: "60px auto", padding: "0 16px" }}>
            <p style={{ fontFamily: "var(--serif)", fontSize: 17 }}>
              There is nothing at <code className="v">{route.path}</code>.
            </p>
            <Link href="/">Back to the start</Link>
          </main>
        </>
      );
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
