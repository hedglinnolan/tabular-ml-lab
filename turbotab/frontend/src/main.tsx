import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/tokens.css";
import "./styles/base.css";
import { App } from "./App";

async function boot() {
  if (import.meta.env.VITE_MOCK === "1") {
    // The mock API (MSW + an in-memory server) exists only in `npm run dev:mock`.
    const { startMocks } = await import("./mocks/browser");
    await startMocks();
  }
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void boot();
