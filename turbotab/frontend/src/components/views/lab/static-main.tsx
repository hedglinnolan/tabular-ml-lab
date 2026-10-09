/**
 * The static build of /lab/views (views.html; `npm run build:views`): every exhibit view kind on its
 * fixtures, light and dark side by side, with no server, no mock worker and nothing fetched, so the
 * folder opens straight from disk for review.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ViewsLab } from "./ViewsLab";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ViewsLab />
  </StrictMode>,
);
