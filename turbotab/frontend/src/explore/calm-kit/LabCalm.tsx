/** /lab/calm (dev:mock only): the calm chooser with links that stay under /lab/calm. */
import { CalmChooser } from "./Chooser";

export function LabCalmChooser() {
  return (
    <CalmChooser
      hrefs={{ qa: "/lab/calm/qa", paper: "/lab/calm/paper", quest: "/lab/calm/quest", map: "/lab/calm/map", kit: "/lab/calm/kit" }}
    />
  );
}
