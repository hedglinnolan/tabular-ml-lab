/**
 * The product's calm kit (plans/W1_PART2_PLAN.md §2, "the kit move"): what the quest shell is drawn
 * with, copied out of src/explore/calm-kit (which stays as it is for the prototypes) and rewritten
 * props-driven where the explore copy was bound to the fixture's walk. Importing it loads the
 * tokens (scoped to `[data-kit]`) and the one family's faces.
 */
import "./tokens.css";
import "./fonts.css";

export { Foot, Plain, POINT_GRACE_MS, Why, plain } from "./parts";
export { ManuscriptBody, ManuscriptOverlay, ManuscriptRail, newestSeq, useManuscriptRail } from "./Manuscript";
export { default as kit } from "./kit.module.css";
export { default as foot } from "./foot.module.css";
