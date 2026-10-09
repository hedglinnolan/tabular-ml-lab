/**
 * The contract each calm structure's walk meets (e2e/calm-protos.spec.ts). From the structure's
 * first draft, after its reset ("Start over", data-testid="proto-reset"), a person's clicks to the
 * locked Table 2, then to "Which of my decisions mattered?". No URL parameter and no keyboard
 * shortcut: buttons, options, slots and nodes a newcomer can see. Each structure's agent replaces
 * its walker when it replaces its Screen.tsx; the marks the cross-check reads stay the kit's
 * (Table 2: data-testid="table2" with data-t2-row rows; what mattered: data-testid="mattered" with
 * data-mattered-row rows; the primary model's interval note: data-testid="t2-inference").
 */
import type { Page } from "@playwright/test";

export interface Walker {
  /** The structure, as the chooser names it. */
  name: string;
  /** Its route in the static entry (calm.html). */
  path: string;
  /** From the first draft (the page loaded and reset) to Table 2 on screen. */
  toTable2: (page: Page) => Promise<void>;
  /** From Table 2 on screen to "Which of my decisions mattered?" on screen. */
  toMattered: (page: Page) => Promise<void>;
}
