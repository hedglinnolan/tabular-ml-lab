/**
 * The contract each living-methods prototype's walk meets (e2e/methods-protos.spec.ts): from the
 * prototype's first draft, after its reset, a person's clicks to the locked Table 2, then to
 * "Which of my decisions mattered?". No URL parameter and no keyboard shortcut: buttons, slots and
 * phrases a newcomer can see.
 */
import type { Page } from "@playwright/test";

export interface Walker {
  /** The prototype, as the chooser names it. */
  name: string;
  /** Its route under dev:mock. */
  path: string;
  /** From the first draft (the page loaded and reset) to Table 2 on screen. */
  toTable2: (page: Page) => Promise<void>;
  /** From Table 2 on screen to "Which of my decisions mattered?" on screen. */
  toMattered: (page: Page) => Promise<void>;
}
