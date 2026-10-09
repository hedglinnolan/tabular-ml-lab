/**
 * Both themes on one page: the calm tokens' light and dark blocks, re-scoped from `:root` to a
 * frame attribute, so the lab shows every view in each theme side by side from the one token file.
 */
export function scopedThemes(css: string): string {
  const light = /:root\s*\{([^}]*)\}/.exec(css)?.[1];
  const dark = /:root\[data-theme="dark"\]\s*\{([^}]*)\}/.exec(css)?.[1];
  if (!light || !dark) throw new Error("tokens.css lost its light or dark block");
  return `[data-lab-theme="light"] {${light} color-scheme: light; }\n[data-lab-theme="dark"] {${dark}}\n`;
}
