/**
 * Contrast as the browser draws it, for the views' tests: a token's hex from tokens.css, a
 * `color-mix(in oklab, …)` of two tokens as CSS Color 4 mixes it, and the WCAG 2 contrast ratio of
 * two colors. The views' fills and inks are written as these expressions, so a test can check every
 * printed value and every mark against its background in both themes.
 */

type RGB = [number, number, number];

const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toGamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** linear sRGB from #RRGGBB */
export function hexToLinear(hex: string): RGB {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) throw new Error(`not a hex color: ${hex}`);
  const n = parseInt(m[1]!, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => toLinear(c / 255)) as RGB;
}

function linearToOklab([r, g, b]: RGB): RGB {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

function oklabToLinear([L, a, b]: RGB): RGB {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const clamp = (x: number) => Math.min(1, Math.max(0, x));
  return [
    clamp(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    clamp(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    clamp(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ];
}

/** relative luminance (WCAG 2) of a linear sRGB color, rounded through 8-bit as a screen shows it */
export function luminance(c: RGB): number {
  const [r, g, b] = c.map((x) => toLinear(Math.round(toGamma(x) * 255) / 255));
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

export function contrast(a: RGB, b: RGB): number {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p);
  return (x! + 0.05) / (y! + 0.05);
}

/** The tokens of one theme from tokens.css: the light :root block, or the forced-dark block. */
export function themeTokens(css: string, theme: "light" | "dark"): Record<string, string> {
  const block = theme === "light" ? /:root\s*\{([^}]*)\}/.exec(css)?.[1] : /:root\[data-theme="dark"\]\s*\{([^}]*)\}/.exec(css)?.[1];
  const out: Record<string, string> = {};
  for (const m of (block ?? "").matchAll(/--([\w-]+):\s*(#[0-9A-Fa-f]{6})/g)) out[m[1]!] = m[2]!;
  return out;
}

/** A view's color expression, resolved: `var(--token)` or `color-mix(in oklab, A p%, B)`. */
export function resolveColor(expr: string, tokens: Record<string, string>): RGB {
  const s = expr.trim();
  const v = /^var\(--([\w-]+)\)$/.exec(s);
  if (v) {
    const hex = tokens[v[1]!];
    if (!hex) throw new Error(`no token --${v[1]}`);
    return hexToLinear(hex);
  }
  const mix = /^color-mix\(in oklab,\s*(var\(--[\w-]+\))\s+([\d.]+)%,\s*(var\(--[\w-]+\))\)$/.exec(s);
  if (mix) {
    const p = Number(mix[2]) / 100;
    const a = linearToOklab(resolveColor(mix[1]!, tokens));
    const b = linearToOklab(resolveColor(mix[3]!, tokens));
    return oklabToLinear([0, 1, 2].map((k) => a[k]! * p + b[k]! * (1 - p)) as RGB);
  }
  throw new Error(`cannot resolve ${expr}`);
}
