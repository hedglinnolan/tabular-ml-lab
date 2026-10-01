/**
 * A few string builders for self-contained SVG: literal colors, system serif, escaped text.
 * A saved figure never reads a CSS variable or a webfont, so it rasterizes the same anywhere
 * (DESIGN_LANGUAGE §07).
 */

export const J = {
  paper: "#ffffff",
  ink: "#1a1a1a",
  dark: "#3d3d3d",
  mid: "#6e6e6e",
  light: "#a6a6a6",
  pale: "#d9d9d9",
  wash: "#efefef",
  font: "Charter, 'Iowan Old Style', 'Source Serif Pro', Georgia, 'Times New Roman', serif",
} as const;

/** Series are told apart by dash pattern first, gray level second (grayscale-safe). */
export const DASHES = ["", "7 3.5", "2 2.5", "9 3 2 3", "4 2"];
export const GRAYS = [J.ink, J.dark, J.mid, J.dark, J.mid];

export function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

type Attr = string | number | undefined | null | false;

export function el(tag: string, attrs: Record<string, Attr>, children: string | string[] = ""): string {
  const a = Object.entries(attrs)
    .filter(([, v]) => v !== undefined && v !== null && v !== false)
    .map(([k, v]) => `${k}="${esc(typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v))}"`)
    .join(" ");
  const body = Array.isArray(children) ? children.join("") : children;
  return body ? `<${tag}${a ? " " + a : ""}>${body}</${tag}>` : `<${tag}${a ? " " + a : ""}/>`;
}

export function text(
  x: number,
  y: number,
  s: string,
  opts: { size?: number; anchor?: "start" | "middle" | "end"; weight?: number; italic?: boolean; fill?: string } = {},
): string {
  return el(
    "text",
    {
      x,
      y,
      "font-size": opts.size ?? 11,
      "text-anchor": opts.anchor ?? "start",
      "font-weight": opts.weight,
      "font-style": opts.italic ? "italic" : undefined,
      fill: opts.fill ?? J.ink,
    },
    esc(s),
  );
}

/** Greedy word wrap by an average glyph width (serif ≈ 0.5 em). */
export function wrap(s: string, width: number, size: number): string[] {
  const max = Math.max(10, Math.floor(width / (size * 0.5)));
  const lines: string[] = [];
  let line = "";
  for (const word of s.split(/\s+/).filter(Boolean)) {
    if ((line + " " + word).trim().length > max && line) {
      lines.push(line);
      line = word;
    } else line = (line + " " + word).trim();
  }
  if (line) lines.push(line);
  return lines;
}

export function hatch(id: string): string {
  return el(
    "pattern",
    { id, width: 4, height: 4, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    el("line", { x1: 0, y1: 0, x2: 0, y2: 4, stroke: J.dark, "stroke-width": 1.2 }),
  );
}

export interface Box {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** A linear scale and its "nice" ticks, without d3 (the export must not depend on layout). */
export function linear(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = (v: number) => r0 + (v - d0) * k;
  return f;
}

export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

export function niceDomain(lo: number, hi: number, count = 5): [number, number] {
  if (!(hi > lo)) return [lo - 1, hi + 1];
  const raw = (hi - lo) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}
