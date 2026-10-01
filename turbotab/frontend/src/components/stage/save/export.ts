/**
 * Turning a figure into a file: SVG as written, PNG rasterized at 2× from that same SVG.
 * Which states a save captures is decided here, from the player's whole steps only.
 */
import { localStep } from "../tracks";

export type SaveWhich = "now" | "step" | "with" | "pair";
export type SaveFormat = "svg" | "png";

/**
 * The real states a save captures on a view whose own storyboard has `localLast` steps, given the
 * global step the player shows or is heading to. Never the player's fractional position.
 */
export function saveIndices(which: SaveWhich, heading: number, globalLast: number, localLast: number): number[] {
  switch (which) {
    case "now":
      return [0];
    case "with":
      return [localLast];
    case "pair":
      return [0, localLast];
    case "step":
      return [Math.max(0, Math.min(localLast, localStep(Math.round(heading), globalLast, localLast)))];
  }
}

export function fileName(base: string, which: SaveWhich | "as-shown", format: SaveFormat): string {
  const slug = base
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
  return `turbotab-${slug || "figure"}-${which}.${format}`;
}

export function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function svgBlob(svg: string): Blob {
  return new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${svg}`], { type: "image/svg+xml" });
}

/** Rasterize a standalone SVG at `scale`× on a white page. */
export async function pngBlob(svg: string, scale = 2): Promise<Blob> {
  const w = Number(/width="([\d.]+)"/.exec(svg)?.[1] ?? 720);
  const h = Number(/height="([\d.]+)"/.exec(svg)?.[1] ?? 480);
  const url = URL.createObjectURL(svgBlob(svg));
  try {
    const img = new Image();
    img.decoding = "async";
    img.src = url;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("This browser cannot draw a PNG here.");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("The PNG could not be made."))), "image/png"),
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function saveFigure(svg: string, name: string, format: SaveFormat): Promise<void> {
  downloadBlob(format === "svg" ? svgBlob(svg) : await pngBlob(svg, 2), name);
}
