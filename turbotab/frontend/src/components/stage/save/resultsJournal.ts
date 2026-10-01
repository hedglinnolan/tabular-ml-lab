/** Journal-format figures of the Results: the comparison, the forest, the substitution curves. */
import type { Coefficient, SubstitutionArtifact } from "../../../api/m1-stage-types";
import { fmtInt, fmtNum } from "../format";
import {
  bandPoints,
  curveDomain,
  curvePoints,
  disagreement,
  effectLabel,
  forestDomain,
  type Comparison,
} from "../results/model";
import { DASHES, GRAYS, J, el, linear, niceDomain, niceTicks, text, type Box } from "./svg";
import { figureSvg } from "./journal";

export function comparisonFigure(data: Comparison, meta: { title: string; caption: string; provenance: string }) {
  const H = 70 + data.rows.length * 40;
  return figureSvg({
    ...meta,
    panelHeight: H,
    panels: [
      {
        label: "",
        body: (b: Box) => {
          const box = { x0: b.x0 + 150, x1: b.x1 - 150, y0: b.y0 + 4, y1: b.y1 - 40 };
          const d = niceDomain(data.domain[0], data.domain[1], 5);
          const x = linear(d, [box.x0, box.x1]);
          const rowH = (box.y1 - box.y0) / Math.max(1, data.rows.length);
          const out: string[] = [];
          if (data.baseline) {
            out.push(el("line", { x1: x(data.baseline.value), x2: x(data.baseline.value), y1: box.y0, y2: box.y1, stroke: J.ink, "stroke-width": 1, "stroke-dasharray": "4 3" }));
            out.push(text(x(data.baseline.value), box.y1 + 30, `baseline (${data.baseline.label}) ${fmtNum(data.baseline.value, 2)}`, { size: 9.5, anchor: "middle", italic: true }));
          }
          data.rows.forEach((r, i) => {
            const y = box.y0 + i * rowH + rowH / 2;
            out.push(text(b.x0, y + 4, r.label, { size: 11.5, weight: 700 }));
            if (r.mean !== null) {
              out.push(el("line", { x1: x(r.mean - (r.sd ?? 0)), x2: x(r.mean + (r.sd ?? 0)), y1: y, y2: y, stroke: J.ink, "stroke-width": 1.4 }));
              out.push(el("circle", { cx: x(r.mean), cy: y, r: 4, fill: J.ink }));
            }
            if (r.holdout !== null) out.push(el("circle", { cx: x(r.holdout), cy: y, r: 3.6, fill: J.paper, stroke: J.ink, "stroke-width": 1.2 }));
            out.push(text(b.x1, y + 4, `${fmtNum(r.mean)} ± ${fmtNum(r.sd, 2)}${r.holdout !== null ? `; ${fmtNum(r.holdout)}` : ""}`, { size: 10.5, anchor: "end" }));
          });
          out.push(el("line", { x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1, stroke: J.dark, "stroke-width": 0.8 }));
          for (const t of niceTicks(d[0], d[1], 5)) {
            out.push(el("line", { x1: x(t), x2: x(t), y1: box.y1, y2: box.y1 + 3, stroke: J.dark }));
            out.push(text(x(t), box.y1 + 14, fmtNum(t, 2), { size: 9.5, anchor: "middle" }));
          }
          return out.join("");
        },
      },
    ],
  });
}

export function forestFigure(
  coefs: Coefficient[],
  inference: boolean,
  meta: { title: string; caption: string; provenance: string },
) {
  const H = 50 + coefs.length * 22;
  return figureSvg({
    ...meta,
    panelHeight: H,
    panels: [
      {
        label: "",
        body: (b: Box) => {
          const box = { x0: b.x0 + 130, x1: b.x1 - 170, y0: b.y0, y1: b.y1 - 24 };
          const d = niceDomain(...forestDomain(coefs), 5);
          const x = linear(d, [box.x0, box.x1]);
          const rowH = (box.y1 - box.y0) / Math.max(1, coefs.length);
          const out: string[] = [el("line", { x1: x(0), x2: x(0), y1: box.y0, y2: box.y1, stroke: J.ink, "stroke-width": 0.9, "stroke-dasharray": "3 2" })];
          coefs.forEach((c, i) => {
            const y = box.y0 + i * rowH + rowH / 2;
            out.push(text(b.x0, y + 4, c.feature, { size: 10.5 }));
            if (inference && c.ci_low !== null && c.ci_high !== null)
              out.push(el("line", { x1: x(c.ci_low), x2: x(c.ci_high), y1: y, y2: y, stroke: J.ink, "stroke-width": 1.2 }));
            if (c.estimate !== null) out.push(el("rect", { x: x(c.estimate) - 3.5, y: y - 3.5, width: 7, height: 7, fill: J.ink }));
            const ci = inference && c.ci_low !== null && c.ci_high !== null ? ` [${fmtNum(c.ci_low)}, ${fmtNum(c.ci_high)}]` : "";
            out.push(text(b.x1, y + 4, `${fmtNum(c.estimate)}${ci}`, { size: 10, anchor: "end" }));
          });
          out.push(el("line", { x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1, stroke: J.dark, "stroke-width": 0.8 }));
          for (const t of niceTicks(d[0], d[1], 5)) out.push(text(x(t), box.y1 + 14, fmtNum(t, 2), { size: 9.5, anchor: "middle" }));
          return out.join("");
        },
      },
    ],
  });
}

export function curvesFigure(
  sub: SubstitutionArtifact,
  target: string,
  meta: { title: string; caption: string; provenance: string },
) {
  return figureSvg({
    ...meta,
    panelHeight: 400,
    panels: [
      {
        label: "",
        body: (b: Box, uid: string) => {
          const box = { x0: b.x0 + 46, x1: b.x1 - 10, y0: b.y0 + 4, y1: b.y1 - 110 };
          const kMax = sub.ks[sub.ks.length - 1] ?? 1;
          const x = linear([0, kMax], [box.x0, box.x1]);
          const d = niceDomain(...curveDomain(sub), 5);
          const y = linear(d, [box.y1, box.y0]);
          const out: string[] = [];
          for (const t of niceTicks(d[0], d[1], 5)) {
            out.push(el("line", { x1: box.x0, x2: box.x1, y1: y(t), y2: y(t), stroke: t === 0 ? J.light : J.wash, "stroke-width": 0.8 }));
            out.push(text(box.x0 - 6, y(t) + 3.5, fmtNum(t, 2), { size: 9.5, anchor: "end" }));
          }
          const gap = disagreement(sub);
          if (gap.length > 1) {
            const top = gap.map((g) => `${x(g.k).toFixed(1)},${y(g.hi).toFixed(1)}`).join("L");
            const bottom = [...gap].reverse().map((g) => `${x(g.k).toFixed(1)},${y(g.lo).toFixed(1)}`).join("L");
            out.push(el("path", { d: `M${top}L${bottom}Z`, fill: `url(#hatch-${uid})`, opacity: 0.35 }));
          }
          sub.models.forEach((m) => {
            const band = bandPoints(sub, m);
            if (band.length > 1) {
              const top = band.map((g) => `${x(g.k).toFixed(1)},${y(g.hi).toFixed(1)}`).join("L");
              const bottom = [...band].reverse().map((g) => `${x(g.k).toFixed(1)},${y(g.lo).toFixed(1)}`).join("L");
              out.push(el("path", { d: `M${top}L${bottom}Z`, fill: J.ink, opacity: 0.07 }));
            }
          });
          sub.models.forEach((m, i) => {
            const pts = curvePoints(sub, m);
            if (!pts.length) return;
            out.push(
              el("path", {
                d: "M" + pts.map((p) => `${x(p.k).toFixed(1)},${y(p.delta).toFixed(1)}`).join("L"),
                fill: "none",
                stroke: GRAYS[i % GRAYS.length],
                "stroke-width": 1.6,
                "stroke-dasharray": DASHES[i % DASHES.length] || undefined,
              }),
            );
            const end = pts[pts.length - 1]!;
            if (m.stopped_at !== null)
              out.push(el("line", { x1: x(end.k), x2: x(end.k), y1: y(end.delta) - 5, y2: y(end.delta) + 5, stroke: J.ink, "stroke-width": 1.4 }));
          });
          out.push(el("line", { x1: box.x0, x2: box.x1, y1: box.y1, y2: box.y1, stroke: J.dark, "stroke-width": 0.8 }));
          const support = sub.models[0]?.on_support_fraction ?? [];
          const cell = sub.ks.length > 1 ? x(sub.ks[1]!) - x(sub.ks[0]!) : 20;
          sub.ks.forEach((k, i) => {
            out.push(text(x(k), box.y1 + 14, fmtInt(k), { size: 9.5, anchor: "middle" }));
            const g = Math.round(255 - 200 * (support[i] ?? 0));
            out.push(el("rect", { x: x(k) - cell / 2 + 1, y: box.y1 + 20, width: Math.max(1, cell - 2), height: 7, fill: `rgb(${g},${g},${g})` }));
          });
          out.push(text(box.x0, box.y1 + 42, `kcal moved from ${sub.donor} to ${sub.recipient}; strip: share of rows on support`, { size: 10.5 }));
          out.push(text(b.x0, box.y0 - 2, `Δ predicted ${target}`, { size: 10.5 }));
          sub.models.forEach((m, i) => {
            const ly = box.y1 + 62 + i * 15;
            out.push(el("line", { x1: box.x0, x2: box.x0 + 28, y1: ly - 3.5, y2: ly - 3.5, stroke: GRAYS[i % GRAYS.length], "stroke-width": 1.6, "stroke-dasharray": DASHES[i % DASHES.length] || undefined }));
            out.push(text(box.x0 + 36, ly, `${m.label}: ${effectLabel(m) || "—"}`, { size: 10.5 }));
          });
          return out.join("");
        },
      },
    ],
  });
}
