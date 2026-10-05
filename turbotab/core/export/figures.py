"""The participant-flow and lineage figures of the bundle, as journal-format SVG (DESIGN_LANGUAGE
§07: "a toggle re-renders the same figure as it will be published — serif type, greyscale, series
distinguished by dash pattern rather than color alone … numbered caption"; "journal rendering uses
literal hex colors rather than CSS variables, the exported file is self-contained").

* **Figure 1, the participant flow** (STROBE 13a–c, TRIPOD+AI 20a; North star 4): the rows at each
  step of the ``cohort`` stage's flow, from the table as read to the rows analyzed, each exclusion
  beside the step that made it with its count and reason, then how the rows were used (under
  prediction, development and any held-out rows).
* **Figure 2, the lineage** (BLUEPRINT §11.1: the canonical lineage; North star 4's "modeling
  decision provenance"): the design stage's lineage from the raw columns the analysis reads, through
  the steps every family shares, to the columns of the model matrix. A line's dash pattern says
  what the step did to that column; an exposure's name is set in bold.

Both are pure functions of the artifacts: the same record draws the same bytes (no clock, no
random identifier), so a replay can compare them too.
"""
from __future__ import annotations

import re
import textwrap
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

FONT = "Charter, 'Iowan Old Style', Georgia, 'Times New Roman', serif"
INK = "#1a1a1a"
MID = "#4d4d4d"
GREY = "#8c8c8c"
RULE = "#b3b3b3"
PAPER = "#ffffff"
SIZE = 11.0
SMALL = 9.5
LINE = 14.0
PAD = 7.0


def _esc(text: Any) -> str:
    return escape(str(text), {'"': "&quot;"})


def plain(text: Any) -> str:
    """A record label as a figure prints it: data values without their backticks."""
    return re.sub(r"`([^`]*)`", r"\1", str(text or ""))


def width_of(text: str, size: float = SIZE) -> float:
    """An estimate of a serif string's width: wide enough for Georgia at ``size``."""
    return len(text) * size * 0.56


def _wrap(text: str, chars: int) -> list[str]:
    return textwrap.wrap(text, chars, break_long_words=True, break_on_hyphens=False) or [""]


def _svg(width: float, height: float, title: str, desc: str, body: Sequence[str]) -> str:
    w, h = round(width), round(height)
    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{_esc(FONT)}" font-size="{SIZE:g}" fill="{INK}">',
        f"<title>{_esc(title)}</title>",
        f"<desc>{_esc(desc)}</desc>",
        "<defs>",
        f'<marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{MID}"/></marker>',
        "</defs>",
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="{PAPER}"/>',
    ]
    return "\n".join([*head, *body, "</svg>"]) + "\n"


def _caption(text: str, x: float, y: float, chars: int) -> tuple[list[str], float]:
    out = []
    for i, line in enumerate(_wrap(text, chars)):
        out.append(f'<text x="{x:.1f}" y="{y + i * LINE:.1f}" font-size="{SIZE - 0.5:g}">'
                   f'{_esc(line)}</text>')
    return out, y + len(out) * LINE


def _box(x: float, y: float, w: float, lines: Sequence[str], *, bold_first: bool = False,
         shade: bool = False) -> tuple[list[str], float]:
    h = PAD * 2 + LINE * len(lines) - 3
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
           f'fill="{"#f2f2f2" if shade else PAPER}" stroke="{INK}" stroke-width="0.8"/>']
    for i, line in enumerate(lines):
        weight = ' font-weight="700"' if bold_first and i == 0 else ""
        out.append(f'<text x="{x + PAD:.1f}" y="{y + PAD + 9 + i * LINE:.1f}"{weight}>'
                   f'{_esc(line)}</text>')
    return out, h


# ── Figure 1: the participant flow ───────────────────────────────────────────


def participant_flow(cohort: Mapping[str, Any], *, purpose: str | None, n_train: int | None,
                     n_holdout: int | None, number: int = 1) -> tuple[str, str]:
    """The participant-flow figure and its caption (module docstring)."""
    width = 660.0
    main_x, main_w = 24.0, 300.0
    side_x, side_w = 372.0, 264.0
    chars_main, chars_side = 44, 38
    body: list[str] = []
    y = 20.0
    steps = list(cohort.get("steps") or [])
    previous_bottom: float | None = None
    for i, step in enumerate(steps):
        lines = [*_wrap(plain(step.get("label")), chars_main), f"n = {int(step.get('n') or 0):,}"]
        if previous_bottom is not None:
            dropped = int(step.get("dropped") or 0)
            gap = 22.0
            if dropped:
                why = plain(step.get("reason") or step.get("label"))
                side = _wrap(f"Excluded: {why}", chars_side) + [f"n = {dropped:,}"]
                side_h = PAD * 2 + LINE * len(side) - 3
                gap = max(gap, side_h + 16.0)
                mid = previous_bottom + gap / 2
                drawn, _ = _box(side_x, mid - side_h / 2, side_w, side)
                body += drawn
                body.append(f'<line x1="{main_x + main_w / 2:.1f}" y1="{mid:.1f}" '
                            f'x2="{side_x:.1f}" y2="{mid:.1f}" stroke="{MID}" stroke-width="0.8" '
                            f'marker-end="url(#arrow)"/>')
            body.append(f'<line x1="{main_x + main_w / 2:.1f}" y1="{previous_bottom:.1f}" '
                        f'x2="{main_x + main_w / 2:.1f}" y2="{previous_bottom + gap:.1f}" '
                        f'stroke="{MID}" stroke-width="0.8" marker-end="url(#arrow)"/>')
            y = previous_bottom + gap
        drawn, h = _box(main_x, y, main_w, lines, bold_first=(i == 0))
        body += drawn
        previous_bottom = y + h
    n_final = int(cohort.get("n_final") or 0)
    assert previous_bottom is not None or not steps
    top = (previous_bottom or y) + 22.0
    centre = main_x + main_w / 2
    if purpose == "prediction" and n_holdout:
        left = [f"Development (cross-validation)", f"n = {int(n_train or 0):,}"]
        right = [f"Held out, opened once for the final model", f"n = {int(n_holdout):,}"]
        lw = 240.0
        drawn, h = _box(main_x, top, lw, left, shade=True)
        body += drawn
        drawn2, h2 = _box(side_x, top, side_w, right, shade=True)
        body += drawn2
        body.append(f'<path d="M{centre:.1f},{(previous_bottom or y):.1f} L{centre:.1f},{top - 10:.1f} '
                    f'L{main_x + lw / 2:.1f},{top - 10:.1f} L{main_x + lw / 2:.1f},{top:.1f}" '
                    f'fill="none" stroke="{MID}" stroke-width="0.8" marker-end="url(#arrow)"/>')
        body.append(f'<path d="M{centre:.1f},{top - 10:.1f} L{side_x + side_w / 2:.1f},'
                    f'{top - 10:.1f} L{side_x + side_w / 2:.1f},{top:.1f}" fill="none" '
                    f'stroke="{MID}" stroke-width="0.8" marker-end="url(#arrow)"/>')
        bottom = top + max(h, h2)
        used = (f"{int(n_train or 0):,} for development by cross-validation and "
                f"{int(n_holdout):,} held out")
    else:
        label = ("Analyzed, every row in each estimate" if purpose == "inference"
                 else "Development, every row cross-validated")
        drawn, h = _box(main_x, top, main_w, [label, f"n = {n_final:,}"], shade=True)
        body += drawn
        body.append(f'<line x1="{centre:.1f}" y1="{(previous_bottom or y):.1f}" x2="{centre:.1f}" '
                    f'y2="{top:.1f}" stroke="{MID}" stroke-width="0.8" marker-end="url(#arrow)"/>')
        bottom = top + h
        used = f"{n_final:,} analyzed"
    first = int(steps[0].get("n") or 0) if steps else n_final
    excluded = sum(int(s.get("dropped") or 0) for s in steps)
    said = (f"Each exclusion is shown beside the step that made it, with its count and its reason "
            f"({excluded:,} excluded in all)." if excluded else "No row was excluded at any step.")
    caption = (f"Figure {number}. Participant flow, from the {first:,} rows of the table as read to "
               f"{used}. {said}")
    if not excluded and not (purpose == "prediction" and n_holdout):
        width = main_x * 2 + main_w + 120.0  # no exclusion boxes to the right
    lines, end = _caption(caption, main_x, bottom + 26.0, int((width - 2 * main_x) / (SIZE * 0.53)))
    body += lines
    svg = _svg(width, end + 8.0, f"Figure {number}. Participant flow", caption, body)
    return svg, caption


# ── Figure 2: the lineage ────────────────────────────────────────────────────

# The operation a line draws, by its dash pattern (greyscale: the pattern carries the meaning).
Style = tuple[str, str, str, str, str]
DASHES: tuple[Style, ...] = (
    # (operation prefix, dasharray, stroke, legend words, the pattern's name in the caption)
    ("kept", "", GREY, "kept as it is", "solid"),
    ("imputed", "1.2,2.4", INK, "missing values filled", "dotted"),
    ("energy-adjusted", "7,3", INK, "energy-adjusted", "long dashes"),
    ("one-hot", "3,2", MID, "coded as indicator columns", "short dashes"),
    ("scaled", "9,2,2,2", MID, "scaled", "dash-dot"),
)
OTHER: Style = ("", "1,1.5", MID, "transformed otherwise", "fine dots")
NODE_H = 17.0
NODE_GAP = 4.0
NODE_CHARS = 30


def _dash(operation: str) -> Style:
    for row in DASHES:
        if operation.startswith(row[0]):
            return row
    return OTHER


def _node_label(node: Mapping[str, Any]) -> str:
    label = plain(node.get("label") or node.get("column") or "")
    if len(label) > NODE_CHARS:
        label = label[: NODE_CHARS - 1] + "…"
    return label


def _reached(nodes: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]],
             columns: Sequence[str]) -> set[str]:
    """The ids of the nodes on a path from a raw column in ``columns``."""
    out = {str(n.get("id")) for n in nodes if n.get("lane") == "raw" and n.get("column") in columns}
    grew = True
    while grew:
        grew = False
        for link in links:
            if str(link.get("source")) in out and str(link.get("target")) not in out:
                out.add(str(link.get("target")))
                grew = True
    return out


def lineage(design: Mapping[str, Any], *, n_rows: int | None = None,
            exposures: Sequence[str] = (), number: int = 2) -> tuple[str, str]:
    """The lineage figure and its caption (module docstring). ``exposures``: the declared
    exposures, whose path through the steps is set in bold (none under prediction)."""
    graph = design.get("lineage") or {}
    nodes = list(graph.get("nodes") or [])
    links = list(graph.get("links") or [])
    emphasized = _reached(nodes, links, exposures)
    lanes = ("raw", "adjusted", "matrix")
    by_lane = {lane: [n for n in nodes if n.get("lane") == lane] for lane in lanes}
    has_out = {link.get("source") for link in links}
    widths = {lane: max([width_of(_node_label(n)) for n in by_lane[lane]] + [60.0]) + 2 * PAD
              for lane in lanes}
    for n in by_lane["raw"]:
        if n.get("id") not in has_out:
            widths["raw"] = max(widths["raw"], width_of(_node_label(n) + " (left out)") + 2 * PAD)
    gap = 120.0
    xs = {"raw": 20.0}
    xs["adjusted"] = xs["raw"] + widths["raw"] + gap
    xs["matrix"] = xs["adjusted"] + widths["adjusted"] + gap
    width = xs["matrix"] + widths["matrix"] + 20.0
    heights = {lane: len(by_lane[lane]) * (NODE_H + NODE_GAP) for lane in lanes}
    tallest = max(heights.values()) if heights else 0.0
    top = 52.0
    pos: dict[str, tuple[float, float]] = {}
    body: list[str] = []
    n_cols = int((design.get("matrix") or {}).get("n_cols") or len(by_lane["matrix"]))
    rows = n_rows if n_rows is not None else (design.get("matrix") or {}).get("n_rows")
    heads = {"raw": f"Raw columns ({sum(int(n.get('count') or 1) for n in by_lane['raw'])})",
             "adjusted": "After the shared steps",
             "matrix": f"Model matrix ({n_cols} columns)"}
    for lane in lanes:
        body.append(f'<text x="{xs[lane]:.1f}" y="{top - 16:.1f}" font-weight="700">'
                    f'{_esc(heads[lane])}</text>')
        y = top + (tallest - heights[lane]) / 2
        for node in by_lane[lane]:
            label = _node_label(node)
            left_out = lane == "raw" and node.get("id") not in has_out
            bold = str(node.get("id")) in emphasized
            dashed = ' stroke-dasharray="2,2"' if left_out else ""
            body.append(f'<rect x="{xs[lane]:.1f}" y="{y:.1f}" width="{widths[lane]:.1f}" '
                        f'height="{NODE_H:.1f}" fill="{PAPER}" stroke="{GREY if left_out else INK}" '
                        f'stroke-width="0.7"{dashed}/>')
            attrs = (' font-weight="700"' if bold else "") + (f' fill="{GREY}"' if left_out else "")
            words = f"{label} (left out)" if left_out else label
            body.append(f'<text x="{xs[lane] + PAD:.1f}" y="{y + 12.2:.1f}"{attrs}>'
                        f'{_esc(words)}</text>')
            pos[str(node.get("id"))] = (xs[lane], y + NODE_H / 2)
            y += NODE_H + NODE_GAP
    lane_of = {str(n.get("id")): n.get("lane") for n in nodes}
    used: dict[str, Style] = {}
    for link in links:
        s, t = str(link.get("source")), str(link.get("target"))
        if s not in pos or t not in pos:
            continue
        x1 = pos[s][0] + widths[str(lane_of[s])]
        y1 = pos[s][1]
        x2, y2 = pos[t]
        op = str(link.get("operation") or "kept")
        style = _dash(op)
        used.setdefault(style[3], style)
        mx = (x1 + x2) / 2
        dash = f' stroke-dasharray="{style[1]}"' if style[1] else ""
        body.append(f'<path d="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} '
                    f'{x2:.1f},{y2:.1f}" fill="none" stroke="{style[2]}" stroke-width="0.9"{dash}/>')
    y = top + tallest + 18.0
    legend = [row for row in (*DASHES, OTHER) if row[3] in used]
    x = 20.0
    for row in legend:
        dash = f' stroke-dasharray="{row[1]}"' if row[1] else ""
        body.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 30:.1f}" y2="{y:.1f}" '
                    f'stroke="{row[2]}" stroke-width="1.1"{dash}/>')
        body.append(f'<text x="{x + 36:.1f}" y="{y + 4:.1f}" font-size="{SMALL:g}">'
                    f'{_esc(row[3])}</text>')
        x += 36 + width_of(row[3], SMALL) + 22
    said = "; ".join(f"{row[4]}, {row[3]}" for row in legend)
    n_raw = sum(int(n.get("count") or 1) for n in by_lane["raw"])
    dropped = sum(1 for n in by_lane["raw"] if n.get("id") not in has_out)
    on = f" ({int(rows):,} rows)" if rows is not None else ""
    notes = [f"Line styles: {said or 'none'}."]
    if emphasized:
        notes.append("The declared exposure and the columns made from it are set in bold.")
    if dropped:
        notes.append(f"The {dropped:,} raw column{'s' if dropped != 1 else ''} left out of the "
                     f"model {'are' if dropped != 1 else 'is'} drawn dashed.")
    notes.append("Drawn from the decision record.")
    caption = (f"Figure {number}. Column lineage, from the {n_raw:,} raw columns the analysis reads "
               f"to the {n_cols:,} columns of the model matrix{on}, through the steps every model "
               f"family shares; a family's own steps (scaling, a spline basis) are in the methods. "
               + " ".join(notes))
    lines, end = _caption(caption, 20.0, y + 30.0, max(60, int((width - 40) / (SIZE * 0.5))))
    body += lines
    svg = _svg(max(width, 480.0), end + 8.0, f"Figure {number}. Column lineage", caption, body)
    return svg, caption


__all__ = ["DASHES", "lineage", "participant_flow", "plain", "width_of"]
