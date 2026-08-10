"""`DRIVE-015` — the separation `DESIGN_LANGUAGE.md:55` asserts, as a check.

> charts use a separate categorical ramp so semantic color stays semantic

That sentence has been in the design language since the start and **nothing has
ever checked it**, which is how `--c1` came to be byte-identical to `--accent`
in all four theme blocks of both files. A design claim with no check is the
condition-three gap `PRODUCT_VISION.md` §06b names.

## The validator ships before the palette, including a palette the adjudicator proposed

That ordering is the whole point of the row. Values are proposed with
measurements attached; the measurement that decides is the one this file makes,
against this tree, on every run. **If it rejects a proposed ramp, that is the
validator working.**

## Four gates, and the fourth is the one that was missed twice

1. **`--c1` is not `--accent`**, per theme, in **both** files.
2. **CVD separation** — every pair, under deuteranopia, protanopia **and**
   tritanopia.
3. **Contrast against its own ground** — a ramp can be perfectly separable and
   invisible. `L56-C2` proposed a ramp that was CVD-clean and theme-inverted:
   `#1F3D82` is 9.62:1 on the light ground and **1.74:1** on the dark.
   Separation and legibility are different axes and both are gates.
4. **Separation from the semantic hues** — `--accent`, `--ok`, `--warn`,
   `--stop`. Measured rather than judged, because "close to the delete red" was
   an observation nobody could act on until it had a number.

## The simulator is validated on controls inside the test

`GUIDED-045`'s positive-control rule, arriving in a new place. **A CVD check
that cannot fail is the thing under test**, so before it judges anything it must
show that red/green collapses and that blue/orange survives. A broken matrix
would otherwise pass every ramp ever proposed.

## Where the tokens live, which is not symmetrical

`--c1`–`--c4`, `--accent`, `--ok` and `--warn` are in the **carried** first
`<style>` block — 30,968 characters, byte-identical between
`turbotab/web/index.html` and `docs/turbotab/prototypes/interview-feed.html`,
asserted by `test_skeleton.py`. **`--stop` is not**: it is a build-added token in
the third block and appears **zero** times in the prototype, with its own comment
saying why it is not `--del`. So the ramp is checked in both files and `--stop`
is read from the app only, and that asymmetry is asserted rather than assumed.
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "turbotab", "web", "index.html")
PROTOTYPE = os.path.join(ROOT, "docs", "turbotab", "prototypes",
                         "interview-feed.html")

#: The ramp, in order. `--c4` is NOT part of it — see `test_the_fourth_slot…`.
RAMP = ("--c1", "--c2", "--c3")
SEMANTIC = ("--accent", "--ok", "--warn", "--stop")

#: Gate values. Both are floors on a measurement, and both are stated with the
#: units and the formula so a later reader can reproduce them:
#: CIE76 ΔE in CIE L*a*b*, and the WCAG 2.x relative-luminance contrast ratio.
CVD_FLOOR = 15.0
CONTRAST_FLOOR = 3.0
SEMANTIC_FLOOR = 15.0

#: Machado, Oliveira & Fernandes (2009), severity 1.0. Published matrices,
#: named rather than hand-tuned, and exercised on controls below.
CVD = {
    "deuteranopia": ((0.367322, 0.860646, -0.227968),
                     (0.280085, 0.672501, 0.047413),
                     (-0.011820, 0.042940, 0.968881)),
    "protanopia": ((0.152286, 1.052583, -0.204868),
                   (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
    "tritanopia": ((1.255528, -0.076749, -0.178779),
                   (-0.078411, 0.930809, 0.147602),
                   (0.004733, 0.691367, 0.303900)),
}


# ── color, from first principles so nothing is imported that could drift ────

def _rgb(value: str) -> Tuple[float, float, float]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lab(value: str) -> Tuple[float, float, float]:
    r, g, b = (_linear(c) for c in _rgb(value))
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29

    fx, fy, fz = f(x / 0.95047), f(y / 1.0), f(z / 1.08883)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    """CIE76 in L*a*b*. Named because the number is meaningless without it."""
    return math.dist(_lab(a), _lab(b))


def luminance(value: str) -> float:
    r, g, b = (_linear(c) for c in _rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def simulate(value: str, kind: str) -> str:
    matrix = CVD[kind]
    rgb = _rgb(value)
    out = [sum(matrix[i][j] * rgb[j] for j in range(3)) for i in range(3)]
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02X}" for c in out)


# ── reading the stylesheet ───────────────────────────────────────────────────

_THEMES = (
    ("light", re.compile(r"^\s*:root\s*\{")),
    ("dark", re.compile(r"^\s*@media\s*\(prefers-color-scheme:\s*dark\)")),
    ("light", re.compile(r'^\s*:root\[data-theme="light"\]')),
    ("dark", re.compile(r'^\s*:root\[data-theme="dark"\]')),
)
_TOKEN = re.compile(r"(--[a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{6})")


def blocks(path: str) -> List[Tuple[str, str, Dict[str, str]]]:
    """Every theme block separately: `[(label, theme, {token: hex})]`.

    **EACH BLOCK IS CHECKED ON ITS OWN, and a revert probe is why.** The first
    draft collapsed these to `{theme: …}` with last-definition-wins, reasoning
    that `:root[data-theme="dark"]` is written after the media query so an
    explicit toggle beats a system preference. That is true of the cascade and
    wrong as a check: reverting **only** the `:root {` block came back GREEN —
    NOT LOAD-BEARING, because the collapsed view never read it.

    `:root {}` is what a reader with a light system preference and no explicit
    toggle actually gets. Four blocks are four surfaces a person can be looking
    at, so all four are gated, and a regression in any one of them is a
    regression somebody sees.
    """
    out: List[Tuple[str, str, Dict[str, str]]] = []
    label = theme = None
    for line in open(path, encoding="utf-8"):
        for name, pattern in _THEMES:
            if pattern.match(line):
                label, theme = line.strip().rstrip("{").strip(), name
                out.append((label, theme, {}))
                break
        if not out:
            continue
        for token, value in _TOKEN.findall(line):
            out[-1][2][token] = value.upper()
    return [b for b in out if b[2]]


def tokens(path: str) -> Dict[str, Dict[str, str]]:
    """`{theme: {token: hex}}` — the cascade's answer, for the carry check only.

    Kept because comparing the two FILES needs one value per theme. Every gate
    below iterates :func:`blocks` instead.
    """
    out: Dict[str, Dict[str, str]] = {"light": {}, "dark": {}}
    for _label, theme, found in blocks(path):
        out[theme].update(found)
    return out


def _pairs(values: Sequence[str]) -> Iterable[Tuple[int, int]]:
    return ((i, j) for i in range(len(values)) for j in range(i + 1, len(values)))


# ═══════════ 0 · the instrument, before it judges anything ═══════════════════

def test_the_simulator_collapses_what_it_should_and_keeps_what_it_should():
    """`GUIDED-045`'s positive control, on a color-vision model.

    A CVD check that cannot fail is the thing under test. A transposed matrix,
    a wrong severity or a linear/sRGB mix-up all produce a simulator that leaves
    every color roughly where it found it — and then every ramp ever proposed
    passes gate 2 for a reason that has nothing to do with color vision.
    """
    red, green, blue, orange = "#FF0000", "#00FF00", "#0000FF", "#FF8C00"

    normal = delta_e(red, green)
    collapsed = delta_e(simulate(red, "deuteranopia"),
                        simulate(green, "deuteranopia"))
    assert normal > 100, f"red and green are {normal:.1f} apart to normal vision"
    assert collapsed < normal / 3, (
        f"red and green stay {collapsed:.1f} apart under the deuteranopia "
        f"simulation, against {normal:.1f} to normal vision. The simulator is "
        f"not simulating anything, so gate 2 cannot fail.")

    kept = delta_e(simulate(blue, "deuteranopia"),
                   simulate(orange, "deuteranopia"))
    assert kept > CVD_FLOOR * 3, (
        f"blue and orange collapse to {kept:.1f} under deuteranopia. They must "
        f"not — a simulator that flattens everything rejects every possible "
        f"ramp and is as useless as one that flattens nothing.")

    # AND THE THIRD KIND IS EXERCISED TOO, because tritanopia is the one a
    # blue/orange ramp is most likely to fail and the easiest to forget.
    tritan = delta_e(simulate(blue, "tritanopia"), simulate(orange, "tritanopia"))
    assert tritan > 0, "the tritanopia matrix produced identical colors"


def test_contrast_is_computed_the_way_the_accessibility_floor_is_stated():
    """Controls on the other instrument. 21:1 and 1:1 are the fixed points."""
    assert abs(contrast("#FFFFFF", "#000000") - 21.0) < 0.01
    assert abs(contrast("#777777", "#777777") - 1.0) < 0.001


# ═══════════ 1 · the ramp is not the accent ═════════════════════════════════

@pytest.mark.parametrize("path,label", [(APP, "index.html"),
                                        (PROTOTYPE, "interview-feed.html")])
def test_the_first_series_is_not_the_accent(path, label):
    """`DRIVE-015`'s original defect, in BOTH files.

    Checking only the app would let the next carry silently reintroduce it,
    which is this project's most-repeated failure shape.
    """
    found = blocks(path)
    assert len(found) >= 4, (
        f"{label}: parsed {len(found)} theme blocks; the file defines four "
        f"({[b[0] for b in found]}). A parser that misses one gates nothing "
        f"for the readers who land in it.")
    for selector, theme, ramp in found:
        if "--c1" not in ramp:
            continue                       # a block that defines no ramp
        assert "--accent" in ramp, (
            f"{label}/{selector}: defines --c1 and no --accent, so the "
            f"comparison below cannot be made in the block it matters in")
        gap = delta_e(ramp["--c1"], ramp["--accent"])
        assert ramp["--c1"] != ramp["--accent"], (
            f"{label}/{selector}: --c1 is {ramp['--c1']} and --accent is "
            f"{ramp['--accent']} — the same color. DESIGN_LANGUAGE.md:55 says "
            f"charts use a SEPARATE categorical ramp so semantic color stays "
            f"semantic; a data series drawn in the 'you are here' hue is the "
            f"app saying two things with one color. DRIVE-015.")
        assert gap >= SEMANTIC_FLOOR, (
            f"{label}/{selector}: --c1 is {gap:.1f} from --accent (floor "
            f"{SEMANTIC_FLOOR}) — different values, same color to a reader")


def test_the_two_files_agree_about_the_ramp():
    """The carry, from the ramp's side.

    `test_skeleton.py` asserts the whole block is byte-identical. This asserts
    the thing this row is about, so a future edit that touched only the app
    fails HERE with a message about the palette rather than in a 30,968-character
    diff.
    """
    # BLOCK BY BLOCK IN ORDER, not by selector name and not by cascade.
    #
    # Not by cascade, because a revert of one block alone leaves the cascade's
    # answer unchanged and two files would read as agreeing when one of their
    # four blocks does not — the same lesson as the gates above.
    #
    # And not keyed by SELECTOR, which was the first attempt and was wrong for a
    # reason worth keeping: `:root {` is not unique. The app reopens it in a
    # later `<style>` for `--stop`, so a dict keyed on the label kept the LAST
    # `:root` — a block with no ramp in it at all — and the comparison failed
    # against `None`. The ramp-bearing blocks are the carried ones, so those are
    # what is compared, in order.
    app = [(label, found) for label, _t, found in blocks(APP) if "--c1" in found]
    proto = [(label, found) for label, _t, found in blocks(PROTOTYPE)
             if "--c1" in found]
    assert len(app) == len(proto) >= 4, (
        f"index.html has {len(app)} ramp-bearing theme blocks and the prototype "
        f"has {len(proto)}; the carried block defines four in each")
    for (label, a), (proto_label, b) in zip(app, proto):
        assert label == proto_label, (
            f"the ramp blocks are in a different order: {label} vs {proto_label}")
        for token in RAMP + ("--accent", "--ok", "--warn"):
            assert a.get(token) == b.get(token), (
                f"{label}/{token}: index.html says {a.get(token)} and the "
                f"prototype says {b.get(token)}. The prototype is carried byte "
                f"for byte — edit it FIRST and re-carry.")


def test_stop_is_an_app_token_and_the_asymmetry_is_deliberate():
    """`--stop` is build-added and absent from the prototype, on purpose.

    Asserted rather than assumed, because the semantic check below reads it from
    one file and a reader would otherwise wonder whether that was an oversight.
    """
    assert "--stop" in tokens(APP)["light"], "the app defines no --stop"
    assert "--stop" not in tokens(PROTOTYPE)["light"], (
        "--stop is now in the prototype; it was a build-added token whose own "
        "comment says why it is not --del, and the semantic check should read "
        "it from both files if it has been carried")


# ═══════════ 2, 3, 4 · the ramp is legible and separable ════════════════════

def test_every_pair_in_the_ramp_survives_every_kind_of_color_blindness():
    """Gate 2. Every pair, all three kinds — not the worst case of one kind.

    `L56-C2` measured the previously-proposed blue/orange/purple ramp at ΔE 2.0
    under deuteranopia: purple is blue plus red, and the red-green axis is what
    deuteranopia removes. Two series became one color, and the legend could not
    rescue them because the legend swatches were those same two colors.
    """
    failures = []
    checked = 0
    for selector, _theme, ramp in blocks(APP):
        values = [ramp[c] for c in RAMP if c in ramp]
        if len(values) != len(RAMP):
            continue
        checked += 1
        for kind in sorted(CVD):
            seen = [simulate(v, kind) for v in values]
            for i, j in _pairs(seen):
                gap = delta_e(seen[i], seen[j])
                if gap < CVD_FLOOR:
                    failures.append(
                        f"{selector}: {RAMP[i]} vs {RAMP[j]} under {kind}: "
                        f"{gap:.1f} ({values[i]} -> {seen[i]}, "
                        f"{values[j]} -> {seen[j]})")
    assert checked >= 4, f"only {checked} block(s) define a full ramp"
    assert not failures, (
        f"{len(failures)} ramp pair(s) below the CVD floor of "
        f"{CVD_FLOOR} (CIE76 ΔE, Machado 2009 severity 1.0):\n  "
        + "\n  ".join(failures) +
        "\n\nTwo series a reader cannot tell apart are one series drawn twice.")


def test_every_series_is_visible_against_its_own_ground():
    """Gate 3, and it is the axis `L56-C2` validated separation on and missed.

    A ramp can be perfectly separable and invisible. The ramp proposed then was
    CVD-clean and theme-inverted: #1F3D82 is 9.62:1 on the light ground and
    1.74:1 on the dark. **Its own** ground, per theme — the whole reason the
    ramp is specified per theme rather than once.
    """
    thin = []
    checked = 0
    for selector, _theme, read in blocks(APP):
        if not all(t in read for t in RAMP):
            continue
        ground = read.get("--surface") or read.get("--ground")
        assert ground, f"{selector}: no --surface or --ground to measure against"
        checked += 1
        for token in RAMP:
            ratio = contrast(read[token], ground)
            if ratio < CONTRAST_FLOOR:
                thin.append(f"{selector}: {token} {read[token]} on {ground}: "
                            f"{ratio:.2f}:1")
    assert checked >= 4, f"only {checked} block(s) define a full ramp and a ground"
    assert not thin, (
        f"{len(thin)} series below the {CONTRAST_FLOOR}:1 floor for "
        f"non-text graphics:\n  " + "\n  ".join(thin) +
        "\n\nA series nobody can see is not a series.")


def test_no_series_is_mistakable_for_a_semantic_hue():
    """Gate 4. The judgment that had no number until `L56-C2` gave it one.

    `--accent` means *you are here*, `--ok` *this is fine*, `--warn` *this has a
    cost*, `--stop` *this is invalid downstream*. A chart series close enough to
    one of them borrows a meaning nobody assigned it — which is
    `DESIGN_LANGUAGE.md` §02's *every hue is a claim*, from the other side.
    """
    cascade = tokens(APP)
    close = []
    checked = 0
    for selector, theme, read in blocks(APP):
        if not all(t in read for t in RAMP):
            continue
        checked += 1
        for token in RAMP:
            for semantic in SEMANTIC:
                # `--stop` is defined in a LATER block than the ramp, so it is
                # taken from the cascade for this theme; the rest are in the
                # block itself. Asserted below rather than assumed.
                value = read.get(semantic) or cascade[theme].get(semantic)
                assert value, f"{selector}: no value anywhere for {semantic}"
                gap = delta_e(read[token], value)
                if gap < SEMANTIC_FLOOR:
                    close.append(f"{selector}: {token} {read[token]} vs "
                                 f"{semantic} {value}: {gap:.1f}")
    assert checked >= 4, f"only {checked} block(s) define a full ramp"
    assert not close, (
        f"{len(close)} series sit within {SEMANTIC_FLOOR} of a "
        f"semantic hue:\n  " + "\n  ".join(close) +
        "\n\nSemantic color stays semantic — DESIGN_LANGUAGE.md:55.")


# ═══════════ 5 · what the ramp does NOT promise ═════════════════════════════

def test_the_fourth_slot_is_not_a_fourth_categorical_series():
    """Three is the maximum this palette can carry, and it is said out loud.

    Teal, green, gold and red are reserved for meaning; what is left is
    essentially blue and orange, and a third hue that survives deuteranopia does
    not exist in the remaining space. So the third series is a **lightness step**
    and anything beyond three is told apart by **dash pattern** —
    `DESIGN_LANGUAGE.md` §07's journal rule promoted in-app rather than a new
    one invented. `--c4` is retained as a low-chroma "Other", deliberately not
    part of the ramp, and this test exists so a later loop does not quietly add
    it to `RAMP` and reintroduce the collapse.
    """
    read = tokens(APP)["light"]
    assert "--c4" in read, "--c4 was removed; the 'Other' slot has no token"
    assert "--c4" not in RAMP, (
        "--c4 has been added to the categorical ramp. Three is the measured "
        "maximum — check it against the gates above before promoting it, and "
        "if it passes, this test is what should change.")
