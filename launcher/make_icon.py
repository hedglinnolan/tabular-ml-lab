"""Distill — the Tabular ML Lab app icon.

Rendered at 4096px with supersampling, downsampled to a 1024px master.
Composition: indigo→violet gradient squircle; a solid white Erlenmeyer
flask; its belly holds a 3x2 data table punched from the glass, with the
bottom-right cell held out as a sealed ring (the test-set lockbox); three
cells rise through the neck — observations distilled into findings.

Outputs (next to this script): icon.png (1024 master), icon.ico,
icon.icns, plus a small-size legibility strip in the system temp dir.

Regenerate with:  python launcher/make_icon.py   (needs numpy + Pillow)
"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 4096            # supersample canvas
OUT = 1024          # master size
R_FRAC = 0.224      # macOS-style squircle corner fraction

HERE = Path(__file__).resolve().parent


def superellipse_mask(size, radius_frac, n=5.0):
    """Continuous-curvature rounded square (|x|^n + |y|^n = 1 corners)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    half = size / 2.0
    r = radius_frac * size
    cx = np.clip(np.abs(xx - half + 0.5) - (half - r), 0, None)
    cy = np.clip(np.abs(yy - half + 0.5) - (half - r), 0, None)
    v = (cx / r) ** n + (cy / r) ** n
    mask = (v <= 1.0).astype(np.uint8) * 255
    return Image.fromarray(mask, "L")


def gradient_bg(size):
    """Diagonal indigo → deep violet, luminous halo behind the flask mouth."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / (size - 1)
    t = np.clip((xx * 0.42 + yy * 0.58), 0, 1)
    top = np.array([124, 140, 240])                      # #7C8CF0
    bot = np.array([64, 52, 168])                        # #4034A8
    rgb = (top[None, None, :] * (1 - t[..., None]) + bot[None, None, :] * t[..., None])
    glow_c = (0.50, 0.22)                                # behind the mouth
    d = np.sqrt((xx - glow_c[0]) ** 2 + (yy - glow_c[1]) ** 2)
    glow = np.clip(1 - d / 0.75, 0, 1) ** 2 * 0.16
    rgb = rgb + (np.array([255, 255, 255])[None, None, :] - rgb) * glow[..., None]
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


# ── flask geometry (1024-space design units) ─────────────────────────────
# rim:  x 414–610, y 178–224
# neck: x 446–578, y 200–450
# cone: (446,430)/(578,430) flaring to (208,838)/(816,838), bottom rounded
CONE_TOP_Y, CONE_BOT_Y = 430.0, 838.0
CONE_HALF_TOP = 66.0        # 512±66  = 446..578
CONE_HALF_BOT = 304.0       # 512±304 = 208..816


def cone_lx(y):
    f = (y - CONE_TOP_Y) / (CONE_BOT_Y - CONE_TOP_Y)
    return 512 - (CONE_HALF_TOP + (CONE_HALF_BOT - CONE_HALF_TOP) * f)


def cone_rx(y):
    f = (y - CONE_TOP_Y) / (CONE_BOT_Y - CONE_TOP_Y)
    return 512 + (CONE_HALF_TOP + (CONE_HALF_BOT - CONE_HALF_TOP) * f)


def flask_silhouette(u):
    sil = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(sil)
    d.rounded_rectangle([414 * u, 178 * u, 610 * u, 224 * u], radius=22 * u, fill=255)
    d.rectangle([446 * u, 200 * u, 578 * u, 450 * u], fill=255)
    d.polygon([(446 * u, 430 * u), (578 * u, 430 * u),
               (816 * u, 838 * u), (208 * u, 838 * u)], fill=255)
    # round the bottom corners: intersect with a rounded rect whose bottom
    # corners coincide with the cone's feet
    clip = Image.new("L", (S, S), 0)
    dc = ImageDraw.Draw(clip)
    dc.rounded_rectangle([208 * u, 100 * u, 816 * u, 838 * u], radius=46 * u, fill=255)
    return ImageChops.multiply(sil, clip)


def table_mask(u):
    """The table in the belly — a 3x2 grid of cells punched from the glyph.

    One cell (bottom-right) is held out: punched only as a thin ring, its
    interior kept sealed in glass — the test-set lockbox.
    """
    holes = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(holes)
    cell, gap = 84.0, 26.0
    cols, rows = 3, 2
    held = (cols - 1, rows - 1)          # bottom-right: where a holdout lives
    ring = 11.0                          # ring thickness for the held cell
    total_w = cols * cell + (cols - 1) * gap
    x0 = 512 - total_w / 2
    y0 = 590.0
    for r in range(rows):
        cy0 = y0 + r * (cell + gap)
        for c in range(cols):
            cx0 = x0 + c * (cell + gap)
            box = [cx0 * u, cy0 * u, (cx0 + cell) * u, (cy0 + cell) * u]
            if (c, r) == held:
                d.rounded_rectangle(box, radius=17 * u, fill=255)
                inner = [(cx0 + ring) * u, (cy0 + ring) * u,
                         (cx0 + cell - ring) * u, (cy0 + cell - ring) * u]
                d.rounded_rectangle(inner, radius=10 * u, fill=0)
            else:
                d.rounded_rectangle(box, radius=17 * u, fill=255)
    return holes


def bubbles_mask(u):
    """Rising cells — square bubbles punched out of the glyph."""
    bubbles = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(bubbles)
    for cx, cy, s, r in [(512, 480, 60, 14), (488, 356, 44, 11), (536, 268, 30, 8)]:
        h = s / 2
        d.rounded_rectangle([(cx - h) * u, (cy - h) * u, (cx + h) * u, (cy + h) * u],
                            radius=r * u, fill=255)
    return bubbles


def main():
    u = S / 1024.0
    bg = gradient_bg(S).convert("RGBA")

    sil = flask_silhouette(u)
    table = table_mask(u)
    bubbles = bubbles_mask(u)

    # glyph mask = silhouette − table cells − bubbles
    glyph_mask = ImageChops.subtract(sil, table)
    glyph_mask = ImageChops.subtract(glyph_mask, bubbles)

    # soft indigo shadow settles the glyph onto the field
    sh = sil.filter(ImageFilter.GaussianBlur(20 * u))
    sh_alpha = sh.point(lambda p: p * 55 // 255)
    shadow = Image.new("RGBA", (S, S), (16, 10, 60, 0))
    shadow.putalpha(sh_alpha)
    bg.alpha_composite(shadow, (0, int(18 * u)))

    glyph = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    glyph.putalpha(glyph_mask)
    bg.alpha_composite(glyph)

    # squircle crop
    mask = superellipse_mask(S, R_FRAC)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(bg, (0, 0), mask)

    master = out.resize((OUT, OUT), Image.LANCZOS)
    master.save(HERE / "icon.png")
    master.save(HERE / "icon.ico",
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                       (64, 64), (128, 128), (256, 256)])
    master.save(HERE / "icon.icns", format="ICNS")

    strip = Image.new("RGBA", (16 + 32 + 64 + 128 + 256 + 5 * 24, 280), (30, 30, 40, 255))
    x = 12
    for s in (16, 32, 64, 128, 256):
        im = master.resize((s, s), Image.LANCZOS)
        strip.paste(im, (x, 270 - s - 10), im)
        x += s + 24
    strip_path = Path(tempfile.gettempdir()) / "tml_icon_strip.png"
    strip.save(strip_path)
    print(f"wrote {HERE / 'icon.png'}, icon.ico, icon.icns; preview strip: {strip_path}")


if __name__ == "__main__":
    main()
