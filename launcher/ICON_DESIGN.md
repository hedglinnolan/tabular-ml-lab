# Distill

*A design philosophy for marks that turn measurement into meaning.*

Distill begins with a refusal: no chart. The rising line, the terminal
point, the candlestick's optimism — these are the borrowed clothes of
finance, and an instrument of research must not wear them. What remains
when the chart is set aside is the honest pair at the heart of this app:
the table, and the lab. So the emblem is a vessel — the Erlenmeyer flask,
science's most legible silhouette — and what it holds is not liquid but
data: a table of cells, resting in the glass where a reagent would be.
The philosophy holds that this is the whole story an emblem may tell:
*the table goes into the flask, and findings rise out of it.*

Form is a single solid gesture. The flask is cut from white in one
uninterrupted silhouette — rim, neck, and cone — because a shape that
must survive sixteen pixels cannot afford an outline's delicacy. Every
detail inside it is made by taking material away, never by adding: the
table is a 3×2 grid of rounded cells punched through to the indigo
beneath; the rising observations are smaller cells punched through the
neck, shrinking as they climb toward the mouth. Detail by subtraction is
the scale discipline — at a thousand pixels the glass is full of
structure; at sixteen the punches close up and the mark returns to what
it always was, a white flask on an indigo tile. Nothing exists that dies
between those two distances.

One cell refuses to open. In the bottom-right of the table — where a
holdout belongs — the punch stops at a thin ring, and the cell stays
sealed in the glass: the test set, kept apart from everything that rises.
Those who know why a cell must be held back will recognize it at once;
everyone else will read only a quiet asymmetry, a table with one cell
ringed as if selected, and sense that the composition is balanced by a
small, principled exception.

Color is unchanged conviction. Indigo — the hour between evidence and
insight — pours diagonally from a luminous crown to a deep violet floor,
one continuous breath of gradient; the luminous source now sits directly
behind the flask's mouth, so the glass appears to stand in front of its
own conclusion. White is spent nowhere except the vessel. A soft indigo
shadow settles the flask onto the field, the only concession to depth a
solid glyph requires. The squircle — the continuous-curvature square
that modern desktops have made the shape of trust — governs the frame as
before; the family of radii inside it (cells, bubbles, rim, feet) is
tuned to one curvature so that no corner argues with another.

Distill's temperament is patience under heat. The table does not leap;
it steeps. Cells leave it one at a time, smaller as they rise, in the
order evidence actually surrenders its findings — slowly, and past a
holdout that never moves. The finished mark should feel less designed
than *settled*: a vessel, its contents, and the one honest thing it
refuses to give up.

---

**Regeneration.** `python launcher/make_icon.py` (needs numpy + Pillow)
renders at 4096px and downsamples to the 1024px master, then writes
`icon.png`, `icon.ico` (16–256px), and `icon.icns` (16–1024px with retina
variants) via PIL. Key geometry, in 1024-space: rim 414–610 × 178–224;
neck 446–578 down to y 450; cone flaring to 208–816 at y 838 with
46-radius feet; table cells 84px with 26px gaps starting at y 590, held
ring 11px thick; rising cells 60/44/30px at (512, 480), (488, 356),
(536, 268); squircle corner fraction 0.224, superellipse n = 5.0.
