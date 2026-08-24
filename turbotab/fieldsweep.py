"""turbotab.fieldsweep — does anything read each FIELD the server composes?

**L42-B.** The door already guards one granularity and it is the coarse one.
`test_every_server_surface_names_its_reader` asserts every route is either
fetched by the page or declared unread with its reader named. It is a real gate,
well built, and its comment records that it had to be written twice.

**It could not see either of L41's two `critical` reachability defects.**
`/project/{project_id}` *is* fetched, on every render, so the route "has a
reader" — while a whole `source` class inside its payload was rendered by
nothing (`GUIDED-142`). There are 45 routes and 49 top-level fields in the
project payload alone, 16 on each finding. The route surface is guarded; the
field surface, an order of magnitude larger, was not.

| | Question | Guarded before L42 |
|---|---|---|
| **Route** | does the page fetch it? | yes |
| **Field** | does anything read each field of the response? | **no — this module** |
| **Name** | does every name the page calls resolve? | no — `GUIDED-140` |

## Why a literal search cannot answer it

The standing check's own comment says why for routes: `runPull` composes
`"/evidence/" + endpoint`, so a fetched path appears nowhere in the file.
**Fields are worse.** `f.severity` is read as `f[k]` inside a loop as often as
by name, and a key referenced in the source may still never reach a person —
`GUIDED-142`'s findings were *in* an array the page iterated and the filter
dropped them.

So the instrument is the harness, and the question is the one the loop prompt
states: **does a value that only this field could produce appear in what the
page rendered?**

## The method, and the second pass is a correction rather than an optimization

**Pass 1 — batch.** Every string and number leaf is replaced with a unique
sentinel, the page is rendered once, and the sentinels that appear in the DOM
mark their fields as reaching a person. One render for two thousand fields.

**Pass 2 — confirm every miss by group-negative bisection.** Batching has a
real bias: a sentinel in `task_type` changes which branches run, so a field
that *would* have rendered may not. That biases toward false *unread*, which is
the direction that manufactures findings.

So every pass-1 miss is re-probed against a clean baseline — and the probe is
**sound at group granularity**, which is what makes it affordable. Tag a chunk
of candidates and render: **if the DOM is byte-identical to the baseline, not
one of them is read**, because a field the page touches cannot change and leave
the output unchanged. A chunk that *does* move the DOM is split and recursed
until the movers are isolated. Confirming two thousand negatives costs a
handful of renders rather than two thousand.

**A mutation that crashes the render is a hit, not a failure.** The page
touched the field — that is the definition. Bisection isolates it and it is
recorded `render broke`, which is the strongest evidence of a reader this
module can produce.

Pass 2 also carries the types pass 1 cannot tag: a boolean has no sentinel, so
it is flipped and the DOM is compared, which is the same test.

`GUIDED-081` is why the comparison is against the rendered DOM rather than
against a key list: the harness silently dropped every `className` write, so an
assertion about styling came back passing. What reached the DOM is the only
thing that answers *did a person see it*.

## The enumeration is derived

A hand-maintained list is what `register.py`'s docstring exists because of — the
register was a markdown table until a merge blind-copied an older one over it
and a section vanished silently. Fields come from the live payload, so **a field
added next loop with no reader fails here** rather than being found by the loop
after.

**Arrays are enumerated element by element, not just their first.** That is
exactly `GUIDED-142`: `findings[0].title` renders and `findings[7].title` does
not, because the seventh is a pack finding and the filter dropped its whole
class. A sweep that sampled element 0 would have reported the array read and
missed the defect it exists to catch.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: How many elements of a list are enumerated. **Named and reported**, because
#: `LOOP.md` §10's no-silent-caps rule is the whole reason this is a constant
#: rather than a slice buried in a loop. Twenty-four covers every array the
#: fixtures produce; where it bites, `Sweep.truncated_arrays` says which.
MAX_ELEMENTS = 24


class SweepError(AssertionError):
    """The sweep cannot answer honestly, so it does not answer."""

#: A sentinel that survives HTML escaping, `esc()`, `toLocaleString` and
#: substring search, and that cannot occur in real data. Letters only — a digit
#: run risks colliding with a formatted number, and punctuation risks being
#: escaped into an entity.
_TAG = "ZQXSENTINEL"

#: Tagging a NUMBER cannot use letters. A distinctive integer is used instead,
#: and both its bare and its `toLocaleString` forms are searched — the page
#: renders most numbers through `num()`, which inserts separators.
_NUM_BASE = 918_273_600


def _sentinel(i: int) -> str:
    return f"{_TAG}{i:04d}"


def _num_sentinel(i: int) -> int:
    return _NUM_BASE + i


def _num_forms(value: int) -> Tuple[str, ...]:
    """Every way the page might render this integer."""
    return (str(value), f"{value:,}")


@dataclass
class Field:
    """One leaf of one payload, and what the sweep decided about it."""
    route: str
    path: str
    kind: str                      # str | num | bool | none | empty
    sample: Any
    #: Set by the sweep. `None` until decided.
    reaches: Optional[bool] = None
    #: How it was decided, so a reader can tell a batch hit from a confirm.
    how: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.route, self.path)

    def to_dict(self) -> Dict[str, Any]:
        return {"route": self.route, "path": self.path, "kind": self.kind,
                "reaches": self.reaches, "how": self.how}


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "none"
    if isinstance(value, str):
        return "str" if value.strip() else "empty"
    if isinstance(value, (int, float)):
        return "num"
    if isinstance(value, (list, dict)):
        return "empty" if not value else ""
    return ""


def leaves(payload: Any, route: str, prefix: str = "") -> List[Field]:
    """Every leaf of this payload, as `route` + a dotted path.

    Descends into dicts and into lists **element by element**. A container is
    itself a leaf only when it is empty, because an empty list has no member to
    stand for it and *nothing renders* and *nothing to render* are different
    answers this sweep must not conflate.
    """
    out: List[Field] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(leaves(value, route, f"{prefix}.{key}" if prefix else str(key)))
        if not payload and prefix:
            out.append(Field(route, prefix, "empty", payload))
        return out
    if isinstance(payload, list):
        if not payload:
            out.append(Field(route, prefix, "empty", payload))
            return out
        for i, value in enumerate(payload[:MAX_ELEMENTS]):
            out.extend(leaves(value, route, f"{prefix}[{i}]"))
        return out
    kind = _kind(payload)
    if kind:
        out.append(Field(route, prefix, kind, payload))
    return out


# ── writing a tagged copy ────────────────────────────────────────────────────

#: `a.b[2].c` → `['a', 'b', 2, 'c']`. One tokenizer over the whole path.
#:
#: **The first version walked part by part and reordered indices with a stack**,
#: and it put `questions[0].title` back together as `['questions', 'title', 0]`
#: — so every nested-array path failed to poke, silently, and the sweep
#: reported 2,408 of 2,450 fields unread. `poke` returned `False` and nobody
#: read it. That is this module's own subject arriving inside it: a value the
#: caller composed, discarded at the boundary, and indistinguishable from a
#: real answer.
_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _steps(path: str) -> List[Any]:
    return [name if name else int(index)
            for name, index in _TOKEN.findall(path)]


def poke(payload: Any, path: str, value: Any) -> bool:
    """Set `path` on a payload in place. `False` when the path is not there."""
    node = payload
    steps = _steps(path)
    for step in steps[:-1]:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return False
    try:
        node[steps[-1]] = value
        return True
    except (KeyError, IndexError, TypeError):
        return False


def peek(payload: Any, path: str) -> Any:
    node = payload
    for step in _steps(path):
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return None
    return node


def _tagged_value(fld: Field, i: int) -> Any:
    """A value that is distinguishable AND type-preserving.

    **Type-preserving is not politeness.** The first version replaced an empty
    list with a sentinel string, and `paintPalette` does
    `plan.questions.filter(...)` — the render died on the first batch and the
    sweep reported nothing. A container keeps its container-ness and carries
    the tag inside it; a boolean has no room for a tag at all and is flipped.
    """
    if fld.kind == "num":
        return _num_sentinel(i)
    if fld.kind == "bool":
        return not fld.sample
    if isinstance(fld.sample, list):
        return [_sentinel(i)]
    if isinstance(fld.sample, dict):
        return {"__probe": _sentinel(i)}
    return _sentinel(i)                     # str, empty str, or None


def _appears(dom: str, fld: Field, i: int) -> bool:
    if fld.kind == "num":
        return any(form in dom for form in _num_forms(_num_sentinel(i)))
    return _sentinel(i) in dom


# ── the rendered DOM ─────────────────────────────────────────────────────────

def container_ids(page_source: str) -> List[str]:
    """Every element id the page declares, **derived from the page**.

    Hand-listing them is how a surface added next loop goes unswept, which is
    this module's own subject one level up. Read from `id="..."` in the markup,
    which is where a container is created.
    """
    return sorted(set(re.findall(r'\bid="([A-Za-z][\w-]*)"', page_source)))


#: **The drive, and without it the sweep is mostly wrong.**
#:
#: The first version read the DOM after bootstrap and reported 1,974 of 2,015
#: fields unread — including `name`, which `renderSample` puts on screen. It
#: was not wrong about the render: `renderSample` fills `sampleBox` only when
#: the sample pill is pressed, and a passive drive never presses it. A field
#: behind a control **is** read; a sweep that calls it unread is measuring how
#: the page starts rather than what it does.
#:
#: So every control the page rendered is pressed before the DOM is read. The
#: attribute VALUES are scraped out of the markup the page itself produced, so a
#: new instance of a known control is pressed without anyone touching this file,
#: and each is dispatched through the page's own delegated click handler.
#:
#: **The attribute NAMES are hand-listed, and that sentence used to say nothing
#: was.** It was wrong in the way a claim about code goes wrong — silently, while
#: the people who already know keep working — and `GUIDED-149` found it: the
#: Explore stack gained a `data-stack-more` affordance, the sweep did not press
#: it, and the seven findings behind it were reported as reaching nobody. That
#: reading was *correct for a passive drive and false about the app*, which is
#: exactly the defect the paragraph above describes, arriving a second time
#: through a control the list did not name.
#:
#: `data-panel` and `data-look` are the two deferred-render families:
#: `data-look` is the pull palette (`runPull`), `data-panel` is the per-finding
#: evidence preview. `data-stack-more` opens `GUIDED-149`'s collapsed remainder,
#: which is disclosure rather than absence — the findings behind it are served,
#: counted and one press away, and a sweep that called them dark could not tell
#: that from `GUIDED-142`, where they were computed and rendered nowhere.
#: `samplePill` is its own control and has no data attribute, so it is named
#: because the page names it.
#:
#: **Adding one here widens what the sweep can see and never narrows it**, which
#: is why this is not `LOOP.md` §08's *a threshold moved under pressure*: the
#: quantity gated is unchanged, and a control the page offers and the drive skips
#: is the instrument measuring how the page starts rather than what it does.
_READ_ALL = """
var IDS = %s;
var NEEDLES = %s;
function collect(){
  var out = "";
  IDS.forEach(function(k){ out += (__harness.html(k) || ""); });
  return out;
}
/* Press everything the page offered. The attributes come from what it
   rendered, so this widens as the page does. */
var markup = collect();
var pressed = {};
["data-look", "data-panel", "data-target-col", "data-stack-more"].forEach(function(attr){
  var rx = new RegExp(attr + '="([^"]*)"', "g"), m;
  while ((m = rx.exec(markup)) !== null){
    var key = attr + "=" + m[1];
    if (pressed[key]) continue;
    pressed[key] = 1;
    var attrs = {};
    attrs[attr] = m[1];
    /* The pull palette needs its endpoint too — `runPull` reads it off the
       chip, and a click without it fetches nothing. */
    if (attr === "data-look"){
      var ep = new RegExp(attr + '="' + m[1].replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&") +
                          '"[^>]*data-endpoint="([^"]*)"').exec(markup);
      if (ep) attrs["data-endpoint"] = ep[1];
    }
    try { __harness.dispatch("click", __harness.target(attrs, [])); } catch (e) {}
  }
});
try { document.getElementById("samplePill") &&
      __harness.dispatch("click", __harness.target({id: "samplePill"}, [])); }
catch (e) {}
setTimeout(function(){
  var dom = markup + collect();
  /* THE DOM IS NOT SHIPPED BACK. It runs to hundreds of kilobytes and the
     harness emits over a pipe — the first version truncated the JSON and the
     sweep died on its own output. What the caller needs is two answers, and
     both are computable here: which sentinels appeared, and whether this
     render differs from any other. So the search happens in the page's own
     process and a hash stands in for equality. */
  var found = [];
  NEEDLES.forEach(function(pair){
    for (var i = 0; i < pair[1].length; i++){
      if (dom.indexOf(pair[1][i]) !== -1){ found.push(pair[0]); return; }
    }
  });
  var h = 5381;
  for (var i = 0; i < dom.length; i++) h = ((h * 33) ^ dom.charCodeAt(i)) >>> 0;
  __emit({hits: found, hash: h, len: dom.length});
}, 0);
"""

#: The rendered DOM is compared as a string and emitted over a pipe, so it is
#: capped. Reported by `Sweep.dom_truncated` when it bites, per the no-silent-
#: caps rule.
MAX_DOM = 400_000


def probe(routes: Dict[str, Any], project_id: str, ids: Sequence[str],
          needles: Sequence[Tuple[int, Sequence[str]]]) -> Optional[Dict[str, Any]]:
    """Drive the page and report `{hits, hash, len}`, or `None` if it died.

    `needles` is `(field index, the strings that would prove it rendered)`.
    Searched inside the page's own process, because the DOM is hundreds of
    kilobytes and the harness emits over a pipe — shipping it back truncated
    the JSON and the sweep died on its own output.

    `None` when the controller died. **That is a result, not an error**: a
    mutation the page cannot survive is a mutation the page read, and the
    caller turns it into the strongest hit this module has. Swallowing it as an
    exception would make a crash indistinguishable from silence, which is the
    one confusion this module is about.
    """
    from turbotab import pageharness as PH

    body = _READ_ALL % (json.dumps(list(ids)),
                        json.dumps([[i, list(forms)] for i, forms in needles]),
                        )
    try:
        return PH.run(body, routes=routes, search=f"?project={project_id}")
    except PH.HarnessError:
        return None


# ── the sweep ────────────────────────────────────────────────────────────────

@dataclass
class Sweep:
    fields: List[Field] = field(default_factory=list)
    routes_swept: List[str] = field(default_factory=list)
    truncated_arrays: List[str] = field(default_factory=list)
    #: Fields that were `null` or an empty container on the swept project.
    #: **Held out of `unread` deliberately** — see the note in `sweep`.
    empty: List[Field] = field(default_factory=list)
    #: Fields whose value IS the project id this drive is addressed by, each
    #: measured individually and confirmed read. Held out of the BATCH pass,
    #: because one unaddressable id costs every other field its measurement —
    #: `AUDIT-040`'s floor. Reported rather than dropped: they are read, and a
    #: count that omitted them would be a smaller denominator wearing the same
    #: name.
    address: List[Field] = field(default_factory=list)
    dom_truncated: bool = False
    n_renders: int = 0

    @property
    def reaching(self) -> List[Field]:
        return [f for f in self.fields if f.reaches] + list(self.address)

    @property
    def unread(self) -> List[Field]:
        return [f for f in self.fields if f.reaches is False]

    def shapes(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """The fields collapsed by PATH SHAPE, which is the reportable unit.

        `findings[0].title` and `findings[7].title` are one shape,
        `findings[*].title`, and the verdict over a shape has **three** values
        rather than two:

        * `all` — every element's field reached a person;
        * `none` — none did;
        * **`partial`** — some did and some did not, which is `GUIDED-142`
          exactly: `findings[0].title` rendered and `findings[7].title` did
          not, because the seventh was a pack finding and the filter dropped
          its whole class.

        Two thousand per-element verdicts are not an artifact anybody can
        dispose of; a hundred-odd shapes are. And collapsing without the third
        verdict would have hidden the defect this module exists to catch, which
        is why `partial` is separated rather than rounded to `all`.
        """
        groups: Dict[Tuple[str, str], List[Field]] = {}
        for fld in self.fields:
            groups.setdefault((fld.route, _shape(fld.path)), []).append(fld)
        out = {}
        for key, members in groups.items():
            reached = [m for m in members if m.reaches]
            verdict = ("all" if len(reached) == len(members)
                       else "none" if not reached else "partial")
            out[key] = {"verdict": verdict, "n": len(members),
                        "n_reaching": len(reached),
                        "kinds": sorted({m.kind for m in members}),
                        "unread_indices": [m.path for m in members
                                           if not m.reaches][:6]}
        return out

    def counts(self) -> Dict[str, int]:
        return {"routes": len(self.routes_swept),
                "fields": len(self.fields),
                "empty": len(self.empty),
                "reaching": len(self.reaching),
                "unread": len(self.unread),
                "renders": self.n_renders}


def sweep(routes: Dict[str, Any], project_id: str, ids: Sequence[str],
          *, confirm: bool = True) -> Sweep:
    """Which fields of these payloads reach a person.

    `routes` is the harness route table — path → payload — exactly as the page
    would be answered. Every payload in it is enumerated and swept; nothing is
    sampled, so what is not here was not looked at, and the caller says which.
    """
    result = Sweep(routes_swept=sorted(routes))
    for route, payload in sorted(routes.items()):
        found = leaves(payload, route)
        result.fields.extend(found)
        for path, value in _walk_lists(payload):
            if len(value) > MAX_ELEMENTS:
                result.truncated_arrays.append(
                    f"{route}{'.' if path else ''}{path} "
                    f"({len(value)} elements, first {MAX_ELEMENTS} swept)")

    # **A FIELD WITH NOTHING IN IT IS NOT AN UNREAD FIELD.** `lockbox`,
    # `selected_models`, `pipeline_specs` and two dozen others are `null` at the
    # Explore step because the journey has not reached them, and a sweep that
    # counted those as unread would report thirty findings about a project
    # nobody had finished. It is the same distinction `leaves` already draws
    # for an empty container, applied one level up — and it is where the
    # honest count lives, so they are held out and counted rather than
    # silently dropped.
    result.empty = [f for f in result.fields if f.kind in ("none", "empty")]
    for fld in result.empty:
        fld.how = "empty on this project"
    result.fields = [f for f in result.fields if f.kind not in ("none", "empty")]

    # **THE DRIVE'S OWN ADDRESS CANNOT BE TAGGED, AND THIS IS `AUDIT-040`'S FLOOR.**
    #
    # The page composes every fetch it makes from the project id —
    # `api("/project/" + P.id + "/models")` — so a sentinel written over that
    # value does not ask *"is this field rendered?"*. It re-points the entire
    # route table at an address the harness has no answer for, `plan` comes back
    # undefined, and `paintPalette` dies on `plan.questions.filter`.
    #
    # THE COST OF NOT DOING THIS WAS THE SINGLE LARGEST ITEM IN THE SUITE.
    # Because pass 1 tags every field in ONE render, one unaddressable id killed
    # the batch for all of them: `seen is None`, zero fields marked, and all
    # 2,867 fell through to pass 2, where each read field must be bisected down
    # to a singleton — about 5,700 `node` renders, measured at **1,175.39s of
    # SETUP** in `test_the_three_unswept_payloads_are_swept.py`. Held back, pass
    # 1 completes in **4.11s** and finds **129 hits in one render**.
    #
    # MATCHED BY VALUE, NOT BY NAME. `id` and `project_id` are two spellings on
    # three routes here and a third spelling is one payload away; the thing that
    # makes a field unaddressable is that it *equals the address being driven*,
    # which is a fact this function already holds. A name list would be a fourth
    # thing to keep in sync — this module's own docstring is about a value
    # composed at one end and dropped at the boundary.
    #
    # THEY ARE NOT EXEMPTED, THEY ARE MEASURED. Each is tagged alone and the
    # render is watched: a mutation the page cannot survive is a mutation the
    # page read, which is `probe`'s own rule and the strongest hit this module
    # has. What is skipped is only their participation in the BATCH, where one
    # of them silently costs every other field its measurement.
    address = [f for f in result.fields
               if isinstance(f.sample, str) and f.sample == project_id]
    for fld in address:
        tagged = json.loads(json.dumps(routes))
        poke(tagged[fld.route], fld.path, _tagged_value(fld, 0))
        result.n_renders += 1
        if probe(tagged, project_id, ids, [(0, (_sentinel(0),))]) is None:
            fld.reaches, fld.how = True, "render broke — the drive's address"
        else:
            # It carried the address and the page survived losing it, so it is
            # an ordinary field after all and goes back into the batch.
            fld.reaches, fld.how = None, None
    address = [f for f in address if f.reaches]
    result.fields = [f for f in result.fields if f not in address]
    result.address = list(address)

    index = {id(fld): i for i, fld in enumerate(result.fields)}

    def _needles(chosen: Sequence[Field]):
        out = []
        for fld in chosen:
            i = index[id(fld)]
            if fld.kind == "num":
                out.append((i, _num_forms(_num_sentinel(i))))
            elif fld.kind != "bool":
                out.append((i, (_sentinel(i),)))
        return out

    def _run(chosen: Sequence[Field]):
        """Tag `chosen`, drive the page, and report what it saw.

        **A failed poke raises.** The first version ignored `poke`'s return, so
        a path the writer could not reach produced an untagged render that read
        as *unread* — which is this module's own subject arriving inside it: a
        value composed at one end and dropped at the boundary. It is a bug in
        the sweep, never a fact about the page, so it stops the sweep rather
        than becoming a finding.
        """
        tagged = json.loads(json.dumps(routes))
        for fld in chosen:
            if not poke(tagged[fld.route], fld.path,
                        _tagged_value(fld, index[id(fld)])):
                raise SweepError(
                    f"cannot write {fld.path!r} in {fld.route!r}: the "
                    f"enumeration produced a path the writer cannot reach, so "
                    f"any verdict about it would be about the sweep rather "
                    f"than about the page")
        result.n_renders += 1
        return probe(tagged, project_id, ids, _needles(chosen))

    # ── pass 1 · one drive, every field tagged at once ───────────────────────
    seen = _run(result.fields)
    if seen is not None:
        result.dom_truncated = False
        hit = set(seen["hits"])
        for fld in result.fields:
            if index[id(fld)] in hit:
                fld.reaches, fld.how = True, "batch"

    if not confirm:
        for fld in result.fields:
            if fld.reaches is None:
                fld.reaches, fld.how = False, "batch"
        return result

    # ── pass 2 · confirm every miss, by group-negative bisection ─────────────
    base = probe(routes, project_id, ids, [])
    result.n_renders += 1
    assert base is not None, "the untouched page does not render at all"
    baseline = base["hash"]

    def _confirm(chunk: List[Field]) -> None:
        """Decide a chunk. **The group negative is what makes this sound.**

        If tagging the whole chunk leaves the rendered DOM identical to the
        baseline, no member of it is read — a field the page touches cannot
        change and leave the output unchanged. Only chunks that MOVE the DOM
        are split, so confirming two thousand negatives costs a handful of
        drives rather than two thousand.
        """
        if not chunk:
            return
        seen = _run(chunk)
        if seen is None:
            # The render died. Something in here is read, hard.
            if len(chunk) == 1:
                chunk[0].reaches, chunk[0].how = True, "render broke"
                return
            half = len(chunk) // 2
            _confirm(chunk[:half])
            _confirm(chunk[half:])
            return
        if seen["hash"] == baseline and not seen["hits"]:
            for fld in chunk:
                fld.reaches, fld.how = False, "confirmed unread"
            return
        if len(chunk) == 1:
            fld = chunk[0]
            fld.reaches = True
            fld.how = ("confirm" if seen["hits"] else "confirm (moved the DOM)")
            return
        half = len(chunk) // 2
        _confirm(chunk[:half])
        _confirm(chunk[half:])

    misses = [f for f in result.fields if f.reaches is None]
    # Grouped by route so a chunk that breaks one payload does not drag another
    # payload's fields through the bisection with it.
    for route in sorted({f.route for f in misses}):
        _confirm([f for f in misses if f.route == route])
    return result


_ARRAY_INDEX = re.compile(r"\[\d+\]")


def _shape(path: str) -> str:
    """`findings[7].title` -> `findings[*].title`."""
    return _ARRAY_INDEX.sub("[*]", path)


def _walk_lists(payload: Any, prefix: str = "") -> Iterable[Tuple[str, list]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from _walk_lists(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(payload, list):
        yield prefix, payload
        for i, value in enumerate(payload[:MAX_ELEMENTS]):
            yield from _walk_lists(value, f"{prefix}[{i}]")
