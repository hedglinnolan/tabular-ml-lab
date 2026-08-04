"""Run the Guided page's own controller, for real, under node.

## Why this exists

`test_the_page_asks_what_the_router_serves.py` says its own limit out loud:

> What it still cannot prove is that a card is VISIBLE; that needs a browser,
> and the honest form of that is the driver.

That honesty was correct and it was also the gap `DRIVE-001` fell through twice.
The second fall is `GUIDED-037`: the Router served `option_values` beside
`options` because the lens's labels are prose and its values are keys, the page
never read the second array, and every generic-channel answer posted a label the
server refuses. The test that was supposed to guard it read

    assert "option_values" in page or "q.options" in page

— a disjunction satisfied by the wrong half, which is `FEATURE_PARITY.md`'s
*a substring of a message is a wildcard wearing an assertion's clothes* arriving
in a new costume.

A text search over `index.html` cannot tell a page that reads a field from a page
that merely mentions it. **Running the code can.** This module loads the page's
real script under a DOM shim, drives its real click handlers, and reports what
its real `fetch` would have sent. The value that comes back is then replayed
against the real API by the caller, so the assertion is *the record changed*
rather than *the page contains a string*.

## What it proves, and what it still does not

It proves **behavior**: what the controller renders, what a click does to that
render, and the exact HTTP body a press produces. That is the whole of
`GUIDED-037` and most of the `DRIVE` interaction backlog.

It does **not** prove **visibility** — that a card is on screen, unclipped,
above the fold, in a section that is not `is-hidden`. Nothing without layout can,
and claiming otherwise here would be this project's own governing failure. The
shim's elements are truthful about what they were told and know nothing about
pixels. The driver remains the check for visibility.

## The shim is deliberately dumb

Every element is the same object and remembers only what was set on it. That is
enough for a controller which writes `innerHTML` and reads attributes, and it
fails loudly rather than silently: an uncaught throw exits node non-zero and the
test reports the stack. A shim that guessed would be a second implementation of
the DOM to keep in sync, which is the two-engines failure one level down.

## Dumb is allowed. Lying is not — `GUIDED-077`

Being ignorant of pixels is a stated limit a reader can work with. **Answering a
question wrongly is not**, because the answer is what a test then asserts, and a
test that asserts a lie reports the page as honest exactly where it has stopped
being honest. This module got that wrong three times, in the same shape each
time: it accepted a write, or invented a value, and no other method agreed.

* `className` was an ordinary property nothing else read, so every assertion
  about how a node is styled came back **passing** (`GUIDED-081`).
* `getElementById` AUTO-CREATED, so it never returned null and `if (!node)` was
  false for every id in the universe. The lattice paid for it: its
  `if (!$("latGrid"))` guard could never be true, the container was never
  written, the grid was built and never attached — `latticeBox` held 0
  characters while `latGrid` held 179 — and the claim about it passed.
* `innerHTML` returned only what had been ASSIGNED, so a surface built by
  `appendChild` probed as an empty container. The workaround was a second
  reader, `__harness.render`, and two readers of one property is two answers to
  one question.

All three are closed. `render` is now an alias for `html` rather than a
different question, and `test_the_shim_says_no_to_an_id_that_does_not_exist`
and its sibling assert the properties rather than the absences.

**The one place it still approximates, stated here rather than found later.**
An id that arrives inside assigned markup — `parent.innerHTML = '<div id="x">'` —
becomes findable, because a browser would make it so and the page relies on it.
The shim does not parse HTML, so content written into such a node does not
appear in the PARENT's serialization: read the id itself and you see it; read
its parent and you see the markup as assigned. Reassigning the parent
un-declares it, so a repaint is still observable. A claim that needs to know a
surface is attached to the page should read the parent — which is exactly the
distinction the lattice claim could not draw before.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PAGE = Path(__file__).resolve().parent / "web" / "index.html"

_SENTINEL = "@@TURBOTAB_HARNESS@@"


def available() -> bool:
    """Whether this machine can run the harness at all.

    A check that cannot run guards nothing, so the callers skip rather than pass
    — `test_engine_is_headless` is the precedent.
    """
    return shutil.which("node") is not None


def page_script() -> str:
    text = PAGE.read_text(encoding="utf-8")
    return text[text.index("<script>") + len("<script>"):text.rindex("</script>")]


_ID_TAG = re.compile(r"<[a-zA-Z][^>]*\bid=\"([^\"]+)\"[^>]*>")
_CLASS_IN = re.compile(r'\bclass="([^"]*)"')


def seed_classes() -> Dict[str, List[str]]:
    """The classes each `id` carries in the markup, so elements start as
    declared.

    Not a nicety. `reveal()` returns early unless its section is `is-hidden`,
    and with classless elements that branch never ran — so the harness reported
    the whole reveal path as dead on every drive, and a textually restored
    `DRIVE-006` nudge probed GREEN. An element that lies about its starting
    state makes every branch keyed on that state unobservable, which is a
    harness asserting something false about the code it instruments.
    """
    text = PAGE.read_text(encoding="utf-8")
    body = text[text.rindex("</style>"):text.index("<script>")]
    out: Dict[str, List[str]] = {}
    for m in _ID_TAG.finditer(body):
        cls = _CLASS_IN.search(m.group(0))
        out[m.group(1)] = cls.group(1).split() if cls else []
    return out


_OPEN_TAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>")
_ATTR_PAIR = re.compile(r'([a-zA-Z][a-zA-Z0-9_:-]*)="([^"]*)"')


def body_elements() -> List[Dict[str, Any]]:
    """Every element the static body declares, as tag, attributes and classes.

    **`TEST-048`.** The shim's `querySelector` returned `null` unconditionally,
    which is not ignorance — it is an answer, and it was the wrong one for every
    element the document declares. `setMap` addressed its eight dots with
    `document.querySelector('.map-step[data-map=…]')`, so **no drive could ever
    observe which step the app claimed the user was on**, and a six-way
    simultaneous `now` survived every drive ever run against this page.

    `seed_classes` already reads this markup for id → classes. This reads the
    same markup one field wider, so a selector over the static body has
    something true to match against instead of a blanket denial.

    Deliberately flat and deliberately not a parser: no nesting, no text, no
    self-closing subtleties. It answers *does the document declare an element
    with these attributes*, which is the only question `querySelector` is asked
    on this page.
    """
    text = PAGE.read_text(encoding="utf-8")
    body = text[text.rindex("</style>"):text.index("<script>")]
    out: List[Dict[str, Any]] = []
    for m in _OPEN_TAG.finditer(body):
        attrs = {k: v for k, v in _ATTR_PAIR.findall(m.group(2))}
        if not attrs:
            continue
        classes = attrs.pop("class", "").split()
        out.append({"tag": m.group(1), "attrs": attrs, "classes": classes})
    return out


_SHIM = r"""
'use strict';
// ── the smallest DOM the Guided controller needs ────────────────────────────
// Truthful about what it was told, ignorant of pixels. See the module docstring
// in turbotab/pageharness.py for what that buys and what it costs.

function El(tag, id){
  this.tagName = (tag || "div").toUpperCase();
  this.id = id || "";
  this._attr = Object.create(null);
  this._classes = Object.create(null);
  this._children = [];
  this._listeners = Object.create(null);
  this.style = new Proxy({}, {get: function(t,k){ return t[k] === undefined ? "" : t[k]; },
                             set: function(t,k,v){ t[k] = v; return true; }});
  // The markup this element was ASSIGNED, kept separate from the children it
  // was APPENDED. `innerHTML` is the two together — see the accessor below.
  this._html = "";
  this._markupIds = [];
  // The elements the assigned markup DECLARES, as flat nodes carrying their
  // own attributes and classes. `TEST-048`: without these, every selector over
  // a rendered surface had nothing true to match and the shim said `null`.
  this._declared = [];
  this.value = "";
  this.title = "";
  this.disabled = false;
  this.inert = false;
  var self = this;
  this.classList = {
    add: function(){ for (var i=0;i<arguments.length;i++) self._classes[arguments[i]] = true; },
    remove: function(){ for (var i=0;i<arguments.length;i++) delete self._classes[arguments[i]]; },
    contains: function(c){ return !!self._classes[c]; },
    toggle: function(c, on){
      if (on === undefined) on = !self._classes[c];
      if (on) self._classes[c] = true; else delete self._classes[c];
      return on;
    }
  };
  this.dataset = new Proxy(Object.create(null), {
    get: function(_, k){
      return self._attr["data-" + String(k).replace(/[A-Z]/g, function(m){ return "-" + m.toLowerCase(); })];
    },
    set: function(_, k, v){
      self._attr["data-" + String(k).replace(/[A-Z]/g, function(m){ return "-" + m.toLowerCase(); })] = v;
      return true;
    }
  });
}
// `innerHTML` IS THE ASSIGNED MARKUP PLUS THE APPENDED CHILDREN.
//
// `GUIDED-077`, second half. It used to return only what was assigned, so a
// surface built by `appendChild` — which is every mutate-in-place renderer §05
// requires — probed as an EMPTY CONTAINER. The workaround was a second reader,
// `__harness.render`, that walked the children; two readers of one property is
// two answers to one question, and a claim written against the wrong one
// asserts nothing.
//
// Setting it replaces the children, as it does in a browser. That is what makes
// a rebuild observable: the nodes are gone, and a renderer that rebuilds can no
// longer pass a test written for one that mutates.
Object.defineProperty(El.prototype, "innerHTML", {
  get: function(){
    var out = this._html;
    for (var i = 0; i < this._children.length; i++) out += this._children[i].__deep();
    return out;
  },
  set: function(v){
    this._html = v === null || v === undefined ? "" : String(v);
    // The children are GONE, so their ids are gone with them. Leaving them
    // findable would make a repaint look like a mutation, which is the exact
    // distinction §05 turns on.
    for (var i = 0; i < this._children.length; i++) __unregister(this._children[i]);
    this._children = [];
    for (var j = 0; j < this._markupIds.length; j++){
      var was = __byId[this._markupIds[j]];
      if (was && was._fromMarkup === this) delete __byId[this._markupIds[j]];
    }
    this._markupIds = __declareMarkupIds(this, this._html);
    // Reassigning un-declares the previous markup's nodes, for the same reason
    // it un-registers its ids: a repaint has to be observable as a repaint.
    this._declared = __declareMarkupNodes(this, this._html);
  }
});

// `textContent`, same contract one level down: assigning it replaces the
// content with text, reading it returns the text with the tags taken out. The
// old version was a plain field, so `node.textContent = "x"` was invisible to
// every other reader — the same defect as `className` before `GUIDED-081`.
Object.defineProperty(El.prototype, "textContent", {
  get: function(){
    return this.innerHTML.replace(/<[^>]*>/g, "");
  },
  set: function(v){
    this.innerHTML = String(v === null || v === undefined ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
});

// `className` IS `_classes`, in both directions.
//
// Without this it was an ordinary JS property: assigning it set something no
// other method here read, so `classList.contains` said no, `__deep` printed no
// class, and every assertion about how a node is styled came back vacuously
// true. A shim that accepts a write and then denies it happened is the one
// thing this file is not allowed to be — it would report the page as honest
// about styling precisely where the page had stopped being honest. Filed as
// `GUIDED-081`; the mutate-in-place renderers §05 requires all set class this
// way, so the gap grew with every one of them.
Object.defineProperty(El.prototype, "className", {
  get: function(){ return Object.keys(this._classes).join(" "); },
  set: function(v){
    this._classes = Object.create(null);
    String(v).split(/\s+/).forEach(function(c){ if (c) this._classes[c] = true; },
                                   this);
  }
});
// `firstChild` was undefined, so `if (!box.firstChild)` — the guard the figure
// surface uses to write its header once — was TRUE on every render, and the
// header was rewritten every time. That was invisible while assigning
// `innerHTML` did not destroy children; now it destroys them, so the guard has
// to be able to answer.
Object.defineProperty(El.prototype, "firstChild", {
  get: function(){
    if (this._children.length) return this._children[0];
    return this._html ? this._html : null;
  }
});
El.prototype.setAttribute = function(k, v){ this._attr[k] = String(v); };
El.prototype.getAttribute = function(k){ return k in this._attr ? this._attr[k] : null; };
El.prototype.hasAttribute = function(k){ return k in this._attr; };
El.prototype.removeAttribute = function(k){ delete this._attr[k]; };
// APPENDING REGISTERS IDS, which is what makes `getElementById` able to say no.
// A node created with an id and put into the tree is findable in a browser; one
// that was never created is not. The shim could not tell those apart while it
// auto-created on lookup, so it answered *is there an element with this id* the
// same way for both.
// AN ID THAT ARRIVES INSIDE ASSIGNED MARKUP.
//
// In a browser `parent.innerHTML = '<div id="x"></div>'` creates a real node
// that `getElementById` finds. This shim does not parse HTML — the module
// docstring says why, and a second DOM implementation to keep in sync is the
// two-engines failure one level down — so it does the smallest true thing: it
// notes which ids the markup declares and answers for them.
//
// THE LIMIT, stated rather than discovered later: content written into such a
// node does NOT appear in the parent's serialization, because the shim does not
// know where in the string it belongs. `__harness.html('<that id>')` reads it;
// `__harness.html('<its parent>')` shows the markup as assigned. Reassigning
// the parent's markup un-declares them, so a repaint stays observable — which
// is the property `GUIDED-077` was really about.
var __MARKUP_ID = /\bid="([^"]+)"/g;
function __declareMarkupIds(parent, html){
  var ids = [], m;
  __MARKUP_ID.lastIndex = 0;
  while ((m = __MARKUP_ID.exec(html)) !== null){
    var id = m[1];
    ids.push(id);
    if (__byId[id]) continue;
    var el = new El("div", id);
    (__seed[id] || []).forEach(function(c){ el.classList.add(c); });
    el._fromMarkup = parent;
    __byId[id] = el;
  }
  return ids;
}

// THE SAME SMALLEST TRUE THING, one field wider. `TEST-048`.
//
// `__declareMarkupIds` noted which ids the markup declares so `getElementById`
// could answer for them. Everything else in that markup — the `data-rg-body`,
// the `data-offer-pv`, the `data-plaus-reason` input — was invisible, so
// `querySelector` had nothing to find and said `null`, which is a claim.
//
// A declared node carries its own tag, attributes and classes and NOT its
// position. It is addressable and writable; `__harness.html('<its id>')` reads
// what was written into it, and the parent's serialization does not show it,
// which is the limit `__declareMarkupIds` already states.
var __OPEN_TAG = /<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>/g;
var __ATTR_PAIR = /([a-zA-Z][a-zA-Z0-9_:-]*)="([^"]*)"/g;
function __declareMarkupNodes(parent, html){
  var out = [], m;
  __OPEN_TAG.lastIndex = 0;
  while ((m = __OPEN_TAG.exec(html)) !== null){
    var attrs = {}, a, any = false;
    __ATTR_PAIR.lastIndex = 0;
    while ((a = __ATTR_PAIR.exec(m[2])) !== null){ attrs[a[1]] = a[2]; any = true; }
    if (!any) continue;
    var id = attrs.id || "";
    // An id already declared has a node; reuse it so one element is one object
    // and a write through `getElementById` is visible to `querySelector`.
    var el = (id && __byId[id]) ? __byId[id] : new El(m[1], id);
    el.tagName = m[1].toUpperCase();
    Object.keys(attrs).forEach(function(k){
      if (k === "class") String(attrs[k]).split(/\s+/).forEach(function(c){
        if (c) el.classList.add(c); });
      else if (k !== "id") el.setAttribute(k, attrs[k]);
    });
    if (attrs.disabled !== undefined) el.disabled = true;
    el._fromMarkup = parent;
    out.push(el);
  }
  return out;
}

function __unregister(el){
  el._parent = null;
  if (el.id && __byId[el.id] === el) delete __byId[el.id];
  for (var i = 0; i < el._children.length; i++) __unregister(el._children[i]);
}
function __register(el){
  if (el.id && !__byId[el.id]) __byId[el.id] = el;
  for (var i = 0; i < el._children.length; i++) __register(el._children[i]);
}
El.prototype.appendChild = function(c){
  this._children.push(c); c._parent = this; __register(c); return c;
};
// A RENDERER THAT MUTATES IN PLACE APPENDS NODES, and `innerHTML` here is what
// was ASSIGNED rather than a serialization of children — so a surface built by
// `appendChild` was invisible to `__harness.html()` and probed as an empty
// container. `DESIGN_LANGUAGE.md` §05 now REQUIRES mutate-in-place, so the
// harness has to be able to see one. Deliberately shallow: tag, id, class and
// the assigned innerHTML, recursively. It is not an HTML serializer and does
// not try to be one — it reports what the element was told, which is the same
// contract every other method here keeps.
El.prototype.__deep = function(){
  var attrs = "";
  var self = this;
  Object.keys(this._attr).forEach(function(k){
    attrs += " " + k + '="' + String(self._attr[k]) + '"';
  });
  var cls = Object.keys(this._classes).join(" ");
  if (cls) attrs += ' class="' + cls + '"';
  if (this.id) attrs = ' id="' + this.id + '"' + attrs;
  return "<" + this.tagName.toLowerCase() + attrs + ">" + this.innerHTML +
         "</" + this.tagName.toLowerCase() + ">";
};
El.prototype.removeChild = function(c){
  var i = this._children.indexOf(c);
  if (i !== -1){
    this._children.splice(i, 1);
    // Out of the tree is out of `getElementById`, or a removed node stays
    // findable and "did this leave?" is unobservable in the other direction.
    __unregister(c);
  }
  return c;
};
El.prototype.addEventListener = function(t, fn){ (this._listeners[t] = this._listeners[t] || []).push(fn); };
El.prototype.removeEventListener = function(){};
// ── querySelector, which used to be a denial ────────────────────────────────
//
// `TEST-048`. This returned `null` unconditionally and `querySelectorAll`
// returned `[]`. That is not the shim being dumb — the module docstring allows
// dumb — it is the shim ANSWERING, and answering *there is no such element*
// about elements that are in `index.html`. Every negative assertion that ever
// went through it was vacuous, and the one that mattered is `setMap`: it
// addressed its eight dots with `document.querySelector('.map-step[data-map=…]')`
// and was therefore a total no-op under every drive this project has run, which
// is why six steps could wear `now` at once for the whole life of the analysis
// map without a single test noticing.
//
// It answers over three populations now, which together are everything the
// controller can address:
//
//   1 · REAL CHILDREN, appended through `createElement` + `appendChild`.
//   2 · ELEMENTS DECLARED IN ASSIGNED MARKUP. `innerHTML = '<button data-x=…>'`
//       creates real nodes in a browser; this shim does not parse HTML, so it
//       does the same smallest-true-thing `__declareMarkupIds` already did for
//       ids and notes the elements with their attributes. Reassigning the
//       parent's markup un-declares them, so a repaint stays observable.
//   3 · THE STATIC BODY, from `body_elements()`. Nothing was modeling it, and
//       it is where the analysis map lives.
//
// THE LIMIT, said here rather than found later: a declared node is flat. It
// carries its own attributes and classes and it does not know its position, so
// descendant selectors (`a b`) and `:nth-child` are not answerable — `matches`
// throws on those already, and throwing is the right answer where `null` was
// the wrong one.
El.prototype.__all = function(out){
  out = out || [];
  for (var i = 0; i < this._children.length; i++){
    out.push(this._children[i]);
    this._children[i].__all(out);
  }
  for (var j = 0; j < this._declared.length; j++){
    out.push(this._declared[j]);
    this._declared[j].__all(out);
  }
  return out;
};
El.prototype.querySelector = function(sel){
  var all = this.__all();
  for (var i = 0; i < all.length; i++) if (matches(all[i], sel)) return all[i];
  return null;
};
El.prototype.querySelectorAll = function(sel){
  return this.__all().filter(function(el){ return matches(el, sel); });
};
El.prototype.closest = function(sel){ return matches(this, sel) ? this : null; };
El.prototype.focus = function(){};
El.prototype.click = function(){};
// DELIBERATELY JUST BELOW THE FOLD, and this is the one place the shim takes a
// position rather than reporting what it was told.
//
// A rect of zeros is not "no layout", it is a CLAIM that the element is at the
// top of the viewport — and code guarded by "is this below the fold?" then never
// runs, so a harness returning zeros reports every scroll-on-reveal path as
// dead. That is how the DRIVE-006 probe first came back GREEN with the nudge
// textually restored: the defect was present, unreachable, and therefore
// invisible, which reads exactly like a fix.
//
// So the default errs toward MAKING the guarded path run. A viewport-moving
// code path is then always observable: this shim can report a scroll that a
// real browser might not have made, and it cannot miss one that it would. For a
// guard that is the correct direction to be wrong in.
El.prototype.getBoundingClientRect = function(){
  var top = this._rect !== undefined ? this._rect : (globalThis.innerHeight || 900) + 40;
  return {top: top, left: 0, right: 0, bottom: top, width: 0, height: 0};
};
Object.defineProperty(El.prototype, "children", {get: function(){ return this._children; }});
Object.defineProperty(El.prototype, "lastChild", {
  get: function(){ return this._children[this._children.length - 1] || null; }});

// Selector support is exactly what the controller's `closest()` calls need:
// a comma-separated list of `[attr]` and `.class` and `tag` tokens. Anything
// richer throws rather than quietly matching nothing, because a selector this
// cannot read is a handler this harness would report as dead when it is not.
function matches(el, sel){
  var parts = String(sel).split(",");
  for (var i = 0; i < parts.length; i++){
    var tok = parts[i].trim();
    if (!tok) continue;
    // PARSED IN FULL BEFORE ANYTHING IS EVALUATED, and that ordering is the
    // whole of a second `TEST-048` defect.
    //
    // This used to evaluate as it parsed and `break` on the first token that
    // did not match — leaving the rest of the token UNCONSUMED, so the
    // `rest.trim()` check below then threw *"selector token not understood"*
    // for a selector it understands perfectly. It fired only on MULTI-TOKEN
    // selectors against NON-MATCHING elements, which is every element but one
    // in any real search, so it was invisible while `querySelector` returned
    // `null` without ever calling this. Repairing `querySelector` alone would
    // have turned a silent wrong answer into a crash — the same blind spot,
    // one layer down.
    // A COMBINATOR IS NOT ANSWERABLE HERE, so it is refused rather than
    // silently reinterpreted. `.map-step .md` is a DESCENDANT selector, and the
    // token parser below would have read it as *an element with both classes*
    // — a different question, answered confidently. A declared node is flat and
    // does not know its position; `null` or `false` there would be the same lie
    // `querySelector` was just repaired for, one shape over.
    if (/[\s>+~]/.test(tok.replace(/\[[^\]]*\]/g, ""))){
      throw new Error("harness: selector token not understood (combinators " +
                      "need position and these nodes are flat): " + tok);
    }
    var rx = /\[([^\]=]+)(?:=["']?([^\]"']*)["']?)?\]|\.([A-Za-z0-9_-]+)|^([a-zA-Z]+)/g;
    var m, parsed = [], rest = tok;
    while ((m = rx.exec(tok)) !== null){
      parsed.push(m);
      rest = rest.replace(m[0], "");
    }
    if (!parsed.length) throw new Error("harness: selector token not understood: " + tok);
    if (rest.trim()) throw new Error("harness: selector token not understood: " + tok);
    var ok = true;
    for (var k = 0; k < parsed.length && ok; k++){
      var p = parsed[k];
      if (p[1] !== undefined){
        if (!el.hasAttribute(p[1])) ok = false;
        else if (p[2] !== undefined && p[2] !== "" && el.getAttribute(p[1]) !== p[2]) ok = false;
      } else if (p[3] !== undefined){
        if (!el.classList.contains(p[3])) ok = false;
      } else if (p[4] !== undefined){
        if (el.tagName !== p[4].toUpperCase()) ok = false;
      }
    }
    if (ok) return true;
  }
  return false;
}

var __byId = Object.create(null);
var __docListeners = Object.create(null);
// The classes each id CARRIES IN THE DOCUMENT, read out of index.html's markup.
//
// Without this every element started classless, so `is-hidden` was never
// present and `reveal()` — which returns early unless the section is hidden —
// never ran its body at all. The harness reported the reveal path as dead code
// on every drive, which is how the DRIVE-006 probe reported a textually
// restored nudge as GREEN. An element that lies about its starting state makes
// every branch keyed on that state unobservable.
var __seed = __SEED__;
// EVERY ELEMENT THE DOCUMENT DECLARES, so a selector over the static body has
// something true to match. `TEST-048`, and the reason `body_elements` exists.
// An element with an id BECOMES the node `getElementById` answers with, so one
// element is one object and a write through either reader is visible to both.
var __static = (__BODY__).map(function(spec){
  var el = new El(spec.tag, spec.attrs.id || "");
  Object.keys(spec.attrs).forEach(function(k){
    if (k !== "id") el.setAttribute(k, spec.attrs[k]);
  });
  spec.classes.forEach(function(c){ el.classList.add(c); });
  if (spec.attrs.disabled !== undefined) el.disabled = true;
  if (el.id && !__byId[el.id]) __byId[el.id] = el;
  return el;
});

var document = {
  documentElement: new El("html"),
  body: new El("body"),
  // `GUIDED-077`, first half. This used to AUTO-CREATE, so it never returned
  // null and `if (!node)` was false for every id in the universe — every branch
  // keyed on *does this node exist yet* was unobservable, and a renderer that
  // asked the DOM whether its own node existed got yes on the first render.
  //
  // The seed table is what makes saying no possible: it holds every id declared
  // in `index.html`'s markup, so an id that is neither declared there nor
  // appended by the page under test is an id that does not exist. Returning an
  // element for it is the harness inventing the thing it was asked about.
  getElementById: function(id){
    if (__byId[id]) return __byId[id];
    if (!(id in __seed)) return null;
    var el = new El("div", id);
    __seed[id].forEach(function(c){ el.classList.add(c); });
    __byId[id] = el;
    return el;
  },
  createElement: function(t){ return new El(t); },
  // OVER THE STATIC BODY AND EVERYTHING RENDERED INTO IT. `TEST-048`.
  //
  // The document's own population is `__static` — every element `index.html`
  // declares, read out of the same markup `seed_classes` reads — plus the
  // subtree of every element the drive has touched. A `null` from here now
  // means the document really does not declare it.
  __all: function(){
    var out = __static.slice(), seen = new Set(__static);
    Object.keys(__byId).forEach(function(id){
      var el = __byId[id];
      if (!seen.has(el)){ out.push(el); seen.add(el); }
      el.__all().forEach(function(c){ if (!seen.has(c)){ out.push(c); seen.add(c); } });
    });
    return out;
  },
  querySelector: function(sel){
    var all = document.__all();
    for (var i = 0; i < all.length; i++) if (matches(all[i], sel)) return all[i];
    return null;
  },
  querySelectorAll: function(sel){
    return document.__all().filter(function(el){ return matches(el, sel); });
  },
  addEventListener: function(t, fn){ (__docListeners[t] = __docListeners[t] || []).push(fn); },
  removeEventListener: function(){}
};

globalThis.document = document;
globalThis.window = globalThis;
globalThis.self = globalThis;
window.matchMedia = function(){ return {matches: false, addEventListener: function(){}}; };
window.addEventListener = function(t, fn){ (__docListeners[t] = __docListeners[t] || []).push(fn); };
window.removeEventListener = function(){};
window.scrollTo = function(o){ __scrolls.push(o || {}); };
window.scrollY = 0;
window.innerHeight = 900;
window.innerWidth = 1400;
window.location = {search: __SEARCH__, reload: function(){}, href: "http://harness/"};
globalThis.requestAnimationFrame = function(fn){ __raf.push(fn); };
globalThis.CSS = {escape: function(s){ return String(s).replace(/["\\]/g, "\\$&"); }};
globalThis.FormData = function(){ this._d = []; };
globalThis.FormData.prototype.append = function(k, v){ this._d.push([k, v]); };
globalThis.Node = El;

var __scrolls = [];
var __raf = [];
var __calls = [];
var __routes = __ROUTES__;

globalThis.fetch = function(path, opts){
  opts = opts || {};
  var method = (opts.method || "GET").toUpperCase();
  var body = null;
  if (typeof opts.body === "string"){ try { body = JSON.parse(opts.body); } catch (_) { body = opts.body; } }
  __calls.push({method: method, path: path, body: body});
  var key = method + " " + path;
  var payload = Object.prototype.hasOwnProperty.call(__routes, key) ? __routes[key]
              : (Object.prototype.hasOwnProperty.call(__routes, path) ? __routes[path] : null);
  // A ROUTE MAY REFUSE. Until `L32` every route answered 200, so the harness
  // could drive every path the page has EXCEPT the one where the server says
  // no — which is the surface `GUIDED-076` is about, and the one an interface
  // is most likely to get wrong. A route declaring `__status` answers with it.
  var status = 200;
  if (payload && typeof payload === "object" && payload.__status){
    status = payload.__status;
    payload = payload.body === undefined ? payload : payload.body;
  }
  var text = JSON.stringify(payload === null ? {} : payload);
  return Promise.resolve({
    ok: status < 400, status: status,
    statusText: status === 200 ? "OK" : "Refused",
    text: function(){ return Promise.resolve(text); },
    json: function(){ return Promise.resolve(JSON.parse(text)); }
  });
};

// A synthetic event target that answers `closest()` the way a real button with
// these attributes would. The handler under test is the page's own.
function target(attrs, classes){
  var el = new El("button");
  Object.keys(attrs || {}).forEach(function(k){ el.setAttribute(k, attrs[k]); });
  (classes || []).forEach(function(c){ el.classList.add(c); });
  return el;
}

function dispatch(type, el){
  var hs = __docListeners[type] || [];
  for (var i = 0; i < hs.length; i++) hs[i]({target: el, preventDefault: function(){},
                                             stopPropagation: function(){}});
}

function drainRaf(){
  var q = __raf.splice(0, __raf.length);
  for (var i = 0; i < q.length; i++) q[i]();
}

globalThis.__harness = {
  el: function(id){ return document.getElementById(id); },
  html: function(id){
    var el = document.getElementById(id);
    return el ? el.innerHTML : null;
  },
  // `render` and `html` ARE THE SAME READ NOW, and that is the point of the
  // `GUIDED-077` fix. `render` existed because `innerHTML` returned only the
  // assigned markup, so a surface built by appending needed a second reader
  // that walked the children — and having two made it possible to write a
  // claim against the one that could not see the thing it was about. It is
  // kept as an alias so the existing claims keep reading, and it no longer
  // answers a different question from `html`.
  render: function(id){
    var el = document.getElementById(id);
    return el ? el.innerHTML : "";
  },
  target: target,
  dispatch: dispatch,
  calls: function(){ return __calls; },
  posts: function(){ return __calls.filter(function(c){ return c.method === "POST"; }); },
  scrolls: function(){ return __scrolls; },
  drainRaf: drainRaf,
  matches: matches
};

var __emitted = null;
// `__emit` ENDS THE RUN, and it has to.
//
// Node exits when its event loop drains, and a page that owns a repeating
// timer never drains one — which is every page that watches a job, so this
// became load-bearing the moment `PRODUCT_VISION.md` §04's progress-and-cancel
// requirement got a surface. Before this, driving the training step hung until
// the 90-second timeout and reported nothing at all.
globalThis.__emit = function(v){
  __emitted = v;
  process.exit(0);
};
process.on("exit", function(){
  process.stdout.write("\n__SENTINEL__" + JSON.stringify(__emitted === undefined ? null : __emitted) + "\n");
});
"""


def run(body: str, *, routes: Optional[Dict[str, Any]] = None,
        search: str = "", timeout: int = 90) -> Any:
    """Load the page's controller, run `body`, and return whatever it emits.

    `routes` answers `fetch` — keyed by ``"POST /path"`` or by bare path. A
    route may REFUSE: give it ``{"__status": 409, "body": {...}}`` and the shim
    answers with that status and `ok: false`, which is how the page's error path
    gets driven at all.
    `search` is `window.location.search`, which is how the page bootstraps a
    project without a test seam: `?project=<id>` is a path the controller already
    has, so the harness uses the page's own code rather than reaching inside it.

    `body` runs after a microtask drain, so promises the bootstrap started have
    settled. It reports its result with ``__emit(value)``.
    """
    script = page_script()
    shim = (_SHIM
            .replace("__ROUTES__", json.dumps(routes or {}))
            .replace("__SEED__", json.dumps(seed_classes()))
            .replace("__BODY__", json.dumps(body_elements()))
            .replace("__SEARCH__", json.dumps(search))
            .replace("__SENTINEL__", _SENTINEL))
    # `setTimeout(..., 0)` chained four deep drains the promise queue the
    # bootstrap's `.then()` chain sits in. Awaiting a fixed number of turns is
    # crude and it is also honest: a body that needs more says so by failing,
    # rather than by passing on a stale render.
    program = (
        shim + "\n" + script + "\n" +
        "(async function(){ for (var i=0;i<8;i++) await new Promise(function(r){ setTimeout(r, 0); });\n"
        + body + "\n})().catch(function(e){ console.error(e && e.stack || String(e)); process.exit(3); });\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(program)
        path = fh.name
    try:
        out = subprocess.run([shutil.which("node"), path], capture_output=True,
                             text=True, timeout=timeout)
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise HarnessError(
            "the page's controller did not run under the harness "
            f"(exit {out.returncode}):\n{(out.stderr or out.stdout)[-2500:]}")
    if _SENTINEL not in out.stdout:
        raise HarnessError(
            "the harness produced no result; the controller may have exited "
            f"early:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    tail = out.stdout[out.stdout.rindex(_SENTINEL) + len(_SENTINEL):]
    return json.loads(tail.strip())


class HarnessError(AssertionError):
    """The page could not be driven — reported, never swallowed."""


# ── reading the render back ──────────────────────────────────────────────────

_TAG = re.compile(r"<(button|div|article|span)\b([^>]*)>", re.I)
_ATTR = re.compile(r'([a-zA-Z-]+)="([^"]*)"')


def elements(html: str, tag: str = "button") -> List[Dict[str, str]]:
    """Every `tag` in a rendered fragment, as its attribute map.

    Structure, not prose: a caller asserts on `data-answer-value`, never on a
    substring of the label (`FEATURE_PARITY.md`, *assert on STRUCTURE*).
    """
    out: List[Dict[str, str]] = []
    for m in _TAG.finditer(html):
        if m.group(1).lower() != tag.lower():
            continue
        out.append({k: v for k, v in _ATTR.findall(m.group(2))})
    return out
