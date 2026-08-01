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
  this.textContent = "";
  this.innerHTML = "";
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
El.prototype.setAttribute = function(k, v){ this._attr[k] = String(v); };
El.prototype.getAttribute = function(k){ return k in this._attr ? this._attr[k] : null; };
El.prototype.hasAttribute = function(k){ return k in this._attr; };
El.prototype.removeAttribute = function(k){ delete this._attr[k]; };
El.prototype.appendChild = function(c){ this._children.push(c); c._parent = this; return c; };
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
  var inner = this.innerHTML || "";
  for (var i = 0; i < this._children.length; i++) inner += this._children[i].__deep();
  return "<" + this.tagName.toLowerCase() + attrs + ">" + inner +
         "</" + this.tagName.toLowerCase() + ">";
};
El.prototype.removeChild = function(c){
  var i = this._children.indexOf(c); if (i !== -1) this._children.splice(i, 1); return c;
};
El.prototype.addEventListener = function(t, fn){ (this._listeners[t] = this._listeners[t] || []).push(fn); };
El.prototype.removeEventListener = function(){};
El.prototype.querySelector = function(){ return null; };
El.prototype.querySelectorAll = function(){ return []; };
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
    var ok = true;
    var rx = /\[([^\]=]+)(?:=["']?([^\]"']*)["']?)?\]|\.([A-Za-z0-9_-]+)|^([a-zA-Z]+)/g;
    var m, seen = 0, rest = tok;
    while ((m = rx.exec(tok)) !== null){
      seen++;
      rest = rest.replace(m[0], "");
      if (m[1] !== undefined){
        if (!el.hasAttribute(m[1])) { ok = false; break; }
        if (m[2] !== undefined && m[2] !== "" && el.getAttribute(m[1]) !== m[2]) { ok = false; break; }
      } else if (m[3] !== undefined){
        if (!el.classList.contains(m[3])) { ok = false; break; }
      } else if (m[4] !== undefined){
        if (el.tagName !== m[4].toUpperCase()) { ok = false; break; }
      }
    }
    if (!seen) throw new Error("harness: selector token not understood: " + tok);
    if (rest.trim()) throw new Error("harness: selector token not understood: " + tok);
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

var document = {
  documentElement: new El("html"),
  body: new El("body"),
  getElementById: function(id){
    if (!__byId[id]){
      var el = new El("div", id);
      (__seed[id] || []).forEach(function(c){ el.classList.add(c); });
      __byId[id] = el;
    }
    return __byId[id];
  },
  createElement: function(t){ return new El(t); },
  querySelector: function(sel){ return null; },
  querySelectorAll: function(sel){ return []; },
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
  html: function(id){ return document.getElementById(id).innerHTML; },
  // The deep read, for a surface built by appending rather than by assigning.
  render: function(id){
    var el = document.getElementById(id);
    if (!el) return "";
    var out = el.innerHTML || "";
    for (var i = 0; i < el.children.length; i++) out += el.children[i].__deep();
    return out;
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
globalThis.__emit = function(v){ __emitted = v; };
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
