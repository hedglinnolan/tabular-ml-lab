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
El.prototype.setAttribute = function(k, v){ this._attr[k] = String(v); };
El.prototype.getAttribute = function(k){ return k in this._attr ? this._attr[k] : null; };
El.prototype.hasAttribute = function(k){ return k in this._attr; };
El.prototype.removeAttribute = function(k){ delete this._attr[k]; };
El.prototype.appendChild = function(c){ this._children.push(c); return c; };
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
El.prototype.getBoundingClientRect = function(){
  return {top:0, left:0, right:0, bottom:0, width:0, height:0};
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

var document = {
  documentElement: new El("html"),
  body: new El("body"),
  getElementById: function(id){
    if (!__byId[id]) __byId[id] = new El("div", id);
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
  var text = JSON.stringify(payload === null ? {} : payload);
  return Promise.resolve({
    ok: true, status: 200, statusText: "OK",
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

    `routes` answers `fetch` — keyed by ``"POST /path"`` or by bare path.
    `search` is `window.location.search`, which is how the page bootstraps a
    project without a test seam: `?project=<id>` is a path the controller already
    has, so the harness uses the page's own code rather than reaching inside it.

    `body` runs after a microtask drain, so promises the bootstrap started have
    settled. It reports its result with ``__emit(value)``.
    """
    script = page_script()
    shim = (_SHIM
            .replace("__ROUTES__", json.dumps(routes or {}))
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
