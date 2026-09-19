"""The app's fonts ship with the app.

`utils/theme.py` carried an `@import` from fonts.googleapis.com on every page,
so a session the README called offline made one outbound request per page load
and an IT allowlist had to carry two Google hostnames for a cosmetic reason.
The two typefaces now live in `static/fonts` under the SIL OFL and Streamlit
serves them from `app/static/`. This holds the three parts together: no web
font import, every referenced file present and a real WOFF2, and static serving
switched on — because a font-face that points at a route Streamlit is not
serving falls back silently, which is the failure this file exists to make loud.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
THEME = ROOT / "utils" / "theme.py"
CONFIG = ROOT / ".streamlit" / "config.toml"


def test_no_page_imports_a_font_from_the_web():
    offenders = []
    for path in [THEME, ROOT / "app.py", *(ROOT / "pages").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if "fonts.googleapis.com" in text or "fonts.gstatic.com" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"web font imports in {offenders}"


def _referenced_font_files():
    return re.findall(r"url\('app/static/fonts/([^']+)'\)", THEME.read_text(encoding="utf-8"))


def test_the_theme_declares_both_families_from_static():
    files = _referenced_font_files()
    assert files, "theme.py declares no @font-face from app/static/fonts"
    text = THEME.read_text(encoding="utf-8")
    assert "font-family: 'Inter'" in text
    assert "font-family: 'JetBrains Mono'" in text


def test_every_referenced_font_file_exists_and_is_woff2():
    missing, not_woff2 = [], []
    for name in _referenced_font_files():
        path = ROOT / "static" / "fonts" / name
        if not path.exists():
            missing.append(name)
        elif path.read_bytes()[:4] != b"wOF2":
            not_woff2.append(name)
    assert not missing, f"theme.py references fonts that are not in static/fonts: {missing}"
    assert not not_woff2, f"not WOFF2 files: {not_woff2}"


def test_static_serving_is_switched_on():
    """Without it Streamlit answers app/static/... with a 404 and the browser
    silently falls back to a system font — which is the same picture as a
    working font to anyone who does not open the network tab."""
    text = CONFIG.read_text(encoding="utf-8")
    assert re.search(r"^enableStaticServing\s*=\s*true", text, re.MULTILINE), (
        ".streamlit/config.toml must set server.enableStaticServing = true")


def test_the_licenses_travel_with_the_files():
    for name in ("OFL-Inter.txt", "OFL-JetBrainsMono.txt"):
        path = ROOT / "static" / "fonts" / name
        assert path.exists(), f"{name} missing"
        assert "SIL Open Font License" in path.read_text(encoding="utf-8")
