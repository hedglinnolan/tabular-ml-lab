# Fonts served by the app

The UI's two typefaces live here so the app makes no request to Google Fonts.
`utils/theme.py` declares them with `@font-face`, and Streamlit serves this
folder at `app/static/` (`server.enableStaticServing` in `.streamlit/config.toml`).
Before this, the theme carried an `@import` from `fonts.googleapis.com` — the
one outbound request an "offline" session would have made.

| File | Family | Weights | Subset | Source |
|------|--------|---------|--------|--------|
| `inter-latin.woff2`, `inter-latin-ext.woff2` | Inter (variable) | 300–900 | latin, latin-ext | Google Fonts, Inter v20 |
| `jetbrains-mono-latin.woff2`, `jetbrains-mono-latin-ext.woff2` | JetBrains Mono (variable) | 400–600 | latin, latin-ext | Google Fonts, JetBrains Mono v24 |

Both are licensed under the SIL Open Font License 1.1: `OFL-Inter.txt` and
`OFL-JetBrainsMono.txt` are the licenses as shipped by the two projects.

`tests/test_the_fonts_ship_with_the_app.py` checks that every file the theme
references exists here, that each is a real WOFF2, and that no page imports a
font from the web.
