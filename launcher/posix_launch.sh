#!/bin/bash
# Tabular ML Lab — macOS/Linux launcher engine.
#
# Called by "Start Tabular ML Lab.command" (macOS) or directly (Linux):
#   bash launcher/posix_launch.sh [--from-app]
#
# What it does:
#   1. Installs uv into ./.tools (first run only; ~35 MB).
#   2. Creates a private Python 3.12 environment in ./.venv and installs the
#      app's dependencies (first run only; ~1-2 GB, several minutes).
#   3. On macOS, generates "Tabular ML Lab.app" next to this folder's
#      launcher so future launches are a double-click on a real app icon.
#      The .app is created ON THIS MACHINE, so it carries no quarantine
#      flag and is immune to Gatekeeper app-translocation.
#   4. Starts the app; your browser opens automatically.
#
# Everything lives inside this folder. To uninstall: delete the folder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$ROOT/.tools/uv"
VENV="$ROOT/.venv"
STAMP="$ROOT/.venv-stamp"
FROM_APP="${1:-}"

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
note() { printf '\033[0;37m%s\033[0m\n' "$*"; }

notify_mac() {
    # Best-effort desktop notification when running without a terminal.
    if [[ "$(uname)" == "Darwin" ]]; then
        osascript -e "display notification \"$1\" with title \"Tabular ML Lab\"" >/dev/null 2>&1 || true
    fi
}

req_hash() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$ROOT/requirements.txt" | cut -d' ' -f1
    else
        sha256sum "$ROOT/requirements.txt" | cut -d' ' -f1
    fi
}

# ── 1. uv ────────────────────────────────────────────────────────────────
if [[ ! -x "$UV" ]]; then
    say "First-time setup 1/2: fetching the environment manager (uv)…"
    note "This is a one-time download (~35 MB)."
    mkdir -p "$ROOT/.tools"

    # Primary: the official binary straight from GitHub Releases.
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64)  UV_TARGET="aarch64-apple-darwin" ;;
        Darwin-x86_64) UV_TARGET="x86_64-apple-darwin" ;;
        Linux-x86_64)  UV_TARGET="x86_64-unknown-linux-gnu" ;;
        Linux-aarch64) UV_TARGET="aarch64-unknown-linux-gnu" ;;
        *) UV_TARGET="" ;;
    esac
    if [[ -n "$UV_TARGET" ]]; then
        TARBALL="$ROOT/.tools/uv.tar.gz"
        if curl -LsSf -o "$TARBALL" \
            "https://github.com/astral-sh/uv/releases/latest/download/uv-$UV_TARGET.tar.gz"; then
            tar -xzf "$TARBALL" -C "$ROOT/.tools"
            mv -f "$ROOT/.tools/uv-$UV_TARGET/uv" "$UV" 2>/dev/null || true
            rm -rf "$TARBALL" "$ROOT/.tools/uv-$UV_TARGET"
        fi
    fi
    # Fallback: the official installer script.
    if [[ ! -x "$UV" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$ROOT/.tools" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
    fi
    [[ -x "$UV" ]] || { echo "uv installation failed — check your internet connection and re-run."; exit 1; }
fi

# ── 2. Python environment ────────────────────────────────────────────────
# TML_PYTHON overrides the interpreter spec (default: pinned 3.12, which uv
# downloads automatically) — used by CI/e2e tests and power users.
PY_SPEC="${TML_PYTHON:-3.12}"
CURRENT_HASH="$(req_hash)"
if [[ ! -x "$VENV/bin/python" || ! -f "$STAMP" || "$(cat "$STAMP")" != "$CURRENT_HASH" ]]; then
    say "First-time setup 2/2: installing Python and the analysis libraries…"
    note "This downloads ~1-2 GB and takes a few minutes — ONE TIME ONLY."
    note "Every launch after this one takes seconds and works offline."
    notify_mac "One-time setup in progress — this takes a few minutes."
    "$UV" venv --python "$PY_SPEC" "$VENV"
    "$UV" pip install --python "$VENV/bin/python" -r "$ROOT/requirements.txt"
    printf '%s' "$CURRENT_HASH" > "$STAMP"
    say "Setup complete."
fi

# ── 3. macOS: materialize the app icon ──────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
    APP="$ROOT/Tabular ML Lab.app"
    APP_STAMP="$APP/Contents/Resources/.root-path"
    if [[ ! -d "$APP" || "$(cat "$APP_STAMP" 2>/dev/null)" != "$ROOT" ]]; then
        rm -rf "$APP"
        mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
        cat > "$APP/Contents/MacOS/TabularMLLab" <<LAUNCH
#!/bin/bash
exec bash "$ROOT/launcher/posix_launch.sh" --from-app
LAUNCH
        chmod +x "$APP/Contents/MacOS/TabularMLLab"
        cp "$ROOT/launcher/icon.icns" "$APP/Contents/Resources/icon.icns"
        printf '%s' "$ROOT" > "$APP_STAMP"
        cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Tabular ML Lab</string>
    <key>CFBundleDisplayName</key><string>Tabular ML Lab</string>
    <key>CFBundleIdentifier</key><string>lab.tabularml.launcher</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleExecutable</key><string>TabularMLLab</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
PLIST
        say "Created 'Tabular ML Lab.app' in this folder."
        note "From now on, double-click that app (and drag it to your Dock if you like)."
    fi
fi

# ── 4. Launch ────────────────────────────────────────────────────────────
if [[ "$FROM_APP" == "--from-app" ]]; then
    notify_mac "Starting Tabular ML Lab — your browser will open shortly."
else
    say "Starting Tabular ML Lab — your browser will open automatically."
    note "Keep this window open while you work; closing it quits the app."
fi

cd "$ROOT"
exec "$VENV/bin/python" -m streamlit run "$ROOT/app.py" \
    --browser.gatherUsageStats false
