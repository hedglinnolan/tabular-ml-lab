#!/usr/bin/env bash
# TurboTab: the one command on macOS and Linux.
#
#     bash turbotab/deploy/turbotab.sh [--port N] [--no-open] [--smoke]
#
# Finds a Python 3.12 or newer and hands over to launch.py, which makes TurboTab's environment
# once, starts it and opens the browser (see launch.py). Without a Python 3.12+ it fetches one
# with uv into $TURBOTAB_HOME/tools (default ~/.turbotab/tools), once. Set TURBOTAB_PYTHON to
# choose the interpreter.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_HOME="${TURBOTAB_HOME:-$HOME/.turbotab}"

new_enough() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}

PY=""
# A double-clicked file does not always get the shell's PATH, so the usual install places too.
for candidate in "${TURBOTAB_PYTHON:-}" python3.13 python3.12 python3 python3.14 python \
    /opt/homebrew/bin/python3 /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    "$HOME/.local/bin/python3.13" "$HOME/.local/bin/python3.12"; do
    [[ -n "$candidate" ]] || continue
    if command -v "$candidate" >/dev/null 2>&1 && new_enough "$candidate"; then
        PY="$(command -v "$candidate")"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo "TurboTab: no Python 3.12 or newer found; fetching one (once, about 60 MB)..."
    UV="$(command -v uv || true)"
    if [[ -z "$UV" ]]; then
        UV="$TT_HOME/tools/uv"
        if [[ ! -x "$UV" ]]; then
            mkdir -p "$TT_HOME/tools"
            curl -LsSf https://astral.sh/uv/install.sh \
                | env UV_INSTALL_DIR="$TT_HOME/tools" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
        fi
    fi
    export UV_PYTHON_INSTALL_DIR="$TT_HOME/tools/python"
    "$UV" python install 3.12
    PY="$("$UV" python find 3.12)"
    if [[ -z "$PY" ]] || ! new_enough "$PY"; then
        echo "TurboTab: could not get Python 3.12. Install it from https://www.python.org/downloads/"
        echo "and start TurboTab again."
        exit 1
    fi
fi

exec "$PY" "$HERE/launch.py" "$@"
