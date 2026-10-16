#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    printf 'Run bash scripts/setup_codespaces.sh first.\n' >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && "${QT_QPA_PLATFORM:-}" != offscreen ]]; then
    printf 'No graphical desktop is running. In Codespaces, rebuild the container using this repository devcontainer configuration.\n' >&2
    exit 1
fi
exec .venv/bin/python iraqi_id_printer.py "$@"
