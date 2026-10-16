#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
printf '\nReady: run bash run.sh\nOpen Ports > 6080 > Open in Browser, then Connect (desktop password: vscode).\nKeep the forwarded port Private.\n'
