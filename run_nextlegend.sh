#!/usr/bin/env bash
#launch the streamlit app
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
APP_DIR="$ROOT_DIR/nextlegend"
PY_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  echo "[NextLegend] Python introuvable dans le venv ($PY_BIN). Crée/active ton venv avant de lancer ce script." >&2
  exit 1
fi

cd "$APP_DIR"
exec "$PY_BIN" -m streamlit run Home.py "$@"
