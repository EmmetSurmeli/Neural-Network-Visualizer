#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
if ! .venv/bin/python -c 'import torch, fastapi, uvicorn' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r backend/requirements.txt
fi
if [ ! -d frontend/node_modules ]; then (cd frontend && npm ci); fi
if [ ! -f backend/models/digits.pt ]; then .venv/bin/python scripts/train_demo.py; fi
(cd frontend && npm run build)
echo 'NeuralScope is ready at http://127.0.0.1:8000'
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
