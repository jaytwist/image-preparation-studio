#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"; export ORT_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1
[ -x .venv/bin/python ] || { python3 -m venv .venv; .venv/bin/pip install torch torchvision; .venv/bin/pip install -r requirements-dev.txt; }
exec .venv/bin/python -m ips.app "$@"
