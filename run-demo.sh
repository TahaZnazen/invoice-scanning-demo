#!/usr/bin/env bash
# Starts the whole demo: scan service (port 8000, in the background) + Expo (port 8081).
# Ctrl+C stops both.
#
#   ./run-demo.sh            normal
#   ./run-demo.sh --debug       also save each photo + detected outline to scan-service/debug-photos/
#   ./run-demo.sh --dev-build   for the installed development build (iPhone document scanner)
#                               instead of Expo Go
#   ./run-demo.sh -c            any other options go to "expo start" (-c = clear the bundler cache)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE="$ROOT/scan-service"
APP="$ROOT/invoice-scanner-demo"
LOG="$SERVICE/service.log"

DEBUG_DIR=""
TARGET="--go" # Expo Go by default (expo-dev-client is installed, so expo would pick the dev build)
EXPO_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG_DIR="$SERVICE/debug-photos" ;;
    --dev-build) TARGET="--dev-client" ;;
    *) EXPO_ARGS+=("$arg") ;;
  esac
done

# First run on a new machine: install dependencies.
if [[ ! -x "$SERVICE/.venv/bin/uvicorn" ]]; then
  echo "› Setting up the Python environment (first run)…"
  python3 -m venv "$SERVICE/.venv"
  "$SERVICE/.venv/bin/pip" install -q -r "$SERVICE/requirements.txt"
fi
if [[ ! -f "$SERVICE/models/doccornernet_lean.ort" ]]; then
  "$SERVICE/.venv/bin/python" "$SERVICE/corner_model.py" --download
fi
if [[ ! -f "$SERVICE/models/sam2.1_hiera_tiny/sam2.1_hiera_tiny.encoder.onnx" ]]; then
  echo "› Downloading the SAM 2.1 page model (~126 MB, first run)…"
  "$SERVICE/.venv/bin/python" "$SERVICE/sam_model.py" --download
fi
if [[ ! -d "$APP/node_modules" ]]; then
  echo "› Installing app dependencies (first run)…"
  (cd "$APP" && npm install --silent)
fi

in_use() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

if in_use 8000; then
  echo "✗ Port 8000 is already in use: the scan service is probably running in another terminal."
  exit 1
fi

if in_use 8081; then
  # Expo already running (e.g. in another terminal): just run the scan service here.
  echo "› Expo is already running on port 8081: starting only the scan service (Ctrl+C to stop)."
  cd "$SERVICE"
  SCAN_DEBUG_DIR="$DEBUG_DIR" exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log
fi

IP="$(ip -4 -br addr 2>/dev/null | awk '$1 !~ /^(lo|docker|br-|veth)/ && $2 == "UP" {split($3, a, "/"); print a[1]; exit}')"
echo "› Laptop address: ${IP:-unknown} (the iPhone must be on the same network)"
echo "› Starting scan service (log: scan-service/service.log)…"
(cd "$SERVICE" && SCAN_DEBUG_DIR="$DEBUG_DIR" exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log) >"$LOG" 2>&1 &
SERVICE_PID=$!
trap 'kill "$SERVICE_PID" 2>/dev/null || true; echo; echo "› Stopped."' EXIT

for _ in $(seq 1 40); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then break; fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "✗ Scan service failed to start:"; tail -20 "$LOG"; exit 1
  fi
  sleep 0.5
done
echo "✓ Scan service ready${DEBUG_DIR:+ (saving debug photos)}"

echo "› Starting Expo: open Invoice Scanner Demo on the iPhone (or scan the QR code)"
cd "$APP"
npx expo start "$TARGET" "${EXPO_ARGS[@]}"
