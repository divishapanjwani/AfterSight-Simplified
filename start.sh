#!/usr/bin/env bash
# AfterSight: start the API (port 8020) and the web app (port 5200). Ctrl-C stops both.
# Frees ports 8020 and 5200 first if something is already listening on them.
set -euo pipefail
cd "$(dirname "$0")"
API_PORT=8020
WEB_PORT=5200
LOGS=outputs/logs

if [[ ! -x .venv/bin/python ]]; then
  echo "No virtualenv at ./.venv — see README → Quick start." >&2; exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  echo "Installing frontend dependencies (npm ci)…"; (cd frontend && npm ci --silent)
fi

free_port() {
  local port=$1 pids
  pids=$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)
  [[ -z "$pids" ]] && return 0
  echo "Port $port is in use by $(ps -o pid=,comm= -p $(echo $pids | tr ' ' ',') | awk '{n=split($2,a,"/"); printf "%s (pid %s) ", a[n], $1}')— stopping it."
  kill $pids 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.3
    [[ -z "$(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true)" ]] && return 0
  done
  kill -9 $(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null) 2>/dev/null || true
}
free_port $API_PORT
free_port $WEB_PORT

mkdir -p "$LOGS"
(cd backend && exec ../.venv/bin/python -m uvicorn app.main:app --port $API_PORT) > "$LOGS/api.log" 2>&1 &
API_PID=$!
(cd frontend && exec npx vite --port $WEB_PORT --strictPort) > "$LOGS/web.log" 2>&1 &
WEB_PID=$!
cleanup() {
  trap - EXIT INT TERM
  # The venv's python and npx both start the real server as a CHILD process, so stop children too,
  # then make sure nothing is left listening on our ports.
  pkill -TERM -P $API_PID 2>/dev/null || true
  pkill -TERM -P $WEB_PID 2>/dev/null || true
  kill $API_PID $WEB_PID 2>/dev/null || true
  free_port $API_PORT > /dev/null
  free_port $WEB_PORT > /dev/null
  echo; echo "AfterSight stopped."
}
trap cleanup EXIT INT TERM

wait_for() {
  for _ in $(seq 1 80); do curl -sf "$1" > /dev/null 2>&1 && return 0; sleep 0.25; done
  return 1
}
if ! wait_for "http://localhost:$API_PORT/api/health"; then
  echo "The API did not start — see $LOGS/api.log" >&2; tail -20 "$LOGS/api.log" >&2; exit 1
fi
if ! wait_for "http://localhost:$WEB_PORT"; then
  echo "The web app did not start — see $LOGS/web.log" >&2; tail -20 "$LOGS/web.log" >&2; exit 1
fi

echo
echo "  AfterSight is running:  http://localhost:$WEB_PORT"
echo "  API + Swagger:          http://localhost:$API_PORT/docs"
echo "  Logs:                   $LOGS/   (Ctrl-C to stop)"
wait
