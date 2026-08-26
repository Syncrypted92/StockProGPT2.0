#!/bin/bash
# Cron helper: one job at a time, hard timeout, unbuffered Python logs.
# Usage: cron_wrap.sh <lock_name> <timeout_sec> <python-args...>
set -u
LOCK_NAME="${1:?lock name}"
TIMEOUT_SEC="${2:?timeout seconds}"
shift 2

cd /opt/stockpro
export PYTHONUNBUFFERED=1
LOCK="/var/lock/stockpro-${LOCK_NAME}.lock"
PY="/opt/stockpro/.venv/bin/python"

set +e
/usr/bin/flock -n -E 75 "$LOCK" /usr/bin/timeout --foreground "$TIMEOUT_SEC" "$PY" "$@"
rc=$?
set -e

ts="$(date -Is)"
if [[ "$rc" -eq 75 ]]; then
  echo "$ts skip_locked lock=${LOCK_NAME}"
  exit 0
fi
if [[ "$rc" -eq 124 ]]; then
  echo "$ts timeout lock=${LOCK_NAME} sec=${TIMEOUT_SEC}"
  exit 124
fi
exit "$rc"
