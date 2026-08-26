#!/bin/bash
# Install StockPro cron (root). Expects app at /opt/stockpro.
#
# IMPORTANT: /etc/cron.d CRON_TZ is unreliable on some Ubuntu/cron builds
# (schedules silently run in the system timezone). Always pin the host to
# America/New_York so weekday hours match RTH.
set -eu

TZ_NAME="${STOCKPRO_TZ:-America/New_York}"

if command -v timedatectl >/dev/null 2>&1; then
  timedatectl set-timezone "$TZ_NAME"
elif [[ -f "/usr/share/zoneinfo/$TZ_NAME" ]]; then
  ln -sf "/usr/share/zoneinfo/$TZ_NAME" /etc/localtime
  echo "$TZ_NAME" >/etc/timezone
else
  echo "ERROR: cannot set timezone to $TZ_NAME" >&2
  exit 1
fi

mkdir -p /var/log/stockpro
chmod 755 /var/log/stockpro

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Strip Windows CRLF if this repo was copied from a PC
sed -i 's/\r$//' "$SCRIPT_DIR/stockpro.cron" "$SCRIPT_DIR/install_linux_cron.sh" "$SCRIPT_DIR/cron_wrap.sh" 2>/dev/null || true
chmod 755 "$SCRIPT_DIR/cron_wrap.sh"

install -m 644 "$SCRIPT_DIR/stockpro.cron" /etc/cron.d/stockpro
chmod 644 /etc/cron.d/stockpro

if command -v systemctl >/dev/null 2>&1; then
  systemctl restart cron 2>/dev/null || systemctl restart crond 2>/dev/null || true
else
  service cron restart 2>/dev/null || service crond restart 2>/dev/null || true
fi

echo "Installed /etc/cron.d/stockpro"
echo "System timezone: $(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || date +%Z)"
echo "Local time: $(date)"
echo "Verify: cat /etc/cron.d/stockpro"
echo "Logs: /var/log/stockpro/"
