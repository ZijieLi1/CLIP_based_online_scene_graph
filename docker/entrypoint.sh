#!/bin/bash
set -e
# Bind-mounting the usbmuxd socket *file* goes stale when the daemon
# exits (phone unplug) and recreates it. /host-run is the live host /run.
if [[ -d /host-run ]]; then
  mkdir -p /var/run
  ln -sfn /host-run/usbmuxd /var/run/usbmuxd
fi
# shellcheck disable=SC1091
source /ws/docker/env.sh
if ! grep -q 'source /ws/docker/env.sh' /root/.bashrc 2>/dev/null; then
  echo 'source /ws/docker/env.sh' >> /root/.bashrc
fi
cd /ws
exec "${@:-sleep}" infinity
