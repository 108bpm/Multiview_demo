#!/bin/bash
# Bring up all 4 already-named Piper CAN interfaces at 1 Mbit/s.
# Names come from the udev rule /etc/udev/rules.d/99-piper-can.rules (by serial).
#
# txqueuelen 1000: deeper TX queue so the bimanual connect burst (4 arms sharing
#   one USB->CAN adapter) doesn't overflow the default 10-frame queue and raise
#   ENOBUFS -> SDK "SendCanMessage(SEND_MESSAGE_FAILED (100017))".
# NOTE: no restart-ms -- the gs_usb USB->CAN adapter doesn't support hardware
#   bus-off auto-restart ("Device doesn't support restart from Bus Off"). If a bus
#   ever goes bus-off, recover by re-running this script (it downs+ups each link).
set -e
for c in left_leader left_follower right_leader right_follower; do
    sudo ip link set "$c" down 2>/dev/null || true
    sudo ip link set "$c" type can bitrate 1000000
    sudo ip link set "$c" txqueuelen 1000
    sudo ip link set "$c" up
    echo "activated $c"
done
echo "---"
ip -br link show type can
