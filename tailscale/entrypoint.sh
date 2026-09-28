#!/bin/bash
set -e

echo "Starting Tailscale daemon..."
tailscaled --tun=userspace-networking --state=${TS_STATE_DIR}/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock &

sleep 3

if [ -n "$TS_AUTHKEY" ]; then
    echo "Authenticating Tailscale..."
    tailscale up --authkey=${TS_AUTHKEY} --hostname=${HOSTNAME} --accept-dns=true
fi

echo "Waiting for Tailscale to connect..."
until tailscale status | grep -q "linux"; do
    sleep 1
done

echo "Configuring Tailscale Serve / Funnel proxy..."
if [ -n "$TS_TARGET_HOST" ] && [ -n "$TS_TARGET_PORT" ]; then
    tailscale serve reset || true
    echo "Enabling Tailscale Funnel for http://${TS_TARGET_HOST}:${TS_TARGET_PORT}..."
    tailscale funnel --bg http://${TS_TARGET_HOST}:${TS_TARGET_PORT} || tailscale serve --bg http://${TS_TARGET_HOST}:${TS_TARGET_PORT}
fi

tailscale status
wait -n
