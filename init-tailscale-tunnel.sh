docker exec pharmagraph tailscale serve reset
docker exec pharmagraph tailscale funnel --bg http://frontend:8501

docker exec pharmagraph_ts tailscale serve reset
docker exec pharmagraph_ts tailscale funnel --bg http://backend:1234
