# 1. Build Docker image locally on the edge target machine
docker build -t ccvnn/watcher-edge-daemon:v17-latest .

# 2. Deploy systemd unit configuration
sudo cp watcher-edge-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload

# 3. Enable and start daemon service
sudo systemctl enable --now watcher-edge-daemon

# 4. Inspect real-time status and journal logs
sudo systemctl status watcher-edge-daemon
sudo journalctl -u watcher-edge-daemon -f -o cat