sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Build image and run detached
docker compose up -d --build

# Inspect real-time streaming pipeline logs
docker logs -f jetson_ds_dualmode

tegrastats

# Copy unit file to systemd directory
sudo cp ds-dualmode.service /etc/systemd/system/ds-dualmode.service
sudo chmod 644 /etc/systemd/system/ds-dualmode.service

# Reload systemd manager configuration
sudo systemctl daemon-reload

# Enable service to auto-start on boot and start immediately
sudo systemctl enable --now ds-dualmode.service

sudo systemctl status ds-dualmode.service

journalctl -u ds-dualmode.service -f -o cat
sudo systemctl restart ds-dualmode.service
sudo reboot