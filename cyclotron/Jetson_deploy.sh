sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Build image and run detached
docker compose up -d --build

# Inspect real-time streaming pipeline logs
docker logs -f jetson_ds_dualmode

tegrastats