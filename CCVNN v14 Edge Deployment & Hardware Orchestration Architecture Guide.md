# CCVNN v14 Edge Deployment & Hardware Orchestration Architecture Guide

**Author:** Mr. Said Sowiny (SowinySoft)

**License:** GNU Affero General Public License v3.0 (AGPLv3)

**Target Platform:** Industrial Edge Nodes, Hybrid K8s/K3s Clusters, NVIDIA Triton Edge Workstations

---

## 1. Orchestration Strategy: Single-Node Edge vs. Multi-Node K8s

For standalone edge stations, **Docker Compose** is recommended to minimize latency and memory overhead. For multi-site fleets, maintenance rollouts, and unified cluster health checks, utilize the existing **Kubernetes (`k8s/`)** and **Helm (`helm/`)** manifests provided in the repository.

```
CCVNN Repository Structure:
├── k8s/
│   ├── deployment.yaml            # Base microservice deployment
│   ├── service.yaml               # Cluster internal IP routing
│   ├── ccvnn-deployment.yaml      # Triton & Watcher edge pod definitions
│   └── ccvnn-service.yaml         # NodePort/LoadBalancer port bindings
└── helm/
    ├── namespace-ccvnn-edge.yaml  # Isolated edge namespace
    ├── secret-ccvnn-tls.yaml      # Encrypted TLS secrets for gRPC/HTTP
    ├── rendered-edge-manifests.yaml
    └── ccvnn-microservice/        # Helm chart values & templates

```

### Deployment Selection Matrix

| Operational Context | Recommended Engine | Justification |
| --- | --- | --- |
| **Single Standalone Workstation** | Docker Compose | Bypasses CNI bridge overhead; direct IPC access to host GPU and hardware sockets. |
| **Multi-Line Factory Fleet** | K3s / Kubernetes | Enables central helm upgrades, rolling model updates, and declarative state enforcement via `k8s/ccvnn-deployment.yaml`. |
| **High Availability & OTA Updates** | Helm (`helm/ccvnn-microservice`) | Templated parameter updates for threshold shifts across multiple factory floors without manual container intervention. |

---

## 2. Dynamic Hardware Auto-Attaching Prerequisites

To ensure camera capture devices, GPU accelerators, and Modbus serial/network relays reattach dynamically after power interruptions or physical hot-plugging, enforce the following host and container prerequisites.

### A. GPU Acceleration Setup (NVIDIA iGPU / Discrete GPU)

1. **Configure Host Docker Daemon for Global GPU Passthrough:**
Update `/etc/docker/daemon.json` on the edge node:
```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

```


2. **Enable GPU Driver Persistence Mode:**
To eliminate model reload initialization delays on boot, run:
```bash
sudo nvidia-smi -pm 1

```



---

### B. Persistent USB & Serial Device Links (`udev` Rules)

USB cameras (`/dev/video*`) and RS485/RS232 serial converters (`/dev/ttyUSB*`) dynamically shift device indices across system reboots. Enforce persistent static symlinks on the host node.

1. **Create Rule File `/etc/udev/rules.d/99-ccvnn-hardware.rules`:**
```udev
# USB Camera Hardware Symlink
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="05a3", ATTRS{idProduct}=="9230", SYMLINK+="ccvnn_cam_main"

# RS485 Modbus Serial Converter Symlink
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="ccvnn_plc_serial"

```


2. **Reload and Trigger `udev` Engine:**
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger

```



---

### C. Industrial Network & Modbus TCP Auto-Recovery

1. **Host Network Mode:**
Avoid docker bridge latency (~0.2ms penalty) by binding container networking directly to the host stack in `docker-compose.yml` or K8s `hostNetwork: true`:
```yaml
network_mode: "host"

```


2. **Socket Auto-Healing in Client Runtime:**
Ensure client daemons wrap Modbus and gRPC connections with exponential backoff handlers:
```python
import time
from pymodbus.client import ModbusTcpClient

def get_plc_client(host="192.168.10.45", port=502):
    client = ModbusTcpClient(host, port=port)
    while not client.connect():
        print("[WARNING] PLC connection offline. Retrying in 2s...")
        time.sleep(2)
    return client

```



---

## 3. Deployment Commands

### Option 1: Native Docker Deployment

```bash
# Launch end-to-end edge microservice stack
docker compose up -d --build

```

### Option 2: Kubernetes Manifest Deployment

```bash
# Apply edge namespace and secrets
kubectl apply -f helm/namespace-ccvnn-edge.yaml
kubectl apply -f helm/secret-ccvnn-tls.yaml

# Deploy Triton and CCVNN Watcher services
kubectl apply -f k8s/ccvnn-service.yaml
kubectl apply -f k8s/ccvnn-deployment.yaml

```

### Option 3: Helm Chart Orchestration

```bash
# Install or upgrade CCVNN edge microservice helm chart
helm upgrade --install ccvnn-edge ./helm/ccvnn-microservice \
  --namespace ccvnn-edge \
  --set triton.image.tag="26.08-py3-igpu"

```