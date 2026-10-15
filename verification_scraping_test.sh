# 1. Test raw Triton metric exposition
curl -s http://127.0.0.1:8002/metrics | grep -E "nv_inference_request_success|nv_cpu_memory_used_bytes|nv_inference_compute_infer"

# 1. Stop existing Prometheus container
docker rm -f v20_prometheus_exporter

# 2. Run Prometheus bound to WSL2 host network
docker run -d \
  --name v20_prometheus_exporter \
  --network host \
  -v $(pwd)/v20_prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest