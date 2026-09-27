# SLA Validation Report: `cyclotron_ensemble`

## 1. Executive Summary

- **Target Model:** `cyclotron_ensemble`
- **Protocol:** Async gRPC
- **Tested Concurrencies:** 1, 2, 4, 8, 16, 32

## 2. Benchmark Metrics

| Concurrency | Throughput (infer/s) | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 17.27 | 57.90 | 55.85 | 64.79 | 78.78 |
| 2 | 27.33 | 73.09 | 68.69 | 96.88 | 122.75 |
| 4 | 21.94 | 180.98 | 154.46 | 268.74 | 353.08 |
| 8 | 9.32 | 855.13 | 226.31 | 666.14 | 8765.85 |
| 16 | 30.16 | 516.51 | 502.70 | 689.78 | 1007.49 |
| 32 | 1.23 | 25886.27 | 9944.52 | 79296.99 | 87758.64 |

## 3. SLA Compliance

- [ ] **P90 Target (<= 10.0 ms)**
- [ ] **P99 Target (<= 20.0 ms)**
