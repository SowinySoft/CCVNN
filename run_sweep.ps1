$MODEL_NAME = "ccvnn_v14_model_a"
$TRITON_URL = "localhost:8001"

foreach ($batch in @(1, 16, 32, 64)) {
    Write-Host "=== Benchmarking Batch Size: $batch ==="
    docker exec tritonserver perf_analyzer `
        -m $MODEL_NAME `
        -i grpc `
        -u $TRITON_URL `
        -b $batch `
        --concurrency-range 1:8:1 `
        --measurement-interval 10000 `
        --async `
        --shape input_vector:14 `
        -f "/models/benchmark_batch_${batch}.csv"
}

Write-Host "Sweep complete. Log files generated in model_repository/benchmark_batch_*.csv"
