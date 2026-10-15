python3 -c "
import subprocess
import csv
import io

def run_perf_debug(model_name, input_name, csv_path):
    print(f'==> Benchmarking {model_name} (Input: {input_name})...')
    cmd = [
        'docker', 'run', 'ccvnn_triton_v20', 'perf_analyzer',
        '-m', model_name,
        '-i', 'http',
        '-u', 'localhost:8000',
        '--shape', f'{input_name}:1,3,640,640',
        '--concurrency-range', '1:8:2',
        '--measurement-interval', '4000',
        '-f', csv_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if res.returncode != 0:
        print(f'  [!] perf_analyzer failed for {model_name} (Exit code {res.returncode}):')
        for line in res.stdout.strip().splitlines()[-10:]:
            print('      ' + line)
        return []

    cat_cmd = ['docker', 'run', 'ccvnn_triton_v20', 'cat', csv_path]
    cat_res = subprocess.run(cat_cmd, capture_output=True, text=True)
    
    results = []
    reader = csv.DictReader(io.StringIO(cat_res.stdout))
    for row in reader:
        conc = int(row.get('Concurrency', row.get('Request Concurrency', 0)))
        tp = float(row.get('Throughput', row.get('Inferences/Second', 0)))
        lat = float(row.get('Avg latency', row.get('Avg Latency', 0))) / 1000.0
        results.append((conc, tp, lat))
    return results

# Benchmark ONNX
onnx_res = run_perf_debug('yolov8n', 'images', '/tmp/res_onnx.csv')
if not onnx_res:
    onnx_res = run_perf_debug('yolov8n', 'x.1', '/tmp/res_onnx.csv')

# Benchmark OpenVINO
ov_res = run_perf_debug('yolov8n_openvino', 'x.1', '/tmp/res_ov.csv')

# Output Table
if onnx_res and ov_res:
    print('\n' + '='*75)
    print(f'{\"Concurrency\":<12} | {\"ONNX Throughput\":<16} | {\"OV Throughput\":<16} | {\"Latency (OV vs ONNX)\":<22}')
    print('='*75)
    for (c1, tp1, lat1), (c2, tp2, lat2) in zip(onnx_res, ov_res):
        lat_diff = ((lat1 - lat2) / lat1) * 100.0 if lat1 > 0 else 0
        print(f'{c1:<12} | {tp1:>8.2f} infer/s | {tp2:>8.2f} infer/s | {lat2:.2f}ms vs {lat1:.2f}ms ({lat_diff:+.1f}%)')
    print('='*75)
"