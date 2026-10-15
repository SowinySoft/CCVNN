import urllib.request
import urllib.error
import json

payload = {
    "inputs": [{
        "name": "84",
        "shape": [1, 1, 1, 128],
        "datatype": "FP32",
        "data": [0.1] * 128
    }]
}

req = urllib.request.Request(
    "http://127.0.0.1:8005/v2/models/ccvnn/infer",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"Status Code: {e.code}")
    print(f"Triton Error Detail: {e.read().decode('utf-8')}")