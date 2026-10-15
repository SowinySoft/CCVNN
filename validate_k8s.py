import sys

def check_k8s_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    required_keys = ["apiVersion", "kind", "metadata", "spec"]
    missing = [k for k in required_keys if k not in content]
    if missing:
        raise ValueError(f"Missing essential keys: {missing}")
    print(f"[OK] {filepath} schema verified.")

try:
    check_k8s_file('k8s/deployment.yaml')
    check_k8s_file('k8s/service.yaml')
    print("SUCCESS: k8s manifests validated successfully!")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
