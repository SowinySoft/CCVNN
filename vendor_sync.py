import os
import shutil

site_packages = ".venv/lib/python3.14/site-packages"
target_dir = "vendor_packages"

# Explicitly empty / clean target directory first
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)

reqs = set()
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
        if pkg:
            reqs.add(pkg.replace("-", "_").lower())

print(f"Target packages to sync: {reqs}")

copied_count = 0
for item in os.listdir(site_packages):
    item_lower = item.replace("-", "_").lower()
    matched = any(item_lower == req or item_lower.startswith(req + "-") or item_lower.startswith(req + "_") for req in reqs)
    
    if matched:
        src = os.path.join(site_packages, item)
        dst = os.path.join(target_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied_count += 1

print(f"Successfully synced {copied_count} packages to {target_dir}")
