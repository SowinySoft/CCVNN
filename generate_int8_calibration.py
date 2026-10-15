import numpy as np

# Generate representative calibration datasets
np.random.seed(42)

# 12D Hardswish calibration data (1000 samples)
calib_12d = np.random.uniform(-3.0, 3.0, (1000, 12)).astype(np.float32)
np.save("calib_12d.npy", calib_12d)

# 9D ReLU6 calibration data (1000 samples)
calib_9d = np.random.uniform(0.0, 6.0, (1000, 9)).astype(np.float32)
np.save("calib_9d.npy", calib_9d)

print("Created calibration datasets: calib_12d.npy (1000x12), calib_9d.npy (1000x9)")
