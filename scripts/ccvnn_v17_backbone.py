import numpy as np
import time

class CCVNNFeatureExtractorV16:
    def __init__(self, roi_height=64, roi_width=64):
        self.h = roi_height
        self.w = roi_width
        
    def extract_vector(self, roi_rgb: np.ndarray, theta_rot: float = 0.0) -> np.ndarray:
        """
        Extracts 16-element feature vector (v_input) from an RGB ROI.
        Includes V16 photometric elements (B_lux, C_ratio).
        """
        start_time = time.perf_counter_ns()
        
        # Convert to Grayscale Luminance
        r, g, b = roi_rgb[:, :, 0], roi_rgb[:, :, 1], roi_rgb[:, :, 2]
        y_lum = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Spatial Shape (e0 - e2)
        grad_x, grad_y = np.gradient(y_lum)
        s_1d = np.mean(np.hypot(grad_x, grad_y)) / 255.0
        s_2d = np.count_nonzero(y_lum > 128) / (self.h * self.w)
        s_3d = np.std(y_lum) / 128.0
        
        # Crystal Depth (e3 - e5)
        c_pri = np.mean(y_lum[::2, ::2]) / 255.0
        c_sec = np.mean(y_lum[1::2, 1::2]) / 255.0
        c_phase = np.arctan2(np.mean(grad_y), np.mean(grad_x)) / np.pi
        
        # Photometric Dynamics (e6 - e8)
        p_light = np.max(y_lum) / 255.0
        p_dark = 1.0 - (np.min(y_lum) / 255.0)
        p_opacity = 1.0 - (np.std(y_lum) / (np.mean(y_lum) + 1e-5))
        p_opacity = np.clip(p_opacity, 0.0, 1.0)
        
        # Auxiliary Moments (e9 - e11)
        m_contrast = np.var(y_lum) / (255.0 ** 2)
        m_density = np.mean(y_lum > (np.mean(y_lum) + np.std(y_lum)))
        m_struct = (np.mean(grad_x**2) + np.mean(grad_y**2)) / (255.0 ** 2)
        
        # Orientation & Chromaticity (e12 - e13)
        hs_chroma = np.mean(np.abs(r.astype(float) - g.astype(float))) / 255.0
        
        # V16 Photometric Extensions (e14 - e15)
        b_lux = np.mean(y_lum) / 255.0
        y_max, y_min = np.max(y_lum), np.min(y_lum)
        c_ratio = (y_max - y_min) / (y_max + y_min + 1e-5)
        
        # Assemble 16-Element Vector
        v16_vector = np.array([
            s_1d, s_2d, s_3d,
            c_pri, c_sec, c_phase,
            p_light, p_dark, p_opacity,
            m_contrast, m_density, m_struct,
            theta_rot / (2 * np.pi),
            hs_chroma,
            b_lux,
            c_ratio
        ], dtype=np.float32)
        
        exec_microsec = (time.perf_counter_ns() - start_time) / 1000.0
        return v16_vector, exec_microsec

# Quick Test
if __name__ == "__main__":
    extractor = CCVNNFeatureExtractorV16()
    dummy_roi = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    vector, latency_us = extractor.extract_vector(dummy_roi, theta_rot=0.785)
    print(f"CCVNN V16 Vector Extracted in {latency_us:.2f} microseconds:")
    print(vector)