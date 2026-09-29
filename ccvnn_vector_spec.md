================================================================================
CCVNN V17 & CUMULATIVE ACTUATION VECTOR (V_cum) MATHEMATICAL SPECIFICATION
================================================================================

1. FACTORIZED VECTOR NOTATION (V_17)
--------------------------------------------------------------------------------
Vector v_i captures local spatial and photometric attributes at pixel coordinate i:
v_i = [ r_i/h, c_i/w, hex_C/255, hex_L/255, hex_U/255, hex_R/255, hex_D/255, 
        c_type, (log2(h) + log2(w))/20, e_10, ..., e_17 ]^T

Element Breakdown (e_1 through e_17):
  * e_1  (Normalized Row):      r_i / h
  * e_2  (Normalized Column):   c_i / w
  * e_3  (Center Hex):          hex_C / 255
  * e_4  (Left Hex):            hex_L / 255
  * e_5  (Upper Hex):           hex_U / 255
  * e_6  (Right Hex):           hex_R / 255
  * e_7  (Down Hex):            hex_D / 255
  * e_8  (Component Type):      c_type
  * e_9  (Scale Factor):        (log2(h) + log2(w)) / 20
  * e_10 (Red Channel Mean):    (1 / (N * 255)) * sum(R_k)
  * e_11 (Green Channel Mean):  (1 / (N * 255)) * sum(G_k)
  * e_12 (Blue Channel Mean):   (1 / (N * 255)) * sum(B_k)
  * e_13 (Luminance Variance):  (1 / N) * sum((Y_k - B_lux * 255)^2)
  * e_14 (Absolute Brightness): B_lux = (1 / (N * 255)) * sum(Y_k)
                                where Y_k = 0.299*R_k + 0.587*G_k + 0.114*B_k
  * e_15 (Dynamic Contrast):    C_ratio = (Y_max - Y_min) / (Y_max + Y_min + epsilon)
                                where epsilon = 10^-5
  * e_16 (Gradient Magnitude):  (1 / N) * sum(sqrt(DX_k^2 + DY_k^2))
  * e_17 (Specular Reflection): R_specular = (1 / N) * sum(I(Y_k > T_sat) * (Y_k - T_sat) / (255 - T_sat))
                                where T_sat = 245


2. CUMULATIVE ACTUATION VECTOR (V_cum)
--------------------------------------------------------------------------------
State vector V_cum manages real-time edge PLC control and smoothing:
V_cum = [ S_anomaly, S_cum, I_severity, C_debounce, R_PLC, T_latency, H_health, Z_hash ]^T

Element Breakdown (V_0 through V_7):
  * V_0 (Instantaneous Anomaly): S_anomaly in [0, 1]
                                 Raw single-frame defect probability.
  * V_1 (Exponential Moving Avg): S_cum^(t) = alpha * S_cum^(t-1) + (1 - alpha) * S_anomaly^(t)
                                 Temporal smoothing parameter minimizing false triggers.
  * V_2 (Hazard Severity Tier):  I_severity mapped from score thresholds (0, 1, 2).
  * V_3 (Debounce Confidence):   C_debounce = (1 / W) * sum(I(S_anomaly > tau_det))
                                 Consecutive frame validation ratio across window W.
  * V_4 (PLC Actuation Flag):    R_PLC = step(C_debounce - tau_actuate) * bitmask
                                 Binary hardware line trigger for physical relays.
  * V_5 (Glass-to-Glass Latency): T_latency = t_render - t_capture
                                 End-to-end timing metric in milliseconds.
  * V_6 (System Health Index):   H_health = DaemonStatus AND BrokerConnected
                                 Background telemetry verification flag.
  * V_7 (Spatial Inspection Hash): Z_hash = CRC32(ROI_coords || FrameCounter)
                                 Spatial integrity hash for active inspection zones.
================================================================================