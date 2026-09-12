python3 rtsp_triton_gst_native.py \
  --rtsp-url "$(python3 -c 'import camera_config; print(camera_config.CameraProfiles.get_unv_rtsp("192.168.1.105", "admin", "YourPassword"))')" \
  --model-name "ccvnn_hardswish_12d" \
  --input-dim 12
  
python3 rtsp_triton_gst_native.py \
  --rtsp-url "rtsp://admin:YourPassword@192.168.1.105:554/unicast/c1/s0/live" \
  --model-name "ccvnn_hardswish_12d" \
  --input-dim 12
  
  
python3 rtsp_multicamera_ingestion.py

python3 rtsp_nvdec_multicamera.py
