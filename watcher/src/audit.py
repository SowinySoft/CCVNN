import hashlib
import json
import logging
import os
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuditSystem")

class AuditLogger:
    def __init__(self, audit_path: str = "logs/audit_trail.jsonl", secret_key: str = "CCVNN_INDUSTRIAL_KEY_2026"):
        self.audit_path = audit_path
        self.secret_key = secret_key
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)

    def generate_sha256_signature(self, payload: Dict[str, Any]) -> str:
        """Computes a SHA-256 signature for tamper-evident audit records."""
        canonical_data = json.dumps(payload, sort_keys=True) + self.secret_key
        return hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()

    def log_incident(self, payload: Dict[str, Any]) -> str:
        """Appends signed incident entry to immutable audit file."""
        signature = self.generate_sha256_signature(payload)
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "signature_sha256": signature,
            "payload": payload
        }
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"AUDIT RECORD SIGNED [SHA-256: {signature[:12]}...]")
        return signature

class RollingFrameBuffer:
    def __init__(self, fps: int = 30, window_seconds: int = 10):
        self.capacity = fps * window_seconds
        self.fps = fps
        self.buffer = deque(maxlen=self.capacity)

    def push_frame(self, frame: np.ndarray):
        """Pushes raw video frame into rolling in-memory ring buffer."""
        self.buffer.append(frame)

    def dump_mp4_clip(self, output_dir: str = "dumps/clips") -> Optional[str]:
        """Dumps buffer contents to MP4 clip upon critical alert trigger."""
        if not self.buffer:
            logger.warning("Buffer empty, skipping clip dump.")
            return None

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"incident_{int(time.time() * 1000)}.mp4")
        
        height, width, _ = self.buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))

        for frame in list(self.buffer):
            writer.write(frame)
        writer.release()

        logger.warning(f"BLACK-BOX DUMP: Saved {len(self.buffer)} frames to {filename}")
        return filename