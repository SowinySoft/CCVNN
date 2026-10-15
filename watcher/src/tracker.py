import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ObjectTracker")


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class SimpleObjectTracker:
    """IoU-based tracker for persistent tracking_id assignment across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Updates tracks with new bounding boxes and injects persistent tracking_id into each detection."""
        # Age existing tracks
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        if not detections:
            return []

        matched_track_ids = set()
        tracked_detections = []

        for det in detections:
            bbox = det.get("bounding_box", [0, 0, 0, 0])
            class_name = det.get("class_name", "unknown")

            best_iou = 0.0
            best_tid = None

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if track["class_name"] != class_name:
                    continue

                iou = compute_iou(bbox, track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                # Matched with existing track
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["age"] = 0
                matched_track_ids.add(best_tid)
                assigned_id = f"tr_{best_tid}"
            else:
                # Register new track
                new_tid = self.next_id
                self.next_id += 1
                self.tracks[new_tid] = {
                    "bbox": bbox,
                    "class_name": class_name,
                    "age": 0
                }
                matched_track_ids.add(new_tid)
                assigned_id = f"tr_{new_tid}"

            det_copy = det.copy()
            det_copy["tracking_id"] = assigned_id
            tracked_detections.append(det_copy)

        return tracked_detections