import pytest
from watcher.src.tracker import SimpleObjectTracker


@pytest.fixture
def tracker():
    t = SimpleObjectTracker(iou_threshold=0.3)
    yield t
    if hasattr(t, "close"):
        t.close()


def test_persistent_tracking_ids(tracker):
    # Frame 1: Detection at [100, 100, 200, 200]
    frame1_dets = [
        {
            "class_name": "fire_smoke",
            "confidence": 0.88,
            "bounding_box": [100, 100, 200, 200],
        }
    ]
    res1 = tracker.update(frame1_dets)
    tid1 = res1[0]["tracking_id"]

    # Frame 2: Slightly shifted bounding box [105, 102, 205, 202]
    frame2_dets = [
        {
            "class_name": "fire_smoke",
            "confidence": 0.91,
            "bounding_box": [105, 102, 205, 202],
        }
    ]
    res2 = tracker.update(frame2_dets)
    tid2 = res2[0]["tracking_id"]

    assert tid1 == tid2 == "tr_1", (
        f"Tracking ID changed unexpectedly: {tid1} vs {tid2}"
    )
    print(
        "✓ Persistent Tracking Verified: Object maintained ID"
        f" '{tid1}' across moving frames."
    )


if __name__ == "__main__":
    t = SimpleObjectTracker(iou_threshold=0.3)
    try:
        # Run manually
        frame1_dets = [
            {
                "class_name": "fire_smoke",
                "confidence": 0.88,
                "bounding_box": [100, 100, 200, 200],
            }
        ]
        res1 = t.update(frame1_dets)
        tid1 = res1[0]["tracking_id"]

        frame2_dets = [
            {
                "class_name": "fire_smoke",
                "confidence": 0.91,
                "bounding_box": [105, 102, 205, 202],
            }
        ]
        res2 = t.update(frame2_dets)
        tid2 = res2[0]["tracking_id"]

        assert tid1 == tid2 == "tr_1", (
            f"Tracking ID changed unexpectedly: {tid1} vs {tid2}"
        )
        print(
            "✓ Persistent Tracking Verified: Object maintained ID"
            f" '{tid1}' across moving frames."
        )
    finally:
        if hasattr(t, "close"):
            t.close()