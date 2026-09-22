"""
MultiFactorRuleEngine: Evaluates vector vision detections combined with 
telemetry hazard scores to determine critical actuation state.
"""

from typing import Dict, List, Any
import numpy as np


"""
MultiFactorRuleEngine: Evaluates vector vision detections combined with 
telemetry hazard scores and applies Watcher Governor debouncing.
"""

from typing import Dict, Any
import numpy as np


class MultiFactorRuleEngine:
    def __init__(
        self, 
        hazard_threshold: float = 0.75, 
        min_vision_confidence: float = 0.5,
        debounce_consecutive_frames: int = 3
    ):
        self.hazard_threshold = hazard_threshold
        self.min_vision_confidence = min_vision_confidence
        self.debounce_consecutive_frames = debounce_consecutive_frames
        
        # Watcher Governor state tracking
        self._consecutive_hazard_count = 0
        self._latched_actuation = False

    def evaluate(self, ensemble_output: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluates combined Triton ensemble outputs with state debouncing.
        """
        hazard_score = float(ensemble_output.get("output_hazard", [[0.0]])[0][0])
        scores = ensemble_output.get("detection_scores", np.zeros((1, 100)))[0]
        classes = ensemble_output.get("detection_classes", np.zeros((1, 100)))[0]

        # Filter valid detections above confidence threshold
        valid_indices = np.where(scores >= self.min_vision_confidence)[0]
        detected_classes = classes[valid_indices].tolist() if len(valid_indices) > 0 else []

        # Instantaneous hazard condition
        is_hazard_active = hazard_score >= self.hazard_threshold
        has_critical_object = len(detected_classes) > 0
        raw_trigger = is_hazard_active or (has_critical_object and hazard_score > 0.4)

        # Watcher Governor Debounce Filter
        if raw_trigger:
            self._consecutive_hazard_count += 1
        else:
            self._consecutive_hazard_count = max(0, self._consecutive_hazard_count - 1)

        debounced_actuation = self._consecutive_hazard_count >= self.debounce_consecutive_frames

        return {
            "raw_trigger": bool(raw_trigger),
            "trigger_actuation": bool(debounced_actuation),
            "consecutive_frames": self._consecutive_hazard_count,
            "hazard_score": hazard_score,
            "detected_objects": len(detected_classes),
            "detected_classes": detected_classes,
            "rule_matched": "DEBOUNCED_CRITICAL_HAZARD" if debounced_actuation else "NORMAL_OPERATIONAL"
        }

    def reset_state(self):
        """Resets the debouncer memory state."""
        self._consecutive_hazard_count = 0
        self._latched_actuation = False