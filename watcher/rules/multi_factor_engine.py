"""
Cyclotron Multi-Factor Rule Engine.
Evaluates VectorPayload items against YAML rule definitions.
"""

import yaml
import logging
from typing import Dict, Any, List, Optional
from watcher.schemas.vector_schema import VectorPayload

logger = logging.getLogger("MultiFactorRuleEngine")


class MultiFactorRuleEngine:
    def __init__(self, config_path: str = "config/watcher_multi_factor_rules.yaml"):
        self.config_path = config_path
        self.rules: List[Dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                self.rules = [r for r in data.get("rules", []) if r.get("enabled", True)]
            logger.info(f"Loaded {len(self.rules)} active rules from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load multi-factor rules: {e}")
            self.rules = []

    def evaluate_vector(self, payload: VectorPayload) -> List[Dict[str, Any]]:
        """
        Evaluates a single VectorPayload against loaded rules and returns triggered actions.
        """
        triggered_actions = []

        for rule in self.rules:
            cond = rule.get("condition", {})
            logic = cond.get("logic", "AND")
            
            telemetry_matched = self._eval_telemetry(cond.get("telemetry"), payload)
            vision_matched = self._eval_vision(cond.get("vision"), payload)

            is_triggered = False
            if logic == "AND":
                is_triggered = telemetry_matched and vision_matched
            elif logic == "OR":
                is_triggered = telemetry_matched or vision_matched

            if is_triggered:
                logger.info(f"[RuleEngine] Triggered rule: {rule['id']} ({rule['name']})")
                triggered_actions.append({
                    "rule_id": rule["id"],
                    "action": rule.get("action", {}),
                    "vector_id": payload.vector_id
                })

        return triggered_actions

    def _eval_telemetry(self, cond: Optional[Dict[str, Any]], payload: VectorPayload) -> bool:
        if not cond:
            return True
        min_score = cond.get("min_hazard_score", 0.0)
        return payload.hazard_score >= min_score

    def _eval_vision(self, cond: Optional[Dict[str, Any]], payload: VectorPayload) -> bool:
        if not cond:
            return True
        
        target_classes = set(cond.get("target_classes", []))
        min_conf = cond.get("min_confidence", 0.0)
        min_count = cond.get("min_entity_count", 1)

        matching_entities = [
            e for e in payload.entities
            if e.confidence >= min_conf and (not target_classes or e.label in target_classes)
        ]

        return len(matching_entities) >= min_count