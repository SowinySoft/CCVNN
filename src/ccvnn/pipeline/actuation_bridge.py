"""
ActuationBridge: Connects MultiFactorRuleEngine with Mosquitto MQTT broker
to trigger hardware safety actuation on debounced hazard detection.
"""

import json
import logging
import paho.mqtt.client as mqtt
from typing import Dict, Any
import numpy as np

from src.ccvnn.rule_engine.multifactor_engine import MultiFactorRuleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ActuationBridge")


class ActuationBridge:
    def __init__(
        self,
        rule_engine: MultiFactorRuleEngine,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        topic: str = "ccvnn/safety/actuation",
    ):
        self.rule_engine = rule_engine
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.topic = topic
        
        # Explicitly set CallbackAPIVersion.VERSION2 to suppress deprecation warning
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            self.client = mqtt.Client()

    def connect(self):
        """Connects to the Mosquitto MQTT broker."""
        try:
            self.client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def process_and_publish(self, ensemble_output: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluates ensemble output through rule engine and publishes MQTT alert if debounced actuation triggers.
        """
        eval_result = self.rule_engine.evaluate(ensemble_output)

        payload = {
            "trigger_actuation": eval_result["trigger_actuation"],
            "raw_trigger": eval_result["raw_trigger"],
            "consecutive_frames": eval_result["consecutive_frames"],
            "hazard_score": round(eval_result["hazard_score"], 4),
            "detected_objects": eval_result["detected_objects"],
            "rule_matched": eval_result["rule_matched"],
        }

        if eval_result["trigger_actuation"]:
            self.client.publish(self.topic, json.dumps(payload), qos=1)
            logger.warning(f"CRITICAL: Safety Actuation Triggered! Published to {self.topic}: {payload}")
        else:
            logger.debug(f"Operational status nominal: {payload}")

        return payload

    def disconnect(self):
        """Disconnects from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker.")
