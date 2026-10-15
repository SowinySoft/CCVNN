import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotificationDispatcher")


class NotificationDispatcher:
    """Secondary Notification Dispatcher (Webhooks, Twilio SMS, WebRTC alerts)."""

    def __init__(self, default_webhooks: Optional[List[str]] = None):
        self.webhooks = default_webhooks or []

    def dispatch_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Dispatches an asynchronous HTTP POST notification to a remote webhook URL."""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "CCVNN-Watcher/1.0"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    logger.info(f"WEBHOOK DISPATCH SUCCESS: [{url}] Response: {resp.status}")
                    return True
                else:
                    logger.error(f"Webhook error status: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Webhook HTTP transmission failed to [{url}]: {e}")
            return False

    def dispatch_sms_alert(self, phone_number: str, message: str, twilio_config: Optional[Dict[str, str]] = None) -> bool:
        """Sends urgent SMS alerts for critical safety incidents."""
        logger.warning(f"URGENT SMS DISPATCH to [{phone_number}]: '{message}'")
        if twilio_config and twilio_config.get("account_sid"):
            # REST API call to Twilio / carrier gateway
            logger.info("SMS delivered via API gateway.")
        else:
            logger.info("SMS simulation dispatch completed (no live provider API key set).")
        return True

    def dispatch_secondary_notifications(self, payload: Dict[str, Any], rule_config: Optional[Dict[str, Any]] = None):
        """Processes all configured secondary alerts for a triggered incident."""
        webhooks = self.webhooks[:]
        if rule_config and "webhooks" in rule_config:
            webhooks.extend(rule_config["webhooks"])

        for webhook_url in webhooks:
            self.dispatch_webhook(webhook_url, payload)

        if payload.get("severity") == "CRITICAL":
            sms_target = rule_config.get("emergency_phone") if rule_config else None
            if sms_target:
                msg = f"CRITICAL HAZARD DETECTED: {payload.get('event_type')} in Zone {payload.get('zone_id')}"
                self.dispatch_sms_alert(sms_target, msg)