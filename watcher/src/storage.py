import sqlite3
import json
import logging
import os
from typing import List, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OfflineStorage")

class OfflineStorage:
    def __init__(self, db_path: str = "data/offline_events.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def buffer_event(self, topic: str, payload: Dict[str, Any]) -> bool:
        """Saves an offline MQTT event to local SQLite buffer."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO pending_events (topic, payload) VALUES (?, ?)",
                    (topic, json.dumps(payload))
                )
                conn.commit()
                logger.warning(f"STORE-AND-FORWARD: Buffered event for topic [{topic}] in SQLite.")
                return True
        except Exception as e:
            logger.error(f"Failed to write offline buffer: {e}")
            return False

    def get_pending_events(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Retrieves all cached events awaiting re-transmission."""
        events = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, topic, payload FROM pending_events ORDER BY id ASC")
            for row_id, topic, payload_str in cursor.fetchall():
                events.append((row_id, topic, json.loads(payload_str)))
        return events

    def remove_event(self, event_id: int):
        """Removes an event after successful re-transmission."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pending_events WHERE id = ?", (event_id,))
            conn.commit()