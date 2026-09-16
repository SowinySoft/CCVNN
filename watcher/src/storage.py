import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OfflineStorage")


class Storage:

    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yields an active connection and guarantees closure on exit."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def execute_write(self, query: str, params: tuple = ()) -> None:
        """Combines transaction safety with guaranteed connection closure."""
        with self.connection() as conn:
            with conn:  # Handles COMMIT / ROLLBACK
                conn.execute(query, params)

    def execute_read(self, query: str, params: tuple = ()) -> list:
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()


class OfflineStorage:

    def __init__(self, db_path: str = "data/offline_events.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yields an active connection and guarantees closure on exit."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self.connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pending_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

    def buffer_event(self, topic: str, payload: Dict[str, Any]) -> bool:
        """Saves an offline MQTT event to local SQLite buffer."""
        try:
            with self.connection() as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO pending_events (topic, payload) VALUES (?, ?)",
                        (topic, json.dumps(payload)),
                    )
            logger.warning(
                f"STORE-AND-FORWARD: Buffered event for topic [{topic}] in SQLite."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to write offline buffer: {e}")
            return False

    def get_pending_events(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        """Retrieves all cached events awaiting re-transmission."""
        events = []
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, topic, payload FROM pending_events ORDER BY id ASC"
            )
            for row_id, topic, payload_str in cursor.fetchall():
                events.append((row_id, topic, json.loads(payload_str)))
        return events

    def remove_event(self, event_id: int):
        """Removes an event after successful re-transmission."""
        with self.connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM pending_events WHERE id = ?", (event_id,)
                )

    def close(self) -> None:
        """Interface hook for publisher and engine teardown routines."""
        pass  # Short-lived connections require no persistent handle cleanup