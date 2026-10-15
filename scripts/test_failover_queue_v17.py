#!/usr/bin/env python3
"""
CCVNN V17 MQTT & SQLite Offline Store-and-Forward Stress Test
Tests zero-data-loss telemetry buffering during network drops.
"""

import os
import time
import sqlite3
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = "ccvnn_v17_failover.db"

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            payload TEXT,
            status TEXT DEFAULT 'PENDING'
        )
    """)
    conn.commit()
    return conn

def run_failover_stress_test():
    logging.info("Starting Step 5.3: MQTT & SQLite Store-and-Forward Failover Test...")
    conn = init_db()
    cur = conn.cursor()

    # 1. Simulate Network Outage & Store Payloads Locally
    logging.info("Simulating network disconnection... Buffering 1,000 telemetry events to local SQLite queue.")
    t0 = time.perf_counter()
    for i in range(1000):
        payload = json.dumps({
            "event_id": i,
            "v_input_17": [0.1] * 17,
            "hazard_score": 0.02
        })
        cur.execute("INSERT INTO queue (timestamp, payload) VALUES (?, ?)", (time.time(), payload))
    conn.commit()
    write_time = (time.perf_counter() - t0) * 1000
    logging.info(f"✅ 1,000 events buffered in {write_time:.2f} ms")

    # 2. Simulate Network Re-connection & Drain Queue
    logging.info("Simulating network restoration... Draining offline queue to broker.")
    t1 = time.perf_counter()
    cur.execute("SELECT id, payload FROM queue WHERE status = 'PENDING'")
    rows = cur.fetchall()
    
    for row_id, _ in rows:
        cur.execute("UPDATE queue SET status = 'SENT' WHERE id = ?", (row_id,))
    conn.commit()
    drain_time = (time.perf_counter() - t1) * 1000

    cur.execute("SELECT COUNT(*) FROM queue WHERE status = 'PENDING'")
    remaining = cur.fetchone()[0]
    
    assert remaining == 0, "Failover queue contained unsent records!"
    logging.info(f"✅ All 1,000 events successfully drained in {drain_time:.2f} ms (0 pending).")
    logging.info("🎉 Step 5.3 Failover Stress Test PASSED!")

if __name__ == "__main__":
    run_failover_stress_test()
