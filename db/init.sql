-- Create safety audit table
CREATE TABLE IF NOT EXISTS safety_audit_logs (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    state VARCHAR(20) NOT NULL,
    reasoning TEXT,
    plc_register_40001 INT NOT NULL,
    PRIMARY KEY (timestamp, id)
);

-- Convert table into TimescaleDB Hypertable
SELECT create_hypertable('safety_audit_logs', 'timestamp', if_not_exists => TRUE);

-- Create index for quick state searches
CREATE INDEX IF NOT EXISTS idx_safety_audit_state ON safety_audit_logs (state, timestamp DESC);