# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# Replace standard RUN pip install with a cached mount:
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dynamic dependencies for OpenCV & SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libglib2.0-0 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy Watcher application code
COPY watcher/ /app/watcher/

# Healthcheck for internal MQTT / service responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import socket; s = socket.socket(); s.connect(('localhost', 1883))" || exit 1

CMD ["python3", "-m", "watcher.src.main"]