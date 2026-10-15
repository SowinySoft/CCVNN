# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
graceful_timeout = 15  # Gives inflight Locust/client requests 15s to finish
timeout = 30