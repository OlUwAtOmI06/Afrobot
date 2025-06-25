import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
backlog = 2048

# Worker processes
workers = 2  # Reduced for memory constraints
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for model loading
keepalive = 5

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "afrobot_api"

# Server mechanics
preload_app = False  # Set to False since you're using lazy loading
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None