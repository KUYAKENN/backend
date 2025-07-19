#!/bin/bash
# Railway startup script with proper PORT handling

# Set default port if PORT is not set
PORT=${PORT:-5000}

echo "Starting application on port $PORT"

# Start gunicorn with the resolved port
exec gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload --log-level info
