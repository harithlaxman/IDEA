#!/bin/sh
set -e
echo "Starting IDEA..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
