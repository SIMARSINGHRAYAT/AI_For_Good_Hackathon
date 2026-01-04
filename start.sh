#!/bin/bash
echo "Starting DataCleanRoom Platform..."

# Wait for database to be ready (if using PostgreSQL)
if [ -n "$DATABASE_URL" ]; then
  echo "Waiting for database to be ready..."
  sleep 5
fi

# Run database migrations (if any)
python -c "
import sys
from app import app, db
with app.app_context():
    db.create_all()
    print('Database tables created/verified')
"

# Start the application
exec gunicorn \
  --bind 0.0.0.0:$PORT \
  --workers 4 \
  --threads 2 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  app:app
