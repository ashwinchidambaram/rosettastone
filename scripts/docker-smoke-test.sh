#!/usr/bin/env bash
set -euo pipefail

# Docker smoke test script
# Builds Docker image, starts the app, verifies health endpoint, and cleans up

HEALTH_URL="http://localhost:8000/api/v1/health"
MAX_ATTEMPTS=30
RETRY_INTERVAL=2
ATTEMPT=0

# Cleanup function - runs on exit or error
cleanup() {
  local exit_code=$?
  echo "Cleaning up Docker containers..."
  docker compose down -v || true
  exit $exit_code
}

trap cleanup EXIT

# Step 1: Build Docker image
echo "Building Docker image..."
docker compose build

# Step 2: Start app service in detached mode
echo "Starting app service..."
docker compose up -d app

# Step 3: Wait for health check to pass
echo "Waiting for health endpoint to respond..."
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "Health check attempt $ATTEMPT/$MAX_ATTEMPTS..."

  # Try to get health endpoint status
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")

  if [ "$HTTP_STATUS" = "200" ]; then
    echo "Health check passed!"
    echo "SUCCESS: Docker smoke test passed"
    exit 0
  fi

  if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
    echo "Health endpoint returned $HTTP_STATUS, retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
  fi
done

# Step 4: Failure case
echo "FAILURE: Health check did not pass after ${MAX_ATTEMPTS} attempts"
exit 1
