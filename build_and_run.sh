#!/bin/bash

# Stop and remove old container if it exists
echo "Stopping and removing old container..."
docker stop interrogator 2>/dev/null || true
docker rm interrogator 2>/dev/null || true

# Build the container
echo "Building the Docker container (forcing no-cache)..."
sudo docker build --no-cache -t lan-interrogator .

# Run with GPU access
echo "Running the container..."
sudo docker run --gpus all -d \
  -p 8000:8000 \
  --name interrogator \
  --restart always \
  lan-interrogator

echo "Container started! Access it at http://localhost:8000"
