#!/bin/bash
# Build script for Z-Image API

set -e

echo "Building base image..."
docker build -f Dockerfile.base -t zimage-base .

echo "Building application image..."
docker build -t zimage-api .

echo "Build complete!"
echo "To run: docker run --gpus all -p 8000:8000 zimage-api"

