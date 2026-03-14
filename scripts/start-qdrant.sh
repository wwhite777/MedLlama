#!/bin/bash
# Start Qdrant Docker container for MedLlama RAG pipeline
set -e

STORAGE_PATH="/home/wjeong/ml/medllama/data/qdrant_storage"
mkdir -p "$STORAGE_PATH"

# Try to start existing container, or create new one
docker start qdrant 2>/dev/null || \
docker run -d --name qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$STORAGE_PATH:/qdrant/storage" \
    qdrant/qdrant:latest

# Wait for health check
echo "Waiting for Qdrant to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "Qdrant is ready!"
        exit 0
    fi
    sleep 1
done

echo "ERROR: Qdrant did not start within 30 seconds"
exit 1
