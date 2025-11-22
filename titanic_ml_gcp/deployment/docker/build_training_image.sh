#!/bin/bash
# Build and push training Docker image to Artifact Registry

set -e  # Exit on error

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-titanic-ml-gcp}"
REGION="${GCP_REGION:-us-central1}"
REPOSITORY_NAME="${ARTIFACT_REGISTRY_REPO:-titanic-ml-repo}"
IMAGE_NAME="xgboost-training"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================"
echo "Building Training Docker Image"
echo "============================================"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY_NAME}"
echo "Image Name: ${IMAGE_NAME}"
echo "Image Tag: ${IMAGE_TAG}"
echo "Full Image URI: ${IMAGE_URI}"
echo "============================================"

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
cd "${PROJECT_ROOT}"

# Build Docker image
echo ""
echo "Building Docker image..."
docker build \
    --platform linux/amd64 \
    -f deployment/docker/Dockerfile.training \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -t "${IMAGE_URI}" \
    .

echo ""
echo "✓ Docker image built successfully"

# Optionally push to Artifact Registry
read -p "Push image to Artifact Registry? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Configuring Docker authentication..."
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
    
    echo ""
    echo "Pushing image to Artifact Registry..."
    docker push "${IMAGE_URI}"
    
    echo ""
    echo "✓ Image pushed successfully"
    echo ""
    echo "Image URI: ${IMAGE_URI}"
fi

echo ""
echo "============================================"
echo "Build Complete"
echo "============================================"

