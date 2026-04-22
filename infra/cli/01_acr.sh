#!/usr/bin/env bash
set -euo pipefail

RG="utility-ai-accelerator"
ACR_NAME="diabetesriskmlops"
LOCATION="eastus"
IMAGE="$ACR_NAME.azurecr.io/diabetes-risk-api:latest"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> Creating Azure Container Registry..."
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$RG" \
  --sku Basic \
  --location "$LOCATION" \
  --admin-enabled false

echo "==> Logging in to ACR..."
az acr login --name "$ACR_NAME"

echo "==> Building image..."
docker build -t "$IMAGE" "$REPO_ROOT"

echo "==> Pushing image..."
docker push "$IMAGE"

echo "==> Done. Image: $IMAGE"
az acr repository list --name "$ACR_NAME" --output table
