#!/usr/bin/env bash
set -euo pipefail

RG="utility-ai-accelerator"
ACR_NAME="diabetesriskmlops"
LOCATION="eastus"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> Creating Azure Container Registry..."
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$RG" \
  --sku Basic \
  --location "$LOCATION" \
  --admin-enabled false

# az acr build uploads the local context to ACR and builds the image there.
# No local Docker daemon required — works in Azure Cloud Shell.
echo "==> Building and pushing image via ACR Tasks..."
az acr build \
  --registry "$ACR_NAME" \
  --image "diabetes-risk-api:latest" \
  "$REPO_ROOT"

echo "==> Done."
az acr repository list --name "$ACR_NAME" --output table
