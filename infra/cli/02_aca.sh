#!/usr/bin/env bash
set -euo pipefail

RG="utility-ai-accelerator"
LOCATION="eastus"
ACR_NAME="diabetesriskmlops"
ACA_ENV="cae-diabetes-risk"
ACA_APP="ca-diabetes-risk-api"
IMAGE="$ACR_NAME.azurecr.io/diabetes-risk-api:latest"

# These must be set in the environment before running this script
: "${API_KEY:?API_KEY must be set}"
: "${DATABRICKS_TOKEN:?DATABRICKS_TOKEN must be set}"
: "${DATABRICKS_HOST:?DATABRICKS_HOST must be set}"
: "${MLFLOW_RUN_ID:?MLFLOW_RUN_ID must be set}"

echo "==> Creating Container Apps Environment..."
az containerapp env create \
  --name "$ACA_ENV" \
  --resource-group "$RG" \
  --location "$LOCATION"

# Create the app with --registry-identity system so ACA knows to use the
# managed identity for registry auth. The identity exists after this call
# but does not yet have AcrPull — the revision will fail to pull the image
# until the role assignment below is applied.
echo "==> Creating Container App (image pull will fail until AcrPull is assigned)..."
az containerapp create \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --environment "$ACA_ENV" \
  --image "$IMAGE" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --registry-server "$ACR_NAME.azurecr.io" \
  --registry-identity system \
  --secrets "api-key=$API_KEY" "databricks-token=$DATABRICKS_TOKEN" \
  --env-vars \
    "API_KEY=secretref:api-key" \
    "DATABRICKS_TOKEN=secretref:databricks-token" \
    "DATABRICKS_HOST=$DATABRICKS_HOST" \
    "MLFLOW_RUN_ID=$MLFLOW_RUN_ID"

echo "==> Assigning AcrPull role to the Container App managed identity..."
PRINCIPAL_ID=$(az containerapp identity show \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --query principalId -o tsv)

ACR_ID=$(az acr show \
  --name "$ACR_NAME" \
  --resource-group "$RG" \
  --query id -o tsv)

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role AcrPull \
  --scope "$ACR_ID"

# Role assignments can take up to 2 minutes to propagate through AAD.
# Force a new revision so the container app retries the pull with the
# now-granted permission.
echo "==> Waiting 60 seconds for role assignment to propagate..."
sleep 60

echo "==> Triggering new revision to retry image pull..."
az containerapp update \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --image "$IMAGE"

ACA_URL=$(az containerapp show \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

echo "==> Container App deployed at: https://$ACA_URL"
