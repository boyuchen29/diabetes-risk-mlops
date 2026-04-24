#!/usr/bin/env bash
set -euo pipefail

RG="utility-ai-accelerator"
LOCATION="eastus"
KV_NAME="kv-diabetes-risk"
ACA_APP="ca-diabetes-risk-api"

: "${API_KEY:?API_KEY must be set}"
: "${DATABRICKS_TOKEN:?DATABRICKS_TOKEN must be set}"

echo "==> Creating Key Vault..."
az keyvault create \
  --name "$KV_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --sku standard \
  --enable-rbac-authorization true

KV_ID=$(az keyvault show --name "$KV_NAME" --resource-group "$RG" --query id -o tsv)

echo "==> Granting current user Key Vault Secrets Officer (needed to write secrets)..."
USER_OID=$(az ad signed-in-user show --query id -o tsv)
az role assignment create \
  --assignee "$USER_OID" \
  --role "Key Vault Secrets Officer" \
  --scope "$KV_ID"

echo "==> Waiting 60 seconds for role assignment to propagate..."
sleep 60

echo "==> Storing secrets in Key Vault..."
az keyvault secret set --vault-name "$KV_NAME" --name "api-key"          --value "$API_KEY"
az keyvault secret set --vault-name "$KV_NAME" --name "databricks-token" --value "$DATABRICKS_TOKEN"

echo "==> Getting Container App managed identity principal ID..."
# The CLI path uses a system-assigned identity (created by --registry-identity system
# in 02_aca.sh). Retrieve its principal ID from the containerapp identity.
PRINCIPAL_ID=$(az containerapp identity show \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --query principalId -o tsv)

echo "==> Granting Key Vault Secrets User role to Container App identity..."
az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Key Vault Secrets User" \
  --scope "$KV_ID"

KV_URI=$(az keyvault show --name "$KV_NAME" --resource-group "$RG" --query properties.vaultUri -o tsv)

echo "==> Updating Container App secrets to use Key Vault references..."
# identityref:system references the system-assigned identity used by the CLI path.
az containerapp secret set \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --secrets \
    "api-key=keyvaultref:${KV_URI}secrets/api-key,identityref:system" \
    "databricks-token=keyvaultref:${KV_URI}secrets/databricks-token,identityref:system"

echo "==> Key Vault migration complete. Secrets are now resolved from: $KV_URI"
