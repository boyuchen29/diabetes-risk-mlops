#!/usr/bin/env bash
set -euo pipefail

GITHUB_ORG="boyuchen29"
GITHUB_REPO="diabetes-risk-mlops"
APP_NAME="sp-diabetes-risk-cicd"
RG="utility-ai-accelerator"
ACR_NAME="diabetesriskmlops"
ACA_APP="ca-diabetes-risk-api"

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

echo "==> Creating service principal..."
EXISTING=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv)
if [[ -n "$EXISTING" ]]; then
  echo "    App '$APP_NAME' already exists (appId=$EXISTING). Skipping creation."
  APP_ID="$EXISTING"
else
  APP_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
  az ad sp create --id "$APP_ID"
fi

echo "==> Configuring federated credential (trusts GitHub Actions on main branch)..."
az ad app federated-credential create \
  --id "$APP_ID" \
  --parameters "{
    \"name\": \"github-main\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"repo:$GITHUB_ORG/$GITHUB_REPO:ref:refs/heads/main\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }"

echo "==> Waiting for service principal to propagate..."
sleep 15

SP_OBJECT_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)

echo "==> Granting AcrPush on ACR..."
ACR_ID=$(az acr show --name "$ACR_NAME" --resource-group "$RG" --query id -o tsv)
az role assignment create \
  --assignee "$SP_OBJECT_ID" \
  --role AcrPush \
  --scope "$ACR_ID"

echo "==> Granting Contributor on Container App..."
ACA_ID=$(az containerapp show \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --query id -o tsv)
# Contributor is the minimum built-in role that covers `az containerapp update`
az role assignment create \
  --assignee "$SP_OBJECT_ID" \
  --role Contributor \
  --scope "$ACA_ID"

echo ""
echo "==> Done! Add these as GitHub repository secrets:"
echo "    Settings → Secrets and variables → Actions → New repository secret"
echo ""
echo "    AZURE_CLIENT_ID=$APP_ID"
echo "    AZURE_TENANT_ID=$TENANT_ID"
echo "    AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
