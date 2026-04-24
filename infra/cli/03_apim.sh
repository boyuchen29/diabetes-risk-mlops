#!/usr/bin/env bash
set -euo pipefail

RG="utility-ai-accelerator"
LOCATION="eastus"
APIM_NAME="apim-diabetes-risk"
ACA_APP="ca-diabetes-risk-api"

: "${API_KEY:?API_KEY must be set}"

ACA_FQDN=$(az containerapp show \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)
ACA_URL="https://$ACA_FQDN"

echo "==> Creating APIM (Consumption tier) — this takes 15–30 minutes..."
az apim create \
  --name "$APIM_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --publisher-name "Diabetes Risk API" \
  --publisher-email "boyu.chen@neudesic.com" \
  --sku-name Consumption

echo "==> Importing OpenAPI spec from ACA..."
az apim api import \
  --resource-group "$RG" \
  --service-name "$APIM_NAME" \
  --api-id "diabetes-risk-api" \
  --path "/" \
  --display-name "Diabetes Risk API" \
  --protocols https \
  --specification-format OpenApi \
  --specification-url "$ACA_URL/openapi.json"

echo "==> Storing backend API key as encrypted Named Value..."
az apim nv create \
  --resource-group "$RG" \
  --service-name "$APIM_NAME" \
  --named-value-id "backend-api-key" \
  --display-name "backend-api-key" \
  --value "$API_KEY" \
  --secret true

echo "==> Applying inbound policy to all API operations..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TMPFILE=$(mktemp /tmp/apim-policy-XXXXXX.json)
cat > "$TMPFILE" <<EOF
{
  "properties": {
    "format": "xml",
    "value": "<policies><inbound><base /><set-backend-service base-url=\"$ACA_URL\" /><set-header name=\"Authorization\" exists-action=\"override\"><value>Bearer {{backend-api-key}}</value></set-header></inbound><backend><base /></backend><outbound><base /></outbound><on-error><base /></on-error></policies>"
  }
}
EOF

az rest \
  --method PUT \
  --url "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.ApiManagement/service/$APIM_NAME/apis/diabetes-risk-api/policies/policy?api-version=2022-08-01" \
  --body "@$TMPFILE" \
  --headers "Content-Type=application/json"

rm -f "$TMPFILE"

echo "==> Creating a subscription for testing..."
az rest \
  --method PUT \
  --url "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.ApiManagement/service/$APIM_NAME/subscriptions/test-subscription?api-version=2022-08-01" \
  --body '{
    "properties": {
      "displayName": "Test Subscription",
      "scope": "/apis/diabetes-risk-api",
      "state": "active"
    }
  }' \
  --headers "Content-Type=application/json"

SUBSCRIPTION_KEY=$(az rest \
  --method POST \
  --url "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.ApiManagement/service/$APIM_NAME/subscriptions/test-subscription/listSecrets?api-version=2022-08-01" \
  --query "primaryKey" -o tsv)

APIM_GW=$(az apim show \
  --name "$APIM_NAME" \
  --resource-group "$RG" \
  --query "gatewayUrl" -o tsv)

echo "==> APIM gateway: $APIM_GW"
echo "==> Test subscription key: $SUBSCRIPTION_KEY"
echo "==> Save this key — use it as Ocp-Apim-Subscription-Key header"
