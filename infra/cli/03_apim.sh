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
  --publisher-email "xoxo.boyu@gmail.com" \
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
POLICY_XML="<policies><inbound><base /><set-backend-service base-url=\"$ACA_URL\" /><set-header name=\"Authorization\" exists-action=\"override\"><value>Bearer {{backend-api-key}}</value></set-header></inbound><backend><base /></backend><outbound><base /></outbound><on-error><base /></on-error></policies>"

az apim api policy create \
  --resource-group "$RG" \
  --service-name "$APIM_NAME" \
  --api-id "diabetes-risk-api" \
  --value "$POLICY_XML" \
  --format xml

echo "==> Creating a subscription for testing..."
az apim subscription create \
  --resource-group "$RG" \
  --service-name "$APIM_NAME" \
  --subscription-id "test-subscription" \
  --display-name "Test Subscription" \
  --scope "/apis/diabetes-risk-api" \
  --state active

SUBSCRIPTION_KEY=$(az apim subscription show \
  --resource-group "$RG" \
  --service-name "$APIM_NAME" \
  --subscription-id "test-subscription" \
  --query "primaryKey" -o tsv)

APIM_GW=$(az apim show \
  --name "$APIM_NAME" \
  --resource-group "$RG" \
  --query "gatewayUrl" -o tsv)

echo "==> APIM gateway: $APIM_GW"
echo "==> Test subscription key: $SUBSCRIPTION_KEY"
echo "==> Save this key — use it as Ocp-Apim-Subscription-Key header"
