resource "azurerm_api_management" "apim" {
  name                = "apim-diabetes-risk"
  location            = var.location
  resource_group_name = var.resource_group_name
  publisher_name      = "Diabetes Risk API"
  publisher_email     = "boyu.chen@neudesic.com"
  sku_name            = "Consumption_0"
}

resource "azurerm_api_management_named_value" "backend_key" {
  name                = "backend-api-key"
  resource_group_name = var.resource_group_name
  api_management_name = azurerm_api_management.apim.name
  display_name        = "backend-api-key"
  value               = var.api_key
  secret              = true
}

# NOTE: The API import fetches /openapi.json from the live ACA URL.
# Apply ACA resources first (Task 3 Step 6), confirm the health endpoint
# responds, then apply this file. Otherwise Terraform may fail to fetch
# the spec before the container is ready.
resource "azurerm_api_management_api" "diabetes_risk" {
  name                  = "diabetes-risk-api"
  resource_group_name   = var.resource_group_name
  api_management_name   = azurerm_api_management.apim.name
  revision              = "1"
  display_name          = "Diabetes Risk API"
  path                  = ""
  protocols             = ["https"]
  subscription_required = true

  import {
    content_format = "openapi+json-link"
    content_value  = "https://${azurerm_container_app.api.ingress[0].fqdn}/openapi.json"
  }
}

resource "azurerm_api_management_api_policy" "all_ops" {
  api_name            = azurerm_api_management_api.diabetes_risk.name
  api_management_name = azurerm_api_management.apim.name
  resource_group_name = var.resource_group_name

  xml_content = <<XML
<policies>
  <inbound>
    <base />
    <set-backend-service base-url="https://${azurerm_container_app.api.ingress[0].fqdn}" />
    <set-header name="Authorization" exists-action="override">
      <value>Bearer {{backend-api-key}}</value>
    </set-header>
  </inbound>
  <backend><base /></backend>
  <outbound><base /></outbound>
  <on-error><base /></on-error>
</policies>
XML
}

resource "azurerm_api_management_subscription" "test" {
  resource_group_name = var.resource_group_name
  api_management_name = azurerm_api_management.apim.name
  api_id              = azurerm_api_management_api.diabetes_risk.id
  subscription_id     = "test-subscription"
  display_name        = "Test Subscription"
  state               = "active"
}
