output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "aca_fqdn" {
  value = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "apim_gateway_url" {
  value = "https://${azurerm_api_management.apim.gateway_url}"
}
