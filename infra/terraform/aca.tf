# User-assigned identity created before the container app so AcrPull
# can be granted before the app's first revision tries to pull the image.
resource "azurerm_user_assigned_identity" "aca_identity" {
  name                = "id-diabetes-risk-api"
  location            = var.location
  resource_group_name = var.resource_group_name
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.aca_identity.principal_id
}

resource "azurerm_container_app_environment" "env" {
  name                = "cae-diabetes-risk"
  location            = var.location
  resource_group_name = var.resource_group_name
}

resource "azurerm_container_app" "api" {
  name                         = "ca-diabetes-risk-api"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = var.resource_group_name
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aca_identity.id]
  }

  registry {
    server   = azurerm_container_registry.acr.login_server
    identity = azurerm_user_assigned_identity.aca_identity.id
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  secret {
    name  = "api-key"
    value = var.api_key
  }

  secret {
    name  = "databricks-token"
    value = var.databricks_token
  }

  template {
    min_replicas = 1
    max_replicas = 3

    container {
      name   = "diabetes-risk-api"
      image  = "${azurerm_container_registry.acr.login_server}/diabetes-risk-api:latest"
      cpu    = 0.5
      memory = "1Gi"

      env {
        name        = "API_KEY"
        secret_name = "api-key"
      }

      env {
        name        = "DATABRICKS_TOKEN"
        secret_name = "databricks-token"
      }

      env {
        name  = "DATABRICKS_HOST"
        value = var.databricks_host
      }

      env {
        name  = "MLFLOW_RUN_ID"
        value = var.mlflow_run_id
      }
    }
  }

  # Explicit dependency ensures the AcrPull role assignment exists before
  # the first revision starts and attempts to pull the image.
  depends_on = [azurerm_role_assignment.acr_pull]
}
