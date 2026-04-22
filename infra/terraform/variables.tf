variable "resource_group_name" {
  default = "utility-ai-accelerator"
}

variable "location" {
  default = "eastus"
}

variable "api_key" {
  description = "Backend API key injected by APIM and validated by the FastAPI middleware"
  sensitive   = true
}

variable "databricks_token" {
  description = "Databricks personal access token"
  sensitive   = true
}

variable "databricks_host" {
  description = "Databricks workspace URL, e.g. https://adb-xxxx.azuredatabricks.net"
}

variable "mlflow_run_id" {
  description = "MLflow run ID to load scoring artifacts from"
}
