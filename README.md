# Diabetes Risk MLOps

An end-to-end MLOps portfolio project demonstrating the full ML lifecycle: from a research notebook to a production-grade API with automated CI/CD.

**Stack:** Python · Databricks · MLflow · FastAPI · Docker · Azure Container Apps · Azure API Management · GitHub Actions

---

## Architecture

```
Databricks (training)          Azure (serving)
─────────────────────          ───────────────────────────────────────
tasks/ingest.py                GitHub Actions
tasks/train.py        ──────►  └─ build image → push ACR
tasks/register.py              └─ deploy → Azure Container Apps
        │                                      │
        │ MLflow artifacts                     │ loads artifacts at startup
        ▼                                      ▼
Databricks MLflow          FastAPI /predict → risk score (0–1000)
Registry                           │
        │                          ▼
        └── MLFLOW_RUN_ID ──► Azure API Management (auth gateway)
```

The model is trained on Databricks and artifacts (scores, weights, encoder) are logged to MLflow. The API loads those artifacts at startup via `MLFLOW_RUN_ID` and serves predictions without running inference at request time — every request is a lookup and weighted sum against pre-computed score tables.

---

## Project Structure

```
├── src/
│   ├── risk_score/          # Core ML library (data, training, scoring)
│   │   ├── data.py          # Data loading, binarization, categorization, encoding
│   │   ├── train.py         # Feature selection and logistic regression training
│   │   ├── score.py         # Score table and weight computation
│   │   ├── mlflow_utils.py  # MLflow logging helpers
│   │   └── cli.py           # Local training entrypoint (risk-score train)
│   └── api/                 # FastAPI serving layer
│       ├── main.py          # App startup — loads MLflow artifacts
│       ├── scorer.py        # Inference: loads artifacts, computes risk scores
│       ├── schemas.py       # Request/response models
│       ├── routers/         # Route handlers (/predict, /predict/explain, /predict/weights)
│       └── middleware/      # API key authentication
├── tasks/                   # Databricks job task scripts (ingest → train → register)
├── infra/
│   ├── cli/                 # One-time Azure provisioning scripts (ACR, ACA, APIM, Key Vault)
│   │   └── setup-oidc.sh    # Bootstrap OIDC service principal for GitHub Actions
│   └── terraform/           # Terraform for ACR, ACA, APIM, Key Vault
├── .github/workflows/
│   └── cicd.yml             # CI/CD: test on PR, build + deploy on merge to main
├── tests/                   # pytest test suite
├── config.yaml              # Databricks training config
├── config_local.yaml        # Local training config (no MLflow remote)
├── Dockerfile
└── docker-compose.yml
```

---

## Sub-projects

| # | Name | Description |
|---|------|-------------|
| 1 | Code Refactoring | Convert research notebook into a structured Python package |
| 2 | Experiment Tracking | Train on Databricks, log to MLflow, register model |
| 3 | Training Pipeline | Databricks Job with discrete ingest → train → register stages |
| 4 | Model Serving | FastAPI app with Docker, loads artifacts from MLflow |
| 5 | Azure Deployment | ACR + ACA + APIM + Key Vault via Terraform and CLI scripts |
| 6 | CI/CD | GitHub Actions: pytest on PR, build + push + deploy on merge |

---

## API Endpoints

All endpoints except `/health` require `Authorization: Bearer <API_KEY>`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Patient features in → risk score out |
| `POST` | `/predict/explain` | Risk score + per-feature breakdown |
| `GET` | `/predict/weights` | Selected features and their model weights |
| `GET` | `/health` | Health check (no auth) |

**Example request:**

```bash
curl -X POST https://<apim-gateway>/predict \
  -H "Ocp-Apim-Subscription-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"AGE":45,"SEX":2,"BMI":32.1,"BP":95.0,"S1":210.0,"S2":130.0,"S3":45.0,"S4":4.2,"S5":4.6,"S6":92.0}'
```

```json
{"risk_score": 76.02}
```

---

## Local Development

### Prerequisites

- Python 3.10+
- Databricks workspace with MLflow (for training)
- Azure CLI (for infrastructure)

### Install

```bash
pip install -e ".[api,dev]"
```

### Run tests

```bash
pytest
```

### Train locally (no Databricks)

```bash
risk-score train --config config_local.yaml
```

Artifacts are written to `mlruns/` (gitignored). This is useful for validating logic locally; production runs use Databricks.

### Run API locally

Copy `.env.example` to `.env` and fill in your Databricks credentials and a `MLFLOW_RUN_ID` from a completed Databricks training run:

```bash
cp .env.example .env
# edit .env
docker-compose up
```

The API will be available at `http://localhost:8000`.

---

## Infrastructure

Infrastructure is managed via Terraform (`infra/terraform/`) and one-time CLI scripts (`infra/cli/`).

### Provision Azure resources

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars
terraform init && terraform apply
```

### Set up GitHub Actions OIDC (one-time)

```bash
az login
chmod +x infra/cli/setup-oidc.sh
./infra/cli/setup-oidc.sh
```

The script prints three values. Add them as GitHub repository secrets:

| Secret | Description |
|--------|-------------|
| `AZURE_CLIENT_ID` | Service principal app ID |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/cicd.yml`) runs on every PR and push to `main`:

- **PR:** runs `pytest` only — no deployment
- **Push to main:** runs `pytest` → builds Docker image → pushes to ACR → deploys to Azure Container Apps → polls revision health state until `Healthy`

Authentication uses OIDC (no stored passwords). The deploy job fails the pipeline if the new revision becomes `Unhealthy`, leaving the previous revision active.

---

## Updating the Model

When a new Databricks training run produces better results:

```bash
az containerapp update \
  --name ca-diabetes-risk-api \
  --resource-group utility-ai-accelerator \
  --set-env-vars "MLFLOW_RUN_ID=<new-run-id>"
```

This restarts the container, which loads the new artifacts. No code change or CI/CD trigger required.
