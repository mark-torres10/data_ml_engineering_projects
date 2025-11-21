# Service Account Setup

## Overview

This project uses **Application Default Credentials (ADC)** for local development instead of service account keys, which is a more secure approach aligned with GCP best practices.

## Authentication Methods

### For Local Development (Current Setup)

We use Application Default Credentials via your personal Google Cloud account:

```bash
gcloud auth application-default login
```

**Benefits:**
- ✅ No JSON keys to manage or secure
- ✅ No risk of accidentally committing credentials
- ✅ Credentials automatically expire with your session
- ✅ Easier to manage and rotate

**How it works:**
- Your Python applications automatically use your authenticated gcloud credentials
- No need to set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Works seamlessly with all Google Cloud client libraries

### Service Account Created

A service account `titanic-ml-sa@titanic-ml-gcp.iam.gserviceaccount.com` has been created with the following roles:

- **Vertex AI User** - For training and deploying models
- **Storage Admin** - For creating and managing GCS buckets
- **Storage Object Admin** - For reading/writing objects
- **Logs Writer** - For writing logs
- **Monitoring Metric Writer** - For writing metrics

**Note:** This service account is available for production deployments but not used for local development.

## For Production Deployments

When deploying to production environments (Cloud Run, Compute Engine, GKE, etc.):

### Option 1: Workload Identity (Recommended for GKE)

```bash
# Link Kubernetes service account to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
  titanic-ml-sa@titanic-ml-gcp.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:titanic-ml-gcp.svc.id.goog[NAMESPACE/KSA_NAME]"
```

### Option 2: Service Account Impersonation

```bash
# Grant your user permission to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
  titanic-ml-sa@titanic-ml-gcp.iam.gserviceaccount.com \
  --member="user:YOUR_EMAIL@domain.com" \
  --role="roles/iam.serviceAccountTokenCreator"
```

### Option 3: Default Service Accounts

Cloud Run, Cloud Functions, and Compute Engine can use the service account directly:

```bash
# Example: Deploy to Cloud Run with specific service account
gcloud run deploy SERVICE_NAME \
  --service-account titanic-ml-sa@titanic-ml-gcp.iam.gserviceaccount.com \
  ...
```

## Organizational Policy

Your organization has the constraint `constraints/iam.disableServiceAccountKeyCreation` enabled, which prevents creating service account JSON keys. This is a **security best practice** that:

- Prevents key leakage through version control
- Eliminates the risk of long-lived credentials being compromised
- Enforces use of more secure authentication methods

## Verifying Your Setup

Test that your credentials work:

```bash
# Verify ADC is configured
gcloud auth application-default print-access-token

# Test with Python
python src/utils/gcp_client.py
```

## Using Credentials in Python

Import the config module to access all GCP settings:

```python
from src.config import config

# Configuration is automatically loaded from .env file
print(f"Project: {config.GCP_PROJECT_ID}")
print(f"Service Account: {config.SERVICE_ACCOUNT_EMAIL}")

# Use with GCP clients
from src.utils.gcp_client import get_storage_client, verify_credentials

# Verify credentials are working
verify_credentials()

# Create authenticated clients
storage_client = get_storage_client()
```

## Environment Variables

All configuration is stored in `.env` file (see `.env` in project root):

```bash
# View current configuration
python src/config.py
```

## Troubleshooting

### "Could not automatically determine credentials"

Run: `gcloud auth application-default login`

### "Permission denied" errors

Verify you're using the correct project: `gcloud config get-value project`

### Need to use a different project

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

## Security Best Practices

✅ **DO:**
- Use Application Default Credentials for local development
- Use Workload Identity for GKE deployments
- Use default service accounts for Cloud Run/Functions
- Keep `.env` file in `.gitignore`
- Rotate credentials regularly via `gcloud auth application-default login`

❌ **DON'T:**
- Create or download service account JSON keys unless absolutely necessary
- Commit `.env` or any credential files to version control
- Share credentials via email or chat
- Use production credentials in development environments

