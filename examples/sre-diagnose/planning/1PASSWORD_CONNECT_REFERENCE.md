# 1Password Integration Reference

This document describes the 1Password integration options for the SRE Diagnose app.

## Overview

The SRE Diagnose app supports two modes for credential retrieval:

1. **Mock Mode** (default): Uses a local mock server for development and testing
2. **SDK Mode**: Uses the official 1Password Python SDK with a service account for production

## SDK Integration (Recommended for Production)

### Architecture

The SDK integration uses 1Password's official Python SDK (`onepassword-sdk`) which:

- Authenticates using a Service Account Token
- Resolves secret references directly via 1Password's API
- Requires no additional containers or infrastructure
- Caches credentials appropriately for performance

### Authentication

The SDK uses a **Service Account Token** for authentication:

```python
from onepassword import Client

client = await Client.authenticate(
  auth=os.environ.get("OP_SERVICE_ACCOUNT_TOKEN"),
  integration_name="sre-diagnose",
  integration_version="0.5.0"
)
```

### Secret Resolution

Secret references use the standard `op://` URI format:

```python
value = await client.secrets.resolve("op://vault/item/field")
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ONEPASSWORD_MODE` | No | Set to `sdk` for production (default: `mock`) |
| `OP_SERVICE_ACCOUNT_TOKEN` | Yes (sdk mode) | 1Password service account token |

### Setup Instructions

1. **Create a Service Account**
   - Go to https://my.1password.com
   - Navigate to **Developer** > **Directory** > **Infrastructure Secrets Management**
   - Select **Create Service Account**
   - Grant access to vaults containing infrastructure credentials

2. **Configure the App**
   
   Create `secrets.yaml`:
   ```yaml
   OP_SERVICE_ACCOUNT_TOKEN: "your-token-here"
   ```

   Update `autonomy.yaml`:
   ```yaml
   containers:
     - name: main
       image: main
       env:
         - ONEPASSWORD_MODE: "sdk"
         - OP_SERVICE_ACCOUNT_TOKEN: secrets.OP_SERVICE_ACCOUNT_TOKEN
   ```

3. **Deploy**
   ```bash
   autonomy zone deploy
   ```

## Mock Mode (Development)

### Architecture

The mock mode uses a simple FastAPI server that returns fake credentials. This allows development and testing without real 1Password access.

### Mock Server Endpoints

```
GET /health              - Health check
GET /secrets/{reference} - Get a secret by op:// reference
GET /vaults              - List available vaults
GET /vaults/{vault}/items - List items in a vault
```

### Mock Credentials

The mock server provides these pre-configured credentials:

| Reference | Value |
|-----------|-------|
| `op://Infrastructure/prod-db/password` | `mock-prod-db-password-x7k9m2` |
| `op://Infrastructure/prod-db/username` | `sre_readonly` |
| `op://Infrastructure/prod-db/host` | `prod-db.internal.example.com` |
| `op://Infrastructure/staging-db/password` | `mock-staging-db-password-j3n8p1` |
| `op://Infrastructure/aws-cloudwatch/access-key` | `AKIAIOSFODNN7EXAMPLE` |
| `op://Infrastructure/aws-cloudwatch/secret-key` | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `op://Infrastructure/k8s-prod/token` | `eyJhbGciOiJSUzI1NiIs...` |
| `op://Infrastructure/datadog/api-key` | `mock-datadog-api-key-abc123` |

## Secret Reference Format

Both modes use the standard 1Password secret reference format:

```
op://vault/item/field
```

Components:
- **vault**: The 1Password vault name or ID
- **item**: The item name or ID within the vault
- **field**: The field name within the item

Examples:
- `op://Infrastructure/prod-db/password`
- `op://Services/api-gateway/api-key`
- `op://DevOps/k8s-cluster/ca-certificate`

## Implementation Details

### Credential Retrieval Function

The main entry point is `retrieve_credential()` which:

1. Normalizes the reference (ensures `op://` prefix)
2. Routes to either SDK or mock implementation based on `ONEPASSWORD_MODE`
3. Implements retry logic (3 attempts with exponential backoff)
4. Stores retrieved credentials in the session dict

```python
async def retrieve_credential(reference: str, session: dict) -> tuple[bool, str]:
    """
    Retrieve a credential from 1Password.
    Returns (success, message) tuple and stores credential in session.
    """
    if ONEPASSWORD_MODE == "sdk":
        return await retrieve_credential_sdk(reference, session)
    else:
        return await retrieve_credential_mock(reference, session)
```

### Error Handling

Both modes implement:
- Retry logic with exponential backoff
- Proper error messages for debugging
- Graceful degradation on failure

### Security Considerations

1. **Credentials are never exposed to the LLM** - stored in session dict, only passed to diagnostic tools
2. **Human-in-the-loop approval** - user must approve credential access before retrieval
3. **Token security** - service account token stored in `secrets.yaml` (gitignored)
4. **Minimal scope** - service accounts should only have access to required vaults

## Alternatives Considered

### 1Password Connect Server

The Connect Server approach was considered but not implemented because:
- Requires two additional containers (`connect-api` and `connect-sync`)
- Needs `1password-credentials.json` file mounted
- More complex deployment and configuration
- SDK approach is simpler and officially supported

### 1Password CLI

The CLI approach was considered but not implemented because:
- Requires `op` binary installed in container
- Shell-based execution is less reliable
- SDK provides native Python integration

## Official Documentation

- [1Password SDKs](https://developer.1password.com/docs/sdks/)
- [Service Accounts](https://developer.1password.com/docs/service-accounts/)
- [Secret References](https://developer.1password.com/docs/cli/secret-references/)
- [Load Secrets with SDKs](https://developer.1password.com/docs/sdks/load-secrets/)

## Troubleshooting

### SDK Mode Issues

**"OP_SERVICE_ACCOUNT_TOKEN environment variable is required"**
- Ensure `ONEPASSWORD_MODE=sdk` and token is set
- Check `secrets.yaml` exists and has correct format
- Verify `autonomy.yaml` references the secret correctly

**"Secret not found"**
- Verify the vault/item/field path is correct
- Ensure service account has access to the vault
- Check the reference format: `op://vault/item/field`

**"onepassword-sdk not installed"**
- Ensure `requirements.txt` includes `onepassword-sdk`
- Rebuild the container image

### Mock Mode Issues

**"Connection refused" to localhost:8080**
- Verify the `onepass` container is running
- Check container logs for errors
- Ensure `autonomy.yaml` includes the mock-1password container

**"Secret not found" in mock mode**
- Add the credential to `MOCK_CREDENTIALS` in `server.py`
- Verify the reference format matches exactly