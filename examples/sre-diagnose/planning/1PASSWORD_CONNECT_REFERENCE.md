# 1Password Connect API Reference

This document captures information about 1Password Connect for future integration.

## Overview

1Password Connect Servers allow secure access to 1Password items and vaults via a private REST API. They're self-hosted in your infrastructure, providing:

- **Reduced latency**: Self-hosted means faster access
- **High availability**: Deploy redundant servers
- **Security**: Only your services can interact with Connect
- **Unlimited re-requests**: Data is cached locally after initial fetch

## Use Cases

1. **Reduce latency and downtime** - Self-hosted, redundant deployment
2. **Provision web services with secrets** - Provide database credentials to services
3. **Automate secrets management** - Scripts can manage secrets programmatically
4. **Load secrets into CI/CD pipelines** - Securely access secrets during builds
5. **Secure infrastructure secrets** - Not tied to personal accounts
6. **Streamline development workflows** - Securely share infrastructure secrets
7. **Secure Kubernetes environments** - Sync 1Password secrets to K8s

## SDK Libraries

Official SDKs available for:
- Go
- Python
- JavaScript

## API Endpoints (Expected)

Based on the mock server we created, the real API likely has similar endpoints:

### Health Check
```
GET /health
```

### Get Secret by Reference
```
GET /v1/vaults/{vault_id}/items/{item_id}
```

### List Vaults
```
GET /v1/vaults
```

### List Items in Vault
```
GET /v1/vaults/{vault_id}/items
```

## Secret Reference Format

1Password uses the `op://` URI scheme:
```
op://vault/item/field
```

Examples:
- `op://Infrastructure/prod-db/password`
- `op://Services/api-gateway/api-key`

## Authentication

Connect servers use bearer token authentication:
```
Authorization: Bearer <connect-token>
```

Tokens are created when setting up the Connect server in 1Password.

## Deployment

Connect servers run as Docker containers. Typical deployment:

```yaml
services:
  op-connect-api:
    image: 1password/connect-api:latest
    ports:
      - "8080:8080"
    volumes:
      - ./1password-credentials.json:/home/opuser/.op/1password-credentials.json
```

## Environment Variables

- `OP_CONNECT_HOST` - Connect server URL
- `OP_CONNECT_TOKEN` - Authentication token

## Future Integration Steps

1. Set up 1Password Connect server in infrastructure
2. Create service account and connect token
3. Replace mock server with real Connect API calls
4. Store token securely (not in code)
5. Update Python client to use official 1Password Python SDK

## Official Documentation

- Main docs: https://developer.1password.com/docs/connect/
- Getting started: https://developer.1password.com/docs/connect/get-started/
- API reference: https://developer.1password.com/docs/connect/connect-api-reference/

## Notes for Real Implementation

The mock server (`images/mock-1password/server.py`) simulates the credential retrieval flow.
When integrating with real 1Password:

1. Install the official SDK: `pip install onepassword-sdk`
2. Use the Connect client instead of direct HTTP calls
3. Handle authentication properly
4. Implement proper error handling for rate limits
5. Consider caching strategies for frequently accessed secrets