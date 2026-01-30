"""Mock 1Password HTTP Server

Simulates 1Password credential retrieval for development and testing.
This is NOT a real 1Password integration - it returns fake credentials.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

app = FastAPI(
  title="Mock 1Password Server",
  description="Simulates 1Password credential retrieval",
  version="0.1.0"
)


# === Response Models ===

class SecretResponse(BaseModel):
  reference: str
  value: str
  vault: str
  item: str
  field: str


class HealthResponse(BaseModel):
  status: str
  service: str


# === Mock Credential Database ===

# Simulated credentials organized by vault/item/field
# Format: op://vault/item/field
MOCK_CREDENTIALS = {
  # Infrastructure vault
  "op://Infrastructure/prod-db/password": {
    "value": "mock-prod-db-password-x7k9m2",
    "vault": "Infrastructure",
    "item": "prod-db",
    "field": "password",
  },
  "op://Infrastructure/prod-db/username": {
    "value": "sre_readonly",
    "vault": "Infrastructure",
    "item": "prod-db",
    "field": "username",
  },
  "op://Infrastructure/prod-db/host": {
    "value": "prod-db.internal.example.com",
    "vault": "Infrastructure",
    "item": "prod-db",
    "field": "host",
  },
  "op://Infrastructure/staging-db/password": {
    "value": "mock-staging-db-password-j3n8p1",
    "vault": "Infrastructure",
    "item": "staging-db",
    "field": "password",
  },
  "op://Infrastructure/staging-db/username": {
    "value": "sre_readonly",
    "vault": "Infrastructure",
    "item": "staging-db",
    "field": "username",
  },
  # AWS credentials
  "op://Infrastructure/aws-cloudwatch/access-key": {
    "value": "AKIAIOSFODNN7EXAMPLE",
    "vault": "Infrastructure",
    "item": "aws-cloudwatch",
    "field": "access-key",
  },
  "op://Infrastructure/aws-cloudwatch/secret-key": {
    "value": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "vault": "Infrastructure",
    "item": "aws-cloudwatch",
    "field": "secret-key",
  },
  "op://Infrastructure/aws-ec2/access-key": {
    "value": "AKIAI44QH8DHBEXAMPLE",
    "vault": "Infrastructure",
    "item": "aws-ec2",
    "field": "access-key",
  },
  "op://Infrastructure/aws-ec2/secret-key": {
    "value": "je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY",
    "vault": "Infrastructure",
    "item": "aws-ec2",
    "field": "secret-key",
  },
  # Kubernetes credentials
  "op://Infrastructure/k8s-prod/token": {
    "value": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.mock-k8s-token",
    "vault": "Infrastructure",
    "item": "k8s-prod",
    "field": "token",
  },
  "op://Infrastructure/k8s-prod/ca-cert": {
    "value": "-----BEGIN CERTIFICATE-----\nMOCK-CA-CERT\n-----END CERTIFICATE-----",
    "vault": "Infrastructure",
    "item": "k8s-prod",
    "field": "ca-cert",
  },
  # Monitoring credentials
  "op://Infrastructure/datadog/api-key": {
    "value": "mock-datadog-api-key-abc123",
    "vault": "Infrastructure",
    "item": "datadog",
    "field": "api-key",
  },
  "op://Infrastructure/grafana/admin-password": {
    "value": "mock-grafana-password-xyz789",
    "vault": "Infrastructure",
    "item": "grafana",
    "field": "admin-password",
  },
  # Service accounts
  "op://Services/payment-api/api-key": {
    "value": "sk_live_mock_payment_key_12345",
    "vault": "Services",
    "item": "payment-api",
    "field": "api-key",
  },
  "op://Services/email-service/smtp-password": {
    "value": "mock-smtp-password-email123",
    "vault": "Services",
    "item": "email-service",
    "field": "smtp-password",
  },
}


def parse_op_reference(reference: str) -> tuple[str, str, str] | None:
  """Parse a 1Password reference in format op://vault/item/field."""
  pattern = r'^op://([^/]+)/([^/]+)/([^/]+)$'
  match = re.match(pattern, reference)
  if match:
    return match.group(1), match.group(2), match.group(3)
  return None


@app.get("/health", response_model=HealthResponse)
async def health():
  """Health check endpoint."""
  return HealthResponse(status="healthy", service="mock-1password")


@app.get("/secrets/{reference:path}", response_model=SecretResponse)
async def get_secret(reference: str):
  """
  Retrieve a secret by its 1Password reference.

  Reference format: op://vault/item/field
  Example: op://Infrastructure/prod-db/password
  """
  # Normalize reference (ensure it starts with op://)
  if not reference.startswith("op://"):
    reference = f"op://{reference}"

  # Validate reference format
  parsed = parse_op_reference(reference)
  if not parsed:
    raise HTTPException(
      status_code=400,
      detail=f"Invalid reference format: {reference}. Expected: op://vault/item/field"
    )

  vault, item, field = parsed

  # Look up credential
  if reference in MOCK_CREDENTIALS:
    cred = MOCK_CREDENTIALS[reference]
    return SecretResponse(
      reference=reference,
      value=cred["value"],
      vault=cred["vault"],
      item=cred["item"],
      field=cred["field"],
    )

  # Not found - return 404
  raise HTTPException(
    status_code=404,
    detail=f"Secret not found: {reference}"
  )


@app.get("/vaults")
async def list_vaults():
  """List available vaults."""
  vaults = set()
  for ref in MOCK_CREDENTIALS.keys():
    parsed = parse_op_reference(ref)
    if parsed:
      vaults.add(parsed[0])
  return {"vaults": sorted(list(vaults))}


@app.get("/vaults/{vault}/items")
async def list_items(vault: str):
  """List items in a vault."""
  items = set()
  for ref in MOCK_CREDENTIALS.keys():
    parsed = parse_op_reference(ref)
    if parsed and parsed[0] == vault:
      items.add(parsed[1])

  if not items:
    raise HTTPException(status_code=404, detail=f"Vault not found: {vault}")

  return {"vault": vault, "items": sorted(list(items))}


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8080)
