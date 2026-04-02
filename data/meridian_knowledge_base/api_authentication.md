# API Authentication & API Keys

## Overview

The Meridian API uses API key-based authentication for all requests. API keys are generated in Settings > API Keys within the Meridian platform. Each key has a configurable scope that controls what actions the key can perform.

## API Key Scopes

API keys can be configured with one of three scope levels:

- **Read-only:** Can query data and read configurations, but cannot modify anything.
- **Read-write:** Can query data, create/update dashboards, run ingestion jobs, and modify most settings.
- **Admin:** Full access including user management, billing, and API key management.

## Authentication Method

Authentication uses a Bearer token in the `Authorization` header of every API request:

```
Authorization: Bearer mk_live_abc123def456
```

All API requests must be made over HTTPS. Requests over plain HTTP will be rejected.

## API Key Rotation

API keys can be rotated without downtime using the key rotation endpoint `POST /api/v2/keys/rotate`. When you rotate a key:

1. Call `POST /api/v2/keys/rotate` with your current key in the Authorization header.
2. A new key is generated and returned in the response.
3. The old key remains valid for a 24-hour grace period.
4. After the grace period, the old key is automatically deactivated.

This allows you to update your applications without any service interruption.

## API Key Expiration

API keys expire after 90 days by default. The expiration period is configurable and can be set to 30, 60, 90, or 365 days. You can configure the expiration period when creating a key or update it later in Settings > API Keys.

Expiring keys trigger email notifications at 14 days, 7 days, and 1 day before expiration.

## OAuth 2.0 Support

OAuth 2.0 client credentials flow is supported for Enterprise tier only. This is recommended for machine-to-machine integrations where you need automated token refresh without manual key management.

To set up OAuth 2.0:
1. Navigate to Settings > API Keys > OAuth Applications.
2. Register a new OAuth application.
3. Use the client ID and client secret to request tokens from the `/oauth/token` endpoint.

## Rate Limits

API key authentication is subject to rate limits based on your pricing tier. See the API Rate Limits & Quotas documentation for details.
