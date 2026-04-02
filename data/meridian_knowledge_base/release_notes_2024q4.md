# Release Notes -- Q4 2024

## Overview

Q4 2024 includes the launch of the Real-Time Streaming add-on, OIDC support for SSO, significant dashboard performance improvements, the new async query API endpoint, and SOC 2 Type II certification renewal.

## Real-Time Streaming Add-On Launch

The Real-Time Streaming add-on is now available for Professional and Enterprise tiers. This add-on enables near-real-time data ingestion from streaming sources.

Launch features:
- **Kafka support** -- Connect to any Apache Kafka cluster including Confluent Cloud and Amazon MSK.
- **Kinesis support** -- Connect to Amazon Kinesis Data Streams.
- Sub-5-second end-to-end latency from source event to dashboard update.
- Maximum throughput of 50,000 events per second per workspace.

Google Pub/Sub support was added shortly after the initial launch and is now fully available.

## SSO: OpenID Connect (OIDC) Support

SSO now supports OpenID Connect (OIDC) alongside existing SAML 2.0. OIDC provides a simpler setup experience for identity providers that support it (e.g., Okta, Auth0, Azure AD).

This addition means Meridian now supports both major SSO protocols, giving organizations flexibility in how they integrate with their identity provider.

## Dashboard Performance Improvements

Dashboard rendering performance has been improved by 3x for dashboards with more than 10 widgets. Key optimizations include:

- Parallel widget query execution (previously sequential).
- Client-side caching for recently-rendered widgets.
- Optimized data serialization format reducing payload sizes by 40%.

Average dashboard load time for complex dashboards (10+ widgets) has dropped from 8 seconds to under 3 seconds.

## API v2.2: Async Query Endpoint

The new async query endpoint `POST /api/v2/query/async` has been introduced for long-running queries. Key features:

- No timeout limit (synchronous queries are limited to 120 seconds).
- Results are available for 24 hours after completion.
- Poll for status using `GET /api/v2/query/async/{query_id}`.
- Ideal for complex analytics, large table scans, and batch reporting queries.

## Security: SOC 2 Type II Certification Renewal

Meridian AI has successfully completed the annual SOC 2 Type II certification renewal. The audit was conducted by an independent third-party firm and covers security, availability, and confidentiality trust service criteria.

The updated audit report is available to customers under NDA. Contact your account manager or security@meridian-ai.com to request a copy.
