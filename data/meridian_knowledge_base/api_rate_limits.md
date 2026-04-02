# API Rate Limits & Quotas

## Overview

API rate limits are enforced per workspace and vary by pricing tier. Rate limits ensure fair usage and platform stability for all customers.

## Rate Limits by Tier

### Starter
- **Per-minute limit:** 100 requests/minute
- **Daily limit:** 10,000 requests/day

### Professional
- **Per-minute limit:** 500 requests/minute
- **Daily limit:** 100,000 requests/day

### Enterprise
- **Per-minute limit:** 2,000 requests/minute
- **Daily limit:** Unlimited

## Rate Limit Headers

Every API response includes rate limit headers so you can monitor your usage:

- **`X-RateLimit-Remaining`** -- The number of requests remaining in the current minute window.
- **`X-RateLimit-Reset`** -- The Unix timestamp when the current rate limit window resets.

## Burst Allowance

All tiers receive a burst allowance of 2x the per-minute limit for 10-second windows. This means:

- Starter can burst up to 200 requests in a 10-second window
- Professional can burst up to 1,000 requests in a 10-second window
- Enterprise can burst up to 4,000 requests in a 10-second window

Burst allowance helps accommodate short traffic spikes without hitting rate limits.

## Rate Limit Exceeded

When you exceed the rate limit, the API returns a `429 Too Many Requests` response. The response includes a `Retry-After` header indicating how many seconds to wait before retrying.

We recommend implementing exponential backoff in your API clients to handle rate limit errors gracefully.
