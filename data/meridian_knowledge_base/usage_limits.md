# Usage Limits & Overage Charges

## Overview

Each Meridian pricing tier has defined usage limits. Understanding these limits helps you choose the right plan and avoid unexpected overage charges.

## Limits by Tier

### Starter
- **Rows:** 10 million rows
- **Data Source Connectors:** 5 connectors
- **Dashboards:** 10 dashboards
- **Scheduled Reports:** 10 scheduled reports
- **API Requests:** 100 requests/minute

### Professional
- **Rows:** 1 billion rows
- **Data Source Connectors:** Unlimited
- **Dashboards:** Unlimited
- **Scheduled Reports:** 50 scheduled reports
- **API Requests:** 500 requests/minute

### Enterprise
- **Rows:** Unlimited
- **Data Source Connectors:** Unlimited
- **Dashboards:** Unlimited
- **Scheduled Reports:** Unlimited
- **API Requests:** Custom rate limits (default 2,000 requests/minute)

## How Row Count Is Measured

Row count is measured as the total rows across all tables in the workspace. This includes data from all connected sources, uploaded files, and streaming ingestion. Deleted rows are excluded from the count after garbage collection runs (typically within 24 hours of deletion).

## Approaching Limit Warning

When your workspace reaches 80% of its row limit, an automatic email warning is sent to all workspace admins. This gives you time to either clean up old data or upgrade your plan before hitting the limit.

## What Happens When You Hit the Limit

When your workspace reaches its row limit:
- New data ingestion jobs are blocked until you are under the limit.
- Existing data remains accessible for queries, dashboards, and reports.
- You will receive email notifications prompting you to upgrade or reduce usage.

## Overage Charges

If overage is enabled on your workspace (opt-in setting):
- Data ingestion continues beyond the limit.
- You are charged $0.10 per 1,000 rows over the plan limit.
- Overage is billed monthly in arrears on the following month's invoice.

Overage can be enabled in Settings > Billing > Overage Settings.
