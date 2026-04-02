# Release Notes -- Q1 2025

## Overview

Q1 2025 brings major upgrades to AskMeridian, a new Databricks connector, enhanced dashboard visualizations, a new API endpoint, and expanded PII detection capabilities.

## AskMeridian v2

We are excited to announce AskMeridian v2, a major upgrade to our natural language query interface. Key improvements include:

- **40% improvement in query accuracy** compared to v1, powered by a new fine-tuned LLM with enhanced schema understanding.
- **Japanese language support added** -- AskMeridian now supports English, Spanish, French, German, and Japanese.
- Improved handling of ambiguous questions with clarification prompts.
- Faster query generation time (average 1.2 seconds, down from 2.5 seconds in v1).

## New Connector: Databricks Unity Catalog

The Databricks Unity Catalog connector is now generally available (GA). This connector provides native integration with Databricks Unity Catalog, allowing you to access tables, views, and volumes across your Databricks workspace.

Key features:
- Schema discovery across Unity Catalog namespaces
- Incremental sync support
- Compatible with Databricks SQL Warehouses and All-Purpose Clusters

## Dashboard Builder Enhancements

Two new chart types have been added to the Dashboard Builder:

- **Heatmap** -- Visualize density and patterns across two dimensions. Ideal for correlation analysis and time-based patterns.
- **Treemap** -- Display hierarchical data as nested rectangles. Useful for showing proportional breakdowns (e.g., revenue by region and product).

These chart types are available on all tiers.

## API v2.3

### New Endpoint: Query Cost Estimation

A new `POST /api/v2/query/explain` endpoint has been added for query cost estimation before execution. This endpoint returns the estimated MCU cost and execution plan for a query without actually running it. Use this to preview the cost of expensive queries before committing compute resources.

### Request Format

```json
{
  "sql": "SELECT * FROM large_table WHERE region = 'EMEA'",
  "workspace_id": "ws_abc123"
}
```

## Data Governance: Enhanced PII Detection

PII auto-detection now supports 12 additional patterns beyond the original set. New patterns include:

- Passport numbers (US, UK, EU formats)
- Driver's license numbers (US state formats)
- National ID numbers (various countries)
- Bank account numbers (IBAN format)
- Vehicle identification numbers (VIN)
- Healthcare member IDs

These patterns are automatically applied to all new data ingestion jobs. Existing data can be re-scanned via Data Governance > PII Detection > Full Scan.
