# Query API -- Running Analytics Queries

## Overview

The Meridian Query API allows you to run analytics queries against your workspace data programmatically. Queries can be executed synchronously for fast results or asynchronously for long-running analytics.

## Synchronous Query

### Endpoint

`POST /api/v2/query`

The synchronous query endpoint executes a query and returns results directly in the response. The maximum execution time for synchronous queries is 120 seconds. If a query exceeds this limit, it will be terminated and you should use the async query endpoint instead.

### Request Format

```json
{
  "sql": "SELECT customer_name, total_spend FROM customers WHERE region = 'EMEA' ORDER BY total_spend DESC LIMIT 100",
  "workspace_id": "ws_abc123"
}
```

## Asynchronous Query

### Endpoint

`POST /api/v2/query/async`

The async query endpoint is designed for long-running queries that may exceed the 120-second synchronous timeout. When you submit an async query, the API returns a query ID immediately. Results are available for 24 hours after the query completes.

### Polling for Results

Use `GET /api/v2/query/async/{query_id}` to check the status and retrieve results when ready.

## Pagination

Query results are paginated at 10,000 rows per page by default. This is configurable up to 100,000 rows per page using the `page_size` parameter. Use the `page_token` from the response to fetch subsequent pages.

## SQL Dialect

The Meridian Query API supports ANSI SQL with Meridian-specific extensions:

- **`EXPLAIN MERIDIAN`** -- Returns the query execution plan with cost estimates in Meridian Compute Units (MCU).
- **`SAMPLE(n)`** -- Returns a random sample of n rows from a table, useful for data exploration.

Standard ANSI SQL features are fully supported including JOINs, CTEs (WITH clauses), window functions, and aggregations.

## Query Cost

Query cost is measured in Meridian Compute Units (MCU). The cost of each query is included in the response headers as `X-MCU-Cost`. You can use `EXPLAIN MERIDIAN` to estimate the cost before executing a query.

MCU consumption counts toward your workspace's monthly compute budget. Enterprise customers can configure custom MCU limits and alerts.

## Best Practices

- Use the async endpoint for queries expected to take more than 30 seconds.
- Add `LIMIT` clauses to exploratory queries to avoid scanning entire tables.
- Use `EXPLAIN MERIDIAN` to estimate cost before running expensive queries.
- Leverage materialized views for frequently-run expensive queries.
