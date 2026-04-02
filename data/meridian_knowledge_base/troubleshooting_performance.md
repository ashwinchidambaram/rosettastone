# Dashboard Performance & Query Optimization

## Overview

This guide covers common performance issues in Meridian dashboards and queries, along with recommended solutions.

## Slow Dashboard Load Times

**Symptom:** Dashboard takes more than 10 seconds to load.

**Possible causes and solutions:**

1. **Unindexed columns:** Check if your dashboard widgets use columns that are not indexed. Add indexes via the Schema Manager (navigate to Data Governance > Schema Manager > Indexes). Indexing frequently-filtered columns can dramatically reduce query times.

2. **Too many widgets:** Dashboards with many widgets (10+) execute multiple queries simultaneously. Consider splitting large dashboards into multiple focused dashboards, or reduce the auto-refresh frequency.

3. **Large result sets:** Widgets that display thousands of rows are slow to render. Add filters or use aggregations to reduce the data volume displayed.

## Query Timeout

**Symptom:** Queries fail with a timeout error after 120 seconds.

**Resolution:**
1. Break complex queries into smaller CTEs (Common Table Expressions) to reduce execution time.
2. Use the async query endpoint (`POST /api/v2/query/async`) for long-running queries that cannot be simplified. Async queries have no timeout limit.
3. Add `LIMIT` clauses to exploratory queries.
4. Consider creating materialized views for frequently-run expensive queries.

## High MCU Consumption

**Symptom:** Your workspace is consuming more Meridian Compute Units (MCU) than expected.

**Resolution:**
1. Check the Query Audit Log for expensive queries. Navigate to Settings > Query Audit Log to see MCU cost per query.
2. Identify and optimize the most expensive queries using `EXPLAIN MERIDIAN`.
3. Consider creating materialized views for queries that are run frequently and scan large amounts of data.
4. Reduce auto-refresh frequency on dashboards to decrease the number of query executions.

## Slow Connector Sync

**Symptom:** Data source syncs are taking longer than expected.

**Resolution:**
1. Verify network allowlisting of Meridian IP ranges. The list of Meridian IP ranges that need to be allowlisted is available in Settings > Network.
2. Check if the source system is under heavy load, which can slow down data extraction.
3. Consider switching to incremental sync mode if full syncs are taking too long.
4. For database connectors, ensure proper indexes exist on the source tables.

## AskMeridian Low Confidence Results

**Symptom:** AskMeridian frequently returns "Low confidence" scores.

**Resolution:**
1. Re-run schema indexing via Settings > AskMeridian > Re-index. This refreshes the LLM's understanding of your workspace schema.
2. Ensure your tables and columns have descriptive names. AskMeridian performs better with clear naming conventions.
3. Add column descriptions in the Schema Manager to provide additional context to the NLQ engine.
