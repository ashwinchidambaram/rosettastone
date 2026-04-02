# Troubleshooting Data Ingestion Errors

## Overview

This guide covers the most common data ingestion errors in Meridian and how to resolve them. If you encounter an error not listed here, contact support with the error code and job ID.

## Error: INGEST_001 -- Payload Too Large

**Error message:** "Payload exceeds 100MB limit"

**Cause:** The batch ingestion request payload is larger than the maximum allowed size of 100MB.

**Resolution:**
1. Split your data into smaller batches, each under 100MB.
2. Use Parquet format to reduce payload size -- Parquet is typically 60-80% smaller than CSV or JSON for the same data.
3. Consider using the streaming ingestion endpoint for continuous data feeds instead of large batches.

## Error: INGEST_002 -- Unsupported Format

**Error message:** "Unsupported file format"

**Cause:** The uploaded file is in a format not supported by the ingestion API.

**Resolution:** The accepted formats are CSV, JSON, Parquet, and Avro. Convert your file to one of these formats before uploading. For Excel files, export as CSV or use the UI drag-and-drop upload which handles Excel conversion automatically.

## Error: INGEST_003 -- Schema Mismatch

**Error message:** "Schema mismatch -- column types in payload don't match target table"

**Cause:** The data types in your upload do not match the existing table schema. For example, a column defined as INTEGER is receiving string values.

**Resolution:**
1. Review your data to ensure column types match the target table schema.
2. If your schema is expected to evolve, enable schema evolution by setting `schema_evolution: true` in the ingestion options. This allows Meridian to automatically adapt the table schema to accommodate new columns or type changes.
3. If you need to change a column type, drop and recreate the table, or create a new table with the updated schema.

## Error: INGEST_004 -- Duplicate Key Violation

**Error message:** "Duplicate key violation"

**Cause:** Records in your upload have `_meridian_id` values that match existing records in the target table, and deduplication is enabled.

**Resolution:**
- If duplicates are expected and you want to overwrite existing records, set `upsert_mode: true` in the ingestion options.
- If you want to allow duplicate records, set `deduplicate: false` in the ingestion options.
- If duplicates are not expected, review your data for unintended duplicates before re-uploading.

## Ingestion Job Stuck in "Pending"

**Symptom:** An ingestion job has been in "pending" status for more than 30 minutes.

**Resolution:**
1. Check connector health in Settings > Connectors > Status. If the connector shows as "Unhealthy" or "Disconnected," reconnect it.
2. Verify that your workspace has not exceeded its row limit. Ingestion jobs may be queued if you are at or near the row limit.
3. Check the platform status page at status.meridian-ai.com for any ongoing maintenance or incidents.
4. If the issue persists, contact support with the job ID from the ingestion response.

## Best Practices

- Always monitor ingestion jobs using the `GET /api/v2/ingest/jobs/{job_id}` endpoint.
- Enable schema evolution for tables where the schema may change over time.
- Use Parquet format for large data loads to maximize throughput and minimize transfer size.
- Test ingestion with a small sample before running large batch jobs.
