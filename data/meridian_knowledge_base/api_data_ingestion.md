# Data Ingestion API -- Endpoints & Examples

## Overview

The Meridian Data Ingestion API allows you to programmatically load data into your workspace. There are two primary ingestion methods: batch ingestion for bulk data loads, and streaming ingestion for real-time data feeds.

## Batch Ingestion

### Endpoint

`POST /api/v2/ingest/batch`

The batch ingestion endpoint accepts up to 100MB payload per request. For larger datasets, split your data into multiple requests.

### Request Format

```json
{
  "target_table": "my_table",
  "format": "json",
  "data": [...],
  "options": {
    "deduplicate": true,
    "schema_evolution": false
  }
}
```

### Supported Formats

The ingestion API supports the following data formats:
- **CSV** -- Comma-separated values with automatic header detection
- **JSON** -- Array of objects or newline-delimited JSON (NDJSON)
- **Parquet** -- Apache Parquet columnar format
- **Avro** -- Apache Avro serialization format

### Deduplication

Deduplication is enabled by default using the `_meridian_id` field. When deduplication is active, records with duplicate `_meridian_id` values are silently dropped. You can disable deduplication by setting `deduplicate: false` in the request options.

If your data does not include a `_meridian_id` field, Meridian will generate one automatically based on a hash of the record contents.

## Streaming Ingestion

### Endpoint

`POST /api/v2/ingest/stream`

The streaming ingestion endpoint requires the Real-Time Streaming add-on. It accepts individual events or small batches for near-real-time data ingestion.

Streaming ingestion supports the same data formats as batch ingestion (CSV, JSON, Parquet, Avro) and uses the same authentication mechanism.

## Async Job Processing

Ingestion jobs are asynchronous. When you submit a batch ingestion request, the API returns a job ID immediately. You can poll the job status using:

`GET /api/v2/ingest/jobs/{job_id}`

The job status endpoint returns the current state of the ingestion:
- `pending` -- Job is queued and waiting to be processed
- `running` -- Job is currently being processed
- `completed` -- Job finished successfully with a summary of rows ingested
- `failed` -- Job failed with error details

## Upsert Mode

For updates to existing records, enable upsert mode by setting `upsert_mode: true` in the options. Upsert mode uses the `_meridian_id` field to match existing records and update them with new values. Fields not included in the update payload are preserved.

## Best Practices

- Keep batch sizes under 50MB for optimal throughput.
- Use Parquet format for large datasets -- it is the most efficient format for both transfer and ingestion.
- Enable `schema_evolution: true` if your schema may change over time to avoid INGEST_003 errors.
- Monitor ingestion jobs via the jobs endpoint rather than assuming immediate completion.
