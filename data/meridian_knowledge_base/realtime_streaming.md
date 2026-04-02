# Real-Time Streaming Add-On

## Overview

The Meridian Real-Time Streaming add-on enables near-real-time data ingestion and dashboard updates from streaming data sources. Instead of waiting for hourly or daily syncs, streaming data flows into MAP continuously with sub-second latency.

## Pricing and Availability

The Real-Time Streaming add-on is priced at $750/month and is available for Professional and Enterprise tiers. It is not available on the Starter plan.

## Supported Streaming Sources

The Real-Time Streaming add-on supports three streaming platforms:

- **Apache Kafka** -- Connect to any Kafka cluster (including Confluent Cloud, Amazon MSK, and self-managed)
- **Amazon Kinesis** -- Connect to Kinesis Data Streams
- **Google Pub/Sub** -- Connect to Google Cloud Pub/Sub topics

## Latency

The end-to-end latency from source event to dashboard update is sub-5-second. This means that when an event is published to your streaming source, it will be reflected in Meridian dashboards within 5 seconds under normal conditions.

## Throughput

The maximum throughput is 50,000 events per second per workspace. This throughput is sufficient for most enterprise use cases. If you need higher throughput, contact your account manager to discuss dedicated streaming infrastructure.

## Streaming Ingestion Endpoint

Real-time streaming uses the streaming ingestion endpoint: `POST /api/v2/ingest/stream`. This endpoint is separate from the batch ingestion endpoint and is optimized for low-latency, high-frequency data writes.

## Setup

To configure real-time streaming:
1. Ensure the Real-Time Streaming add-on is enabled on your workspace.
2. Navigate to Settings > Data Sources > Add New > Streaming.
3. Select your streaming platform (Kafka, Kinesis, or Pub/Sub).
4. Enter the connection details (broker endpoints, topic names, authentication credentials).
5. Configure the target table and schema mapping.
6. Start the streaming connector.

## Monitoring

Streaming connectors can be monitored in Settings > Data Sources > [Connector] > Streaming Health. The monitoring dashboard shows:
- Events processed per second
- Latency (p50, p95, p99)
- Error rates and dropped events
- Consumer lag (for Kafka)
