# Connecting Data Sources -- Supported Connectors

## Overview

Meridian Analytics Platform supports a wide range of data source connectors to bring your data into MAP. Connectors are organized into three categories: Databases, SaaS Applications, and File-based sources.

## Database Connectors

MAP supports the following database connectors:

- **PostgreSQL** -- Connect to PostgreSQL 10+ instances
- **MySQL** -- Connect to MySQL 5.7+ and MariaDB 10.2+
- **SQL Server** -- Connect to SQL Server 2016+ (including Azure SQL Database)
- **Oracle** -- Connect to Oracle Database 12c+
- **Snowflake** -- Native connector with one-click OAuth
- **BigQuery** -- Native connector with one-click OAuth
- **Redshift** -- Connect to Amazon Redshift clusters
- **Databricks** -- Connect to Databricks Unity Catalog (GA as of Q1 2025)

## SaaS Application Connectors

MAP integrates with popular SaaS applications:

- **Salesforce** -- One-click OAuth, syncs objects like Accounts, Contacts, Opportunities
- **HubSpot** -- Syncs CRM, marketing, and sales data
- **Stripe** -- Syncs payments, subscriptions, and customer data
- **Zendesk** -- Syncs tickets, users, and satisfaction ratings
- **Jira** -- Syncs issues, sprints, and project data
- **Google Analytics** -- Syncs web analytics data via GA4 API
- **Shopify** -- Syncs orders, products, and customer data

## File-Based Sources

You can upload data directly via the platform UI or API:

- **CSV** -- Drag-and-drop upload in the UI or upload via API
- **Excel** -- Supports .xlsx files with multiple sheets
- **Parquet** -- Apache Parquet columnar format for efficient data transfer
- **JSON** -- JSON arrays or newline-delimited JSON (NDJSON)

File uploads support drag-and-drop in the UI and are also available programmatically via the Data Ingestion API.

## Custom Connectors

Custom connectors can be built using the Connector SDK. The Connector SDK is available on the Enterprise tier only. It allows you to build custom connectors for proprietary data sources or internal systems that are not covered by the built-in connectors.

The Connector SDK provides a Python-based framework with hooks for authentication, schema discovery, and incremental data sync.

## Sync Frequency

Data sync frequency options depend on your setup:

- **Real-time** -- Available with the Real-Time Streaming add-on (Kafka, Kinesis, Pub/Sub)
- **Hourly** -- Syncs data every hour on the hour
- **Daily** -- Syncs data once per day at a configurable time
- **Manual** -- Sync only when manually triggered by a user or API call

Sync frequency can be configured per connector in Settings > Data Sources > [Connector Name] > Sync Schedule.
