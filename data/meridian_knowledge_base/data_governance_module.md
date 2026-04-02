# Data Governance Add-On -- Access Controls & Lineage

## Overview

The Meridian Data Governance module provides enterprise-grade data management capabilities including column-level access controls, data lineage visualization, audit logging, and PII auto-detection. It is designed for organizations with strict data access and compliance requirements.

## Pricing and Availability

The Data Governance module is an add-on:
- **Professional tier:** $300/month
- **Enterprise tier:** Included at no additional cost

The module is not available on the Starter tier.

## Column-Level Access Controls

Column-level access controls allow you to restrict which users or groups can see specific columns in your data. This is essential for protecting sensitive data like salaries, Social Security numbers, or customer financial details.

To configure column-level access:
1. Navigate to Data Governance > Access Controls.
2. Select the table and column you want to restrict.
3. Choose which user groups have access to view the column.
4. Users without access will see the column masked (e.g., "****") in queries and dashboards.

## Data Lineage Visualization

The data lineage feature provides a visual trace of data from source connector through transformations to dashboard widgets. This helps you understand where your data comes from, how it has been transformed, and which dashboards depend on which data sources.

Lineage is automatically tracked for all data that flows through MAP. You can view the lineage graph for any table, column, or dashboard widget.

## Audit Log Retention

The Data Governance module includes comprehensive audit logging:

- **Professional tier:** Audit log retention for 1 year
- **Enterprise tier:** Audit log retention for 7 years

Audit logs capture all data access events, configuration changes, user logins, and API calls. Logs can be exported for external compliance auditing.

## PII Auto-Detection

The PII auto-detection feature scans ingested data for patterns that indicate personally identifiable information. Detection patterns include:

- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- Passport numbers
- Driver's license numbers
- And 12 additional patterns added in Q1 2025

When PII is detected, Meridian can automatically mask the data in queries and dashboards. PII detection runs on every data ingestion job and can be configured to block ingestion if PII is found in unexpected columns.
