# Automated Reports & Scheduled Exports

## Overview

Meridian's automated reporting feature allows you to schedule regular delivery of data reports to your team via email. Reports can include dashboard widgets, custom queries, and conditional alerts.

## Scheduling

Reports can be scheduled on the following frequencies:
- **Daily** -- Delivered every day at a specified time
- **Weekly** -- Delivered on a chosen day of the week
- **Monthly** -- Delivered on a chosen date each month

All schedule times are configured in the workspace's time zone.

## Delivery Format

Reports are delivered via email with attachments in your chosen format:
- **PDF** -- Formatted report with visualizations rendered as static images
- **CSV** -- Raw data export as a comma-separated values file

Each report email includes a summary of key metrics in the email body and the full report as an attachment.

## Report Templates

Custom report templates can be created using the Meridian Report Language (MRL) -- a YAML-based DSL (domain-specific language). MRL allows you to define:

- Which data sources and queries to include
- Layout and formatting options
- Conditional sections (show/hide based on data values)
- Custom headers and footers

## Report Limits

- **Professional tier:** Maximum 50 scheduled reports per workspace
- **Enterprise tier:** Unlimited scheduled reports

Reports can include up to 20 widgets per report. If you need more widgets in a single report, consider splitting it into multiple reports.

## Conditional Alerts

Reports support conditional alerts that trigger delivery only when a metric crosses a defined threshold. For example, you can configure a report to be sent only when:

- Revenue drops below a target value
- Error rates exceed a specified percentage
- Inventory levels fall below a minimum threshold

Conditional alerts are configured in the report settings under "Delivery Conditions."

## Managing Reports

To create a new scheduled report:
1. Navigate to Reports > Scheduled > Create New.
2. Select the widgets or queries to include.
3. Choose the delivery frequency and time.
4. Add recipient email addresses.
5. Optionally configure conditional alerts.
6. Save and activate the report.

Existing reports can be edited, paused, or deleted from the Reports > Scheduled page.
