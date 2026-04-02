# Dashboard Builder -- Creating & Sharing Dashboards

## Overview

The Meridian Dashboard Builder provides a drag-and-drop interface for creating interactive, data-driven dashboards. Build visualizations, share them with your team, and embed them in external applications.

## Widget Types

The Dashboard Builder includes a drag-and-drop widget builder with 15 chart types:

- Bar chart (vertical and horizontal)
- Line chart
- Area chart
- Pie chart
- Donut chart
- Scatter plot
- Heatmap
- Treemap
- Table
- KPI card
- Gauge
- Funnel chart
- Histogram
- Combo chart (bar + line)
- Map (geographic)

Each widget can be configured with custom colors, labels, axes, and data source queries.

## Sharing and Embedding

Dashboards can be shared via link (view-only) or embedded via iframe. Link sharing allows anyone with the link to view the dashboard without logging in. Iframe embedding is available on Professional and Enterprise tiers and allows you to embed dashboards directly in your internal applications, wikis, or portals.

To share a dashboard:
1. Open the dashboard and click "Share" in the top right.
2. Choose "Share Link" for a view-only link, or "Embed" for an iframe code snippet.
3. Configure access controls (public, workspace-only, or password-protected).

## Auto-Refresh

Dashboards support automatic refresh at configurable intervals:

- **1 minute** -- For real-time monitoring dashboards
- **5 minutes** -- For frequently updated operational dashboards
- **15 minutes** -- For standard business dashboards
- **1 hour** -- For executive summary dashboards
- **Manual** -- Refresh only when the user clicks the refresh button

Auto-refresh intervals can be set per dashboard in the dashboard settings.

## Cross-Widget Filtering

Dashboard filters support cross-widget filtering with a single click. When you click on a data point in one widget (for example, a specific region in a bar chart), all other widgets on the dashboard automatically filter to that selection. This enables interactive data exploration without configuring each widget separately.

## Export Options

Dashboards can be exported in multiple formats:

- **PDF** -- Full dashboard export with all widgets rendered as static images
- **PNG** -- Screenshot of the dashboard as an image
- **CSV** -- Data-only export of the underlying data for each widget

Export is available from the dashboard menu under "Export."
