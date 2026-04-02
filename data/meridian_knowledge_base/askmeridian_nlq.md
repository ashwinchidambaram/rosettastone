# AskMeridian Natural Language Query Interface

## Overview

AskMeridian is Meridian's natural language query (NLQ) feature that allows users to ask questions about their data in plain language. Instead of writing SQL, users type questions like "What were our top 10 products by revenue last quarter?" and AskMeridian generates and executes the appropriate SQL query against the connected data.

## How It Works

Users type natural language questions into the AskMeridian interface. The system generates and executes SQL against the connected data sources in your workspace. Results are displayed as tables or auto-generated visualizations.

AskMeridian uses a fine-tuned LLM that is aware of your workspace schema. When you connect data sources and run schema indexing, AskMeridian learns the table names, column names, data types, and relationships in your data. This enables it to generate accurate SQL for your specific dataset.

## Confidence Scores

A confidence score is displayed with each answer. Confidence levels are categorized as:

- **High** -- The system is confident in the generated query and results.
- **Medium** -- The system has moderate confidence; review the generated SQL.
- **Low** -- The system is uncertain; the query may not accurately answer your question.

Users can flag incorrect results by clicking the "Flag" button. Flagged results help improve AskMeridian's accuracy over time.

## Query History

The history of all NLQ queries is stored for 90 days on the Professional tier. Enterprise customers get unlimited query history retention. You can revisit previous queries, re-run them, or share them with colleagues.

## Tier Availability

AskMeridian is available on Professional and Enterprise tiers only. The feature is not included in the Starter plan.

## Tips for Best Results

- Be specific in your questions. "What was revenue in Q4 2024?" is better than "How are things going?"
- Use column names or business terms that match your data schema when possible.
- If you get low-confidence results frequently, re-run schema indexing via Settings > AskMeridian > Re-index to ensure the LLM has up-to-date schema information.
