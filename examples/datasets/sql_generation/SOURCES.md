# SQL Generation Dataset — Sources & Documentation

## Overview

This dataset contains 300 natural language to SQL generation pairs across 6 complexity
variants and 3 schema domains. Each pair was sent to two models (GPT-4o and Claude Haiku)
to collect real model responses for migration benchmarking.

## Data Sources

### Custom-Authored Pairs (All 300)

All 300 question/answer pairs were authored specifically for this dataset, designed to
work with the three custom schema domains defined below. Question patterns for the
`simple`, `join`, `aggregation`, and `cte_subquery` variants are inspired by patterns
found in the Spider 1.0 dataset (see citation below), but all questions are original
compositions written to match our custom schemas.

**Variant breakdown:**

| Variant | Count | Source |
|---|---|---|
| `simple` | 60 | Custom-authored, Spider-inspired patterns |
| `join` | 70 | Custom-authored, Spider-inspired patterns |
| `aggregation` | 60 | Custom-authored, Spider-inspired patterns |
| `cte_subquery` | 50 | Custom-authored, Spider-inspired patterns |
| `window_function` | 30 | Fully custom-authored (original work) |
| `unanswerable` | 30 | Fully custom-authored (original work) |

### Spider 1.0 — Inspiration Reference

The question patterns for simple, join, aggregation, and CTE/subquery variants draw
inspiration from the Spider benchmark's question taxonomy and difficulty classification.

- **Dataset:** Spider 1.0 (A Large-Scale Human-Labeled Dataset for Complex and
  Cross-Database Semantic Parsing and Text-to-SQL Task)
- **License:** CC-BY-SA-4.0
- **HuggingFace:** `xlangai/spider`
- **Citation:**
  > Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li,
  > James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev.
  > "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Database
  > Semantic Parsing and Text-to-SQL Task."
  > In Proceedings of the 2018 Conference on Empirical Methods in Natural Language
  > Processing (EMNLP 2018).

**Note:** No Spider questions or SQL queries were used directly. Spider was loaded
during development to study question patterns and difficulty classifications. All
questions in this dataset are original compositions.

## Schema Domains

Three PostgreSQL schemas are used across the dataset, with roughly equal distribution
(~100 pairs per domain).

### E-commerce Schema

```sql
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    total_amount NUMERIC(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled')),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    price NUMERIC(10,2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(10,2) NOT NULL
);
```

### HR Schema

```sql
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget NUMERIC(12,2),
    location VARCHAR(100)
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    salary NUMERIC(10,2) NOT NULL,
    hire_date DATE NOT NULL,
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE performance_reviews (
    id SERIAL PRIMARY KEY,
    employee_id INTEGER REFERENCES employees(id),
    review_date DATE NOT NULL,
    score INTEGER CHECK (score BETWEEN 1 AND 5),
    reviewer_id INTEGER REFERENCES employees(id)
);
```

### SaaS Schema

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    plan VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP
);

CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    plan_name VARCHAR(50) NOT NULL,
    mrr NUMERIC(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active','paused','cancelled')),
    started_at TIMESTAMP DEFAULT NOW(),
    cancelled_at TIMESTAMP
);

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    event_type VARCHAR(100) NOT NULL,
    properties JSONB,
    occurred_at TIMESTAMP DEFAULT NOW()
);
```

## Models

| Model | Provider | Records | Total Cost |
|---|---|---|---|
| GPT-4o | OpenAI (via OpenRouter) | 300 | ~$0.48 |
| Claude Haiku 4.5 | Anthropic | 300 | ~$0.27 |

## Quality Metrics

### GPT-4o (`sql_generation_gpt4o.jsonl`)
- SQL syntax validity: 270/270 (100.0%)
- Unanswerable error JSON validity: 30/30 (100.0%)
- Window functions with OVER clause: 29/30 (96.7%)
- Explicit JOIN usage in join variant: 68/70 (97.1%)

### Haiku (`sql_generation_haiku.jsonl`)
- SQL syntax validity: 270/270 (100.0%)
- Unanswerable error JSON validity: 30/30 (100.0%)
- Window functions with OVER clause: 30/30 (100.0%)
- Explicit JOIN usage in join variant: 68/70 (97.1%)

## JSONL Record Format

```json
{
  "prompt": "<schema DDL + natural language question>",
  "response": "<SQL query or error JSON>",
  "source_model": "openai/gpt-4o",
  "metadata": {
    "dataset": "sql_generation",
    "variant": "join",
    "domain": "ecommerce",
    "source": "custom_authored_spider_inspired"
  }
}
```

## Generation Date

2026-04-01
