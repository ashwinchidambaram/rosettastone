# fintech_extraction Dataset — Provenance & Methodology

## Overview

This dataset contains 400 synthetic invoice extraction tasks, each processed by two models:
- **GPT-4o** (`openai/gpt-4o` via OpenRouter) — `fintech_extraction_gpt4o.jsonl`
- **Claude Haiku** (`anthropic/claude-haiku-4-5-20251001`) — `fintech_extraction_haiku.jsonl`

Each record contains an unstructured invoice text (prompt) and the model's structured JSON extraction (response).

## Generation Methodology

### Synthetic Invoice Generation

All invoice texts are **synthetically generated** using:
- **Faker** (v40.x) for addresses, phone numbers, emails, and company names
- **Custom templates** for 6 invoice layout variants
- **53 realistic vendor company names** (manually curated, not from any external dataset)
- **24 line item categories** with realistic descriptions and price ranges
- **Random seed: 42** for full reproducibility

### Variant Distribution (400 total)

| Variant          | Count | Description |
|------------------|-------|-------------|
| `clean`          | 80    | Well-formatted tabular invoices, all fields present, US-style |
| `noisy`          | 80    | OCR-degraded text: character substitutions (0/O, 1/l, rn/m), misaligned columns, garbled characters |
| `multi_currency` | 60    | EUR, GBP, JPY, AUD, CAD invoices with correct currency symbols |
| `missing_fields` | 70    | Intentionally null due_date, tax_rate/tax_amount, or sparse line items |
| `edge_case`      | 60    | Credit memos (negative amounts), zero-quantity lines, long descriptions, PO numbers as invoice numbers |
| `ambiguous`      | 50    | Non-invoice documents: purchase orders, delivery notes, statements of account, proformas, quotes |

### LLM Response Collection

- **System prompt**: Instructs the model to extract structured JSON from invoice text
- **Temperature**: 0.0 (deterministic)
- **Prompt tuning**: 30 samples tested with GPT-4o before production run (96.7% valid JSON)
- **Production quality**: GPT-4o 99.75% valid JSON, Haiku 99.5% valid JSON

### Extraction Schema

```json
{
  "vendor_name": "string or null",
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "line_items": [{"description": "string", "quantity": number, "unit_price": number, "amount": number}],
  "subtotal": "number or null",
  "tax_rate": "decimal (0.0825, not 8.25) or null",
  "tax_amount": "number or null",
  "total": "number or null",
  "currency": "ISO 4217 code or null"
}
```

## Licensing & Attribution

- **This is 100% original synthetic data.** No external datasets were used.
- All vendor names, addresses, line items, and invoice content are generated programmatically.
- No real PII, financial data, or copyrighted material is included.
- No external dataset licensing requirements apply.

## Reproducibility

```bash
# Regenerate the invoice texts (deterministic with seed)
uv run python scripts/generate_fintech_extraction.py --seed 42 --generate-only

# Full regeneration (requires API keys)
uv run python scripts/generate_fintech_extraction.py --seed 42
```

## Cost Summary

| Model | Tuning | Production | Total |
|-------|--------|------------|-------|
| GPT-4o | $0.13 | $1.70 | $1.83 |
| Haiku  | $0.00 | $0.93 | $0.93 |
| **Combined** | **$0.13** | **$2.63** | **$2.76** |

## Generation Date

2026-04-01
