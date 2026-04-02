#!/usr/bin/env python3
"""
Generate Enterprise RAG Simulation dataset for RosettaStone benchmarks.

Produces 300 RAG prompt-response pairs across 6 variants and 5 user personas,
using a synthetic Meridian AI knowledge base with BM25 retrieval.

Calls GPT-4o and Haiku for model responses. Outputs JSONL files.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from litellm import completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/ashwinchidambaram/dev/projects/rosettastone/.env")
BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "data" / "meridian_knowledge_base"
OUTPUT_DIR = BASE_DIR / "examples" / "datasets" / "enterprise_rag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_PAIRS = 300
DISTRIBUTION = {
    "single_turn_factual": 80,
    "single_turn_procedural": 60,
    "multi_turn_clarification": 50,
    "multi_turn_complex": 40,
    "unanswerable": 40,
    "conflicting_context": 30,
}

SEED = 42
TOP_K = 3
CHUNK_TARGET_WORDS = 150

# ---------------------------------------------------------------------------
# System prompt for RAG
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful support assistant for Meridian AI, a B2B SaaS data analytics "
    "platform. Answer questions using ONLY the information provided in the retrieved "
    "context below. If the context does not contain enough information to answer the "
    "question, say so clearly -- do not make up information. Be concise, accurate, "
    "and professional."
)

# ---------------------------------------------------------------------------
# Knowledge Base Documents (inline content matching the files on disk)
# ---------------------------------------------------------------------------
KB_DOCUMENTS = {
    "product_overview.md": {
        "doc_id": "KB-001",
        "title": "Meridian Analytics Platform -- Product Overview",
    },
    "pricing_tiers.md": {
        "doc_id": "KB-002",
        "title": "Pricing Plans & Feature Comparison",
    },
    "api_authentication.md": {
        "doc_id": "KB-003",
        "title": "API Authentication & API Keys",
    },
    "api_data_ingestion.md": {
        "doc_id": "KB-004",
        "title": "Data Ingestion API -- Endpoints & Examples",
    },
    "api_query_endpoints.md": {
        "doc_id": "KB-005",
        "title": "Query API -- Running Analytics Queries",
    },
    "api_rate_limits.md": {
        "doc_id": "KB-006",
        "title": "API Rate Limits & Quotas",
    },
    "getting_started.md": {
        "doc_id": "KB-007",
        "title": "Getting Started -- First 30 Minutes",
    },
    "sso_setup.md": {
        "doc_id": "KB-008",
        "title": "Single Sign-On (SSO) Configuration Guide",
    },
    "data_source_connectors.md": {
        "doc_id": "KB-009",
        "title": "Connecting Data Sources -- Supported Connectors",
    },
    "askmeridian_nlq.md": {
        "doc_id": "KB-010",
        "title": "AskMeridian Natural Language Query Interface",
    },
    "dashboard_builder.md": {
        "doc_id": "KB-011",
        "title": "Dashboard Builder -- Creating & Sharing Dashboards",
    },
    "automated_reports.md": {
        "doc_id": "KB-012",
        "title": "Automated Reports & Scheduled Exports",
    },
    "data_governance_module.md": {
        "doc_id": "KB-013",
        "title": "Data Governance Add-On -- Access Controls & Lineage",
    },
    "realtime_streaming.md": {
        "doc_id": "KB-014",
        "title": "Real-Time Streaming Add-On",
    },
    "security_overview.md": {
        "doc_id": "KB-015",
        "title": "Security Architecture & Certifications",
    },
    "data_encryption.md": {
        "doc_id": "KB-016",
        "title": "Data Encryption -- At Rest & In Transit",
    },
    "hipaa_compliance.md": {
        "doc_id": "KB-017",
        "title": "HIPAA Compliance & BAA Details",
    },
    "troubleshooting_ingestion.md": {
        "doc_id": "KB-018",
        "title": "Troubleshooting Data Ingestion Errors",
    },
    "troubleshooting_performance.md": {
        "doc_id": "KB-019",
        "title": "Dashboard Performance & Query Optimization",
    },
    "billing_faq.md": {
        "doc_id": "KB-020",
        "title": "Billing FAQ -- Invoices, Upgrades & Cancellations",
    },
    "usage_limits.md": {
        "doc_id": "KB-021",
        "title": "Usage Limits & Overage Charges",
    },
    "pto_policy.md": {
        "doc_id": "KB-022",
        "title": "Paid Time Off & Leave Policy",
    },
    "remote_work_policy.md": {
        "doc_id": "KB-023",
        "title": "Remote Work & Office Attendance Policy",
    },
    "release_notes_2025q1.md": {
        "doc_id": "KB-024",
        "title": "Release Notes -- Q1 2025",
    },
    "release_notes_2024q4.md": {
        "doc_id": "KB-025",
        "title": "Release Notes -- Q4 2024",
    },
}


# ---------------------------------------------------------------------------
# Chunking and BM25 Index
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, split on word boundaries."""
    return re.findall(r"\w+", text.lower())


def chunk_documents() -> list[dict]:
    """Read KB Markdown files and split into chunks for BM25 retrieval."""
    chunks = []
    for filename, meta in KB_DOCUMENTS.items():
        filepath = KB_DIR / filename
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping")
            continue
        text = filepath.read_text()

        # Strip markdown headers but keep the text
        lines = text.split("\n")
        current_section = meta["title"]
        paragraphs = []
        current_para = []

        for line in lines:
            if line.startswith("#"):
                if current_para:
                    paragraphs.append((current_section, "\n".join(current_para).strip()))
                    current_para = []
                current_section = line.lstrip("#").strip()
            elif line.strip() == "":
                if current_para:
                    paragraphs.append((current_section, "\n".join(current_para).strip()))
                    current_para = []
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append((current_section, "\n".join(current_para).strip()))

        # Merge small paragraphs, split large ones
        merged = []
        buffer_section = None
        buffer_text = ""
        for section, para_text in paragraphs:
            if not para_text.strip():
                continue
            word_count = len(para_text.split())
            if word_count < 50 and buffer_text:
                buffer_text += "\n\n" + para_text
            elif word_count < 50:
                buffer_section = section
                buffer_text = para_text
            else:
                if buffer_text:
                    merged.append((buffer_section, buffer_text))
                    buffer_text = ""
                    buffer_section = None
                merged.append((section, para_text))

        if buffer_text:
            if merged:
                last_section, last_text = merged[-1]
                merged[-1] = (last_section, last_text + "\n\n" + buffer_text)
            else:
                merged.append((buffer_section or current_section, buffer_text))

        # Split chunks > 200 words at sentence boundaries
        for section, para_text in merged:
            words = para_text.split()
            if len(words) > 200:
                sentences = re.split(r"(?<=[.!?])\s+", para_text)
                current_chunk = []
                current_words = 0
                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > 200 and current_chunk:
                        chunks.append({
                            "doc_id": meta["doc_id"],
                            "title": meta["title"],
                            "section": section,
                            "text": " ".join(current_chunk),
                        })
                        current_chunk = [sent]
                        current_words = sent_words
                    else:
                        current_chunk.append(sent)
                        current_words += sent_words
                if current_chunk:
                    chunks.append({
                        "doc_id": meta["doc_id"],
                        "title": meta["title"],
                        "section": section,
                        "text": " ".join(current_chunk),
                    })
            else:
                chunks.append({
                    "doc_id": meta["doc_id"],
                    "title": meta["title"],
                    "section": section,
                    "text": para_text,
                })

    print(f"Total chunks: {len(chunks)}")
    return chunks


def build_bm25_index(chunks: list[dict]):
    """Build BM25 index from chunks."""
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    return bm25


def retrieve(query: str, bm25, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Run BM25 retrieval, return top-k chunks."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Prompt Assembly
# ---------------------------------------------------------------------------
def format_retrieved_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks for inclusion in prompt."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[{i}] (Source: {chunk['title']})\n{chunk['text']}")
    return "\n\n".join(parts)


def build_single_turn_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Build a single-turn RAG prompt."""
    context = format_retrieved_context(retrieved_chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"Retrieved Context:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"User Question: {question}"
    )


def build_multi_turn_prompt(
    turns: list[dict], retrieved_chunks: list[dict]
) -> str:
    """Build a multi-turn RAG prompt with conversation history."""
    context = format_retrieved_context(retrieved_chunks)

    # Separate conversation history from the final user message
    history_turns = turns[:-1]
    final_message = turns[-1]["content"]

    history_parts = []
    for turn in history_turns:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {turn['content']}")

    history_str = "\n".join(history_parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"Retrieved Context:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Conversation History:\n\n"
        f"{history_str}\n\n"
        f"---\n\n"
        f"Current User Message: {final_message}"
    )


# ---------------------------------------------------------------------------
# All 300 Questions -- Hand-Authored
# ---------------------------------------------------------------------------

# === SINGLE_TURN_FACTUAL (80) ===
SINGLE_TURN_FACTUAL_QUESTIONS = [
    # Dana - Data Engineer (20)
    {"persona": "dana", "question": "What's the max payload size for the batch ingestion endpoint?", "target_docs": ["KB-004"], "expected_fact": "100MB"},
    {"persona": "dana", "question": "What data formats does the ingestion API support?",  "target_docs": ["KB-004"], "expected_fact": "CSV, JSON, Parquet, Avro"},
    {"persona": "dana", "question": "What is the rate limit for Professional tier API requests per minute?", "target_docs": ["KB-006"], "expected_fact": "500 requests/minute"},
    {"persona": "dana", "question": "How long are async query results available after completion?", "target_docs": ["KB-005"], "expected_fact": "24 hours"},
    {"persona": "dana", "question": "What's the default API key expiration period?", "target_docs": ["KB-003"], "expected_fact": "90 days"},
    {"persona": "dana", "question": "What is the max execution time for synchronous queries?", "target_docs": ["KB-005"], "expected_fact": "120 seconds"},
    {"persona": "dana", "question": "What field is used for deduplication in the ingestion API?", "target_docs": ["KB-004"], "expected_fact": "_meridian_id"},
    {"persona": "dana", "question": "What streaming sources does the Real-Time Streaming add-on support?", "target_docs": ["KB-014"], "expected_fact": "Kafka, Kinesis, Pub/Sub"},
    {"persona": "dana", "question": "What is the maximum throughput for the streaming add-on?", "target_docs": ["KB-014"], "expected_fact": "50,000 events/second"},
    {"persona": "dana", "question": "What endpoint do I use for key rotation?", "target_docs": ["KB-003"], "expected_fact": "POST /api/v2/keys/rotate"},
    {"persona": "dana", "question": "What is the burst allowance for the Enterprise tier?", "target_docs": ["KB-006"], "expected_fact": "4,000 requests in 10-second window"},
    {"persona": "dana", "question": "What SQL dialect does the Meridian Query API support?", "target_docs": ["KB-005"], "expected_fact": "ANSI SQL with Meridian extensions"},
    {"persona": "dana", "question": "What header contains the query cost in MCU?", "target_docs": ["KB-005"], "expected_fact": "X-MCU-Cost"},
    {"persona": "dana", "question": "What is the default pagination size for query results?", "target_docs": ["KB-005"], "expected_fact": "10,000 rows per page"},
    {"persona": "dana", "question": "How many database connectors does MAP support?", "target_docs": ["KB-009"], "expected_fact": "PostgreSQL, MySQL, SQL Server, Oracle, Snowflake, BigQuery, Redshift, Databricks"},
    {"persona": "dana", "question": "What is the daily API request limit for the Starter tier?", "target_docs": ["KB-006"], "expected_fact": "10,000 requests/day"},
    {"persona": "dana", "question": "How long does the old API key remain valid after rotation?", "target_docs": ["KB-003"], "expected_fact": "24 hours"},
    {"persona": "dana", "question": "What are the configurable API key expiration options?", "target_docs": ["KB-003"], "expected_fact": "30, 60, 90, or 365 days"},
    {"persona": "dana", "question": "What rate limit headers are included in API responses?", "target_docs": ["KB-006"], "expected_fact": "X-RateLimit-Remaining, X-RateLimit-Reset"},
    {"persona": "dana", "question": "What is the end-to-end latency for streaming ingestion?", "target_docs": ["KB-014"], "expected_fact": "sub-5-second"},
    # Marcus - VP Sales (16)
    {"persona": "marcus", "question": "What is the price per user per month for the Starter plan?", "target_docs": ["KB-002"], "expected_fact": "$49/user/month"},
    {"persona": "marcus", "question": "What's the minimum user count for the Enterprise plan?", "target_docs": ["KB-002"], "expected_fact": "25 users"},
    {"persona": "marcus", "question": "What's the annual billing discount?", "target_docs": ["KB-002"], "expected_fact": "20% off monthly price"},
    {"persona": "marcus", "question": "How much is the Advanced ML add-on?", "target_docs": ["KB-002"], "expected_fact": "$500/month"},
    {"persona": "marcus", "question": "What uptime SLA does the Enterprise plan offer?", "target_docs": ["KB-001"], "expected_fact": "99.95%"},
    {"persona": "marcus", "question": "Is HIPAA BAA available for Professional tier customers?", "target_docs": ["KB-017"], "expected_fact": "No, Enterprise only"},
    {"persona": "marcus", "question": "What payment methods are accepted for Enterprise accounts?", "target_docs": ["KB-020"], "expected_fact": "credit card, ACH/wire"},
    {"persona": "marcus", "question": "How much notice is required to cancel an annual plan?", "target_docs": ["KB-020"], "expected_fact": "30 days"},
    {"persona": "marcus", "question": "What is the overage charge per 1,000 rows?", "target_docs": ["KB-020"], "expected_fact": "$0.10"},
    {"persona": "marcus", "question": "What compliance certifications does Meridian hold?", "target_docs": ["KB-015"], "expected_fact": "SOC 2 Type II"},
    {"persona": "marcus", "question": "What's the row limit on the Professional plan?", "target_docs": ["KB-002"], "expected_fact": "1 billion rows"},
    {"persona": "marcus", "question": "How many data source connectors does MAP support out of the box?", "target_docs": ["KB-001"], "expected_fact": "40+"},
    {"persona": "marcus", "question": "When was the SOC 2 Type II certification first obtained?", "target_docs": ["KB-015"], "expected_fact": "2023"},
    {"persona": "marcus", "question": "What is the maximum number of users on the Starter plan?", "target_docs": ["KB-002"], "expected_fact": "10 users"},
    {"persona": "marcus", "question": "How long is the free trial period?", "target_docs": ["KB-007"], "expected_fact": "14 days"},
    {"persona": "marcus", "question": "Is a credit card required for the free trial?", "target_docs": ["KB-007"], "expected_fact": "No"},
    # Priya - Analyst (18)
    {"persona": "priya", "question": "How many chart types are available in the Dashboard Builder?", "target_docs": ["KB-011"], "expected_fact": "15"},
    {"persona": "priya", "question": "What export formats are available for dashboards?", "target_docs": ["KB-011"], "expected_fact": "PDF, PNG, CSV"},
    {"persona": "priya", "question": "What languages does AskMeridian support?", "target_docs": ["KB-010"], "expected_fact": "English, Spanish, French, German, Japanese"},
    {"persona": "priya", "question": "What is the max number of scheduled reports on Professional?", "target_docs": ["KB-012"], "expected_fact": "50"},
    {"persona": "priya", "question": "What confidence levels does AskMeridian display?", "target_docs": ["KB-010"], "expected_fact": "Low, Medium, High"},
    {"persona": "priya", "question": "How long is NLQ query history retained on Professional?", "target_docs": ["KB-010"], "expected_fact": "90 days"},
    {"persona": "priya", "question": "Can dashboards be embedded via iframe?", "target_docs": ["KB-011"], "expected_fact": "Yes, on Professional & Enterprise"},
    {"persona": "priya", "question": "What are the auto-refresh interval options for dashboards?", "target_docs": ["KB-011"], "expected_fact": "1 min, 5 min, 15 min, 1 hour, manual"},
    {"persona": "priya", "question": "What formats can scheduled reports be delivered in?", "target_docs": ["KB-012"], "expected_fact": "PDF, CSV"},
    {"persona": "priya", "question": "What is the max number of widgets per report?", "target_docs": ["KB-012"], "expected_fact": "20"},
    {"persona": "priya", "question": "What is MRL in the context of automated reports?", "target_docs": ["KB-012"], "expected_fact": "Meridian Report Language, a YAML-based DSL"},
    {"persona": "priya", "question": "Which tiers include AskMeridian?", "target_docs": ["KB-010"], "expected_fact": "Professional and Enterprise"},
    {"persona": "priya", "question": "What new chart types were added in Q1 2025?", "target_docs": ["KB-024"], "expected_fact": "heatmap, treemap"},
    {"persona": "priya", "question": "How much did AskMeridian accuracy improve in v2?", "target_docs": ["KB-024"], "expected_fact": "40%"},
    {"persona": "priya", "question": "What scheduling frequencies are available for reports?", "target_docs": ["KB-012"], "expected_fact": "daily, weekly, monthly"},
    {"persona": "priya", "question": "What is the sample dataset pre-loaded in new workspaces?", "target_docs": ["KB-007"], "expected_fact": "Meridian Demo Retail"},
    {"persona": "priya", "question": "What improvement was made to dashboard render time in Q4 2024?", "target_docs": ["KB-025"], "expected_fact": "3x improvement"},
    {"persona": "priya", "question": "Can AskMeridian users flag incorrect results?", "target_docs": ["KB-010"], "expected_fact": "Yes"},
    # James - IT Security (14)
    {"persona": "james", "question": "What encryption standard is used for data at rest?", "target_docs": ["KB-016"], "expected_fact": "AES-256"},
    {"persona": "james", "question": "What TLS versions are accepted for data in transit?", "target_docs": ["KB-016"], "expected_fact": "TLS 1.3 enforced, TLS 1.2 accepted"},
    {"persona": "james", "question": "Are customer-managed encryption keys available?", "target_docs": ["KB-016"], "expected_fact": "Yes, Enterprise tier only"},
    {"persona": "james", "question": "How often is backup key rotation performed?", "target_docs": ["KB-016"], "expected_fact": "90-day cycle"},
    {"persona": "james", "question": "What SSO protocols does Meridian support?", "target_docs": ["KB-008"], "expected_fact": "SAML 2.0, OIDC"},
    {"persona": "james", "question": "Is SCIM provisioning available on Professional?", "target_docs": ["KB-008"], "expected_fact": "No, Enterprise only"},
    {"persona": "james", "question": "How long are HIPAA audit logs retained?", "target_docs": ["KB-017"], "expected_fact": "7 years"},
    {"persona": "james", "question": "What AWS regions host Meridian data?", "target_docs": ["KB-015"], "expected_fact": "us-east-1, eu-west-1"},
    {"persona": "james", "question": "Is the infrastructure single-tenant for Enterprise?", "target_docs": ["KB-015"], "expected_fact": "Yes"},
    {"persona": "james", "question": "How long are audit logs retained on the Professional tier with Data Governance?", "target_docs": ["KB-013"], "expected_fact": "1 year"},
    {"persona": "james", "question": "What is the session timeout for HIPAA workspaces?", "target_docs": ["KB-017"], "expected_fact": "15 minutes"},
    {"persona": "james", "question": "What is the bug bounty reporting email?", "target_docs": ["KB-015"], "expected_fact": "security@meridian-ai.com"},
    {"persona": "james", "question": "How often is penetration testing conducted?", "target_docs": ["KB-015"], "expected_fact": "annually"},
    {"persona": "james", "question": "What OAuth flow is supported for API auth?", "target_docs": ["KB-003"], "expected_fact": "OAuth 2.0 client credentials"},
    # Alex - New Hire (12)
    {"persona": "alex", "question": "How many PTO days do full-time employees get?", "target_docs": ["KB-022"], "expected_fact": "20 days"},
    {"persona": "alex", "question": "How many sick days per year do I get?", "target_docs": ["KB-022"], "expected_fact": "5"},
    {"persona": "alex", "question": "What is the PTO rollover limit?", "target_docs": ["KB-022"], "expected_fact": "10 days"},
    {"persona": "alex", "question": "Do I have to come into the office?", "target_docs": ["KB-023"], "expected_fact": "Remote-first, no mandatory in-office for most roles"},
    {"persona": "alex", "question": "How much is the home office stipend for new hires?", "target_docs": ["KB-023"], "expected_fact": "$1,500"},
    {"persona": "alex", "question": "What are the core collaboration hours?", "target_docs": ["KB-023"], "expected_fact": "10 AM - 3 PM local time"},
    {"persona": "alex", "question": "How much parental leave does a primary caregiver get?", "target_docs": ["KB-022"], "expected_fact": "16 weeks"},
    {"persona": "alex", "question": "Which days are the in-office days for hybrid employees?", "target_docs": ["KB-023"], "expected_fact": "Tuesday and Thursday"},
    {"persona": "alex", "question": "How many days of international remote work are allowed per year?", "target_docs": ["KB-023"], "expected_fact": "30 days"},
    {"persona": "alex", "question": "Where do I submit PTO requests?", "target_docs": ["KB-022"], "expected_fact": "Workday"},
    {"persona": "alex", "question": "How many company holidays does Meridian observe?", "target_docs": ["KB-022"], "expected_fact": "10"},
    {"persona": "alex", "question": "What is the annual home office stipend after the first year?", "target_docs": ["KB-023"], "expected_fact": "$500/year"},
]

# === SINGLE_TURN_PROCEDURAL (60) ===
SINGLE_TURN_PROCEDURAL_QUESTIONS = [
    # Dana (16)
    {"persona": "dana", "question": "How do I rotate my API key without downtime?", "target_docs": ["KB-003"]},
    {"persona": "dana", "question": "How do I submit a batch ingestion job via the API?", "target_docs": ["KB-004"]},
    {"persona": "dana", "question": "How do I check the status of an ingestion job?", "target_docs": ["KB-004"]},
    {"persona": "dana", "question": "How do I enable schema evolution for data ingestion?", "target_docs": ["KB-018"]},
    {"persona": "dana", "question": "How do I set up a streaming data source with Kafka?", "target_docs": ["KB-014"]},
    {"persona": "dana", "question": "How do I run an async query and retrieve the results?", "target_docs": ["KB-005"]},
    {"persona": "dana", "question": "How do I connect a Snowflake data source to MAP?", "target_docs": ["KB-009"]},
    {"persona": "dana", "question": "How do I estimate query cost before running it?", "target_docs": ["KB-005", "KB-024"]},
    {"persona": "dana", "question": "How do I resolve the INGEST_003 schema mismatch error?", "target_docs": ["KB-018"]},
    {"persona": "dana", "question": "How do I resolve a duplicate key violation (INGEST_004)?", "target_docs": ["KB-018"]},
    {"persona": "dana", "question": "How do I configure API key scopes?", "target_docs": ["KB-003"]},
    {"persona": "dana", "question": "How do I set up OAuth 2.0 client credentials for API access?", "target_docs": ["KB-003"]},
    {"persona": "dana", "question": "How do I build a custom connector using the Connector SDK?", "target_docs": ["KB-009"]},
    {"persona": "dana", "question": "How do I troubleshoot an ingestion job stuck in pending for over 30 minutes?", "target_docs": ["KB-018"]},
    {"persona": "dana", "question": "How do I configure data sync frequency for a connector?", "target_docs": ["KB-009"]},
    {"persona": "dana", "question": "How do I use the upsert mode for data ingestion?", "target_docs": ["KB-004"]},
    # Marcus (10)
    {"persona": "marcus", "question": "How do I upgrade from Starter to Professional mid-cycle?", "target_docs": ["KB-020"]},
    {"persona": "marcus", "question": "How do I cancel an annual subscription?", "target_docs": ["KB-020"]},
    {"persona": "marcus", "question": "How do I request a SOC 2 Type II audit report for a prospect?", "target_docs": ["KB-015"]},
    {"persona": "marcus", "question": "How do I set up a new Enterprise account with HIPAA BAA?", "target_docs": ["KB-017"]},
    {"persona": "marcus", "question": "How do I check a customer's current usage against their plan limits?", "target_docs": ["KB-021"]},
    {"persona": "marcus", "question": "How do I update payment information for a customer?", "target_docs": ["KB-020"]},
    {"persona": "marcus", "question": "How do I enable overage charges for a workspace?", "target_docs": ["KB-021"]},
    {"persona": "marcus", "question": "How do I add the Advanced ML add-on to a Professional account?", "target_docs": ["KB-002"]},
    {"persona": "marcus", "question": "How do I downgrade a customer from Professional to Starter?", "target_docs": ["KB-020"]},
    {"persona": "marcus", "question": "How do I scope an Enterprise deal with HIPAA, SSO, and custom rate limits?", "target_docs": ["KB-002", "KB-017"]},
    # Priya (16)
    {"persona": "priya", "question": "How do I set up a weekly automated report delivered as PDF via email?", "target_docs": ["KB-012"]},
    {"persona": "priya", "question": "How do I create a new dashboard in the Dashboard Builder?", "target_docs": ["KB-011"]},
    {"persona": "priya", "question": "How do I embed a dashboard in our internal wiki using an iframe?", "target_docs": ["KB-011"]},
    {"persona": "priya", "question": "How do I configure cross-widget filtering on a dashboard?", "target_docs": ["KB-011"]},
    {"persona": "priya", "question": "How do I set up conditional alerts on a scheduled report?", "target_docs": ["KB-012"]},
    {"persona": "priya", "question": "How do I create a custom report template using MRL?", "target_docs": ["KB-012"]},
    {"persona": "priya", "question": "How do I export a dashboard as a PDF?", "target_docs": ["KB-011"]},
    {"persona": "priya", "question": "How do I set the auto-refresh interval on a dashboard?", "target_docs": ["KB-011"]},
    {"persona": "priya", "question": "How do I share a dashboard via a view-only link?", "target_docs": ["KB-011"]},
    {"persona": "priya", "question": "How do I fix AskMeridian low-confidence results?", "target_docs": ["KB-019"]},
    {"persona": "priya", "question": "How do I add a heatmap widget to my dashboard?", "target_docs": ["KB-024", "KB-011"]},
    {"persona": "priya", "question": "How do I use AskMeridian to query my data?", "target_docs": ["KB-010"]},
    {"persona": "priya", "question": "How do I upload a CSV file to Meridian?", "target_docs": ["KB-009"]},
    {"persona": "priya", "question": "How do I manage scheduled reports -- edit, pause, or delete them?", "target_docs": ["KB-012"]},
    {"persona": "priya", "question": "How do I change the delivery frequency of an existing scheduled report?", "target_docs": ["KB-012"]},
    {"persona": "priya", "question": "How do I explore data using the sample dataset?", "target_docs": ["KB-007"]},
    # James (10)
    {"persona": "james", "question": "How do I set up SAML 2.0 SSO for my organization?", "target_docs": ["KB-008"]},
    {"persona": "james", "question": "How do I set up OpenID Connect (OIDC) SSO?", "target_docs": ["KB-008"]},
    {"persona": "james", "question": "How do I enable SCIM provisioning for automated user management?", "target_docs": ["KB-008"]},
    {"persona": "james", "question": "How do I configure column-level access controls?", "target_docs": ["KB-013"]},
    {"persona": "james", "question": "How do I view the data lineage for a dashboard widget?", "target_docs": ["KB-013"]},
    {"persona": "james", "question": "How do I enable PII auto-detection on ingested data?", "target_docs": ["KB-013"]},
    {"persona": "james", "question": "How do I export HIPAA audit logs for compliance review?", "target_docs": ["KB-017"]},
    {"persona": "james", "question": "How do I configure customer-managed encryption keys (CMK)?", "target_docs": ["KB-016"]},
    {"persona": "james", "question": "How do I create a HIPAA-compliant workspace?", "target_docs": ["KB-017"]},
    {"persona": "james", "question": "How do I enable IP allowlisting for Enterprise?", "target_docs": ["KB-015"]},
    # Alex (8)
    {"persona": "alex", "question": "How do I create a new workspace as a new user?", "target_docs": ["KB-007"]},
    {"persona": "alex", "question": "How do I connect my first data source?", "target_docs": ["KB-007"]},
    {"persona": "alex", "question": "How do I submit a PTO request?", "target_docs": ["KB-022"]},
    {"persona": "alex", "question": "How do I request international remote work approval?", "target_docs": ["KB-023"]},
    {"persona": "alex", "question": "How do I invite team members to my workspace?", "target_docs": ["KB-007"]},
    {"persona": "alex", "question": "How do I claim my home office stipend?", "target_docs": ["KB-023"]},
    {"persona": "alex", "question": "How do I get started with the Meridian Demo Retail sample dataset?", "target_docs": ["KB-007"]},
    {"persona": "alex", "question": "How do I build my first dashboard?", "target_docs": ["KB-007"]},
]

# === MULTI_TURN_CLARIFICATION (50) ===
MULTI_TURN_CLARIFICATION = [
    # Dana (12)
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What are the API rate limits?"},
        {"role": "assistant", "content": "Rate limits depend on your pricing tier. Could you tell me which plan you're on -- Starter, Professional, or Enterprise?"},
        {"role": "user", "content": "We're on Professional."},
    ], "target_docs": ["KB-006"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "I need to ingest a large dataset. What are the limits?"},
        {"role": "assistant", "content": "The batch ingestion endpoint accepts payloads up to 100MB per request. What format is your data in?"},
        {"role": "user", "content": "It's in Parquet format, about 500MB total."},
    ], "target_docs": ["KB-004"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "I'm getting an ingestion error. Can you help?"},
        {"role": "assistant", "content": "Sure! What error code are you seeing? Common ones include INGEST_001 through INGEST_004."},
        {"role": "user", "content": "It's INGEST_003 on a Parquet upload."},
    ], "target_docs": ["KB-018"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can I use streaming ingestion?"},
        {"role": "assistant", "content": "Streaming ingestion is available with the Real-Time Streaming add-on. Do you currently have that add-on enabled?"},
        {"role": "user", "content": "I don't think so. What does it cost and what does it support?"},
    ], "target_docs": ["KB-014"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I handle duplicate records in ingestion?"},
        {"role": "assistant", "content": "Meridian uses the _meridian_id field for deduplication by default. Are you seeing duplicate key violations, or do you want to allow duplicates?"},
        {"role": "user", "content": "I'm seeing INGEST_004 duplicate key violations and I want to update existing records."},
    ], "target_docs": ["KB-004", "KB-018"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What query endpoints are available?"},
        {"role": "assistant", "content": "There are two main query endpoints: synchronous and asynchronous. How long do your queries typically take to run?"},
        {"role": "user", "content": "Some of them run for 3-4 minutes."},
    ], "target_docs": ["KB-005"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What database connectors are supported?"},
        {"role": "assistant", "content": "MAP supports several database connectors. Are you looking for a specific database, or would you like the full list?"},
        {"role": "user", "content": "I need Databricks specifically. Is it supported?"},
    ], "target_docs": ["KB-009"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "My query keeps timing out."},
        {"role": "assistant", "content": "Synchronous queries have a 120-second timeout. Is your query exceeding that limit?"},
        {"role": "user", "content": "Yes, it takes about 3 minutes. What should I do?"},
    ], "target_docs": ["KB-019", "KB-005"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I configure API key expiration?"},
        {"role": "assistant", "content": "API keys expire after 90 days by default. Do you need a shorter or longer expiration period?"},
        {"role": "user", "content": "I need them to last a full year. Is that possible?"},
    ], "target_docs": ["KB-003"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I monitor API usage?"},
        {"role": "assistant", "content": "API responses include rate limit headers. Are you looking to track rate limit usage, or query costs in MCU?"},
        {"role": "user", "content": "I want to track MCU costs for our queries."},
    ], "target_docs": ["KB-005", "KB-019"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What's the best format for large data ingestion?"},
        {"role": "assistant", "content": "The ingestion API supports CSV, JSON, Parquet, and Avro. How large is your dataset?"},
        {"role": "user", "content": "About 2GB. I need the most efficient option."},
    ], "target_docs": ["KB-004"]},
    {"persona": "dana", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How does data sync frequency work?"},
        {"role": "assistant", "content": "Sync frequency can be real-time, hourly, daily, or manual. Which connector type are you configuring?"},
        {"role": "user", "content": "A PostgreSQL connector. I want hourly syncs."},
    ], "target_docs": ["KB-009"]},
    # Marcus (10)
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What's included in the Enterprise plan?"},
        {"role": "assistant", "content": "The Enterprise plan includes unlimited rows, connectors, dashboards, and reports, plus a dedicated CSM, phone support, HIPAA BAA, and 99.95% SLA. Is there a specific feature you're looking to confirm?"},
        {"role": "user", "content": "Does it include SSO and the Data Governance module?"},
    ], "target_docs": ["KB-002", "KB-013"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How does billing work for plan changes?"},
        {"role": "assistant", "content": "Plan changes work differently depending on whether you're upgrading or downgrading. Which scenario are you asking about?"},
        {"role": "user", "content": "A customer wants to upgrade from Starter to Professional mid-cycle."},
    ], "target_docs": ["KB-020"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "A prospect is asking about security certifications."},
        {"role": "assistant", "content": "Meridian holds SOC 2 Type II certification. What specific compliance requirements does the prospect have?"},
        {"role": "user", "content": "They need SOC 2 and HIPAA. They're a healthcare company."},
    ], "target_docs": ["KB-015", "KB-017"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What happens if a customer exceeds their row limit?"},
        {"role": "assistant", "content": "When a workspace hits its row limit, new ingestion is blocked. Are you asking about a specific tier, or about overage charges?"},
        {"role": "user", "content": "Tell me about overage charges and how they're billed."},
    ], "target_docs": ["KB-021", "KB-020"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can Professional customers get the Real-Time Streaming add-on?"},
        {"role": "assistant", "content": "Yes, Real-Time Streaming is available on both Professional and Enterprise. Would you like pricing details?"},
        {"role": "user", "content": "Yes, what's the monthly cost?"},
    ], "target_docs": ["KB-014"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "A customer wants to cancel. What's the process?"},
        {"role": "assistant", "content": "The cancellation process depends on the billing type. Is the customer on a monthly or annual plan?"},
        {"role": "user", "content": "They're on an annual plan."},
    ], "target_docs": ["KB-020"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How many connectors does Starter include?"},
        {"role": "assistant", "content": "The Starter plan includes 5 data source connectors. Do you need more detail on what's available at higher tiers?"},
        {"role": "user", "content": "Yes, what about Professional?"},
    ], "target_docs": ["KB-002", "KB-021"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What's the difference between Professional and Enterprise support?"},
        {"role": "assistant", "content": "Professional includes email and chat support. Enterprise adds a dedicated CSM and phone support. Are you comparing for a specific deal?"},
        {"role": "user", "content": "Yes, a customer is deciding between the two. They want to know about SLA differences too."},
    ], "target_docs": ["KB-002", "KB-001"]},
    {"persona": "marcus", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Does Meridian offer any discounts?"},
        {"role": "assistant", "content": "Yes, we offer an annual billing discount. Would you like the specific percentage?"},
        {"role": "user", "content": "Yes, and does it apply to all tiers?"},
    ], "target_docs": ["KB-002"]},
    # Priya (12)
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "I want to share a dashboard with an external stakeholder."},
        {"role": "assistant", "content": "You can share dashboards via a view-only link or embed them via iframe. Does the stakeholder need just viewing access, or do they need to interact with the data?"},
        {"role": "user", "content": "Just viewing. Can I make the link password-protected?"},
    ], "target_docs": ["KB-011"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "I need a report that only sends when a metric crosses a threshold."},
        {"role": "assistant", "content": "That sounds like a conditional alert on a scheduled report. What metric and threshold are you tracking?"},
        {"role": "user", "content": "Revenue dropping below $50,000 per day. How do I set that up?"},
    ], "target_docs": ["KB-012"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "AskMeridian isn't giving me good results."},
        {"role": "assistant", "content": "AskMeridian shows confidence scores with each answer. Are you seeing mostly Low confidence results?"},
        {"role": "user", "content": "Yes, almost all my queries come back as Low confidence."},
    ], "target_docs": ["KB-019", "KB-010"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can I schedule a report to be sent on a specific day each week?"},
        {"role": "assistant", "content": "Yes, reports can be scheduled daily, weekly, or monthly. For weekly, you choose the day of the week. What format do you need it in?"},
        {"role": "user", "content": "I'd like it sent every Monday as a PDF attachment."},
    ], "target_docs": ["KB-012"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I add a visualization to my dashboard?"},
        {"role": "assistant", "content": "You can drag and drop widgets from the widget palette. What type of chart are you looking to add?"},
        {"role": "user", "content": "I need a scatter plot showing correlation between two metrics."},
    ], "target_docs": ["KB-011"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How many reports can I schedule?"},
        {"role": "assistant", "content": "The limit depends on your tier. Which plan are you on?"},
        {"role": "user", "content": "Professional."},
    ], "target_docs": ["KB-012"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "I want to export my dashboard data."},
        {"role": "assistant", "content": "Dashboards can be exported as PDF, PNG, or CSV. Which format do you need?"},
        {"role": "user", "content": "I just need the underlying data, not the visuals."},
    ], "target_docs": ["KB-011"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can I use AskMeridian in languages other than English?"},
        {"role": "assistant", "content": "Yes! AskMeridian supports multiple languages. Which language are you interested in?"},
        {"role": "user", "content": "Japanese. Our team in Tokyo needs it."},
    ], "target_docs": ["KB-010", "KB-024"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "My dashboard is loading really slowly."},
        {"role": "assistant", "content": "Slow dashboard loading can have several causes. How many widgets does your dashboard have, and how long is it taking to load?"},
        {"role": "user", "content": "It has 12 widgets and takes about 15 seconds."},
    ], "target_docs": ["KB-019"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What chart types are available?"},
        {"role": "assistant", "content": "The Dashboard Builder offers 15 chart types. Are you looking for a specific type of visualization?"},
        {"role": "user", "content": "I need something to show hierarchical data breakdown, like revenue by region then by product."},
    ], "target_docs": ["KB-011", "KB-024"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I make my report template reusable?"},
        {"role": "assistant", "content": "You can create custom templates using MRL (Meridian Report Language). Are you familiar with YAML syntax?"},
        {"role": "user", "content": "Yes, I'm comfortable with YAML. What can I configure in MRL?"},
    ], "target_docs": ["KB-012"]},
    {"persona": "priya", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can I add filters that work across all widgets on a dashboard?"},
        {"role": "assistant", "content": "Yes, Meridian supports cross-widget filtering. When you click a data point in one widget, all others filter automatically. Is there a specific filter you want to set up?"},
        {"role": "user", "content": "I want a date range filter that applies to every widget."},
    ], "target_docs": ["KB-011"]},
    # James (10)
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What's the encryption posture for data at rest?"},
        {"role": "assistant", "content": "We use AES-256 encryption for all data at rest. Do you need details about key management options?"},
        {"role": "user", "content": "Yes, can we use our own encryption keys?"},
    ], "target_docs": ["KB-016"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "We need to set up SSO. What options do you support?"},
        {"role": "assistant", "content": "We support SAML 2.0 and OpenID Connect. Which protocol does your identity provider use?"},
        {"role": "user", "content": "We use Azure AD with SAML. What info do you need from us?"},
    ], "target_docs": ["KB-008"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How are user accounts managed with SSO?"},
        {"role": "assistant", "content": "SSO supports JIT provisioning for automatic account creation on first login. Do you also need automated lifecycle management?"},
        {"role": "user", "content": "Yes, we want automatic deactivation when users leave. Is SCIM available?"},
    ], "target_docs": ["KB-008"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What audit logging capabilities exist?"},
        {"role": "assistant", "content": "Audit logging is available through the Data Governance module. The retention period varies by tier. Which tier are you on?"},
        {"role": "user", "content": "Enterprise. And we're specifically concerned about HIPAA audit requirements."},
    ], "target_docs": ["KB-013", "KB-017"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Is the platform single-tenant or multi-tenant?"},
        {"role": "assistant", "content": "It depends on your tier. Which plan are you evaluating?"},
        {"role": "user", "content": "Enterprise. We need infrastructure isolation from other customers."},
    ], "target_docs": ["KB-015"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What PII protection capabilities does Meridian have?"},
        {"role": "assistant", "content": "The Data Governance module includes PII auto-detection and column-level access controls. Are you looking for detection, masking, or both?"},
        {"role": "user", "content": "Both. What patterns does the PII detector recognize?"},
    ], "target_docs": ["KB-013", "KB-024"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can you walk me through the HIPAA workspace setup?"},
        {"role": "assistant", "content": "HIPAA workspaces have special security controls. Do you already have an Enterprise plan with a signed BAA?"},
        {"role": "user", "content": "Yes, our BAA is signed. What's different about the HIPAA workspace?"},
    ], "target_docs": ["KB-017"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What TLS versions are supported?"},
        {"role": "assistant", "content": "We enforce TLS 1.3 as the primary protocol. Are you asking about general platform access or a specific compliance context?"},
        {"role": "user", "content": "For HIPAA compliance. What's the minimum TLS version in HIPAA workspaces?"},
    ], "target_docs": ["KB-016", "KB-017"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I restrict data access for certain columns?"},
        {"role": "assistant", "content": "Column-level access controls are available in the Data Governance module. Do you want to restrict by user or by group?"},
        {"role": "user", "content": "By group. How do I set that up?"},
    ], "target_docs": ["KB-013"]},
    {"persona": "james", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Do you do regular security testing?"},
        {"role": "assistant", "content": "Yes, we conduct annual penetration testing and quarterly internal assessments. Would you like details about the pen test reports?"},
        {"role": "user", "content": "Yes, can we get a copy of the pen test report?"},
    ], "target_docs": ["KB-015"]},
    # Alex (6)
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What's the PTO policy?"},
        {"role": "assistant", "content": "Full-time employees get 20 PTO days, 10 holidays, and 5 sick days per year. Do you want to know about any specific aspect, like rollover or parental leave?"},
        {"role": "user", "content": "Yes, how does PTO rollover work?"},
    ], "target_docs": ["KB-022"]},
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "I just got access to Meridian. Where do I start?"},
        {"role": "assistant", "content": "Welcome! The Getting Started guide walks you through setup in under 30 minutes. Have you created your workspace yet?"},
        {"role": "user", "content": "Yes, but I haven't connected any data. What should I do next?"},
    ], "target_docs": ["KB-007"]},
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Can I work from another country for a few weeks?"},
        {"role": "assistant", "content": "International remote work is allowed under certain conditions. How long would you be working internationally?"},
        {"role": "user", "content": "About 3 weeks, visiting family in Portugal."},
    ], "target_docs": ["KB-023"]},
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "What happens if I don't use all my PTO?"},
        {"role": "assistant", "content": "Unused PTO can roll over to the next year with some limits. Would you like the specific rollover rules?"},
        {"role": "user", "content": "Yes, what's the max rollover?"},
    ], "target_docs": ["KB-022"]},
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I request bereavement leave?"},
        {"role": "assistant", "content": "I'm sorry for your loss. Bereavement leave varies by the relationship. Is this for an immediate family member or extended family?"},
        {"role": "user", "content": "It's for my grandfather."},
    ], "target_docs": ["KB-022"]},
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "Do I need to come into the office?"},
        {"role": "assistant", "content": "Meridian is remote-first, but it depends on your location. Are you based near one of our offices (Austin, NYC, or London)?"},
        {"role": "user", "content": "I'm in Austin."},
    ], "target_docs": ["KB-023"]},
    {"persona": "alex", "variant": "multi_turn_clarification", "turns": [
        {"role": "user", "content": "How do I get reimbursed for my home office setup?"},
        {"role": "assistant", "content": "New hires get a $1,500 one-time stipend for home office setup. Are you looking for details on how to submit the reimbursement?"},
        {"role": "user", "content": "Yes, where do I submit the receipts?"},
    ], "target_docs": ["KB-023"]},
]

# === MULTI_TURN_COMPLEX (40) ===
MULTI_TURN_COMPLEX = [
    # Dana (10)
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I'm setting up a new data pipeline for our analytics team."},
        {"role": "assistant", "content": "Great! Meridian supports multiple ingestion methods. What data source are you connecting?"},
        {"role": "user", "content": "We have a PostgreSQL database and a Kafka stream."},
        {"role": "assistant", "content": "PostgreSQL is supported as a native connector with hourly, daily, or manual sync. For Kafka, you'll need the Real-Time Streaming add-on. Are you on Professional or Enterprise?"},
        {"role": "user", "content": "Professional. What will the streaming add-on cost us?"},
        {"role": "assistant", "content": "The Real-Time Streaming add-on is $750/month. It supports Kafka, Kinesis, and Pub/Sub with sub-5-second latency."},
        {"role": "user", "content": "What's the max throughput for streaming?"},
    ], "target_docs": ["KB-009", "KB-014"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to optimize our query performance. Some dashboards are very slow."},
        {"role": "assistant", "content": "Slow dashboards can be caused by unindexed columns, too many widgets, or expensive queries. How many widgets are on the slow dashboard?"},
        {"role": "user", "content": "About 15 widgets. Some queries take over 2 minutes."},
        {"role": "assistant", "content": "For queries exceeding the 120-second sync limit, use the async endpoint. For the dashboard itself, consider splitting it into multiple dashboards."},
        {"role": "user", "content": "How do I identify which queries are most expensive?"},
        {"role": "assistant", "content": "Check the Query Audit Log in Settings to see MCU cost per query. You can also use EXPLAIN MERIDIAN to estimate cost before running."},
        {"role": "user", "content": "Can I create materialized views to speed things up?"},
    ], "target_docs": ["KB-019", "KB-005"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We're migrating from CSV uploads to the API for data ingestion."},
        {"role": "assistant", "content": "The batch ingestion endpoint POST /api/v2/ingest/batch supports CSV, JSON, Parquet, and Avro. What kind of volumes are you dealing with?"},
        {"role": "user", "content": "Around 200MB per day, currently in CSV."},
        {"role": "assistant", "content": "Since the max payload is 100MB per request, you'll need to split into at least 2-3 batches. Consider converting to Parquet for better compression."},
        {"role": "user", "content": "Good idea. How do I handle schema changes over time?"},
    ], "target_docs": ["KB-004", "KB-018"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to set up authentication for our API integrations."},
        {"role": "assistant", "content": "Meridian supports API key auth with Bearer tokens and OAuth 2.0 client credentials (Enterprise only). Which approach works for your setup?"},
        {"role": "user", "content": "We need automated token refresh. Is OAuth available on Professional?"},
        {"role": "assistant", "content": "OAuth 2.0 client credentials flow is Enterprise-only. On Professional, you can use API keys with configurable expiration and key rotation."},
        {"role": "user", "content": "How do I rotate keys without service interruption?"},
    ], "target_docs": ["KB-003"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We're building a real-time monitoring dashboard."},
        {"role": "assistant", "content": "Real-time dashboards work best with the Streaming add-on for data and 1-minute auto-refresh intervals. What data source are you streaming from?"},
        {"role": "user", "content": "Amazon Kinesis. We need events to appear on the dashboard within seconds."},
        {"role": "assistant", "content": "Kinesis is supported with sub-5-second latency. You'll need the Real-Time Streaming add-on at $750/month."},
        {"role": "user", "content": "What's the maximum event throughput, and what happens if we exceed it?"},
    ], "target_docs": ["KB-014", "KB-011"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to troubleshoot why our connector sync is slow."},
        {"role": "assistant", "content": "There are several potential causes. Have you verified that Meridian's IP ranges are allowlisted in your source system's firewall?"},
        {"role": "user", "content": "Where can I find the IP ranges to allowlist?"},
        {"role": "assistant", "content": "The Meridian IP ranges are listed in Settings > Network. Make sure all listed IPs are allowlisted on your source database."},
        {"role": "user", "content": "Done, but it's still slow. What else could it be?"},
    ], "target_docs": ["KB-019", "KB-009"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to build a custom connector for our proprietary data system."},
        {"role": "assistant", "content": "Custom connectors can be built via the Connector SDK. Are you on the Enterprise tier? The SDK is Enterprise-only."},
        {"role": "user", "content": "Yes, we're on Enterprise. What language is the SDK in?"},
        {"role": "assistant", "content": "The Connector SDK is Python-based. It provides hooks for authentication, schema discovery, and incremental data sync."},
        {"role": "user", "content": "Can I set the custom connector to sync on an hourly schedule?"},
    ], "target_docs": ["KB-009"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We're hitting rate limits during peak hours."},
        {"role": "assistant", "content": "What tier are you on? Rate limits vary significantly between plans."},
        {"role": "user", "content": "Professional. We hit 500 req/min during ETL windows."},
        {"role": "assistant", "content": "Professional allows 500 requests/minute with burst up to 1,000 for 10-second windows. If you need more, Enterprise offers 2,000 req/min."},
        {"role": "user", "content": "Is there any way to spread our requests more evenly to stay under the limit?"},
    ], "target_docs": ["KB-006"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We need to ingest data from Salesforce and HubSpot."},
        {"role": "assistant", "content": "Both are supported as SaaS connectors. Salesforce has one-click OAuth. What sync frequency do you need?"},
        {"role": "user", "content": "Hourly for Salesforce, daily for HubSpot."},
        {"role": "assistant", "content": "Both frequencies are available per-connector in Settings > Data Sources. Salesforce syncs Accounts, Contacts, and Opportunities. HubSpot syncs CRM, marketing, and sales data."},
        {"role": "user", "content": "Can I control which Salesforce objects are synced?"},
    ], "target_docs": ["KB-009"]},
    {"persona": "dana", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to use the new query explain endpoint from API v2.3."},
        {"role": "assistant", "content": "The POST /api/v2/query/explain endpoint returns estimated MCU cost and execution plan without running the query. What kind of queries are you looking to estimate?"},
        {"role": "user", "content": "Large analytical queries that scan millions of rows."},
        {"role": "assistant", "content": "That's a perfect use case. You can also use EXPLAIN MERIDIAN in SQL syntax for similar functionality."},
        {"role": "user", "content": "What's the difference between the API endpoint and the SQL command?"},
    ], "target_docs": ["KB-024", "KB-005"]},
    # Marcus (6)
    {"persona": "marcus", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I'm scoping a deal for a large healthcare organization."},
        {"role": "assistant", "content": "Healthcare orgs typically need Enterprise tier for HIPAA compliance. What are their key requirements?"},
        {"role": "user", "content": "HIPAA BAA, SSO with SAML, and at least 500 users."},
        {"role": "assistant", "content": "Enterprise covers all of that: BAA during onboarding, SAML SSO included, and no user maximum. They'll also get a dedicated CSM."},
        {"role": "user", "content": "What about data encryption? They're very security-conscious."},
        {"role": "assistant", "content": "Enterprise includes AES-256 encryption at rest with the option for customer-managed keys (CMK). TLS 1.3 for transit. HIPAA workspaces have additional controls."},
        {"role": "user", "content": "What are the additional HIPAA workspace controls?"},
    ], "target_docs": ["KB-017", "KB-016", "KB-002"]},
    {"persona": "marcus", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "A retail prospect wants to compare Starter vs Professional."},
        {"role": "assistant", "content": "The key differences are in scale, support, and features. Starter is $49/user with 10M rows and 5 connectors. Professional is $149/user with 1B rows and unlimited connectors."},
        {"role": "user", "content": "They have about 50 million rows of data. Which tier do they need?"},
        {"role": "assistant", "content": "With 50M rows, they exceed Starter's 10M limit. They'll need Professional at minimum."},
        {"role": "user", "content": "What add-ons should I recommend for a retail analytics use case?"},
    ], "target_docs": ["KB-002", "KB-021"]},
    {"persona": "marcus", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "A customer is upset about an unexpected overage charge on their invoice."},
        {"role": "assistant", "content": "Overage is charged at $0.10 per 1,000 rows over the plan limit, billed monthly in arrears. What plan are they on?"},
        {"role": "user", "content": "Professional. They didn't realize they had exceeded 1 billion rows."},
        {"role": "assistant", "content": "An automatic email warning is sent at 80% usage. They should have received a warning. Do they want to upgrade to Enterprise for unlimited rows?"},
        {"role": "user", "content": "Maybe. What's the Enterprise pricing structure?"},
    ], "target_docs": ["KB-021", "KB-020", "KB-002"]},
    {"persona": "marcus", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "A financial services prospect needs to understand our security posture."},
        {"role": "assistant", "content": "We're SOC 2 Type II certified and support HIPAA BAA for Enterprise. What specific security questions do they have?"},
        {"role": "user", "content": "They want to know about encryption, penetration testing, and data residency."},
        {"role": "assistant", "content": "AES-256 at rest, TLS 1.3 in transit. Annual pen testing by a third party. Data hosted on AWS us-east-1 and eu-west-1, with region selection for Enterprise."},
        {"role": "user", "content": "Can they get a copy of the pen test report and SOC 2 audit report?"},
    ], "target_docs": ["KB-015", "KB-016"]},
    {"persona": "marcus", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to explain the free trial to a new prospect."},
        {"role": "assistant", "content": "We offer a 14-day free trial of the Professional tier. No credit card required."},
        {"role": "user", "content": "What happens at the end of the trial if they don't upgrade?"},
        {"role": "assistant", "content": "The workspace transitions to Starter tier features. They can upgrade anytime during or after the trial."},
        {"role": "user", "content": "If they decide to go annual during the trial, do they get the 20% discount?"},
    ], "target_docs": ["KB-007", "KB-002"]},
    {"persona": "marcus", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "A prospect wants to know about data governance capabilities."},
        {"role": "assistant", "content": "The Data Governance module includes column-level access controls, data lineage, PII auto-detection, and audit logging. It's $300/month on Professional or included in Enterprise."},
        {"role": "user", "content": "They're on Professional. What does PII detection cover?"},
        {"role": "assistant", "content": "PII auto-detection scans for email, phone, SSN, credit cards, and 12 additional patterns added in Q1 2025 including passports and driver's licenses."},
        {"role": "user", "content": "Can it automatically mask detected PII?"},
    ], "target_docs": ["KB-013", "KB-024"]},
    # Priya (10)
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to build a comprehensive executive dashboard for our C-suite."},
        {"role": "assistant", "content": "The Dashboard Builder has 15 chart types including KPI cards, line charts, and tables. What metrics do your executives want to see?"},
        {"role": "user", "content": "Revenue trends, customer acquisition, and regional breakdowns."},
        {"role": "assistant", "content": "I'd suggest line charts for trends, KPI cards for key numbers, and a geographic map for regional data. You can add auto-refresh too."},
        {"role": "user", "content": "I also want to schedule a weekly email of this dashboard to the executives."},
    ], "target_docs": ["KB-011", "KB-012"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I'm trying to understand our data pipeline end to end."},
        {"role": "assistant", "content": "The Data Governance module includes data lineage visualization that traces data from source to dashboard. Have you set that up?"},
        {"role": "user", "content": "I didn't know about it. How does lineage tracking work?"},
        {"role": "assistant", "content": "Lineage is automatically tracked for all data flowing through MAP. You can view the lineage graph for any table, column, or dashboard widget."},
        {"role": "user", "content": "Can I see which dashboards depend on a specific data source?"},
    ], "target_docs": ["KB-013"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to set up automated reporting for my team."},
        {"role": "assistant", "content": "You can schedule reports daily, weekly, or monthly, delivered via email as PDF or CSV. How many reports do you need?"},
        {"role": "user", "content": "About 30 different reports for different teams."},
        {"role": "assistant", "content": "On Professional you can have up to 50 scheduled reports. That's within your limit. Each report can include up to 20 widgets."},
        {"role": "user", "content": "Can some reports only send when something unusual happens?"},
    ], "target_docs": ["KB-012"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to analyze data from Shopify and Google Analytics together."},
        {"role": "assistant", "content": "Both are supported as SaaS connectors. Once connected, you can join and analyze data from both sources in a single dashboard."},
        {"role": "user", "content": "How do I connect Shopify?"},
        {"role": "assistant", "content": "Go to Settings > Data Sources > Add New, select Shopify, and follow the connection wizard. Shopify syncs orders, products, and customer data."},
        {"role": "user", "content": "Once both are connected, can I use AskMeridian to query across both data sources?"},
    ], "target_docs": ["KB-009", "KB-010"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to create an interactive dashboard that external partners can explore."},
        {"role": "assistant", "content": "Dashboards can be shared via view-only link or embedded via iframe. External partners can view shared dashboards without logging in."},
        {"role": "user", "content": "Can I restrict what data they see?"},
        {"role": "assistant", "content": "Yes, column-level access controls in the Data Governance module let you restrict specific columns. You can also use dashboard-level filters."},
        {"role": "user", "content": "What about password-protecting the shared link?"},
    ], "target_docs": ["KB-011", "KB-013"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "Our team in France needs to use AskMeridian."},
        {"role": "assistant", "content": "AskMeridian supports French as one of its 5 languages. They can ask questions in French and get results."},
        {"role": "user", "content": "How accurate is it in French compared to English?"},
        {"role": "assistant", "content": "AskMeridian v2 improved accuracy by 40% overall. The same fine-tuned LLM handles all supported languages, though English typically has the highest accuracy."},
        {"role": "user", "content": "Can they also view dashboards and reports in French?"},
    ], "target_docs": ["KB-010", "KB-024"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to monitor inventory levels and get alerted when stock is low."},
        {"role": "assistant", "content": "You can set up a scheduled report with conditional alerts that trigger when a metric crosses a threshold. What threshold do you want?"},
        {"role": "user", "content": "Alert me when any product drops below 100 units."},
        {"role": "assistant", "content": "Configure a conditional alert in your report settings under Delivery Conditions. Set the threshold for stock quantity < 100."},
        {"role": "user", "content": "Can I have this alert go to multiple people on my team?"},
    ], "target_docs": ["KB-012"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to upload a large Excel file with multiple sheets."},
        {"role": "assistant", "content": "MAP supports Excel (.xlsx) uploads with multiple sheets via the UI drag-and-drop. Each sheet is treated as a separate table."},
        {"role": "user", "content": "The file is about 150MB. Will that work?"},
        {"role": "assistant", "content": "The batch ingestion API has a 100MB limit per request. For the UI upload, similar limits apply. You may need to split the file or convert to Parquet for better compression."},
        {"role": "user", "content": "Can I convert to Parquet and upload via the API instead?"},
    ], "target_docs": ["KB-004", "KB-009"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to build a dashboard that auto-refreshes every minute."},
        {"role": "assistant", "content": "1-minute auto-refresh is supported. Keep in mind that each refresh executes all widget queries, which can increase MCU consumption."},
        {"role": "user", "content": "How do I reduce the cost of frequent refreshes?"},
        {"role": "assistant", "content": "Consider using materialized views for expensive queries and reducing the number of widgets on the dashboard."},
        {"role": "user", "content": "What's the impact on our API rate limits with 1-minute refresh?"},
    ], "target_docs": ["KB-011", "KB-019", "KB-006"]},
    {"persona": "priya", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to create a report that includes both visual charts and raw data."},
        {"role": "assistant", "content": "Reports can include up to 20 widgets which can be charts, tables, or KPI cards. For raw data, add a table widget to the report."},
        {"role": "user", "content": "Can I deliver it in multiple formats at once?"},
        {"role": "assistant", "content": "Reports are delivered in a single format per schedule -- either PDF or CSV. You could create two schedules for the same report in different formats."},
        {"role": "user", "content": "Is there a way to create a custom template so I don't have to set this up manually each time?"},
    ], "target_docs": ["KB-012"]},
    # James (8)
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We're evaluating Meridian for our healthcare organization."},
        {"role": "assistant", "content": "Meridian supports healthcare customers with HIPAA compliance. Do you have specific compliance requirements?"},
        {"role": "user", "content": "Yes, we need HIPAA BAA, and all PHI must be in isolated infrastructure."},
        {"role": "assistant", "content": "Enterprise tier provides BAA and single-tenant infrastructure. PHI goes in a dedicated HIPAA workspace with extra security controls."},
        {"role": "user", "content": "What about data encryption? We need CMK for our encryption keys."},
        {"role": "assistant", "content": "Enterprise supports customer-managed keys (CMK) via AWS KMS. AES-256 encryption at rest."},
        {"role": "user", "content": "And can we control which columns our analysts can see to protect patient data?"},
    ], "target_docs": ["KB-017", "KB-016", "KB-013"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to understand the full SSO and user provisioning setup."},
        {"role": "assistant", "content": "SSO supports SAML 2.0 and OIDC. User provisioning can be JIT (auto-create on first login) or SCIM (full lifecycle management, Enterprise only)."},
        {"role": "user", "content": "We want SCIM with our Azure AD. What's needed?"},
        {"role": "assistant", "content": "Generate a SCIM API token in Settings > Security > Provisioning, then configure your Azure AD SCIM integration with the Meridian endpoint and token."},
        {"role": "user", "content": "What happens when a user is deactivated in Azure AD?"},
    ], "target_docs": ["KB-008"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We need to do a security review of Meridian before procurement approval."},
        {"role": "assistant", "content": "We can provide SOC 2 Type II report (under NDA), pen test report (Enterprise), and answer specific security questionnaire items."},
        {"role": "user", "content": "What about network security and DDoS protection?"},
        {"role": "assistant", "content": "All traffic uses TLS 1.3, WAF protects against web exploits, and DDoS protection is provided via AWS Shield. IP allowlisting is available for Enterprise."},
        {"role": "user", "content": "Is there a bug bounty program?"},
    ], "target_docs": ["KB-015"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "Our CISO wants to understand data residency options."},
        {"role": "assistant", "content": "Data is hosted on AWS in us-east-1 and eu-west-1. Enterprise customers can select their preferred region."},
        {"role": "user", "content": "We need EU data residency for GDPR. Can we ensure all data stays in eu-west-1?"},
        {"role": "assistant", "content": "Yes, Enterprise customers can select eu-west-1 during workspace setup. All data processing stays in the selected region."},
        {"role": "user", "content": "What about backups? Are they in the same region?"},
    ], "target_docs": ["KB-015", "KB-016"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "We need field-level encryption for some sensitive columns."},
        {"role": "assistant", "content": "Field-level encryption is available via the Data Governance module. Which columns need encryption?"},
        {"role": "user", "content": "SSN and credit card columns. How does it work?"},
        {"role": "assistant", "content": "Specific columns are encrypted at the field level with separate keys. Only users with explicit access permissions can decrypt."},
        {"role": "user", "content": "Can this be combined with the PII auto-detection feature?"},
    ], "target_docs": ["KB-016", "KB-013"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to set up comprehensive audit logging for compliance."},
        {"role": "assistant", "content": "The Data Governance module provides audit logging. Enterprise retains logs for 7 years, Professional for 1 year."},
        {"role": "user", "content": "We're on Enterprise. Are the audit logs immutable?"},
        {"role": "assistant", "content": "In HIPAA workspaces, audit logs are immutable and retained for 7 years. For standard workspaces, logs are retained for 7 years but may not have the same immutability guarantees."},
        {"role": "user", "content": "Can we export audit logs to our SIEM?"},
    ], "target_docs": ["KB-013", "KB-017"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "What's the session management policy for HIPAA workspaces?"},
        {"role": "assistant", "content": "HIPAA workspaces enforce a 15-minute session timeout, compared to 30 minutes for standard workspaces."},
        {"role": "user", "content": "Can we customize the timeout period?"},
        {"role": "assistant", "content": "The 15-minute timeout is the default for HIPAA compliance. Customization may be available through compliance admin settings."},
        {"role": "user", "content": "What about MFA enforcement? Is it required for HIPAA workspaces?"},
    ], "target_docs": ["KB-017"]},
    {"persona": "james", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to understand the API key security model."},
        {"role": "assistant", "content": "API keys have configurable scopes (read-only, read-write, admin) and configurable expiration (30-365 days)."},
        {"role": "user", "content": "How do we ensure keys aren't compromised?"},
        {"role": "assistant", "content": "Keys can be rotated via POST /api/v2/keys/rotate with a 24-hour grace period. Expiration notifications are sent at 14, 7, and 1 day before."},
        {"role": "user", "content": "For highly sensitive integrations, is there an alternative to API keys?"},
    ], "target_docs": ["KB-003"]},
    # Alex (6)
    {"persona": "alex", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I just joined Meridian AI as a new developer. How do I get set up?"},
        {"role": "assistant", "content": "Welcome! Start by creating your workspace at app.meridian-ai.com/signup. You'll need your company email. Then follow the onboarding checklist."},
        {"role": "user", "content": "Done. I see there's a sample dataset. What's in it?"},
        {"role": "assistant", "content": "The Meridian Demo Retail dataset is pre-loaded with sample retail data including customers, orders, products, and revenue metrics."},
        {"role": "user", "content": "Can I use AskMeridian to explore it? I don't know SQL."},
        {"role": "assistant", "content": "AskMeridian lets you type natural language questions instead of SQL. It's available on Professional and Enterprise tiers."},
        {"role": "user", "content": "Is there a free trial I can use to try Professional features?"},
    ], "target_docs": ["KB-007", "KB-010"]},
    {"persona": "alex", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I need to understand my benefits as a new employee."},
        {"role": "assistant", "content": "Full-time employees get 20 PTO days, 10 company holidays, 5 sick days, and generous parental leave. What would you like to know more about?"},
        {"role": "user", "content": "What about the remote work policy? I was told we're remote-first."},
        {"role": "assistant", "content": "Meridian is remote-first with no mandatory in-office days for most roles. Employees in Austin, NYC, and London are expected in-office 2 days/week."},
        {"role": "user", "content": "I'm in Austin. Which days are in-office?"},
        {"role": "assistant", "content": "Tuesday and Thursday are the in-office days for Austin, NYC, and London employees."},
        {"role": "user", "content": "Do I get anything for setting up my home office?"},
    ], "target_docs": ["KB-022", "KB-023"]},
    {"persona": "alex", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I want to take some time off next month."},
        {"role": "assistant", "content": "You can request PTO through Workday. How many days are you planning to take?"},
        {"role": "user", "content": "About 5 days. Do I need manager approval?"},
        {"role": "assistant", "content": "Yes, PTO requests for more than 3 consecutive days require manager approval. Submit via Workday at least 2 weeks in advance."},
        {"role": "user", "content": "How many PTO days do I have available? I started 2 months ago."},
    ], "target_docs": ["KB-022"]},
    {"persona": "alex", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I'm thinking about visiting my family in India for 4 weeks while working remote."},
        {"role": "assistant", "content": "International remote work is allowed for up to 30 days per year with manager and Legal approval."},
        {"role": "user", "content": "How do I request this?"},
        {"role": "assistant", "content": "Submit a request in Workday at least 2 weeks in advance with the country, dates, and duration. Your manager and Legal will review it."},
        {"role": "user", "content": "What if I want to stay longer than 30 days?"},
    ], "target_docs": ["KB-023"]},
    {"persona": "alex", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I'm confused about the difference between PTO, sick days, and company holidays."},
        {"role": "assistant", "content": "PTO is 20 discretionary days for vacation or personal time. Sick days are 5 days for illness. Company holidays are 10 fixed days where the whole company is off."},
        {"role": "user", "content": "Do sick days roll over like PTO?"},
        {"role": "assistant", "content": "No, sick days do not accrue and do not roll over to the next year. PTO rolls over up to 10 days."},
        {"role": "user", "content": "What if I get really sick and need more than 5 days?"},
    ], "target_docs": ["KB-022"]},
    {"persona": "alex", "variant": "multi_turn_complex", "turns": [
        {"role": "user", "content": "I just finished the onboarding checklist. What should I learn next?"},
        {"role": "assistant", "content": "Now that you've set up your workspace and built your first dashboard, you could explore AskMeridian for natural language queries or try connecting additional data sources."},
        {"role": "user", "content": "How do I use AskMeridian? I saw it in the nav bar."},
        {"role": "assistant", "content": "Click on AskMeridian and type a question in plain language, like 'What were total sales last month?'. It generates and runs SQL for you."},
        {"role": "user", "content": "What if it gives me wrong results?"},
    ], "target_docs": ["KB-010", "KB-019"]},
]

# === UNANSWERABLE (40) ===
UNANSWERABLE_QUESTIONS = [
    # Topic 1: Stock price / IPO (8)
    {"persona": "marcus", "question": "What is Meridian AI's current stock price?", "topic": "stock_ipo"},
    {"persona": "marcus", "question": "When is Meridian AI planning to go public?", "topic": "stock_ipo"},
    {"persona": "marcus", "question": "What was Meridian AI's revenue last quarter?", "topic": "stock_ipo"},
    {"persona": "dana", "question": "How many shares of Meridian AI are outstanding?", "topic": "stock_ipo"},
    {"persona": "alex", "question": "Does Meridian AI offer employee stock options?", "topic": "stock_ipo"},
    {"persona": "marcus", "question": "What is Meridian AI's market capitalization?", "topic": "stock_ipo"},
    {"persona": "james", "question": "Who are Meridian AI's major institutional investors?", "topic": "stock_ipo"},
    {"persona": "priya", "question": "What was Meridian AI's valuation at the last funding round?", "topic": "stock_ipo"},
    # Topic 2: Competitor comparison (8)
    {"persona": "marcus", "question": "How does Meridian compare to Tableau in terms of features?", "topic": "competitor"},
    {"persona": "marcus", "question": "Is Meridian better than Looker for enterprise analytics?", "topic": "competitor"},
    {"persona": "priya", "question": "What are the main differences between Meridian and Power BI?", "topic": "competitor"},
    {"persona": "priya", "question": "Can you compare Meridian's pricing to Snowflake's?", "topic": "competitor"},
    {"persona": "dana", "question": "Does Meridian support dbt like Databricks does?", "topic": "competitor"},
    {"persona": "marcus", "question": "Why should I choose Meridian over ThoughtSpot?", "topic": "competitor"},
    {"persona": "dana", "question": "How does Meridian's query performance compare to BigQuery?", "topic": "competitor"},
    {"persona": "james", "question": "Is Meridian more secure than Looker?", "topic": "competitor"},
    # Topic 3: On-premise deployment (8)
    {"persona": "james", "question": "Can we deploy Meridian on-premise in our own data center?", "topic": "on_premise"},
    {"persona": "james", "question": "Does Meridian offer a self-hosted version?", "topic": "on_premise"},
    {"persona": "dana", "question": "Can we run Meridian on our own Kubernetes cluster?", "topic": "on_premise"},
    {"persona": "james", "question": "Is there an air-gapped deployment option for Meridian?", "topic": "on_premise"},
    {"persona": "dana", "question": "Can Meridian be installed on AWS GovCloud?", "topic": "on_premise"},
    {"persona": "marcus", "question": "Does Meridian offer a private cloud deployment for government clients?", "topic": "on_premise"},
    {"persona": "james", "question": "Can we host Meridian on Azure instead of AWS?", "topic": "on_premise"},
    {"persona": "dana", "question": "Is there a Docker image for self-hosting Meridian?", "topic": "on_premise"},
    # Topic 4: Mobile app (8)
    {"persona": "priya", "question": "Is there a Meridian mobile app for iOS?", "topic": "mobile_app"},
    {"persona": "priya", "question": "Can I view dashboards on my Android phone?", "topic": "mobile_app"},
    {"persona": "marcus", "question": "When will the Meridian mobile app be released?", "topic": "mobile_app"},
    {"persona": "alex", "question": "Can I get push notifications from Meridian on my phone?", "topic": "mobile_app"},
    {"persona": "priya", "question": "Is there a tablet-optimized version of Meridian?", "topic": "mobile_app"},
    {"persona": "marcus", "question": "Does Meridian have an Apple Watch companion app?", "topic": "mobile_app"},
    {"persona": "alex", "question": "Can I access AskMeridian from my phone?", "topic": "mobile_app"},
    {"persona": "dana", "question": "Is there a mobile SDK for building custom mobile dashboards?", "topic": "mobile_app"},
    # Topic 5: Internal engineering architecture (8)
    {"persona": "dana", "question": "What database does Meridian use internally for storing customer data?", "topic": "internal_arch"},
    {"persona": "dana", "question": "Is the Meridian backend built with Python or Go?", "topic": "internal_arch"},
    {"persona": "james", "question": "What message queue does Meridian use internally?", "topic": "internal_arch"},
    {"persona": "dana", "question": "How does Meridian's query engine work under the hood?", "topic": "internal_arch"},
    {"persona": "james", "question": "What container orchestration system does Meridian use?", "topic": "internal_arch"},
    {"persona": "dana", "question": "Does Meridian use a microservices or monolithic architecture?", "topic": "internal_arch"},
    {"persona": "james", "question": "What CI/CD pipeline does Meridian use for deployments?", "topic": "internal_arch"},
    {"persona": "dana", "question": "What ML framework powers the AskMeridian query generation?", "topic": "internal_arch"},
]

# === CONFLICTING_CONTEXT (30) ===
# Each has a question, two manually-specified conflicting chunks, and the conflict type
CONFLICTING_CONTEXT_QUESTIONS = [
    # Numeric discrepancy (10)
    {"persona": "dana", "question": "What is the API rate limit for the Professional tier?",
     "conflict_type": "numeric", "target_docs": ["KB-006", "KB-021"],
     "chunk_override": "Professional tier: 1,000 requests per minute, 100,000 requests per day.",
     "chunk_override_source": "API Rate Limits & Quotas (updated draft)"},
    {"persona": "marcus", "question": "What is the row limit for the Starter plan?",
     "conflict_type": "numeric", "target_docs": ["KB-002", "KB-021"],
     "chunk_override": "Starter plan: 5 million rows, 5 data source connectors.",
     "chunk_override_source": "Usage Limits & Overage Charges (internal revision)"},
    {"persona": "dana", "question": "What is the maximum payload size for batch ingestion?",
     "conflict_type": "numeric", "target_docs": ["KB-004", "KB-018"],
     "chunk_override": "The batch ingestion endpoint accepts payloads up to 200MB per request.",
     "chunk_override_source": "Data Ingestion API -- Endpoints & Examples (beta docs)"},
    {"persona": "priya", "question": "How many scheduled reports can Professional users create?",
     "conflict_type": "numeric", "target_docs": ["KB-012", "KB-021"],
     "chunk_override": "Professional tier: 100 scheduled reports per workspace.",
     "chunk_override_source": "Usage Limits & Overage Charges (draft update)"},
    {"persona": "dana", "question": "What is the maximum query execution time for synchronous queries?",
     "conflict_type": "numeric", "target_docs": ["KB-005", "KB-019"],
     "chunk_override": "Synchronous queries have a maximum execution time of 60 seconds.",
     "chunk_override_source": "Query API -- Running Analytics Queries (deprecated v2.1)"},
    {"persona": "marcus", "question": "What is the daily API request limit for the Enterprise tier?",
     "conflict_type": "numeric", "target_docs": ["KB-006"],
     "chunk_override": "Enterprise: 2,000 requests/minute, 500,000 requests/day.",
     "chunk_override_source": "API Rate Limits & Quotas (pre-release notes)"},
    {"persona": "dana", "question": "What is the default page size for query result pagination?",
     "conflict_type": "numeric", "target_docs": ["KB-005"],
     "chunk_override": "Query results are paginated at 5,000 rows per page by default, configurable up to 50,000.",
     "chunk_override_source": "Query API -- Running Analytics Queries (v2.0 docs)"},
    {"persona": "priya", "question": "How many chart types does the Dashboard Builder support?",
     "conflict_type": "numeric", "target_docs": ["KB-011", "KB-024"],
     "chunk_override": "The Dashboard Builder includes 17 chart types including the new heatmap and treemap.",
     "chunk_override_source": "Release Notes -- Q1 2025 (marketing materials)"},
    {"persona": "marcus", "question": "What is the price of the Real-Time Streaming add-on?",
     "conflict_type": "numeric", "target_docs": ["KB-014", "KB-002"],
     "chunk_override": "Real-Time Streaming add-on: $500/month for Professional and Enterprise tiers.",
     "chunk_override_source": "Pricing Plans & Feature Comparison (promotional pricing)"},
    {"persona": "dana", "question": "How many events per second can the streaming add-on handle?",
     "conflict_type": "numeric", "target_docs": ["KB-014"],
     "chunk_override": "Maximum throughput: 100,000 events/second per workspace for Enterprise, 25,000 for Professional.",
     "chunk_override_source": "Real-Time Streaming Add-On (capacity planning guide)"},
    # Feature availability discrepancy (10)
    {"persona": "james", "question": "Is SSO available on the Starter plan?",
     "conflict_type": "feature", "target_docs": ["KB-008", "KB-002"],
     "chunk_override": "SSO is available on all tiers. Starter includes basic SAML 2.0 SSO.",
     "chunk_override_source": "Single Sign-On (SSO) Configuration Guide (sales enablement)"},
    {"persona": "marcus", "question": "Which tiers support the Data Governance add-on?",
     "conflict_type": "feature", "target_docs": ["KB-013", "KB-002"],
     "chunk_override": "The Data Governance module is available on all tiers: $150/month for Starter, $300/month for Professional, included in Enterprise.",
     "chunk_override_source": "Data Governance Add-On (partner portal docs)"},
    {"persona": "james", "question": "Is OAuth 2.0 supported on the Professional plan?",
     "conflict_type": "feature", "target_docs": ["KB-003"],
     "chunk_override": "OAuth 2.0 client credentials flow is supported on Professional and Enterprise tiers.",
     "chunk_override_source": "API Authentication & API Keys (developer preview)"},
    {"persona": "priya", "question": "Is AskMeridian available on the Starter plan?",
     "conflict_type": "feature", "target_docs": ["KB-010", "KB-002"],
     "chunk_override": "AskMeridian basic mode is available on all tiers. Advanced NLQ features require Professional or Enterprise.",
     "chunk_override_source": "AskMeridian Natural Language Query Interface (marketing brief)"},
    {"persona": "dana", "question": "Can Starter plan customers build custom connectors?",
     "conflict_type": "feature", "target_docs": ["KB-009"],
     "chunk_override": "The Connector SDK is available on Professional and Enterprise tiers for building custom connectors.",
     "chunk_override_source": "Connecting Data Sources -- Supported Connectors (updated FAQ)"},
    {"persona": "james", "question": "Is customer-managed encryption available on Professional?",
     "conflict_type": "feature", "target_docs": ["KB-016"],
     "chunk_override": "Customer-managed keys (CMK) are available on Professional ($200/month add-on) and Enterprise (included).",
     "chunk_override_source": "Data Encryption -- At Rest & In Transit (sales quote template)"},
    {"persona": "james", "question": "Which tiers get access to penetration test reports?",
     "conflict_type": "feature", "target_docs": ["KB-015"],
     "chunk_override": "Penetration test reports are available to Professional and Enterprise customers upon request under NDA.",
     "chunk_override_source": "Security Architecture & Certifications (customer FAQ)"},
    {"persona": "priya", "question": "Can Starter plan users embed dashboards via iframe?",
     "conflict_type": "feature", "target_docs": ["KB-011"],
     "chunk_override": "Dashboard embedding via iframe is available on all tiers with a shared link.",
     "chunk_override_source": "Dashboard Builder -- Creating & Sharing Dashboards (help center draft)"},
    {"persona": "marcus", "question": "Is SCIM provisioning available on Professional?",
     "conflict_type": "feature", "target_docs": ["KB-008"],
     "chunk_override": "SCIM provisioning for automated user management is available on Professional and Enterprise tiers.",
     "chunk_override_source": "Single Sign-On (SSO) Configuration Guide (integration guide)"},
    {"persona": "dana", "question": "Does the Starter plan support real-time streaming ingestion?",
     "conflict_type": "feature", "target_docs": ["KB-014"],
     "chunk_override": "Real-Time Streaming is available as an add-on on all tiers: $250/month for Starter, $500/month for Professional, $750/month for Enterprise.",
     "chunk_override_source": "Real-Time Streaming Add-On (early access pricing)"},
    # Stale vs current info (10)
    {"persona": "priya", "question": "Does AskMeridian support Japanese?",
     "conflict_type": "stale", "target_docs": ["KB-010", "KB-024"],
     "chunk_override": "AskMeridian supports English, Spanish, French, and German. Additional languages are planned for future releases.",
     "chunk_override_source": "AskMeridian Natural Language Query Interface (Q4 2024 docs)"},
    {"persona": "dana", "question": "Is the Databricks connector available?",
     "conflict_type": "stale", "target_docs": ["KB-009", "KB-024"],
     "chunk_override": "Databricks Unity Catalog connector is in beta and expected to be generally available in Q1 2025.",
     "chunk_override_source": "Release Notes -- Q4 2024"},
    {"persona": "james", "question": "Does Meridian support OIDC for SSO?",
     "conflict_type": "stale", "target_docs": ["KB-008", "KB-025"],
     "chunk_override": "SSO is supported via SAML 2.0 only. OpenID Connect support is planned for Q4 2024.",
     "chunk_override_source": "Single Sign-On (SSO) Configuration Guide (pre-Q4 2024)"},
    {"persona": "priya", "question": "Are heatmap and treemap chart types available?",
     "conflict_type": "stale", "target_docs": ["KB-011", "KB-024"],
     "chunk_override": "The Dashboard Builder includes 13 chart types: bar, line, area, pie, donut, scatter, table, KPI card, gauge, funnel, histogram, combo, and map.",
     "chunk_override_source": "Dashboard Builder -- Creating & Sharing Dashboards (pre-Q1 2025)"},
    {"persona": "dana", "question": "Is there a query cost estimation endpoint?",
     "conflict_type": "stale", "target_docs": ["KB-005", "KB-024"],
     "chunk_override": "Query cost estimation is available via the EXPLAIN MERIDIAN SQL command. A dedicated API endpoint is planned for API v2.3.",
     "chunk_override_source": "Query API -- Running Analytics Queries (pre-Q1 2025)"},
    {"persona": "james", "question": "How many PII patterns does the auto-detection support?",
     "conflict_type": "stale", "target_docs": ["KB-013", "KB-024"],
     "chunk_override": "PII auto-detection supports 5 patterns: email, phone, SSN, credit card numbers, and IP addresses.",
     "chunk_override_source": "Data Governance Add-On (Q3 2024 docs)"},
    {"persona": "priya", "question": "Has dashboard rendering performance been optimized recently?",
     "conflict_type": "stale", "target_docs": ["KB-019", "KB-025"],
     "chunk_override": "Dashboard rendering for complex dashboards (10+ widgets) currently takes 8-10 seconds. Performance optimization is on the Q4 2024 roadmap.",
     "chunk_override_source": "Dashboard Performance & Query Optimization (pre-Q4 2024)"},
    {"persona": "dana", "question": "Does the streaming add-on support Google Pub/Sub?",
     "conflict_type": "stale", "target_docs": ["KB-014", "KB-025"],
     "chunk_override": "The Real-Time Streaming add-on currently supports Kafka and Kinesis. Google Pub/Sub support is planned for early 2025.",
     "chunk_override_source": "Release Notes -- Q4 2024 (initial release)"},
    {"persona": "james", "question": "Has the SOC 2 Type II certification been renewed recently?",
     "conflict_type": "stale", "target_docs": ["KB-015", "KB-025"],
     "chunk_override": "Meridian AI has been SOC 2 Type II certified since 2023. The next renewal audit is scheduled for Q4 2024.",
     "chunk_override_source": "Security Architecture & Certifications (mid-2024 docs)"},
    {"persona": "dana", "question": "Does the async query endpoint exist?",
     "conflict_type": "stale", "target_docs": ["KB-005", "KB-025"],
     "chunk_override": "Long-running queries are limited by the 120-second synchronous timeout. An async query endpoint is planned for API v2.2.",
     "chunk_override_source": "Query API -- Running Analytics Queries (pre-Q4 2024)"},
]


# ---------------------------------------------------------------------------
# Build all prompts
# ---------------------------------------------------------------------------
def build_all_prompts(bm25, chunks: list[dict]) -> list[dict]:
    """Build all 300 prompts with BM25 retrieval."""
    all_prompts = []

    # Single-turn factual (80)
    for q in SINGLE_TURN_FACTUAL_QUESTIONS:
        retrieved = retrieve(q["question"], bm25, chunks, TOP_K)
        prompt = build_single_turn_prompt(q["question"], retrieved)
        all_prompts.append({
            "prompt": prompt,
            "variant": "single_turn_factual",
            "persona": q["persona"],
            "target_docs": q.get("target_docs", []),
        })

    # Single-turn procedural (60)
    for q in SINGLE_TURN_PROCEDURAL_QUESTIONS:
        retrieved = retrieve(q["question"], bm25, chunks, TOP_K)
        prompt = build_single_turn_prompt(q["question"], retrieved)
        all_prompts.append({
            "prompt": prompt,
            "variant": "single_turn_procedural",
            "persona": q["persona"],
            "target_docs": q.get("target_docs", []),
        })

    # Multi-turn clarification (50)
    for q in MULTI_TURN_CLARIFICATION:
        # Use the full conversation to build a combined query for retrieval
        user_turns = [t["content"] for t in q["turns"] if t["role"] == "user"]
        combined_query = " ".join(user_turns)
        retrieved = retrieve(combined_query, bm25, chunks, TOP_K)
        prompt = build_multi_turn_prompt(q["turns"], retrieved)
        all_prompts.append({
            "prompt": prompt,
            "variant": "multi_turn_clarification",
            "persona": q["persona"],
            "target_docs": q.get("target_docs", []),
        })

    # Multi-turn complex (40)
    for q in MULTI_TURN_COMPLEX:
        user_turns = [t["content"] for t in q["turns"] if t["role"] == "user"]
        combined_query = " ".join(user_turns)
        retrieved = retrieve(combined_query, bm25, chunks, TOP_K)
        prompt = build_multi_turn_prompt(q["turns"], retrieved)
        all_prompts.append({
            "prompt": prompt,
            "variant": "multi_turn_complex",
            "persona": q["persona"],
            "target_docs": q.get("target_docs", []),
        })

    # Unanswerable (40)
    for q in UNANSWERABLE_QUESTIONS:
        retrieved = retrieve(q["question"], bm25, chunks, TOP_K)
        prompt = build_single_turn_prompt(q["question"], retrieved)
        all_prompts.append({
            "prompt": prompt,
            "variant": "unanswerable",
            "persona": q["persona"],
            "target_docs": [],
        })

    # Conflicting context (30)
    for q in CONFLICTING_CONTEXT_QUESTIONS:
        # Get one real chunk via BM25
        retrieved = retrieve(q["question"], bm25, chunks, 1)
        # Create the override chunk (the conflicting one)
        override_chunk = {
            "doc_id": "OVERRIDE",
            "title": q["chunk_override_source"],
            "section": "Override",
            "text": q["chunk_override"],
        }
        # Get a second real chunk for additional context
        extra_retrieved = retrieve(q["question"], bm25, chunks, 3)
        # Pick a real chunk that isn't the same as the first
        second_real = None
        for c in extra_retrieved:
            if c["text"] != retrieved[0]["text"]:
                second_real = c
                break
        if second_real is None:
            second_real = extra_retrieved[-1] if len(extra_retrieved) > 1 else retrieved[0]

        # Assemble: override chunk + real chunk + second real chunk
        conflict_chunks = [override_chunk, retrieved[0], second_real]
        prompt = build_single_turn_prompt(q["question"], conflict_chunks)
        all_prompts.append({
            "prompt": prompt,
            "variant": "conflicting_context",
            "persona": q["persona"],
            "target_docs": q.get("target_docs", []),
        })

    print(f"Total prompts built: {len(all_prompts)}")

    # Verify distribution
    variant_counts: dict[str, int] = {}
    for p in all_prompts:
        v = p["variant"]
        variant_counts[v] = variant_counts.get(v, 0) + 1
    print(f"Variant distribution: {variant_counts}")

    persona_counts: dict[str, int] = {}
    for p in all_prompts:
        per = p["persona"]
        persona_counts[per] = persona_counts.get(per, 0) + 1
    print(f"Persona distribution: {persona_counts}")

    return all_prompts


# ---------------------------------------------------------------------------
# Model calling with cost tracking
# ---------------------------------------------------------------------------
class CostTracker:
    def __init__(self):
        self.costs: dict[str, dict[str, float]] = {}

    def add(self, model: str, phase: str, cost: float):
        if model not in self.costs:
            self.costs[model] = {"tuning": 0.0, "production": 0.0}
        self.costs[model][phase] += cost

    def summary(self) -> list[dict]:
        result = []
        for model, phases in self.costs.items():
            result.append({
                "model": model,
                "tuning_cost_usd": round(phases["tuning"], 6),
                "production_cost_usd": round(phases["production"], 6),
                "total_cost_usd": round(phases["tuning"] + phases["production"], 6),
                "pairs_generated": 300,
            })
        return result


def call_model(
    model: str,
    api_key: str,
    prompt: str,
    tracker: CostTracker,
    phase: str = "production",
    max_retries: int = 5,
) -> tuple[str, float]:
    """Call an LLM model with retry logic. Returns (response_text, cost)."""
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model,
                api_key=api_key,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content.strip()
            cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
            tracker.add(model, phase, cost)
            return text, cost
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower():
                wait = min(60, 5 * (attempt + 1))
                print(f"  Rate limited, waiting {wait}s... (attempt {attempt + 1})")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2**attempt
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}")
                return f"Error: API call failed after {max_retries} attempts.", 0.0


# ---------------------------------------------------------------------------
# Prompt tuning
# ---------------------------------------------------------------------------
def run_tuning(prompts: list[dict], tracker: CostTracker) -> None:
    """Run prompt tuning on 30 mixed samples using GPT-4o."""
    import random

    print("\n=== Prompt Tuning (30 samples with GPT-4o) ===")

    # Select 5 per variant
    by_variant: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        v = p["variant"]
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(i)

    random.seed(99)
    tuning_samples = []
    for variant in DISTRIBUTION:
        indices = by_variant.get(variant, [])
        sample_size = min(5, len(indices))
        chosen = random.sample(indices, sample_size)
        tuning_samples.extend(chosen)

    print(f"Tuning on {len(tuning_samples)} samples...")

    api_key = os.environ["OPENROUTER_API_KEY"]
    model = "openrouter/openai/gpt-4o"

    non_empty = 0
    refusals = 0
    total = len(tuning_samples)

    for idx in tuning_samples:
        pair = prompts[idx]
        prompt = pair["prompt"]
        variant = pair["variant"]

        response, cost = call_model(model, api_key, prompt, tracker, phase="tuning")

        if response and len(response) > 10:
            non_empty += 1
        else:
            print(f"  [TUNE WARN] Empty/short response for {variant}: {response[:80]}")

        if variant == "unanswerable":
            # Check if model correctly says it can't answer
            lower = response.lower()
            if any(
                phrase in lower
                for phrase in [
                    "don't have",
                    "not available",
                    "not contain",
                    "no information",
                    "doesn't contain",
                    "does not contain",
                    "cannot find",
                    "not covered",
                    "does not provide",
                    "doesn't provide",
                    "not included",
                    "not mentioned",
                    "no documentation",
                    "don't see",
                ]
            ):
                pass  # Good
            else:
                refusals += 1
                print(f"  [TUNE WARN] Unanswerable but model answered: {response[:100]}")

        time.sleep(1.0)

    response_rate = non_empty / total * 100 if total > 0 else 0
    print(f"\nTuning Results:")
    print(f"  Non-empty responses: {non_empty}/{total} ({response_rate:.1f}%)")
    print(f"  Unanswerable issues: {refusals}")
    print(f"  Tuning cost: ${tracker.costs.get(model, {}).get('tuning', 0):.4f}")


# ---------------------------------------------------------------------------
# Full production run
# ---------------------------------------------------------------------------
def run_production(
    prompts: list[dict],
    model: str,
    api_key: str,
    output_path: Path,
    tracker: CostTracker,
    source_model_label: str,
    sleep_between: float = 1.0,
) -> None:
    """Run full production for one model, writing JSONL incrementally."""
    print(f"\n=== Production Run: {source_model_label} -> {output_path.name} ===")

    # Check for existing records to support resume
    existing_count = 0
    if output_path.exists():
        with open(output_path) as f:
            existing_count = sum(1 for line in f if line.strip())
        if existing_count > 0:
            print(f"  Resuming from record {existing_count}")

    if existing_count >= len(prompts):
        print(f"  Already complete ({existing_count} records). Skipping.")
        return

    mode = "a" if existing_count > 0 else "w"
    non_empty = 0
    fail_count = 0

    with open(output_path, mode) as fout:
        for i, pair in enumerate(prompts):
            if i < existing_count:
                continue

            prompt_text = pair["prompt"]
            response, cost = call_model(model, api_key, prompt_text, tracker)

            # Clean response: strip markdown fences
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                response = response.strip()

            metadata: dict[str, Any] = {
                "dataset": "enterprise_rag",
                "variant": pair["variant"],
                "persona": pair["persona"],
            }
            if pair.get("target_docs"):
                metadata["target_docs"] = pair["target_docs"]

            record = {
                "prompt": prompt_text,
                "response": response,
                "source_model": source_model_label,
                "metadata": metadata,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if response and len(response) > 10:
                non_empty += 1
            else:
                fail_count += 1

            # Progress
            total_done = i + 1
            if total_done % 50 == 0 or total_done == len(prompts):
                current_cost = tracker.costs.get(model, {}).get("production", 0)
                print(
                    f"  [{total_done}/{len(prompts)}] "
                    f"non_empty={non_empty} "
                    f"fails={fail_count} "
                    f"cost=${current_cost:.4f}"
                )

            time.sleep(sleep_between)

    print(f"\n  Final: non_empty={non_empty}, fails={fail_count}")


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(output_path: Path, model: str, api_key: str, tracker: CostTracker,
                sleep_between: float = 1.0) -> None:
    """Fix empty or low-quality responses."""
    print(f"\n=== Post-processing: {output_path.name} ===")

    records = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    fixes = 0
    for i, rec in enumerate(records):
        response = rec["response"]

        # Fix empty or error responses
        if not response or len(response) < 10 or response.startswith("Error:"):
            print(f"  Re-running record {i} (variant={rec['metadata']['variant']})")
            new_resp, _ = call_model(model, api_key, rec["prompt"], tracker)
            if new_resp and len(new_resp) > 10:
                records[i]["response"] = new_resp
                fixes += 1
            time.sleep(sleep_between)

    # Write back
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Total fixes: {fixes}")


# ---------------------------------------------------------------------------
# Spot check
# ---------------------------------------------------------------------------
def spot_check(output_path: Path) -> dict[str, int]:
    """Spot-check and return quality metrics."""
    print(f"\n=== Spot-Check: {output_path.name} ===")

    records = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    by_variant: dict[str, list[dict]] = {}
    for r in records:
        v = r["metadata"]["variant"]
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(r)

    metrics = {"total": len(records), "non_empty": 0, "issues": 0}

    for variant, vrecs in sorted(by_variant.items()):
        print(f"\n  Variant: {variant} ({len(vrecs)} records)")
        samples = vrecs[:3]
        for j, rec in enumerate(samples):
            resp = rec["response"]
            if resp and len(resp) > 10:
                preview = resp[:100].replace("\n", " ")
                print(f"    [{j}] OK: {preview}")
            else:
                print(f"    [{j}] ISSUE: empty/short response")
                metrics["issues"] += 1

    for r in records:
        if r["response"] and len(r["response"]) > 10:
            metrics["non_empty"] += 1

    rate = metrics["non_empty"] / metrics["total"] * 100 if metrics["total"] > 0 else 0
    print(f"\n  Summary: {metrics['non_empty']}/{metrics['total']} non-empty ({rate:.1f}%)")
    print(f"  Issues: {metrics['issues']}")

    return metrics


# ---------------------------------------------------------------------------
# SOURCES.md
# ---------------------------------------------------------------------------
def write_sources(output_dir: Path) -> None:
    """Write SOURCES.md provenance document."""
    content = """\
# Enterprise RAG Dataset -- Sources & Provenance

## Dataset
- **Name:** enterprise_rag
- **Records:** 300 prompt-response pairs per model (600 total)
- **Models:** openai/gpt-4o, anthropic/claude-haiku-4-5-20251001

## Knowledge Base
- **25 synthetic Markdown documents** describing the fictional Meridian AI company
- All content is original, hand-authored for this dataset
- No real company data or copyrighted material is used

## Retrieval
- **BM25** (rank_bm25 library) for deterministic keyword-based retrieval
- Top-3 chunks per query
- Sentence-aware paragraph chunking (~150 words per chunk)

## Questions
- All 300 questions are hand-authored in the generation script
- No LLM-generated questions
- 5 user personas: Dana (Data Engineer), Marcus (VP Sales), Priya (Analyst),
  James (IT Security), Alex (New Hire)

## Variants
| Variant | Count | Description |
|---------|-------|-------------|
| single_turn_factual | 80 | Direct fact lookup from KB |
| single_turn_procedural | 60 | How-to / step-by-step questions |
| multi_turn_clarification | 50 | 3-turn conversations with clarification |
| multi_turn_complex | 40 | 4-5 turn conversations with topic evolution |
| unanswerable | 40 | Questions about topics not in KB |
| conflicting_context | 30 | Contradictory retrieved context |

## Conflicting Context Design
- 10 numeric discrepancies (different numbers for same fact)
- 10 feature availability discrepancies (different tier requirements)
- 10 stale vs. current information conflicts

## Licensing
- All synthetic content: original, no external license required
- rank_bm25: Apache 2.0
- LiteLLM: MIT
"""
    (output_dir / "SOURCES.md").write_text(content)
    print(f"SOURCES.md written to {output_dir / 'SOURCES.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Enterprise RAG dataset")
    parser.add_argument("--tuning-only", action="store_true", help="Run only the tuning step")
    args = parser.parse_args()

    load_dotenv(ENV_PATH)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openrouter_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("API keys loaded successfully.")

    tracker = CostTracker()

    # Step 1: Chunk documents and build BM25 index
    print("\n=== Step 1: Chunking Documents ===")
    chunks = chunk_documents()

    print("\n=== Step 2: Building BM25 Index ===")
    bm25 = build_bm25_index(chunks)

    # Step 3: Build all prompts
    print("\n=== Step 3: Building Prompts ===")
    all_prompts = build_all_prompts(bm25, chunks)
    assert len(all_prompts) == TOTAL_PAIRS, f"Expected {TOTAL_PAIRS} prompts, got {len(all_prompts)}"

    # Step 4: Tuning
    run_tuning(all_prompts, tracker)

    if args.tuning_only:
        print("\n=== Tuning complete. Exiting (--tuning-only). ===")
        cost_summary = tracker.summary()
        print(json.dumps(cost_summary, indent=2))
        return

    # Step 5: Production runs
    gpt4o_path = OUTPUT_DIR / "enterprise_rag_gpt4o.jsonl"
    haiku_path = OUTPUT_DIR / "enterprise_rag_haiku.jsonl"

    # GPT-4o
    run_production(
        prompts=all_prompts,
        model="openrouter/openai/gpt-4o",
        api_key=openrouter_key,
        output_path=gpt4o_path,
        tracker=tracker,
        source_model_label="openai/gpt-4o",
        sleep_between=1.0,
    )
    postprocess(gpt4o_path, "openrouter/openai/gpt-4o", openrouter_key, tracker, sleep_between=1.0)

    # Haiku
    run_production(
        prompts=all_prompts,
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=anthropic_key,
        output_path=haiku_path,
        tracker=tracker,
        source_model_label="anthropic/claude-haiku-4-5-20251001",
        sleep_between=1.5,
    )
    postprocess(haiku_path, "anthropic/claude-haiku-4-5-20251001", anthropic_key, tracker, sleep_between=1.5)

    # Step 6: Write cost summary
    cost_summary = tracker.summary()
    cost_path = OUTPUT_DIR / "cost_summary.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"\nCost summary written to {cost_path}")
    print(json.dumps(cost_summary, indent=2))

    # Write SOURCES.md
    write_sources(OUTPUT_DIR)

    # Spot-checks
    spot_check(gpt4o_path)
    spot_check(haiku_path)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
