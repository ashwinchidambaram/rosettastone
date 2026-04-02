#!/usr/bin/env python3
"""Generate fintech_extraction dataset: 400 synthetic invoice texts + LLM responses.

Usage:
    uv run python scripts/generate_fintech_extraction.py --seed 42 --output examples/datasets/fintech_extraction
    uv run python scripts/generate_fintech_extraction.py --seed 42 --generate-only  # skip LLM calls
    uv run python scripts/generate_fintech_extraction.py --seed 42 --tune-only      # only prompt tuning
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from faker import Faker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VENDOR_COMPANIES: list[str] = [
    "Meridian Technology Solutions LLC",
    "Pacific Rim Manufacturing Co.",
    "Bluewater Consulting Group",
    "Summit Ridge Logistics Inc.",
    "Cascade Data Systems",
    "Ironclad Security Partners",
    "NovaBridge Financial Services",
    "Pinnacle Cloud Infrastructure",
    "Sterling Digital Media Corp.",
    "Atlas Workforce Solutions",
    "Harborview Analytics Group",
    "Redwood Enterprise Software",
    "Evergreen Supply Chain Management",
    "Vanguard Compliance Services LLC",
    "Cobalt Engineering Consultants",
    "Northstar IT Staffing Solutions",
    "Apex Industrial Components Inc.",
    "Silverline Communications Ltd.",
    "Terraforming Data Solutions",
    "Crescent Bay Health Technologies",
    "Whitfield & Associates Consulting",
    "Stratford Global Sourcing",
    "Keystone Project Management Inc.",
    "Blue Horizon Renewable Energy",
    "Granite Peak Construction Services",
    "Falcon Cybersecurity Solutions",
    "Oaktree Legal Technologies",
    "Bayshore Medical Devices Corp.",
    "Lakeshore Software Development",
    "Pathfinder Marketing Analytics LLC",
    "Eclipse Semiconductor Inc.",
    "Trident Aerospace Components",
    "Ridgeline Capital Advisors",
    "Birchwood Packaging Solutions",
    "Clearwater Environmental Services",
    "Nexus Industrial Automation",
    "Sagebrush Freight Logistics",
    "Brightfield Solar Technologies",
    "Copperline Electrical Contractors",
    "Windhaven Telecommunications",
    "Stonewall Architectural Design",
    "Riverbend Agricultural Systems",
    "Highpoint SaaS Platforms",
    "Ironwood Precision Manufacturing",
    "Magellan Navigation Software",
    "Driftwood Creative Agency",
    "Sunridge Pharmaceutical Supplies",
    "Blackrock Mining Equipment Co.",
    "Cloudvault Storage Solutions",
    "Tidewater Marine Engineering",
    "Benchmark Quality Assurance Inc.",
    "Prairie Wind Energy Systems",
    "Alpine Research Instruments",
]

LINE_ITEM_CATEGORIES: list[dict[str, Any]] = [
    {"desc": "Professional Services — Senior Consultant", "min_price": 150, "max_price": 350, "unit": "hours"},
    {"desc": "Software License — Enterprise Edition (Annual)", "min_price": 2000, "max_price": 25000, "unit": "licenses"},
    {"desc": "Hardware Component — Server Rack Unit", "min_price": 800, "max_price": 5000, "unit": "units"},
    {"desc": "Consulting Retainer — Monthly Advisory", "min_price": 3000, "max_price": 15000, "unit": "months"},
    {"desc": "Maintenance Contract — Annual Support Plan", "min_price": 1200, "max_price": 8000, "unit": "contracts"},
    {"desc": "Cloud Hosting — Dedicated Instance (Monthly)", "min_price": 200, "max_price": 2500, "unit": "instances"},
    {"desc": "Data Migration Services — Per-Table ETL", "min_price": 500, "max_price": 3000, "unit": "tables"},
    {"desc": "Training & Onboarding — Workshop Session", "min_price": 800, "max_price": 4000, "unit": "sessions"},
    {"desc": "Network Equipment — Managed Switch 48-Port", "min_price": 400, "max_price": 2000, "unit": "units"},
    {"desc": "Security Audit — Penetration Testing Engagement", "min_price": 5000, "max_price": 25000, "unit": "engagements"},
    {"desc": "API Integration — Custom Connector Development", "min_price": 2000, "max_price": 12000, "unit": "connectors"},
    {"desc": "Graphic Design — Branding Package", "min_price": 1500, "max_price": 8000, "unit": "packages"},
    {"desc": "Office Supplies — Bulk Paper & Toner", "min_price": 50, "max_price": 500, "unit": "boxes"},
    {"desc": "Legal Review — Contract Drafting & Negotiation", "min_price": 300, "max_price": 1500, "unit": "hours"},
    {"desc": "Quality Assurance Testing — Test Suite Execution", "min_price": 100, "max_price": 500, "unit": "hours"},
    {"desc": "Project Management — Sprint Planning & Oversight", "min_price": 120, "max_price": 300, "unit": "hours"},
    {"desc": "Database Administration — Performance Tuning", "min_price": 150, "max_price": 400, "unit": "hours"},
    {"desc": "Technical Writing — Documentation Deliverable", "min_price": 80, "max_price": 250, "unit": "hours"},
    {"desc": "Freight & Shipping — Expedited Delivery", "min_price": 50, "max_price": 800, "unit": "shipments"},
    {"desc": "Printed Circuit Board Assembly — Custom Run", "min_price": 10, "max_price": 150, "unit": "units"},
    {"desc": "Renewable Energy Credits — Carbon Offset Bundle", "min_price": 15, "max_price": 60, "unit": "credits"},
    {"desc": "Subscription — SaaS Analytics Platform (Monthly)", "min_price": 99, "max_price": 999, "unit": "seats"},
    {"desc": "Catering Services — Corporate Event", "min_price": 500, "max_price": 5000, "unit": "events"},
    {"desc": "Insurance Premium — Professional Liability (Quarterly)", "min_price": 1000, "max_price": 8000, "unit": "quarters"},
]

TAX_RATES: list[float] = [0.0, 0.05, 0.07, 0.0825, 0.10, 0.13, 0.20]

CURRENCIES: list[dict[str, Any]] = [
    {"code": "USD", "symbol": "$", "locale": "en_US"},
    {"code": "EUR", "symbol": "€", "locale": "en_GB"},
    {"code": "GBP", "symbol": "£", "locale": "en_GB"},
    {"code": "JPY", "symbol": "¥", "locale": "ja_JP"},
    {"code": "AUD", "symbol": "A$", "locale": "en_AU"},
    {"code": "CAD", "symbol": "C$", "locale": "en_CA"},
]

# Variant distribution
VARIANT_COUNTS: dict[str, int] = {
    "clean": 80,
    "noisy": 80,
    "multi_currency": 60,
    "missing_fields": 70,
    "edge_case": 60,
    "ambiguous": 50,
}

assert sum(VARIANT_COUNTS.values()) == 400


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineItem:
    description: str
    quantity: float
    unit_price: float
    amount: float


@dataclass
class InvoiceData:
    vendor_name: str
    vendor_address: str
    vendor_phone: str
    vendor_email: str
    bill_to_name: str
    bill_to_address: str
    invoice_number: str
    invoice_date: str  # ISO 8601
    due_date: str | None  # ISO 8601 or None
    line_items: list[LineItem]
    subtotal: float
    tax_rate: float | None
    tax_amount: float | None
    total: float
    currency_code: str
    currency_symbol: str
    variant: str
    po_number: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Invoice Generator
# ---------------------------------------------------------------------------

class InvoiceGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.fake = Faker()
        Faker.seed(seed)

    def _pick_vendor(self) -> str:
        return self.rng.choice(VENDOR_COMPANIES)

    def _pick_currency(self, variant: str) -> dict[str, Any]:
        if variant == "multi_currency":
            return self.rng.choice([c for c in CURRENCIES if c["code"] != "USD"])
        return CURRENCIES[0]  # USD

    def _pick_line_items(self, count: int, allow_negative: bool = False, long_desc: bool = False) -> list[LineItem]:
        items = []
        chosen = self.rng.sample(LINE_ITEM_CATEGORIES, min(count, len(LINE_ITEM_CATEGORIES)))
        for cat in chosen:
            qty = self.rng.choice([1, 1, 2, 3, 5, 10])
            price = round(self.rng.uniform(cat["min_price"], cat["max_price"]), 2)
            if allow_negative and self.rng.random() < 0.4:
                qty = -qty  # credit/return
            desc = cat["desc"]
            if long_desc:
                extra_words = " ".join(self.fake.words(nb=self.rng.randint(30, 50)))
                desc = f"{cat['desc']} — Additional details: {extra_words}"
            amount = round(qty * price, 2)
            items.append(LineItem(description=desc, quantity=float(qty), unit_price=price, amount=amount))
        return items

    def _gen_invoice_number(self, variant: str) -> str:
        year = self.rng.randint(2023, 2025)
        seq = self.rng.randint(1, 99999)
        if variant == "edge_case" and self.rng.random() < 0.3:
            # PO number as invoice number
            return f"PO-{year}-{seq:05d}"
        prefixes = ["INV", "INV", "INV", "BILL", "REC"]
        prefix = self.rng.choice(prefixes)
        return f"{prefix}-{year}-{seq:05d}"

    def _gen_dates(self, variant: str) -> tuple[str, str | None]:
        start = date(2023, 1, 1)
        inv_date = start + timedelta(days=self.rng.randint(0, 800))
        if variant == "missing_fields" and self.rng.random() < 0.5:
            return inv_date.isoformat(), None
        due_date = inv_date + timedelta(days=self.rng.choice([15, 30, 30, 45, 60, 90]))
        return inv_date.isoformat(), due_date.isoformat()

    def _gen_tax(self, subtotal: float, variant: str) -> tuple[float | None, float | None]:
        if variant == "missing_fields" and self.rng.random() < 0.4:
            return None, None
        rate = self.rng.choice(TAX_RATES)
        amount = round(subtotal * rate, 2)
        return rate, amount

    def generate_invoice_data(self, variant: str) -> InvoiceData:
        currency = self._pick_currency(variant)
        vendor = self._pick_vendor()

        # Generate addresses
        vendor_address = self.fake.address().replace("\n", ", ")
        vendor_phone = self.fake.phone_number()
        vendor_email = self.fake.company_email()
        bill_to_name = self.fake.company()
        bill_to_address = self.fake.address().replace("\n", ", ")

        inv_number = self._gen_invoice_number(variant)
        inv_date, due_date = self._gen_dates(variant)

        # Line item count
        if variant == "edge_case":
            n_items = self.rng.choice([1, 3, 5, 8])
        elif variant == "missing_fields":
            n_items = self.rng.choice([1, 1, 2])
        else:
            n_items = self.rng.choice([2, 3, 4, 5])

        allow_negative = variant == "edge_case"
        long_desc = variant == "edge_case" and self.rng.random() < 0.4

        line_items = self._pick_line_items(n_items, allow_negative=allow_negative, long_desc=long_desc)

        # Handle edge case: zero-quantity line
        if variant == "edge_case" and self.rng.random() < 0.3:
            cat = self.rng.choice(LINE_ITEM_CATEGORIES)
            price = round(self.rng.uniform(cat["min_price"], cat["max_price"]), 2)
            line_items.append(LineItem(description=f"{cat['desc']} [CANCELLED]", quantity=0.0, unit_price=price, amount=0.0))

        subtotal = round(sum(li.amount for li in line_items), 2)
        tax_rate, tax_amount = self._gen_tax(subtotal, variant)
        total = round(subtotal + (tax_amount or 0.0), 2)

        po_number = None
        if self.rng.random() < 0.3:
            po_number = f"PO-{self.rng.randint(10000, 99999)}"

        notes = None
        if self.rng.random() < 0.25:
            notes = self.rng.choice([
                "Payment due upon receipt. Late payments subject to 1.5% monthly interest.",
                "Please reference invoice number on all correspondence.",
                "Wire transfer preferred. Bank details on file.",
                "Net 30 terms apply. Thank you for your business.",
                "Discount of 2% if paid within 10 days.",
            ])

        return InvoiceData(
            vendor_name=vendor,
            vendor_address=vendor_address,
            vendor_phone=vendor_phone,
            vendor_email=vendor_email,
            bill_to_name=bill_to_name,
            bill_to_address=bill_to_address,
            invoice_number=inv_number,
            invoice_date=inv_date,
            due_date=due_date,
            line_items=line_items,
            subtotal=subtotal,
            tax_rate=tax_rate,
            tax_amount=tax_amount,
            total=total,
            currency_code=currency["code"],
            currency_symbol=currency["symbol"],
            variant=variant,
            po_number=po_number,
            notes=notes,
        )

    # -------------------------------------------------------------------
    # Text rendering by variant
    # -------------------------------------------------------------------

    def render_clean(self, inv: InvoiceData) -> str:
        """Well-formatted tabular invoice."""
        lines = []
        lines.append("=" * 72)
        lines.append(f"  {inv.vendor_name}")
        lines.append(f"  {inv.vendor_address}")
        lines.append(f"  Phone: {inv.vendor_phone}  |  Email: {inv.vendor_email}")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"  INVOICE")
        lines.append(f"  Invoice Number:  {inv.invoice_number}")
        lines.append(f"  Invoice Date:    {inv.invoice_date}")
        if inv.due_date:
            lines.append(f"  Due Date:        {inv.due_date}")
        if inv.po_number:
            lines.append(f"  PO Number:       {inv.po_number}")
        lines.append("")
        lines.append(f"  Bill To:")
        lines.append(f"    {inv.bill_to_name}")
        lines.append(f"    {inv.bill_to_address}")
        lines.append("")
        lines.append("-" * 72)
        lines.append(f"  {'Description':<36} {'Qty':>6} {'Unit Price':>12} {'Amount':>12}")
        lines.append("-" * 72)
        for li in inv.line_items:
            desc = li.description[:36]
            lines.append(f"  {desc:<36} {li.quantity:>6.1f} {inv.currency_symbol}{li.unit_price:>11,.2f} {inv.currency_symbol}{li.amount:>11,.2f}")
        lines.append("-" * 72)
        lines.append(f"  {'Subtotal':>56} {inv.currency_symbol}{inv.subtotal:>11,.2f}")
        if inv.tax_rate is not None:
            pct = f"{inv.tax_rate * 100:.2f}%"
            lines.append(f"  {'Tax (' + pct + ')':>56} {inv.currency_symbol}{inv.tax_amount:>11,.2f}")
        lines.append(f"  {'TOTAL':>56} {inv.currency_symbol}{inv.total:>11,.2f}")
        lines.append("=" * 72)
        if inv.notes:
            lines.append(f"\n  Notes: {inv.notes}")
        lines.append(f"\n  Currency: {inv.currency_code}")
        return "\n".join(lines)

    def render_noisy(self, inv: InvoiceData) -> str:
        """OCR-degraded version of the clean invoice."""
        clean_text = self.render_clean(inv)
        return self._apply_ocr_noise(clean_text)

    def _apply_ocr_noise(self, text: str) -> str:
        """Apply realistic OCR-style degradation."""
        result = list(text)
        noise_map = {
            "O": "0", "0": "O",
            "l": "1", "1": "l",
            "I": "l", "5": "S", "S": "5",
            "8": "B", "B": "8",
            "g": "9", "6": "b",
        }
        # Character substitution (roughly 3-5% of alphanumeric chars)
        for i in range(len(result)):
            if result[i] in noise_map and self.rng.random() < 0.04:
                result[i] = noise_map[result[i]]

        text = "".join(result)

        # Misaligned columns: randomly add/remove spaces in some lines
        lines = text.split("\n")
        noisy_lines = []
        for line in lines:
            if self.rng.random() < 0.15 and len(line) > 10:
                # Insert random extra spaces
                pos = self.rng.randint(2, max(3, len(line) - 2))
                extra = " " * self.rng.randint(1, 4)
                line = line[:pos] + extra + line[pos:]
            if self.rng.random() < 0.08 and "  " in line:
                # Collapse some double spaces
                line = line.replace("  ", " ", 1)
            noisy_lines.append(line)

        text = "\n".join(noisy_lines)

        # rn -> m substitution (classic OCR error)
        if self.rng.random() < 0.3:
            text = text.replace("rn", "m", 1)

        # Occasional garbled characters
        result = list(text)
        for i in range(len(result)):
            if result[i].isalpha() and self.rng.random() < 0.008:
                result[i] = self.rng.choice(["#", "@", "&", "~", "^"])
        text = "".join(result)

        return text

    def render_multi_currency(self, inv: InvoiceData) -> str:
        """Invoice with non-USD currency, potentially different formatting."""
        lines = []
        lines.append(f"{'─' * 60}")
        lines.append(f"  {inv.vendor_name}")
        lines.append(f"  {inv.vendor_address}")
        lines.append(f"  Tel: {inv.vendor_phone}")
        lines.append(f"{'─' * 60}")
        lines.append("")
        lines.append(f"  INVOICE / FACTURE / RECHNUNG")
        lines.append(f"  Ref: {inv.invoice_number}")
        lines.append(f"  Date: {inv.invoice_date}")
        if inv.due_date:
            lines.append(f"  Payment Due: {inv.due_date}")
        lines.append("")
        lines.append(f"  Client: {inv.bill_to_name}")
        lines.append(f"  Address: {inv.bill_to_address}")
        lines.append("")

        # For JPY, no decimal places
        fmt = ".0f" if inv.currency_code == "JPY" else ",.2f"

        lines.append(f"  {'Item':<34} {'Qty':>5} {'Price':>14} {'Total':>14}")
        lines.append(f"  {'─' * 67}")
        for li in inv.line_items:
            desc = li.description[:34]
            price_str = f"{inv.currency_symbol}{li.unit_price:{fmt}}"
            amt_str = f"{inv.currency_symbol}{li.amount:{fmt}}"
            lines.append(f"  {desc:<34} {li.quantity:>5.1f} {price_str:>14} {amt_str:>14}")
        lines.append(f"  {'─' * 67}")
        sub_str = f"{inv.currency_symbol}{inv.subtotal:{fmt}}"
        lines.append(f"  {'Subtotal':>53} {sub_str:>14}")
        if inv.tax_rate is not None and inv.tax_rate > 0:
            pct = f"{inv.tax_rate * 100:.1f}%"
            tax_str = f"{inv.currency_symbol}{inv.tax_amount:{fmt}}"
            lines.append(f"  {'Tax/VAT (' + pct + ')':>53} {tax_str:>14}")
        tot_str = f"{inv.currency_symbol}{inv.total:{fmt}}"
        lines.append(f"  {'TOTAL DUE':>53} {tot_str:>14}")
        lines.append(f"\n  All amounts in {inv.currency_code}.")
        if inv.notes:
            lines.append(f"  {inv.notes}")
        return "\n".join(lines)

    def render_missing_fields(self, inv: InvoiceData) -> str:
        """Invoice with some fields intentionally missing."""
        lines = []
        lines.append(f"INVOICE")
        lines.append(f"From: {inv.vendor_name}")
        if self.rng.random() > 0.3:
            lines.append(f"{inv.vendor_address}")
        lines.append("")
        lines.append(f"Invoice #: {inv.invoice_number}")
        lines.append(f"Date: {inv.invoice_date}")
        # due_date may already be None from generation
        if inv.due_date:
            lines.append(f"Due: {inv.due_date}")
        lines.append("")
        lines.append(f"To: {inv.bill_to_name}")
        lines.append("")
        for li in inv.line_items:
            lines.append(f"  - {li.description}")
            lines.append(f"    Qty: {li.quantity:.1f}  @  ${li.unit_price:,.2f}  =  ${li.amount:,.2f}")
        lines.append("")
        lines.append(f"Subtotal: ${inv.subtotal:,.2f}")
        if inv.tax_rate is not None:
            lines.append(f"Tax ({inv.tax_rate * 100:.2f}%): ${inv.tax_amount:,.2f}")
        # Note: no tax line if tax is None (intentionally missing)
        lines.append(f"Total: ${inv.total:,.2f}")
        lines.append(f"Currency: {inv.currency_code}")
        return "\n".join(lines)

    def render_edge_case(self, inv: InvoiceData) -> str:
        """Edge-case invoice: credit memos, zero-qty, long descriptions."""
        is_credit = any(li.quantity < 0 for li in inv.line_items)
        doc_type = "CREDIT MEMO" if is_credit else "INVOICE"

        lines = []
        lines.append(f"*** {doc_type} ***")
        lines.append(f"Vendor: {inv.vendor_name}")
        lines.append(f"Address: {inv.vendor_address}")
        lines.append(f"Contact: {inv.vendor_phone} / {inv.vendor_email}")
        lines.append("")
        lines.append(f"Document Number: {inv.invoice_number}")
        lines.append(f"Issue Date: {inv.invoice_date}")
        if inv.due_date:
            lines.append(f"Due Date: {inv.due_date}")
        lines.append("")
        lines.append(f"Billed To: {inv.bill_to_name}")
        lines.append(f"           {inv.bill_to_address}")
        lines.append("")
        lines.append(f"+{'-'*46}+{'-'*8}+{'-'*14}+{'-'*14}+")
        lines.append(f"| {'Description':<44} | {'Qty':>6} | {'Unit Price':>12} | {'Amount':>12} |")
        lines.append(f"+{'-'*46}+{'-'*8}+{'-'*14}+{'-'*14}+")
        for li in inv.line_items:
            # Wrap long descriptions
            desc = li.description
            if len(desc) > 44:
                desc_lines = [desc[i:i+44] for i in range(0, len(desc), 44)]
                lines.append(f"| {desc_lines[0]:<44} | {li.quantity:>6.1f} | {inv.currency_symbol}{li.unit_price:>11,.2f} | {inv.currency_symbol}{li.amount:>11,.2f} |")
                for dl in desc_lines[1:]:
                    lines.append(f"| {dl:<44} | {'':>6} | {'':>12} | {'':>12} |")
            else:
                lines.append(f"| {desc:<44} | {li.quantity:>6.1f} | {inv.currency_symbol}{li.unit_price:>11,.2f} | {inv.currency_symbol}{li.amount:>11,.2f} |")
        lines.append(f"+{'-'*46}+{'-'*8}+{'-'*14}+{'-'*14}+")
        lines.append(f"  Subtotal: {inv.currency_symbol}{inv.subtotal:>12,.2f}")
        if inv.tax_rate is not None:
            lines.append(f"  Tax ({inv.tax_rate*100:.2f}%): {inv.currency_symbol}{inv.tax_amount:>12,.2f}")
        lines.append(f"  TOTAL: {inv.currency_symbol}{inv.total:>12,.2f}")
        lines.append(f"  Currency: {inv.currency_code}")
        if inv.notes:
            lines.append(f"\n  {inv.notes}")
        return "\n".join(lines)

    def render_ambiguous(self, inv: InvoiceData) -> str:
        """Render non-invoice documents or ambiguous documents."""
        doc_type = self.rng.choice([
            "purchase_order", "delivery_note", "statement_of_account",
            "partial_invoice", "proforma", "quote"
        ])

        if doc_type == "purchase_order":
            return self._render_purchase_order(inv)
        elif doc_type == "delivery_note":
            return self._render_delivery_note(inv)
        elif doc_type == "statement_of_account":
            return self._render_statement_of_account(inv)
        elif doc_type == "partial_invoice":
            return self._render_partial_invoice(inv)
        elif doc_type == "proforma":
            return self._render_proforma(inv)
        else:  # quote
            return self._render_quote(inv)

    def _render_purchase_order(self, inv: InvoiceData) -> str:
        lines = [
            f"PURCHASE ORDER",
            f"PO Number: {inv.invoice_number.replace('INV', 'PO').replace('BILL', 'PO')}",
            f"Date: {inv.invoice_date}",
            f"",
            f"From: {inv.bill_to_name}",
            f"To: {inv.vendor_name}",
            f"",
            f"Please supply the following items:",
            f"",
        ]
        for li in inv.line_items:
            lines.append(f"  {li.description}")
            lines.append(f"    Quantity requested: {li.quantity:.1f}")
            lines.append(f"    Agreed price: {inv.currency_symbol}{li.unit_price:,.2f}")
            lines.append("")
        lines.append(f"Estimated total: {inv.currency_symbol}{inv.total:,.2f} ({inv.currency_code})")
        lines.append(f"\nDelivery requested by: {inv.due_date or 'TBD'}")
        lines.append(f"This is a purchase order, not an invoice.")
        return "\n".join(lines)

    def _render_delivery_note(self, inv: InvoiceData) -> str:
        lines = [
            f"DELIVERY NOTE / PACKING SLIP",
            f"Reference: DN-{self.rng.randint(10000, 99999)}",
            f"Date: {inv.invoice_date}",
            f"",
            f"Shipped From: {inv.vendor_name}",
            f"Shipped To: {inv.bill_to_name}, {inv.bill_to_address}",
            f"",
            f"Items Delivered:",
        ]
        for li in inv.line_items:
            lines.append(f"  [{li.quantity:.0f}x] {li.description}")
        lines.append(f"\nRelated Invoice: {inv.invoice_number}")
        lines.append(f"Note: This delivery note does not constitute an invoice.")
        lines.append(f"No payment information is included.")
        return "\n".join(lines)

    def _render_statement_of_account(self, inv: InvoiceData) -> str:
        lines = [
            f"STATEMENT OF ACCOUNT",
            f"Account: {inv.bill_to_name}",
            f"Statement Date: {inv.invoice_date}",
            f"From: {inv.vendor_name}",
            f"",
            f"{'Date':<14} {'Reference':<20} {'Debit':>12} {'Credit':>12} {'Balance':>12}",
            f"{'-'*70}",
        ]
        balance = 0.0
        # Generate a few fake transactions
        for i in range(self.rng.randint(3, 6)):
            ref = f"INV-{self.rng.randint(2023, 2025)}-{self.rng.randint(100, 99999):05d}"
            amt = round(self.rng.uniform(500, 15000), 2)
            if self.rng.random() < 0.3:
                # Payment
                balance -= amt
                lines.append(f"  {inv.invoice_date:<12} {'PMT-' + str(self.rng.randint(1000,9999)):<18} {'':>12} {inv.currency_symbol}{amt:>11,.2f} {inv.currency_symbol}{balance:>11,.2f}")
            else:
                balance += amt
                lines.append(f"  {inv.invoice_date:<12} {ref:<18} {inv.currency_symbol}{amt:>11,.2f} {'':>12} {inv.currency_symbol}{balance:>11,.2f}")
        lines.append(f"{'-'*70}")
        lines.append(f"  {'Balance Due:':>44} {inv.currency_symbol}{balance:>11,.2f}")
        lines.append(f"\n  Currency: {inv.currency_code}")
        lines.append(f"  This is a statement of account, not an invoice.")
        return "\n".join(lines)

    def _render_partial_invoice(self, inv: InvoiceData) -> str:
        """Truncated/partial invoice — missing the bottom section."""
        lines = [
            f"INVOICE",
            f"",
            f"  {inv.vendor_name}",
            f"  Invoice No: {inv.invoice_number}",
            f"  Date: {inv.invoice_date}",
            f"",
            f"  Bill To: {inv.bill_to_name}",
            f"",
        ]
        # Only show some line items
        shown = inv.line_items[:max(1, len(inv.line_items) // 2)]
        for li in shown:
            lines.append(f"    {li.description}  x{li.quantity:.0f}  {inv.currency_symbol}{li.amount:,.2f}")
        lines.append("")
        lines.append(f"  [Page 1 of 2 — continued on next page]")
        lines.append(f"  Currency: {inv.currency_code}")
        return "\n".join(lines)

    def _render_proforma(self, inv: InvoiceData) -> str:
        lines = [
            f"PROFORMA INVOICE",
            f"(This is not a tax invoice)",
            f"",
            f"From: {inv.vendor_name}",
            f"Proforma #: {inv.invoice_number.replace('INV', 'PRO')}",
            f"Date: {inv.invoice_date}",
            f"Valid Until: {inv.due_date or 'N/A'}",
            f"",
            f"To: {inv.bill_to_name}",
            f"",
        ]
        for li in inv.line_items:
            lines.append(f"  {li.description}")
            lines.append(f"    {li.quantity:.1f} x {inv.currency_symbol}{li.unit_price:,.2f} = {inv.currency_symbol}{li.amount:,.2f}")
        lines.append("")
        lines.append(f"  Estimated Subtotal: {inv.currency_symbol}{inv.subtotal:,.2f}")
        if inv.tax_rate is not None:
            lines.append(f"  Estimated Tax ({inv.tax_rate*100:.1f}%): {inv.currency_symbol}{inv.tax_amount:,.2f}")
        lines.append(f"  Estimated Total: {inv.currency_symbol}{inv.total:,.2f} {inv.currency_code}")
        lines.append(f"\n  This proforma is for informational purposes only.")
        return "\n".join(lines)

    def _render_quote(self, inv: InvoiceData) -> str:
        lines = [
            f"QUOTATION / ESTIMATE",
            f"Quote #: QT-{self.rng.randint(10000, 99999)}",
            f"Date: {inv.invoice_date}",
            f"Valid for 30 days",
            f"",
            f"Prepared by: {inv.vendor_name}",
            f"Prepared for: {inv.bill_to_name}",
            f"",
            f"Scope of Work:",
        ]
        for li in inv.line_items:
            lines.append(f"  - {li.description}: {li.quantity:.0f} units @ {inv.currency_symbol}{li.unit_price:,.2f} each = {inv.currency_symbol}{li.amount:,.2f}")
        lines.append(f"\n  Subtotal: {inv.currency_symbol}{inv.subtotal:,.2f}")
        if inv.tax_rate:
            lines.append(f"  Tax ({inv.tax_rate*100:.1f}%): {inv.currency_symbol}{inv.tax_amount:,.2f}")
        lines.append(f"  Total Estimate: {inv.currency_symbol}{inv.total:,.2f} {inv.currency_code}")
        lines.append(f"\n  This is a quotation only and does not represent an obligation to pay.")
        return "\n".join(lines)

    def render(self, inv: InvoiceData) -> str:
        """Render invoice data to text based on variant."""
        renderers = {
            "clean": self.render_clean,
            "noisy": self.render_noisy,
            "multi_currency": self.render_multi_currency,
            "missing_fields": self.render_missing_fields,
            "edge_case": self.render_edge_case,
            "ambiguous": self.render_ambiguous,
        }
        return renderers[inv.variant](inv)

    def ground_truth(self, inv: InvoiceData) -> dict[str, Any]:
        """Return the expected extraction JSON for this invoice."""
        return {
            "vendor_name": inv.vendor_name,
            "invoice_number": inv.invoice_number,
            "invoice_date": inv.invoice_date,
            "due_date": inv.due_date,
            "line_items": [
                {
                    "description": li.description,
                    "quantity": li.quantity,
                    "unit_price": li.unit_price,
                    "amount": li.amount,
                }
                for li in inv.line_items
            ],
            "subtotal": inv.subtotal,
            "tax_rate": inv.tax_rate,
            "tax_amount": inv.tax_amount,
            "total": inv.total,
            "currency": inv.currency_code,
        }

    def generate_all(self) -> list[tuple[InvoiceData, str]]:
        """Generate all 400 invoice (data, text) pairs."""
        all_pairs: list[tuple[InvoiceData, str]] = []
        for variant, count in VARIANT_COUNTS.items():
            for _ in range(count):
                inv = self.generate_invoice_data(variant)
                text = self.render(inv)
                all_pairs.append((inv, text))
        # Shuffle to mix variants
        self.rng.shuffle(all_pairs)
        return all_pairs


# ---------------------------------------------------------------------------
# Extraction Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial document extraction system. You extract structured data from invoice text.

Your task: Given unstructured invoice text (which may contain OCR noise, formatting inconsistencies, or missing information), extract the relevant fields into a JSON object.

IMPORTANT RULES:
1. Return ONLY a valid JSON object — no markdown, no explanation, no code fences.
2. Use null for any field that is genuinely missing or cannot be determined from the text.
3. Do NOT hallucinate or guess values that are not present in the text.
4. Dates must be in ISO 8601 format: YYYY-MM-DD
5. Currency codes must be ISO 4217 (e.g., USD, EUR, GBP, JPY, AUD, CAD)
6. tax_rate must be a decimal (e.g., 0.0825 for 8.25%), NOT a percentage
7. All monetary amounts should be numbers (not strings)
8. For OCR-degraded text, correct obvious character errors (0↔O, 1↔l, rn→m, etc.) when extracting values
9. If the document is NOT an invoice (e.g., purchase order, delivery note, statement of account, quote), still extract whatever fields are present but set missing/inapplicable fields to null
10. For credit memos, amounts may be negative — preserve the sign

Required JSON schema:
{
  "vendor_name": "string or null",
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "line_items": [{"description": "string", "quantity": number, "unit_price": number, "amount": number}],
  "subtotal": number or null,
  "tax_rate": decimal or null (e.g., 0.0825),
  "tax_amount": number or null,
  "total": number or null,
  "currency": "ISO 4217 code or null"
}"""


def build_user_prompt(invoice_text: str) -> str:
    return f"Extract structured data from the following document:\n\n{invoice_text}"


# ---------------------------------------------------------------------------
# LLM Calling
# ---------------------------------------------------------------------------

def call_llm(
    model: str,
    api_key: str,
    invoice_text: str,
    max_retries: int = 3,
) -> tuple[str, float]:
    """Call LLM and return (response_text, cost)."""
    from litellm import completion

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(invoice_text)},
    ]

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                api_key=api_key,
                messages=messages,
                temperature=0.0,
                max_tokens=2000,
            )
            text = response.choices[0].message.content.strip()
            cost = response._hidden_params.get("response_cost", 0.0) or 0.0
            return text, cost
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")


def validate_response(response_text: str) -> tuple[bool, dict | None, str]:
    """Validate that the response is valid JSON matching our schema.
    Returns (is_valid, parsed_dict_or_none, error_message).
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"

    if not isinstance(data, dict):
        return False, None, "Response is not a JSON object"

    required_keys = {"vendor_name", "invoice_number", "invoice_date", "due_date",
                     "line_items", "subtotal", "tax_rate", "tax_amount", "total", "currency"}
    missing = required_keys - set(data.keys())
    if missing:
        return False, None, f"Missing keys: {missing}"

    # Check tax_rate is decimal not percentage (if present)
    if data.get("tax_rate") is not None:
        if isinstance(data["tax_rate"], (int, float)) and data["tax_rate"] > 1:
            return False, None, f"tax_rate={data['tax_rate']} looks like a percentage, should be decimal"

    # Check line_items is a list
    if not isinstance(data.get("line_items"), list):
        return False, None, "line_items is not a list"

    return True, data, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tuning(
    generator: InvoiceGenerator,
    all_pairs: list[tuple[InvoiceData, str]],
    gpt4o_key: str,
) -> float:
    """Run prompt tuning on a sample of invoices. Returns total tuning cost."""
    print("\n=== PROMPT TUNING PHASE ===")

    # Select a diverse sample: ~30 across variants
    by_variant: dict[str, list[tuple[InvoiceData, str]]] = {}
    for inv, text in all_pairs:
        by_variant.setdefault(inv.variant, []).append((inv, text))

    sample: list[tuple[InvoiceData, str]] = []
    for variant, pairs in by_variant.items():
        # Take 5 from each variant
        n = min(5, len(pairs))
        sample.extend(pairs[:n])

    total_cost = 0.0
    valid_count = 0
    total_count = len(sample)

    for i, (inv, text) in enumerate(sample):
        print(f"  Tuning [{i+1}/{total_count}] variant={inv.variant} vendor={inv.vendor_name[:30]}...", end=" ")
        try:
            resp_text, cost = call_llm(
                model="openrouter/openai/gpt-4o",
                api_key=gpt4o_key,
                invoice_text=text,
            )
            total_cost += cost
            is_valid, parsed, err = validate_response(resp_text)
            if is_valid:
                valid_count += 1
                print(f"OK (${cost:.4f})")
            else:
                print(f"INVALID: {err} (${cost:.4f})")
        except Exception as e:
            print(f"ERROR: {e}")

    accuracy = valid_count / total_count * 100 if total_count > 0 else 0
    print(f"\n  Tuning results: {valid_count}/{total_count} valid ({accuracy:.1f}%)")
    print(f"  Tuning cost: ${total_cost:.4f}")

    if accuracy < 95:
        print("  WARNING: Accuracy below 95% target. Consider revising prompt.")

    return total_cost


def run_production(
    all_pairs: list[tuple[InvoiceData, str]],
    model: str,
    api_key: str,
    output_path: Path,
    source_model_label: str,
) -> float:
    """Run production LLM calls and write JSONL incrementally. Returns total cost."""
    print(f"\n=== PRODUCTION RUN: {source_model_label} ===")
    print(f"  Output: {output_path}")

    total_cost = 0.0
    total = len(all_pairs)

    # Clear file if it exists
    output_path.write_text("")

    for i, (inv, text) in enumerate(all_pairs):
        variant = inv.variant
        print(f"  [{i+1}/{total}] variant={variant} ", end="")

        try:
            resp_text, cost = call_llm(
                model=model,
                api_key=api_key,
                invoice_text=text,
            )
            total_cost += cost

            # Build the prompt that was sent (user message only, without system)
            full_prompt = build_user_prompt(text)

            record = {
                "prompt": full_prompt,
                "response": resp_text,
                "source_model": source_model_label,
                "metadata": {
                    "dataset": "fintech_extraction",
                    "variant": variant,
                },
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            is_valid, _, err = validate_response(resp_text)
            status = "OK" if is_valid else f"WARN({err[:40]})"
            print(f"{status} (${cost:.4f}) cumulative=${total_cost:.4f}")

        except Exception as e:
            print(f"ERROR: {e}")
            # Write a record with the error
            record = {
                "prompt": build_user_prompt(text),
                "response": f"ERROR: {e}",
                "source_model": source_model_label,
                "metadata": {
                    "dataset": "fintech_extraction",
                    "variant": variant,
                },
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Small delay to avoid rate limiting
        if i % 10 == 9:
            time.sleep(0.5)

    print(f"\n  {source_model_label} production cost: ${total_cost:.4f}")
    return total_cost


def spot_check(output_path: Path, n_per_variant: int = 1) -> None:
    """Read the JSONL and spot-check records."""
    print(f"\n=== SPOT CHECK: {output_path.name} ===")
    records_by_variant: dict[str, list[dict]] = {}
    with open(output_path) as f:
        for line in f:
            rec = json.loads(line)
            v = rec["metadata"]["variant"]
            records_by_variant.setdefault(v, []).append(rec)

    for variant, records in records_by_variant.items():
        print(f"\n  --- {variant} ({len(records)} records) ---")
        sample = records[:n_per_variant]
        for rec in sample:
            prompt_preview = rec["prompt"][:120].replace("\n", " ")
            is_valid, parsed, err = validate_response(rec["response"])
            if is_valid:
                vendor = parsed.get("vendor_name", "?")
                total = parsed.get("total", "?")
                currency = parsed.get("currency", "?")
                n_items = len(parsed.get("line_items", []))
                print(f"    Valid | vendor={vendor} | total={total} {currency} | {n_items} items")
            else:
                print(f"    INVALID: {err}")
                print(f"    Response preview: {rec['response'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate fintech extraction dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="examples/datasets/fintech_extraction",
                        help="Output directory")
    parser.add_argument("--generate-only", action="store_true", help="Only generate invoices, skip LLM calls")
    parser.add_argument("--tune-only", action="store_true", help="Only run prompt tuning")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip tuning, go straight to production")
    parser.add_argument("--model", type=str, choices=["gpt4o", "haiku", "both"], default="both",
                        help="Which model(s) to run")
    args = parser.parse_args()

    # Resolve paths relative to the script's project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load env
    env_path = Path("/Users/ashwinchidambaram/dev/projects/rosettastone/.env")
    load_dotenv(env_path)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Generate invoices
    print(f"Generating 400 invoices with seed={args.seed}...")
    generator = InvoiceGenerator(seed=args.seed)
    all_pairs = generator.generate_all()

    # Print variant distribution
    variant_counts: dict[str, int] = {}
    for inv, _ in all_pairs:
        variant_counts[inv.variant] = variant_counts.get(inv.variant, 0) + 1
    print(f"  Variant distribution: {json.dumps(variant_counts, indent=2)}")

    if args.generate_only:
        # Just write sample invoices for inspection
        sample_path = output_dir / "sample_invoices.txt"
        with open(sample_path, "w") as f:
            for inv, text in all_pairs[:20]:
                f.write(f"\n{'='*80}\nVariant: {inv.variant}\n{'='*80}\n{text}\n")
        print(f"  Wrote 20 sample invoices to {sample_path}")
        return

    # --- Server reporting (silent no-op) ---
    sys.path.insert(0, str(project_root / "scripts"))
    try:
        from dataset_cost_client import report_run_start, report_run_update
        run_id = report_run_start("fintech_extraction", "openai/gpt-4o")
    except Exception:
        run_id = None

    costs = {
        "gpt4o_tuning": 0.0,
        "gpt4o_production": 0.0,
        "haiku_production": 0.0,
    }

    # --- Tuning ---
    if not args.skip_tuning:
        if not openrouter_key:
            print("ERROR: OPENROUTER_API_KEY not set. Cannot run tuning.")
            sys.exit(1)
        costs["gpt4o_tuning"] = run_tuning(generator, all_pairs, openrouter_key)

        if run_id:
            try:
                report_run_update(run_id, tuning_cost=costs["gpt4o_tuning"], status="tuning_complete")
            except Exception:
                pass

    if args.tune_only:
        print("\nTuning-only mode. Exiting.")
        return

    # --- Production: GPT-4o ---
    if args.model in ("gpt4o", "both"):
        if not openrouter_key:
            print("ERROR: OPENROUTER_API_KEY not set.")
            sys.exit(1)
        gpt4o_path = output_dir / "fintech_extraction_gpt4o.jsonl"
        costs["gpt4o_production"] = run_production(
            all_pairs,
            model="openrouter/openai/gpt-4o",
            api_key=openrouter_key,
            output_path=gpt4o_path,
            source_model_label="openai/gpt-4o",
        )
        spot_check(gpt4o_path, n_per_variant=1)

    # --- Production: Haiku ---
    if args.model in ("haiku", "both"):
        if not anthropic_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        haiku_path = output_dir / "fintech_extraction_haiku.jsonl"
        costs["haiku_production"] = run_production(
            all_pairs,
            model="anthropic/claude-haiku-4-5-20251001",
            api_key=anthropic_key,
            output_path=haiku_path,
            source_model_label="anthropic/claude-haiku-4-5-20251001",
        )
        spot_check(haiku_path, n_per_variant=1)

    # --- Cost summary ---
    cost_summary = [
        {
            "model": "openai/gpt-4o",
            "tuning_cost_usd": round(costs["gpt4o_tuning"], 4),
            "production_cost_usd": round(costs["gpt4o_production"], 4),
            "total_cost_usd": round(costs["gpt4o_tuning"] + costs["gpt4o_production"], 4),
            "pairs_generated": 400,
        },
        {
            "model": "anthropic/claude-haiku-4-5-20251001",
            "tuning_cost_usd": 0.0,
            "production_cost_usd": round(costs["haiku_production"], 4),
            "total_cost_usd": round(costs["haiku_production"], 4),
            "pairs_generated": 400,
        },
    ]

    cost_path = output_dir / "cost_summary.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"\n  Cost summary written to {cost_path}")

    total_all = sum(c["total_cost_usd"] for c in cost_summary)
    print(f"\n  TOTAL COST: ${total_all:.4f}")

    if total_all > 5.0:
        print("  WARNING: Total cost exceeds $5 budget!")

    # Server reporting
    if run_id:
        try:
            report_run_update(
                run_id,
                production_cost=costs["gpt4o_production"] + costs["haiku_production"],
                pairs=800,
                status="complete",
                output_path=str(output_dir),
            )
        except Exception:
            pass

    print("\nDONE.")


if __name__ == "__main__":
    main()
