#!/usr/bin/env python3
"""Fix 12 quality issues in sql_generation dataset."""
from __future__ import annotations
import json, time, re, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/Users/ashwinchidambaram/dev/projects/rosettastone/.env")

GPT4O_FILE = Path("examples/datasets/sql_generation/sql_generation_gpt4o.jsonl")
HAIKU_FILE = Path("examples/datasets/sql_generation/sql_generation_haiku.jsonl")

gpt4o = [json.loads(l) for l in GPT4O_FILE.read_text().splitlines() if l.strip()]
haiku = [json.loads(l) for l in HAIKU_FILE.read_text().splitlines() if l.strip()]

def strip_markdown(text: str) -> str:
    """Extract clean SQL from markdown-contaminated response."""
    # Remove ```sql ... ``` or ``` ... ``` fences, keep last SQL block
    blocks = re.findall(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Strip "Wait, let me reconsider..." and everything before final query
    if 'reconsider' in text.lower() or 'wait' in text.lower():
        # Find last line that looks like SQL
        lines = text.strip().split('\n')
        # Find last non-empty, non-prose line
        return text.split('reconsider')[-1].strip().lstrip(':').strip()
    return text.strip()

def regenerate_gpt4o(prompt: str) -> str:
    """Re-call GPT-4o for a false-refusal fix."""
    from litellm import completion
    for attempt in range(3):
        response = completion(
            model="openrouter/openai/gpt-4o",
            api_key=os.environ["OPENROUTER_API_KEY"],
            messages=[{"role": "user", "content": prompt}],
        )
        time.sleep(1.0)
        result = response.choices[0].message.content.strip()
        # Accept if it doesn't look like an error JSON
        if '"error"' not in result or 'Cannot answer' not in result:
            return result
    # If still refuses after 3 attempts, keep the error (may be genuinely unanswerable)
    return response.choices[0].message.content.strip()

# Fix GPT-4o false refusals (0-indexed: lines 82,199,219,229,231,232 → indices 81,198,218,228,230,231)
FALSE_REFUSAL_INDICES = [81, 198, 218, 228, 230, 231]
for idx in FALSE_REFUSAL_INDICES:
    print(f"Regenerating GPT-4o record {idx+1}...")
    gpt4o[idx]["response"] = regenerate_gpt4o(gpt4o[idx]["prompt"])

# Fix GPT-4o structural bugs (optional but recommended)
STRUCTURAL_BUG_INDICES = [242, 246, 259]  # lines 243, 247, 260
for idx in STRUCTURAL_BUG_INDICES:
    print(f"Regenerating GPT-4o structural bug record {idx+1}...")
    gpt4o[idx]["response"] = regenerate_gpt4o(gpt4o[idx]["prompt"])

# Fix GPT-4o markdown contamination (line 206 → index 205)
print("Stripping GPT-4o markdown contamination record 206...")
gpt4o[205]["response"] = strip_markdown(gpt4o[205]["response"])

# Fix Haiku markdown contamination (lines 100,179,197,229,259 → indices 99,178,196,228,258)
HAIKU_MARKDOWN_INDICES = [99, 178, 196, 228, 258]
for idx in HAIKU_MARKDOWN_INDICES:
    print(f"Stripping Haiku markdown contamination record {idx+1}...")
    haiku[idx]["response"] = strip_markdown(haiku[idx]["response"])

# Write back
GPT4O_FILE.write_text('\n'.join(json.dumps(r) for r in gpt4o) + '\n')
HAIKU_FILE.write_text('\n'.join(json.dumps(r) for r in haiku) + '\n')

print(f"GPT-4o: {len(gpt4o)} records")
print(f"Haiku: {len(haiku)} records")
print("Done.")
