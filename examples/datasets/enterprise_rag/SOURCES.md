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
