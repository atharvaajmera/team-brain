# `ask.py` End-to-End Audit

This document maps the current end-to-end flow that already exists in the repo and identifies what should be reused for the CLI demo.

It is the implementation guide for step 2 of the `ask.py` demo plan.

## Current Pipeline

The current system already has a working backend pipeline:

1. Slack ingestion
2. Chroma storage
3. Retrieval
4. Intent classification
5. Thread expansion
6. LLM answer generation

The main issue is not missing functionality. The issue is that the CLI presentation is still too raw for demo use.

## Reusable Components

### 1. Ingestion

File:

- [brain.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/brain.py:1)

What it does:

- pulls Slack threads via `get_threads_from_channel(...)`
- normalizes message text and metadata
- writes messages into Chroma through `add_messages(...)`

CLI relevance:

- reusable as the corpus-building step
- should **not** be auto-triggered inside the first `ask.py` demo version
- the CLI should assume an index already exists and fail gracefully if it does not

### 2. Storage

File:

- [memory/storage.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/memory/storage.py:1)

What it does:

- creates a persistent Chroma client rooted at `chroma_db/`
- exposes `collection`
- exposes `add_messages(...)`

CLI relevance:

- this is already the correct storage entrypoint
- `ask.py` should continue using the shared persistent collection through existing downstream functions

### 3. Retrieval

File:

- [memory/retrieval.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/memory/retrieval.py:1)

What it does:

- builds Chroma filters from query intent
- performs candidate retrieval from Chroma
- optionally applies PRF
- attaches PRF debug metadata when enabled

CLI relevance:

- should be reused as-is through the higher-level decision path
- the demo should not call retrieval directly unless we need extra evidence formatting later

### 4. Thread Ranking and Intent Decision

File:

- [memory/ranking.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/memory/ranking.py:1)

What it does:

- groups candidate messages by thread
- computes thread-level scores
- computes runtime decision features:
  - `entropy`
  - `coherence`
  - `rel_gap`
  - `abs_ratio`
- classifies the query into:
  - `NARROW`
  - `AMBIGUOUS`
  - `BROAD`
  - fallback `None`/reject path
- chooses thread ids for expansion

CLI relevance:

- this is already the core runtime classifier
- the demo should surface these metrics in a friendlier way
- no new classifier path should be introduced for the CLI

### 5. Query Orchestration

File:

- [memory/decision.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/memory/decision.py:1)

What it does:

- `get_top_convo_id(query)`:
  - analyzes intent
  - retrieves candidates
  - selects threads
  - retries without temporal filter on fallback
- `query_text_phase_2(query)`:
  - expands selected thread ids into full thread messages
  - returns:
    - `type`
    - `threads`
    - `is_fallback`
    - `fallback_reason`
    - `stats`

CLI relevance:

- this is the best current backend entrypoint for `ask.py`
- the demo should continue calling `query_text_phase_2(...)`
- this keeps retrieval, classification, and fallback logic centralized

### 6. Answer Generation

File:

- [memory/llm.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/memory/llm.py:1)

What it does:

- converts retrieved threads into prompt context
- generates category-specific prompts
- calls Ollama
- returns either a full response or a token stream

CLI relevance:

- this is already the correct answer generation layer
- it should be reused directly
- the CLI needs a fallback if this call fails or the model is unavailable

### 7. Current CLI

File:

- [ask.py](/c:/Users/Atharva/OneDrive/Desktop/WebD/team-brain-python/ask.py:1)

What it does now:

- starts an interactive loop
- calls `query_text_phase_2(...)`
- prints:
  - category
  - raw-ish metric line
  - LLM answer
  - source thread ids

Current gaps:

- no explicit confidence score
- no formatted `Top Threads` section
- no readable evidence block
- no graceful handling of LLM failure
- no explicit corpus-readiness check
- labels are shown as `Category` instead of `Intent`
- sources are just thread ids, not useful evidence snippets

## Recommended Reuse Strategy

The `ask.py` demo should reuse this pipeline unchanged:

1. `query_text_phase_2(query)`
2. `generate_response(query, category, threads)`

The implementation work should be mostly presentation-layer work in `ask.py`:

- format intent and confidence
- render top threads
- render evidence snippets
- handle failures cleanly

## What Does Not Need To Be Rebuilt

Do not rebuild:

- ingestion
- retrieval
- classifier
- thread expansion
- Ollama prompt construction

The repo already has these pieces.

## What Must Be Added In Step 3+

### Presentation helpers

Needed for:

- confidence formatting
- top-thread formatting
- evidence rendering

### Error handling

Needed for:

- empty collection / missing index
- no retrieved threads
- LLM request failure

### Human-readable evidence

Needed for:

- representative snippets per thread
- compact references instead of only thread ids

## Conclusion

The current system is already functionally end-to-end.

The usable demo work is mostly:

1. better CLI contract
2. better output formatting
3. graceful fallback behavior

The backend query path should continue to be driven by `query_text_phase_2(...)`.
