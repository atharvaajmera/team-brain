# `ask.py` CLI Contract

This document defines the user-facing behavior for the `python ask.py` demo CLI.

## Goal

Make `python ask.py` the primary end-to-end demo entrypoint for:

1. indexed Slack corpus retrieval
2. intent classification
3. answer generation
4. evidence display

The output must be human-readable and must not print raw JSON.

## Invocation

Primary mode:

```bash
python ask.py
```

Behavior:

- starts an interactive prompt
- accepts one natural-language query per line
- exits on `exit`, `quit`, or `close`

Optional future mode:

- one-shot query arguments may be added later
- not required for the first usable demo

## Required Flow

For each user query, the CLI must follow this sequence:

1. retrieve relevant Slack threads from the indexed corpus
2. classify the query intent as `NARROW`, `AMBIGUOUS`, `BROAD`, or `REJECT`
3. generate an answer or summary
4. show supporting evidence

The CLI should not re-run ingestion automatically in the first version.
If no indexed corpus is available, it should print a helpful setup message.

## Required Output Shape

The CLI output should use sections like this:

```text
Intent: BROAD
Confidence: 0.74

Top Threads:
1. ...
2. ...

Summary:
...

Evidence:
- ...
- ...
```

This is a formatting contract, not a literal byte-for-byte requirement.

## Section Requirements

### Intent

- always shown when classification succeeds
- one of:
  - `NARROW`
  - `AMBIGUOUS`
  - `BROAD`
  - `REJECT`

### Confidence

- shown as a compact numeric score in the range `0.00` to `1.00`
- should be derived from classifier/retrieval geometry
- it is a user-facing confidence heuristic, not a calibrated probability

### Top Threads

- show the selected thread list in ranked order
- each item should include:
  - thread label or id
  - short representative description or snippet
- keep each item compact enough for terminal reading

### Summary

- for `NARROW`: direct answer using the best thread
- for `AMBIGUOUS`: concise answer with disambiguating framing
- for `BROAD`: cross-thread summary of themes
- for `REJECT`: a polite message saying relevant Slack evidence was not found

### Evidence

- show why the answer was produced
- should be grounded in retrieved Slack messages
- may include:
  - representative snippets
  - authors
  - timestamps
  - thread references

## Error and Empty-State Behavior

### No corpus indexed

Print a clear setup message, for example:

```text
No indexed Slack corpus was found. Build the local index first, then try again.
```

### No relevant result

Print a clean, non-JSON message, for example:

```text
Intent: REJECT
Confidence: 0.32

Summary:
I could not find relevant Slack discussions for that question.
```

### LLM failure

If answer generation fails:

- do not crash the CLI
- fall back to a deterministic evidence-first summary

## Non-Goals For First Demo

- automatic Slack ingestion from the CLI
- JSON output mode
- web UI
- multi-command orchestration
- probability calibration

## Implementation Boundary

The first usable CLI/demo should focus on presentation and resilience, not classifier redesign.

Use the existing pipeline:

1. `query_text_phase_2(...)`
2. retrieve thread messages
3. `generate_response(...)`
4. render human-readable output
