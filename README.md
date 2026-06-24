# TeamBrain

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/ChromaDB-0.5.5-blue?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Groq-Llama%204%20Scout-orange?style=for-the-badge" alt="Groq API">
  <img src="https://img.shields.io/badge/Local-Ollama-purple?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/Tests-40/40_Passing-brightgreen?style=for-the-badge" alt="Tests 40/40">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License MIT">
</p>

A privacy-aware Retrieval-Augmented Generation (RAG) system that ingests your team's Slack conversations and lets you search, summarize, and ask questions about them using natural language. TeamBrain connects directly to your Slack workspace, processes engineering discussions, and answers complex queries without leaking private data to the cloud.

---

## Features

- **LLM-Powered Query Understanding:** Natural language queries are mapped to strict JSON execution plans via Groq.
- **Intelligent Evidence Gating:** An AI-powered reasoning engine breaks down broad queries, evaluates retrieval evidence, and clarifies ambiguity before attempting to answer. Calculates composite confidence scores (cosine distance, lexical overlap, thread density). If confidence is low, the bot clarifies rather than hallucinates.
- **Privacy-Aware Routing:** Automatic PII detection safely diverts sensitive queries to local offline models (Ollama).
- **Advanced Retrieval:** MMR diversification, Pseudo-Relevance Feedback (PRF), and thread-level message-count weighting.
- **Incremental Syncing:** Cursor-based pagination that remembers its last sync state. Only new Slack messages are fetched.
- **Offline Diagnostics:** Run fully deterministic eval benchmarks without burning LLM API credits.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Vector DB** | ChromaDB 0.5.5 |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Cloud LLM** | Groq API (Llama 4 Scout 17B) |
| **Local LLM** | Ollama (llama3.2) |
| **Slack** | Slack SDK (Bolt) |
| **Testing** | Pytest |

## Installation

**1. Clone and create virtual environment**
```bash
git clone https://github.com/yourusername/team-brain-python.git
cd team-brain-python
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**
Create a `.env` file in the root directory:
```env
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_CHANNEL_ID_AUTO=C0A766EQALV
GROQ_API_KEY=gsk_your-groq-api-key
```

**4. Build the knowledge base**
```bash
# Incremental sync (first run = full index)
python brain.py

# Force full re-index
python brain.py --full
```

## Usage

### Slack Bot (Live Mode)
Run the Slack application to listen for app mentions:
```bash
python slack_bot.py
```

### CLI Interactive Mode
Run the local CLI to query your corpus directly:
```bash
python ask.py
```

### One-Shot Query & Debugging
Run a query immediately from the terminal. Use `--debug` to see exactly how the orchestrator evaluates your question:
```bash
python ask.py "what are the recent deployment issues"
python ask.py "redis cache problems" --debug
```

## Project Structure

```text
team-brain-python/
├── ask.py                  # CLI query interface
├── slack_bot.py            # Slack presentation and events layer
├── sync_job.py             # Scheduled incremental ingestion job
├── requirements.txt        # Python dependencies
├── .env                    # API keys
├── config/
│   ├── sync_state.json     # Incremental sync checkpoint
│   └── diagnostics_cache.json # Offline LLM test fixtures
├── docs/                   # Architecture, Setup, and Multilingual plans
├── tests/                  # Unit and integration test suite
├── memory/
│   ├── storage.py          # ChromaDB collection management
│   ├── service.py          # Core orchestrator and access control
│   ├── privacy.py          # PII detection, redaction, routing
│   └── ...                 # Core reasoning/RAG modules
└── scripts/
    └── diagnostics.py      # System diagnostics and evaluation
```



## License

This project is licensed under the [MIT License](LICENSE).
