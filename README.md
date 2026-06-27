# TeamBrain

<p align="center">
  <img src="assets/team-brain-logo.jpg" alt="TeamBrain Logo" width="200" style="border-radius: 50%; margin-bottom: 20px;" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/ChromaDB-0.5.5-blue?style=for-the-badge" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Enabled">
  <img src="https://img.shields.io/badge/Groq-Llama%204%20Scout-orange?style=for-the-badge" alt="Groq API">
  <img src="https://img.shields.io/badge/Local-Ollama-purple?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/Tests-40/40_Passing-brightgreen?style=for-the-badge" alt="Tests 40/40">
</p>

A privacy-aware Retrieval-Augmented Generation (RAG) system that ingests your team's Slack conversations and lets you search, summarize, and ask questions about them using natural language. TeamBrain connects directly to your Slack workspace, processes engineering discussions, and answers complex queries without leaking private data to the cloud.

---

## Features

- **LLM-Powered Query Understanding:** Natural language queries are mapped to strict JSON execution plans via Groq.
- **Intelligent Evidence Gating:** An AI-powered reasoning engine breaks down broad queries, evaluates retrieval evidence, and clarifies ambiguity before attempting to answer. Calculates composite confidence scores (cosine distance, lexical overlap, thread density). If confidence is low, the bot clarifies rather than hallucinates.
- **Privacy-Aware Routing:** Automatic PII detection safely diverts sensitive queries to local offline models (Ollama). Protected from leaking API keys and other secrets.
- **Advanced Retrieval:** MMR diversification, Pseudo-Relevance Feedback (PRF), and thread-level message-count weighting.
- **Clean Ingestion:** Excludes bot messages and queries directed *at* the bot to prevent polluting the knowledge base with noise.
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
| **Deployment**| Docker & Docker Compose |

## Installation & Deployment (Production)

The easiest way to run TeamBrain is via Docker Compose, which automatically spins up the vector database, the background sync worker, and the Slack bot.

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/team-brain.git
cd team-brain
```

**2. Configure environment variables**
Create a `.env` file in the root directory:
```env
# Required Slack Tokens
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_APP_TOKEN=xapp-your-slack-app-token

# APIs and Models
GROQ_API_KEY=gsk_your-groq-api-key
OLLAMA_URL=https://your-ngrok-id.ngrok-free.app/api/generate
MODEL=llama3.2

# Configuration
PORT=8090
SLACK_WORKSPACE=your-workspace-name
```

*Note: If exposing local Ollama via Ngrok, run ngrok with `--host-header="localhost:11434"` to bypass Ollama's Host origin protections.*

**3. Launch with Docker**
```bash
sudo docker compose up -d
```
This will start 3 containers: `chroma-db` (vector store), `bot` (listens for Slack messages), and `sync-worker` (syncs new messages every 60 minutes).

## Usage

### Slack Bot (Live Mode)
Invite `@Exocortex_Local` to any channels you want it to index and answer questions in. Ask it questions directly by mentioning it: `@Exocortex_Local what was the decision on the instance size?`

### Local Development (Without Docker)
If you want to run the python scripts manually:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the bot
python slack_bot.py

# Run a CLI query
python ask.py "what are the recent deployment issues" --debug
```

## Project Structure

```text
team-brain-python/
├── assets/                 # Project images and logos
├── ask.py                  # CLI query interface
├── slack_bot.py            # Slack presentation and events layer
├── sync_job.py             # Scheduled incremental ingestion job
├── requirements.txt        # Python dependencies
├── .env                    # API keys and environment variables
├── docker-compose.yml      # Multi-container orchestration
├── Dockerfile              # App containerization
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
