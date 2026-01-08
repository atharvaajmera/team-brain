# TeamBrain - Slack RAG Bot

A Retrieval-Augmented Generation (RAG) bot that ingests Slack messages and enables semantic search over your team's conversation history using ChromaDB and sentence transformers.

## 🚀 Features Implemented

### 1. **Slack Integration** ([ingest.py](ingest.py))

- Connects to Slack workspace using Slack SDK
- Fetches conversation history from specified channels
- Retrieves threaded replies and nested conversations
- SSL-secured connection with certificate verification
- Configurable message limit for data ingestion

### 2. **Vector Database Storage** ([memory.py](memory.py))

- Uses ChromaDB for persistent vector storage
- Stores messages with metadata (user, timestamp, thread_id)
- Semantic search using sentence transformers
- Efficient upsert operations for message updates
- Query functionality with configurable result limits (default: top 5)

### 3. **Knowledge Base Builder** ([brain.py](brain.py))

- Automated pipeline to ingest and store Slack messages
- Processes messages from specified channels
- Extracts and structures message metadata
- Batch processing for efficient storage
- Converts raw Slack data into searchable embeddings

### 4. **Interactive Query Interface** ([ask.py](ask.py))

- Command-line interface for querying the knowledge base
- Semantic search with confidence scoring
- Configurable relevance threshold (default: 1.2)
- Displays user, timestamp, and message content
- Filters low-confidence results
- Exit commands: `exit`, `quit`, `close`

## 🛠️ Tech Stack

- **Slack SDK**: Slack workspace integration
- **ChromaDB (v0.5.5)**: Vector database for semantic search
- **Sentence Transformers**: Text embedding generation
- **Python dotenv**: Environment variable management
- **Certifi**: SSL certificate verification

## 📋 Prerequisites

- Python 3.8+
- Slack Bot Token with appropriate permissions
- Access to a Slack workspace

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd team-brain-python
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the root directory:

   ```
   SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
   SLACK_CHANNEL_ID=C0A766EQALV
   ```

## 🎯 Usage

### Building the Knowledge Base

Ingest messages from your Slack channel:

```bash
python brain.py
```

This will:

- Connect to the specified Slack channel
- Fetch the last 50 messages (configurable)
- Store them in the ChromaDB vector database

### Querying the Knowledge Base

Start the interactive query interface:

```bash
python ask.py
```

Then ask questions about your team's conversations:

```
>> What did we discuss about the project deadline?
>> Who mentioned the API integration?
>> exit
```

## 📁 Project Structure

```
team-brain-python/
├── ask.py              # Interactive query interface
├── brain.py            # Knowledge base builder
├── ingest.py           # Slack data ingestion
├── memory.py           # ChromaDB vector operations
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not tracked)
├── chroma_db/         # Vector database storage (generated)
└── README.md          # Project documentation
```

## 🔧 Configuration

### Confidence Threshold ([ask.py](ask.py))

```python
threshold = 1.2  # Lower = stricter matching
```

### Message Limit ([brain.py](brain.py))

```python
messages = get_threads_from_channel(CHANNEL_ID, limit=50)
```

### Number of Results ([memory.py](memory.py))

```python
results = collection.query(query_texts=[query], n_results=5)
```

## 🎓 How It Works

1. **Ingestion**: Messages are fetched from Slack using the Slack SDK
2. **Embedding**: Text is converted to vector embeddings using sentence transformers
3. **Storage**: Embeddings are stored in ChromaDB with metadata
4. **Retrieval**: User queries are embedded and matched against stored vectors
5. **Ranking**: Results are ranked by similarity (distance) and filtered by confidence

## 🔐 Security

- Uses SSL certificate verification for Slack connections
- Stores sensitive tokens in environment variables
- Anonymous telemetry disabled for ChromaDB

## 🚧 Future Enhancements

- Multi-channel support
- Real-time message syncing
- Web interface for queries
- Advanced filtering (date ranges, users, threads)
- Export search results
- Integration with LLM for answer generation

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]
