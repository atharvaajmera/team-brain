"""
Mock data seeding script for testing without hitting Slack API.
Simulates the same data structure that ingest.py gets from Slack.
"""

from memory.storage import add_messages


def get_mock_threads():
    """
    Returns mock Slack messages in the same format as Slack API.
    Simulates realistic threaded conversations for testing.
    """
    # Mock messages in Slack API format
    mock_messages = [
        # 🧵 Thread 1 — Google OAuth Bug (Strong, Specific Signal)
        {"user": "alice", "text": "Google OAuth login is failing in production.", "ts": "1773000000", "thread_ts": "1773000000"},
        {"user": "bob", "text": "Is this the redirect URI issue again?", "ts": "1773000060", "thread_ts": "1773000000"},
        {"user": "alice", "text": "Yes, it's returning 401 Unauthorized.", "ts": "1773000120", "thread_ts": "1773000000"},
        {"user": "alice", "text": "Fixed. The OAUTH_REDIRECT_URI was using HTTP instead of HTTPS.", "ts": "1773000180", "thread_ts": "1773000000"},
        
        # 🧵 Thread 2 — General API Issue Discussion (Vague, Weak Signal)
        {"user": "charlie", "text": "We've been seeing multiple API issues this week.", "ts": "1773100000", "thread_ts": "1773100000"},
        {"user": "frank", "text": "Is it related to pagination or deployment?", "ts": "1773100060", "thread_ts": "1773100000"},
        {"user": "diana", "text": "Some endpoints are timing out intermittently.", "ts": "1773100120", "thread_ts": "1773100000"},
        {"user": "frank", "text": "Might be database connection pool limits.", "ts": "1773100180", "thread_ts": "1773100000"},
        
        # 🧵 Thread 3 — API Documentation Review (Specific)
        {"user": "frank", "text": "Updated API documentation for new endpoints.", "ts": "1773200000", "thread_ts": "1773200000"},
        {"user": "diana", "text": "Reviewing the API docs now.", "ts": "1773200060", "thread_ts": "1773200000"},
        {"user": "diana", "text": "Looks good. Examples are clear.", "ts": "1773200120", "thread_ts": "1773200000"},
        {"user": "frank", "text": "Great, pushing to main branch.", "ts": "1773200180", "thread_ts": "1773200000"},
        
        # 🧵 Thread 4 — Deployment Incident (Specific)
        {"user": "eve", "text": "Deploying version 3.0.0 to production in 5 minutes.", "ts": "1773300000", "thread_ts": "1773300000"},
        {"user": "alice", "text": "Holding off on DB migrations.", "ts": "1773300060", "thread_ts": "1773300000"},
        {"user": "eve", "text": "Deployment successful.", "ts": "1773300120", "thread_ts": "1773300000"},
        {"user": "eve", "text": "Monitoring logs for API errors.", "ts": "1773300180", "thread_ts": "1773300000"},
        
        # 🧵 Thread 5 — Performance Issue (Specific)
        {"user": "bob", "text": "Dashboard is loading slowly.", "ts": "1773400000", "thread_ts": "1773400000"},
        {"user": "eve", "text": "Query taking 2 seconds on users table.", "ts": "1773400060", "thread_ts": "1773400000"},
        {"user": "eve", "text": "Missing index found.", "ts": "1773400120", "thread_ts": "1773400000"},
        {"user": "eve", "text": "Added index. Performance improved.", "ts": "1773400180", "thread_ts": "1773400000"},
        
        # 🧵 Thread 6 — Multi-Bug Release Thread (Multiple Topics)
        {"user": "alice", "text": "Latest release introduced several bugs.", "ts": "1773500000", "thread_ts": "1773500000"},
        {"user": "alice", "text": "Bug 1: File upload fails.", "ts": "1773500060", "thread_ts": "1773500000"},
        {"user": "alice", "text": "Bug 2: API search results duplicated.", "ts": "1773500120", "thread_ts": "1773500000"},
        {"user": "bob", "text": "Taking file upload bug.", "ts": "1773500180", "thread_ts": "1773500000"},
        {"user": "eve", "text": "Fixing API search duplication.", "ts": "1773500240", "thread_ts": "1773500000"},
        {"user": "bob", "text": "All bugs fixed and deployed.", "ts": "1773500300", "thread_ts": "1773500000"},
        
        # 🧵 Thread 7 — Single Strong Message Trap
        {"user": "frank", "text": "Minor API issue resolved.", "ts": "1773600000", "thread_ts": "1773600000"},
        
        # 🧵 Thread 8 — Pagination Discussion
        {"user": "charlie", "text": "Should we use cursor-based pagination?", "ts": "1773700000", "thread_ts": "1773700000"},
        {"user": "frank", "text": "Cursor-based is better for large datasets.", "ts": "1773700060", "thread_ts": "1773700000"},
        {"user": "charlie", "text": "Implementing cursor-based in API layer.", "ts": "1773700120", "thread_ts": "1773700000"},
        
        # 🧵 Thread 9 — Caching Proposal
        {"user": "frank", "text": "We should add caching to API responses.", "ts": "1773800000", "thread_ts": "1773800000"},
        {"user": "diana", "text": "Redis would help reduce repeated DB queries.", "ts": "1773800060", "thread_ts": "1773800000"},
        {"user": "frank", "text": "Let's build a POC this sprint.", "ts": "1773800120", "thread_ts": "1773800000"},
        
        # 🧵 Thread 10 — Noise / Unrelated
        {"user": "alice", "text": "Planning team offsite next month.", "ts": "1773900000", "thread_ts": "1773900000"},
        {"user": "bob", "text": "Goa sounds good.", "ts": "1773900060", "thread_ts": "1773900000"},
        {"user": "eve", "text": "I'll check hotel options.", "ts": "1773900120", "thread_ts": "1773900000"},
    ]
    
    return mock_messages


def seed():
    """Seed ChromaDB with mock threads. Callable from benchmark.py or standalone."""
    print("Fetching mock threads for testing...")
    
    # Get mock messages (simulating Slack API response)
    messages = get_mock_threads()
    
    # Process messages (same as brain.py)
    texts = []
    ids = []
    metadatas = []
    
    for msg in messages:
        text = msg.get('text', '')
        if not text.strip():
            continue
        
        user = msg.get('user', 'unknown_user')
        ts = float(msg.get('ts', 'no_ts'))
        thread_id = float(msg.get('thread_ts', ts))
        
        # Log messages (same as ingest.py)
        print(f"User: {user}, Text: {text}")
        
        text_to_embed = f"{text}"
        texts.append(text_to_embed)
        ids.append(f"{user}_{ts}")
        metadatas.append({
            "user": user,
            "ts": ts,
            "text": text,
            "thread_id": thread_id
        })
    
    # Add to memory
    add_messages(texts, ids, metadatas)
    print("Mock threads saved to chromadb successfully.")


if __name__ == "__main__":
    seed()
