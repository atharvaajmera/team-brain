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
        # Thread 1 — Google OAuth Bug (4 msgs)
        {"user": "alice", "text": "Google OAuth login is failing in production.", "ts": "1773000000", "thread_ts": "1773000000"},
        {"user": "bob", "text": "Is this the redirect URI issue again?", "ts": "1773000060", "thread_ts": "1773000000"},
        {"user": "alice", "text": "Yes, it's returning 401 Unauthorized.", "ts": "1773000120", "thread_ts": "1773000000"},
        {"user": "alice", "text": "Fixed. The OAUTH_REDIRECT_URI was using HTTP instead of HTTPS.", "ts": "1773000180", "thread_ts": "1773000000"},
        
        # Thread 2 — General API Issues (5 msgs)
        {"user": "charlie", "text": "We've been seeing multiple API issues this week.", "ts": "1773100000", "thread_ts": "1773100000"},
        {"user": "frank", "text": "Is it related to pagination or deployment?", "ts": "1773100060", "thread_ts": "1773100000"},
        {"user": "diana", "text": "Some endpoints are timing out intermittently.", "ts": "1773100120", "thread_ts": "1773100000"},
        {"user": "frank", "text": "Might be database connection pool limits.", "ts": "1773100180", "thread_ts": "1773100000"},
        {"user": "charlie", "text": "Opened a ticket for the DB team to investigate.", "ts": "1773100240", "thread_ts": "1773100000"},
        
        # Thread 3 — API Documentation Review (4 msgs)
        {"user": "frank", "text": "Updated API documentation for new endpoints.", "ts": "1773200000", "thread_ts": "1773200000"},
        {"user": "diana", "text": "Reviewing the API docs now.", "ts": "1773200060", "thread_ts": "1773200000"},
        {"user": "diana", "text": "Looks good. Examples are clear.", "ts": "1773200120", "thread_ts": "1773200000"},
        {"user": "frank", "text": "Great, pushing to main branch.", "ts": "1773200180", "thread_ts": "1773200000"},
        
        # Thread 4 — Deployment v3.0 (5 msgs)
        {"user": "eve", "text": "Deploying version 3.0.0 to production in 5 minutes.", "ts": "1773300000", "thread_ts": "1773300000"},
        {"user": "alice", "text": "Holding off on DB migrations.", "ts": "1773300060", "thread_ts": "1773300000"},
        {"user": "eve", "text": "Deployment successful.", "ts": "1773300120", "thread_ts": "1773300000"},
        {"user": "eve", "text": "Monitoring logs for API errors.", "ts": "1773300180", "thread_ts": "1773300000"},
        {"user": "bob", "text": "No issues seen so far from QA.", "ts": "1773300240", "thread_ts": "1773300000"},
        
        # Thread 5 — Dashboard Performance (5 msgs)
        {"user": "bob", "text": "Dashboard is loading slowly.", "ts": "1773400000", "thread_ts": "1773400000"},
        {"user": "eve", "text": "Query taking 2 seconds on users table.", "ts": "1773400060", "thread_ts": "1773400000"},
        {"user": "eve", "text": "Missing index found.", "ts": "1773400120", "thread_ts": "1773400000"},
        {"user": "eve", "text": "Added index. Performance improved.", "ts": "1773400180", "thread_ts": "1773400000"},
        {"user": "bob", "text": "Dashboard loads in under 200ms now. Great fix.", "ts": "1773400240", "thread_ts": "1773400000"},
        
        # Thread 6 — Multi-Bug Release (6 msgs)
        {"user": "alice", "text": "Latest release introduced several bugs.", "ts": "1773500000", "thread_ts": "1773500000"},
        {"user": "alice", "text": "Bug 1: File upload fails.", "ts": "1773500060", "thread_ts": "1773500000"},
        {"user": "alice", "text": "Bug 2: API search results duplicated.", "ts": "1773500120", "thread_ts": "1773500000"},
        {"user": "bob", "text": "Taking file upload bug.", "ts": "1773500180", "thread_ts": "1773500000"},
        {"user": "eve", "text": "Fixing API search duplication.", "ts": "1773500240", "thread_ts": "1773500000"},
        {"user": "bob", "text": "All bugs fixed and deployed.", "ts": "1773500300", "thread_ts": "1773500000"},
        
        # Thread 7 — CI/CD Pipeline (4 msgs)
        {"user": "frank", "text": "CI pipeline is broken after the linter update.", "ts": "1773600000", "thread_ts": "1773600000"},
        {"user": "diana", "text": "The ESLint config needs the new flat config format.", "ts": "1773600060", "thread_ts": "1773600000"},
        {"user": "frank", "text": "Migrated to flat config. Build passes now.", "ts": "1773600120", "thread_ts": "1773600000"},
        {"user": "diana", "text": "Nice. Also added a pre-commit hook.", "ts": "1773600180", "thread_ts": "1773600000"},
        
        # Thread 8 — Pagination Discussion (3 msgs)
        {"user": "charlie", "text": "Should we use cursor-based pagination?", "ts": "1773700000", "thread_ts": "1773700000"},
        {"user": "frank", "text": "Cursor-based is better for large datasets.", "ts": "1773700060", "thread_ts": "1773700000"},
        {"user": "charlie", "text": "Implementing cursor-based in API layer.", "ts": "1773700120", "thread_ts": "1773700000"},
        
        # Thread 9 — Caching Proposal (4 msgs)
        {"user": "frank", "text": "We should add caching to API responses.", "ts": "1773800000", "thread_ts": "1773800000"},
        {"user": "diana", "text": "Redis would help reduce repeated DB queries.", "ts": "1773800060", "thread_ts": "1773800000"},
        {"user": "frank", "text": "Let's build a POC this sprint.", "ts": "1773800120", "thread_ts": "1773800000"},
        {"user": "charlie", "text": "I can set up a Redis cluster on staging.", "ts": "1773800180", "thread_ts": "1773800000"},
        
        # Thread 10 — Team Offsite (3 msgs)
        {"user": "alice", "text": "Planning team offsite next month.", "ts": "1773900000", "thread_ts": "1773900000"},
        {"user": "bob", "text": "Goa sounds good.", "ts": "1773900060", "thread_ts": "1773900000"},
        {"user": "eve", "text": "I'll check hotel options.", "ts": "1773900120", "thread_ts": "1773900000"},
        
        # Thread 11 — Database Migration (4 msgs)
        {"user": "eve", "text": "Running database migration for user profiles schema.", "ts": "1774000000", "thread_ts": "1774000000"},
        {"user": "alice", "text": "Make sure to back up before running ALTER TABLE.", "ts": "1774000060", "thread_ts": "1774000000"},
        {"user": "eve", "text": "Backup done. Migration completed without downtime.", "ts": "1774000120", "thread_ts": "1774000000"},
        {"user": "bob", "text": "Verified. All user profile fields are intact.", "ts": "1774000180", "thread_ts": "1774000000"},
        
        # Thread 12 — Security Vulnerability (3 msgs)
        {"user": "diana", "text": "Found a critical XSS vulnerability in the search bar.", "ts": "1774100000", "thread_ts": "1774100000"},
        {"user": "frank", "text": "Sanitizing all user inputs now. Patch incoming.", "ts": "1774100060", "thread_ts": "1774100000"},
        {"user": "diana", "text": "Patch deployed. Pentest confirms the fix works.", "ts": "1774100120", "thread_ts": "1774100000"},
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
