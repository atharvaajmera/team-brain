import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.storage import collection

DEFAULT_OUTPUT = REPO_ROOT / "config" / "benchmark_queries.json"
DEFAULT_MIN_PER_THREAD = 5
DEFAULT_MAX_PER_THREAD = 10

STOPWORDS = {
    "about", "after", "again", "against", "also", "another", "any", "are", "back",
    "because", "been", "before", "being", "between", "both", "but", "could",
    "does", "doing", "down", "during", "each", "even", "from", "further", "have",
    "having", "into", "its", "just", "more", "most", "only", "other", "over",
    "same", "some", "such", "than", "that", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "under", "until", "very", "want",
    "were", "what", "when", "where", "which", "while", "with", "would", "your",
    "team", "thread", "message", "messages", "issue", "problem", "please", "help",
    "need", "still", "getting", "cannot", "cant", "doesnt", "dont", "should",
    "will", "today", "yesterday", "tomorrow", "thanks", "thank", "update",
}

TECH_WORDS = {
    "oauth", "redis", "docker", "slack", "api", "auth", "token", "login", "deploy",
    "deployment", "rollback", "staging", "production", "postgres", "mysql", "db",
    "database", "migration", "cache", "caching", "worker", "queue", "lambda",
    "s3", "kafka", "nginx", "gunicorn", "celery", "pagination", "dashboard",
    "charts", "ci", "cd", "pipeline", "build", "release", "security", "vulnerability",
    "patch", "docs", "documentation", "endpoint", "timeout", "latency", "bug",
    "bugs", "error", "errors", "fix", "fixed", "failing", "fails", "failure",
}

CODE_HINTS = (
    "error", "exception", "traceback", "failed", "failing", "timeout", "timed out",
    "401", "403", "404", "429", "500", "502", "503", "504", "fix", "fixed",
    "resolved", "patch", "rollback", "retry", "stack", "crash", "cannot", "can't",
)

FIX_HINTS = (
    "fix", "fixed", "resolved", "solution", "patch", "workaround", "rollback",
    "deployed", "merged", "closing", "closed",
)


def _tokenize(text):
    if not text:
        return []

    tokens = re.findall(r"[a-z0-9][a-z0-9._/-]*", text.lower())
    kept = []
    seen = set()

    for token in tokens:
        token = token.strip("._/-")
        if not token:
            continue

        is_number = token.isdigit()
        is_tech = token in TECH_WORDS

        if token in STOPWORDS and not is_tech:
            continue
        if len(token) <= 2 and not is_number and not is_tech:
            continue

        if token not in seen:
            kept.append(token)
            seen.add(token)

    return kept


def _pick_representative_messages(messages):
    if not messages:
        return []

    sorted_messages = sorted(messages, key=lambda item: float(item.get("ts", 0)))
    picks = []

    first = sorted_messages[0]
    picks.append(first)

    code_message = next(
        (
            message for message in sorted_messages
            if any(hint in message.get("text", "").lower() for hint in CODE_HINTS)
        ),
        None,
    )
    if code_message and code_message not in picks:
        picks.append(code_message)

    fix_message = next(
        (
            message for message in reversed(sorted_messages)
            if any(hint in message.get("text", "").lower() for hint in FIX_HINTS)
        ),
        None,
    )
    if fix_message and fix_message not in picks:
        picks.append(fix_message)

    for message in sorted_messages:
        if len(picks) >= 3:
            break
        if message not in picks:
            picks.append(message)

    return picks


def _score_token(token, representative_text):
    score = 0
    if token in TECH_WORDS:
        score += 3
    if token.isdigit():
        score += 3
    if any(ch.isdigit() for ch in token):
        score += 2
    if token in representative_text:
        score += 1
    score += min(len(token), 12) / 12
    return score


def _generate_queries_from_tokens(tokens, min_queries, max_queries, representative_text):
    unique_tokens = []
    seen = set()
    for token in tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    ranked_tokens = sorted(
        unique_tokens,
        key=lambda token: (-_score_token(token, representative_text), token),
    )

    selected = ranked_tokens[:8]
    if len(selected) < 2:
        return []

    queries = []
    seen_queries = set()

    for size in (3, 2, 4):
        for combo in combinations(selected, size):
            query = " ".join(combo).strip()
            if query and query not in seen_queries:
                queries.append(query)
                seen_queries.add(query)
            if len(queries) >= max_queries:
                return queries

    return queries[:max(min_queries, 0)]


def _load_threads_from_db():
    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    threads = defaultdict(list)

    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        thread_id = metadata.get("thread_id")
        if thread_id is None:
            continue

        message = {
            "text": metadata.get("text") or document or "",
            "ts": metadata.get("ts", 0),
            "author": metadata.get("author", metadata.get("user", "unknown")),
        }
        threads[str(int(float(thread_id)))].append(message)

    return threads


def build_query_records(min_per_thread=DEFAULT_MIN_PER_THREAD, max_per_thread=DEFAULT_MAX_PER_THREAD):
    threads = _load_threads_from_db()
    records = []

    for thread_id, messages in sorted(threads.items(), key=lambda item: int(item[0])):
        representatives = _pick_representative_messages(messages)
        representative_text = " ".join(message.get("text", "").lower() for message in representatives)

        tokens = []
        for message in representatives:
            tokens.extend(_tokenize(message.get("text", "")))

        queries = _generate_queries_from_tokens(tokens, min_per_thread, max_per_thread, representative_text)

        for query in queries:
            records.append({
                "query": query,
                "expected_thread_id": thread_id,
            })

    return records


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate benchmark queries from the Chroma thread store."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write benchmark queries JSON",
    )
    parser.add_argument(
        "--min-per-thread",
        type=int,
        default=DEFAULT_MIN_PER_THREAD,
        help="Minimum target queries per thread",
    )
    parser.add_argument(
        "--max-per-thread",
        type=int,
        default=DEFAULT_MAX_PER_THREAD,
        help="Maximum queries per thread",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.min_per_thread < 1 or args.max_per_thread < 1:
        raise ValueError("min/max queries per thread must be positive integers")
    if args.min_per_thread > args.max_per_thread:
        raise ValueError("min-per-thread cannot be greater than max-per-thread")

    records = build_query_records(
        min_per_thread=args.min_per_thread,
        max_per_thread=args.max_per_thread,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    thread_count = len({record["expected_thread_id"] for record in records})
    print(f"Wrote {len(records)} queries across {thread_count} threads to {output_path}")


if __name__ == "__main__":
    main()
