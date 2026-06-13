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
from memory.shared import tokenize

DEFAULT_OUTPUT = REPO_ROOT / "config" / "benchmark_queries.json"
DEFAULT_MIN_PER_THREAD = 5
DEFAULT_MAX_PER_THREAD = 10

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
    """Wrapper around shared.tokenize that removes duplicates (keeps order)."""
    return list(dict.fromkeys(tokenize(text)))


def _is_bad_query(query):
    tokens = query.split()
    if len(tokens) < 2:
        return True
    numeric_ratio = sum(token.isdigit() for token in tokens) / len(tokens)
    if numeric_ratio > 0.5:
        return True
    if all(token in STOPWORDS for token in tokens):
        return True
    return False


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
        if len(token) >= 3:
            score += 3
    if any(ch.isdigit() for ch in token):
        score += 2
    if token in representative_text:
        score += 1
    score += min(len(token), 12) / 12
    return score


def _rank_tokens(tokens, representative_text):
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1

    ranked_tokens = sorted(
        token_counts,
        key=lambda token: (
            -(
                token_counts[token] * 2
                + _score_token(token, representative_text)
                - (1 if token not in TECH_WORDS and token_counts[token] == 1 else 0)
            ),
            token,
        ),
    )
    return ranked_tokens, token_counts


def _generate_queries_from_tokens(thread_id, tokens, min_queries, max_queries, representative_text):
    ranked_tokens, token_counts = _rank_tokens(tokens, representative_text)

    selected = ranked_tokens[:min(10, len(ranked_tokens))]
    if len(selected) < 2:
        print(f"Skipping thread {thread_id}: insufficient tokens")
        return []

    queries = []
    seen_queries = set()

    def add_query(query):
        query = " ".join(query.split()).strip()
        if query and not _is_bad_query(query) and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)
            return len(queries) >= max_queries
        return False

    top_tokens = selected[:5]
    strong_tokens = [token for token in top_tokens if token_counts[token] > 1 or token in TECH_WORDS]
    if len(strong_tokens) < 2:
        strong_tokens = top_tokens

    for size in (2, 3, 4):
        for combo in combinations(top_tokens, size):
            if add_query(" ".join(combo)):
                return queries

    for combo in combinations(strong_tokens[:4], 2):
        t1, t2 = combo
        for template in (
            f"{t1} {t2} issue",
            f"{t1} {t2} failing",
            f"{t1} problem {t2}",
            f"{t1} broken",
            f"{t2} problem",
            f"{t1} not working",
        ):
            if add_query(template):
                return queries

    for token in strong_tokens[:2]:
        for template in (
            f"{token} issue",
            f"{token} failing",
            f"{token} broken",
            f"{token} problem",
        ):
            if add_query(template):
                return queries

    for token in selected[:2]:
        if add_query(token):
            return queries

    for size in (3, 2, 4):
        for combo in combinations(selected, size):
            query = " ".join(reversed(combo))
            if add_query(query):
                return queries

    if len(queries) < min_queries:
        for combo in combinations(selected, 2):
            t1, t2 = combo
            for template in (
                f"{t1} {t2} auth",
                f"{t1} {t2} error",
                f"{t1} {t2} fix",
            ):
                if add_query(template):
                    return queries
                if len(queries) >= min_queries:
                    break
            if len(queries) >= min_queries:
                break

    return queries[:max_queries]


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

    if not threads:
        print("No threads found in Chroma collection 'slack_archive'.")

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

        queries = _generate_queries_from_tokens(
            thread_id,
            tokens,
            min_per_thread,
            max_per_thread,
            representative_text,
        )

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

    if not records:
        raise RuntimeError(
            "No benchmark queries were generated. Check that the repo-root Chroma DB "
            "contains ingested messages with thread_id metadata."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    thread_count = len({record["expected_thread_id"] for record in records})
    print(f"Wrote {len(records)} queries across {thread_count} threads to {output_path}")


if __name__ == "__main__":
    main()
