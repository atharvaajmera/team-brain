"""Shared constants and utility functions used across retrieval, ranking, and evaluation.

This module is the single source of truth for:
- Constants (ALPHA, ENTROPY_TEMP, MIN_THREAD_SIZE, STOPWORDS, TECH_WORDS)
- softmax_entropy computation
- Text tokenization
- Thread grouping and scoring
"""

import math
import re

import numpy as np

# ── Constants ──

ALPHA = 0.25              # Thread score discount factor for message count
ENTROPY_TEMP = 0.1        # Default softmax temperature
MIN_THREAD_SIZE = 2       # Minimum messages for a thread to be considered in NORMAL mode
MAX_RETRIEVAL_RESULTS = 40

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


# ── Utility Functions ──

def softmax_entropy(values, temp=ENTROPY_TEMP):
    """Compute normalized softmax entropy over a list of scores.

    Lower values indicate one dominant item; higher values indicate a flat distribution.
    Returns a value in [0, 1] where 1 = maximum entropy (uniform distribution).
    """
    v = np.array(values, dtype=float)
    neg_over_t = -v / temp
    neg_over_t -= neg_over_t.max()
    exp_s = np.exp(neg_over_t)
    probs = exp_s / exp_s.sum()
    ent = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_ent = float(math.log2(len(v))) if len(v) > 1 else 1.0
    return ent / max_ent


def tokenize(text):
    """Tokenize text for retrieval and PRF use.

    Lowercases, removes stopwords, keeps tech terms and numeric tokens.
    Returns a list of cleaned token strings.
    """
    if not text:
        return []

    # Use \w to support Unicode letters (CJK, Cyrillic, accented, etc.)
    # Start with a letter/number [^\W_], followed by word chars or . _ / -
    tokens = re.findall(r"[^\W_][\w._/-]*", text.lower())
    cleaned = []

    for token in tokens:
        token = token.strip("._/-")
        if not token:
            continue

        is_number = token.isdigit()
        is_tech = token in TECH_WORDS

        if token in STOPWORDS and not is_tech:
            continue
        if len(token) <= 2 and token.isascii() and not is_number and not is_tech:
            continue

        cleaned.append(token)

    return cleaned


def group_threads(candidates):
    """Group retrieval candidates by thread_id and compute thread-level scores.

    Each candidate must have candidate['metadata']['thread_id'] and candidate['distance'].

    Thread score formula: avg_distance - log(message_count + 1) * ALPHA
    Lower score = better thread (threads with more messages and lower distances win).

    Returns:
        List of thread aggregate dicts sorted by thread_score ascending (best first).
        Each dict contains:
            thread_id, avg_distance, min_distance, message_count,
            thread_score, best_candidate
    """
    threads = {}
    for c in candidates:
        meta = c.get("metadata") or {}
        tid = meta.get("thread_id", meta.get("ts", c.get("id")))
        if tid is None:
            continue
        threads.setdefault(tid, {"candidates": [], "distances": []})
        threads[tid]["candidates"].append(c)
        threads[tid]["distances"].append(c.get("distance", 1.0))

    aggregates = []
    for tid, td in threads.items():
        avg_d = float(np.mean(td["distances"]))
        aggregates.append({
            "thread_id": tid,
            "avg_distance": avg_d,
            "min_distance": float(np.min(td["distances"])),
            "message_count": len(td["candidates"]),
            "thread_score": avg_d - math.log(len(td["candidates"]) + 1) * ALPHA,
            "best_candidate": min(td["candidates"], key=lambda x: x["distance"]),
        })

    return sorted(aggregates, key=lambda x: x["thread_score"])
