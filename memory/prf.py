import re
from collections import Counter, defaultdict

try:
    from scripts.query_generator import _tokenize as generator_tokenize
except Exception:
    generator_tokenize = None

PRF_STOPWORDS = {
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

PRF_TECH_WORDS = {
    "oauth", "redis", "docker", "slack", "api", "auth", "token", "login", "deploy",
    "deployment", "rollback", "staging", "production", "postgres", "mysql", "db",
    "database", "migration", "cache", "caching", "worker", "queue", "lambda",
    "s3", "kafka", "nginx", "gunicorn", "celery", "pagination", "dashboard",
    "charts", "ci", "cd", "pipeline", "build", "release", "security", "vulnerability",
    "patch", "docs", "documentation", "endpoint", "timeout", "latency", "bug",
    "bugs", "error", "errors", "fix", "fixed", "failing", "fails", "failure",
}


def tokenize_prf_text(text):
    if generator_tokenize is not None:
        return generator_tokenize(text)

    if not text:
        return []

    tokens = re.findall(r"[a-z0-9][a-z0-9._/-]*", text.lower())
    cleaned = []

    for token in tokens:
        token = token.strip("._/-")
        if not token:
            continue

        is_number = token.isdigit()
        is_tech = token in PRF_TECH_WORDS

        if token in PRF_STOPWORDS and not is_tech:
            continue
        if len(token) <= 2 and not is_number and not is_tech:
            continue

        cleaned.append(token)

    return cleaned


def extract_expansion_terms(query, candidates, max_terms=6, top_k_messages=12):
    top_candidates = candidates[:top_k_messages]
    if not top_candidates:
        return []

    query_tokens = set(tokenize_prf_text(query))
    token_frequency = Counter()
    message_coverage = defaultdict(int)

    for candidate in top_candidates:
        text = candidate.get("document") or candidate.get("metadata", {}).get("text", "")
        message_tokens = tokenize_prf_text(text)
        if not message_tokens:
            continue

        token_frequency.update(message_tokens)
        for token in set(message_tokens):
            message_coverage[token] += 1

    scored = []
    for token, freq in token_frequency.items():
        if token in query_tokens:
            continue

        coverage = message_coverage[token]
        score = 0.0

        # Core PRF signal: repeated terms across top retrieved messages.
        score += freq
        score += coverage * 1.5

        # Light bonuses for technical/error-code style terms.
        if token in PRF_TECH_WORDS:
            score += 1.5
        if token.isdigit() and len(token) >= 3:
            score += 2.0
        elif any(ch.isdigit() for ch in token):
            score += 1.0

        # Slight penalty for generic one-off words.
        if coverage == 1 and token not in PRF_TECH_WORDS:
            score -= 1.0

        scored.append((token, score, freq, coverage))

    scored.sort(key=lambda item: (-item[1], -item[3], -item[2], item[0]))
    return [token for token, _, _, _ in scored[:max_terms]]
