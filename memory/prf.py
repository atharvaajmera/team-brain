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


def extract_expansion_terms(query, candidates, max_terms=4, top_k_messages=12):
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


def build_expanded_queries(query, expansion_terms, max_queries=2):
    base_query = " ".join(query.split()).strip()
    if not base_query or not expansion_terms or max_queries < 1:
        return []

    terms = [term.strip() for term in expansion_terms if term and term.strip()]
    if not terms:
        return []

    expanded_queries = []
    seen = set()

    def add_query(parts):
        expanded = " ".join(part for part in parts if part).strip()
        expanded = " ".join(expanded.split())
        if not expanded or expanded == base_query or expanded in seen:
            return False
        seen.add(expanded)
        expanded_queries.append(expanded)
        return len(expanded_queries) >= max_queries

    top_terms = terms[:4]

    if len(top_terms) >= 2:
        if add_query([base_query, top_terms[0], top_terms[1]]):
            return expanded_queries

    if len(top_terms) >= 3:
        if add_query([base_query, top_terms[0], top_terms[2]]):
            return expanded_queries

    if len(top_terms) >= 4:
        if add_query([base_query, top_terms[2], top_terms[3]]):
            return expanded_queries

    for term in top_terms:
        if add_query([base_query, term]):
            return expanded_queries

    return expanded_queries[:max_queries]

def merge_prf_candidates(candidate_lists, limit=40):
    merged_by_id = {}

    for query_index, candidates in enumerate(candidate_lists):
        for rank, candidate in enumerate(candidates):
            candidate_id = candidate.get("id")
            if not candidate_id:
                continue

            distance = candidate.get("distance", float("inf"))
            existing = merged_by_id.get(candidate_id)

            if existing is None:
                merged = dict(candidate)
                merged["prf_hits"] = 1
                merged["best_distance"] = distance
                merged["original_rank"] = rank if query_index == 0 else None
                merged["original_distance"] = distance if query_index == 0 else None
                merged["expansion_hits"] = 0
                merged["best_expansion_distance"] = None
                merged_by_id[candidate_id] = merged
                continue

            existing["prf_hits"] += 1
            if query_index == 0:
                existing.update(candidate)
                existing["best_distance"] = distance
                existing["original_rank"] = rank
                existing["original_distance"] = distance
            else:
                existing["expansion_hits"] += 1
                best_expansion_distance = existing.get("best_expansion_distance")
                if best_expansion_distance is None or distance < best_expansion_distance:
                    existing["best_expansion_distance"] = distance
                if existing.get("original_rank") is None and distance < existing.get("best_distance", float("inf")):
                    existing.update(candidate)
                    existing["best_distance"] = distance

    merged = list(merged_by_id.values())

    for candidate in merged:
        original_rank = candidate.get("original_rank")
        original_distance = candidate.get("original_distance")
        expansion_hits = candidate.get("expansion_hits", 0)
        best_expansion_distance = candidate.get("best_expansion_distance")

        if original_rank is not None:
            boost = 0.0
            boost += min(expansion_hits, 2) * 0.05
            if best_expansion_distance is not None and original_distance is not None and best_expansion_distance < original_distance:
                boost += 0.03
            candidate["prf_merge_score"] = original_rank - boost
        else:
            candidate["prf_merge_score"] = 1000 + candidate.get("best_distance", float("inf"))

    merged.sort(
        key=lambda candidate: (
            candidate.get("prf_merge_score", float("inf")),
            candidate.get("best_distance", float("inf")),
            -candidate.get("prf_hits", 1),
        )
    )
    return merged[:limit]

def run_prf_retrieval(
    query,
    first_pass_candidates,
    retrieve_fn,
    max_terms=4,
    max_queries=2,
    limit=40,
):
    if not first_pass_candidates:
        return {
            "expanded_queries": [],
            "expansion_terms": [],
            "merged_candidates": [],
        }

    expansion_terms = extract_expansion_terms(
        query,
        first_pass_candidates,
        max_terms=max_terms,
    )
    expanded_queries = build_expanded_queries(
        query,
        expansion_terms,
        max_queries=max_queries,
    )

    candidate_lists = [first_pass_candidates]
    for expanded_query in expanded_queries:
        expanded_candidates = retrieve_fn(expanded_query)
        if expanded_candidates:
            candidate_lists.append(expanded_candidates)

    merged_candidates = merge_prf_candidates(candidate_lists, limit=limit)
    return {
        "expanded_queries": expanded_queries,
        "expansion_terms": expansion_terms,
        "merged_candidates": merged_candidates,
    }
