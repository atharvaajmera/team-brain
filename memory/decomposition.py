"""Query decomposition for broad/summarize queries.

When a user asks something broad like "Summarize last week's issues" or
"Give me an overview of backend problems", a single semantic search may
miss entire topic clusters.  This module uses Groq to break a broad query
into 2-5 focused sub-queries, retrieves candidates for each, then merges
and deduplicates the result set so the downstream ranker sees a wider,
more representative pool of threads.
"""

import json
import logging
from groq import Groq
from memory.settings import settings

logger = logging.getLogger("decomposition")
_client = Groq(api_key=settings.GROQ_API_KEY)
_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_DECOMPOSE_PROMPT = """\
You are a query decomposition engine for a Slack archive search bot.

The user asked a broad question that likely spans multiple topics.
Break it into 2-5 focused sub-queries that together cover the full scope
of the original question.  Each sub-query should be a short, specific
search string suitable for semantic vector search over Slack messages.

Output ONLY valid JSON, no markdown fences, no explanation.

Schema:
{{
    "sub_queries": ["query1", "query2", ...],
    "reasoning": "one-line explanation of how you split the query"
}}

Rules:
- Each sub-query should target a distinct topic/aspect.
- Keep sub-queries short (3-8 words).
- Do NOT repeat the same concept in multiple sub-queries.
- If the query is already narrow, return just 1 sub-query.
- Maximum 5 sub-queries.

Examples:

User: "Give me an overview of all the issues this week"
Output: {{"sub_queries": ["deployment failures and rollbacks", "database migration issues", "frontend bugs and errors", "infrastructure and scaling problems", "security vulnerabilities and patches"], "reasoning": "Split by common engineering issue categories"}}

User: "What happened with the backend?"
Output: {{"sub_queries": ["backend API errors and timeouts", "backend deployment and scaling", "backend database issues"], "reasoning": "Split backend concerns into API, deployment, and database"}}

User: "Summarize everything"
Output: {{"sub_queries": ["deployment and release issues", "bug fixes and error reports", "infrastructure and performance", "security and dependency updates"], "reasoning": "Broad catch-all split into four major engineering areas"}}
"""


def decompose_query(user_query: str) -> dict:
    """Break a broad query into focused sub-queries via Groq.

    Returns:
        dict with keys:
            - sub_queries: list[str] of focused search strings
            - reasoning: str explaining the decomposition
            - decomposed: bool indicating whether decomposition was applied
    """
    try:
        response = _client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": _DECOMPOSE_PROMPT + f"\n\nUser query: {user_query}",
            }],
            model=_MODEL,
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed = json.loads(text)
        sub_queries = parsed.get("sub_queries", [user_query])

        # Validate: must be a list of strings, 1-5 items
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [user_query]
        sub_queries = [q for q in sub_queries if isinstance(q, str) and q.strip()]
        if not sub_queries:
            sub_queries = [user_query]
        sub_queries = sub_queries[:5]

        return {
            "sub_queries": sub_queries,
            "reasoning": parsed.get("reasoning", ""),
            "decomposed": len(sub_queries) > 1,
        }

    except Exception as e:
        logger.error(f"[decomposition] Groq call failed: {e}")
        return {
            "sub_queries": [user_query],
            "reasoning": "fallback — decomposition failed",
            "decomposed": False,
        }
