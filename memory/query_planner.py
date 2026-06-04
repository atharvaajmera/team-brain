"""LLM-powered query planner using Groq API."""

import json
import os
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_SYSTEM_PROMPT = """\
You are a query parser for a Slack archive search bot.

Output ONLY valid JSON, no markdown fences, no explanation.

Schema:
{{
  "action": "search" | "recent" | "summarize" | "reject",
  "search_query": "semantic search string or null",
  "filters": {{
    "author": "username or null",
    "after": "YYYY-MM-DD or null",
    "before": "YYYY-MM-DD or null",
    "limit": integer or null
  }}
}}

Action definitions:
- "search": find specific conversations by topic/content (most common)
- "recent": latest N messages, no semantic search needed
- "summarize": summary of conversations, possibly filtered by topic/author/time
- "reject": clearly unrelated to Slack (weather, jokes, etc.)

Rules:
- For "search": set search_query to a clean version of what to search for
- For "recent": set limit (default 10). search_query is null
- For "summarize": set search_query to the topic if there is one
- For time references: compute actual dates relative to today
- For author mentions: extract just the name, lowercase
- If vague but about Slack content, default to "search", not "reject"

Today's date: {today}
"""


def plan_query(user_query: str) -> dict:
    """Parse a user query into structured intent via Groq.
    Returns dict with: action, search_query, filters.
    Falls back to search on any error."""
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = _SYSTEM_PROMPT.format(today=today)

    try:
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": prompt + f"\n\nUser query: {user_query}"}],
            model=_MODEL,
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed = json.loads(text)
        action = parsed.get("action", "search")
        if action not in ("search", "recent", "summarize", "reject"):
            action = "search"

        return {
            "action": action,
            "search_query": parsed.get("search_query"),
            "filters": {
                "author": parsed.get("filters", {}).get("author"),
                "after": parsed.get("filters", {}).get("after"),
                "before": parsed.get("filters", {}).get("before"),
                "limit": parsed.get("filters", {}).get("limit"),
            },
        }
    except Exception as e:
        print(f"[query_planner] Groq call failed: {e}")
        return {
            "action": "search",
            "search_query": user_query,
            "filters": {"author": None, "after": None, "before": None, "limit": None},
        }
