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
    "goal": "answer" | "catch_up" | "analysis" | "clarify" | "reject",
    "retrieval_steps": [
        {{
            "tool": "semantic_search" | "recent_threads" | "author_search",
            "query": "search string (null if not needed)",
            "filters": {{
                "author": "username or null",
                "after": "YYYY-MM-DD or null",
                "before": "YYYY-MM-DD or null"
            }},
            "limit": integer
        }}
    ],
    "answer_requirements": {{
        "format": "direct" | "summary" | "timeline" | "comparison" | "decision",
        "cite_sources": true
    }}
}}

Tool guidelines:
- semantic_search: default for topic-based queries. Set query to the core topic.
- recent_threads: use for "what happened today", "catch me up". No query needed.
- author_search: use for "what did alice say?". Put username in filters.author, and topic in query.

Format guidelines:
- direct: concise answer
- summary: thematic overview of a topic
- timeline: chronological events
- comparison: comparing multiple topics
- decision: highlighting agreed outcomes

Examples:
User: "what happened with redis?"
Output: {{"goal": "answer", "retrieval_steps": [{{"tool": "semantic_search", "query": "redis", "filters": {{}}, "limit": 40}}], "answer_requirements": {{"format": "summary", "cite_sources": true}}}}

User: "catch me up on backend today"
Output: {{"goal": "catch_up", "retrieval_steps": [{{"tool": "recent_threads", "query": null, "filters": {{"after": "{today}"}}, "limit": 40}}], "answer_requirements": {{"format": "timeline", "cite_sources": true}}}}

User: "what did alice say about deploys?"
Output: {{"goal": "answer", "retrieval_steps": [{{"tool": "author_search", "query": "deploys", "filters": {{"author": "alice"}}, "limit": 40}}], "answer_requirements": {{"format": "direct", "cite_sources": true}}}}

Today's date: {today}
"""


def plan_query(user_query: str) -> dict:
    """Parse a user query into structured intent via Groq.
    Returns dict with: goal, retrieval_steps, answer_requirements.
    Falls back to a semantic search plan on any error."""
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = _SYSTEM_PROMPT.format(today=today)

    try:
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": prompt + f"\n\nUser query: {user_query}"}],
            model=_MODEL,
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed = json.loads(text)
        
        # Safe clamp limits
        for step in parsed.get("retrieval_steps", []):
            limit = step.get("limit")
            if not isinstance(limit, int):
                step["limit"] = 40
            else:
                step["limit"] = min(max(limit, 1), 100)

        return parsed
    except Exception as e:
        print(f"[query_planner] Groq call failed: {e}")
        return {
            "goal": "answer",
            "retrieval_steps": [{
                "tool": "semantic_search",
                "query": user_query,
                "filters": {},
                "limit": 40
            }],
            "answer_requirements": {
                "format": "direct",
                "cite_sources": True
            }
        }
