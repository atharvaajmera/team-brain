"""LLM-powered query planner using Groq API."""

import json
import logging
from datetime import datetime
from pydantic import ValidationError

from groq import Groq
from memory.settings import settings
from memory.models import QueryPlan, RetrievalStep, AnswerRequirements

_client = Groq(api_key=settings.GROQ_API_KEY)
_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_SYSTEM_PROMPT = """\
You are a query parser for a Slack archive search bot.

Output ONLY valid JSON, no markdown fences, no explanation.

Schema:
{{
    "goal": "answer" | "catch_up" | "summarize" | "analysis" | "clarify" | "reject",
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

Goal guidelines:
- answer: specific question about a topic, person, role, decision, or anything that COULD be discussed in Slack (e.g. "what caused the redis outage?", "who is the CTO?", "who is prakhar?")
- catch_up: temporal recency queries (e.g. "catch me up", "what happened today?")
- summarize: broad overviews spanning multiple topics (e.g. "summarize all issues", "give an overview of backend problems", "what are all the things the team discussed?")
- reject: ONLY for queries that are completely unrelated to workplace topics (e.g. "tell me a joke", "what's the weather?", "solve 2+2"). Questions about people, roles, teams, projects, decisions, or anything that could appear in Slack should NEVER be rejected — use "answer" instead.

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

User: "give me an overview of all the issues this week"
Output: {{"goal": "summarize", "retrieval_steps": [{{"tool": "semantic_search", "query": "issues and problems", "filters": {{}}, "limit": 40}}], "answer_requirements": {{"format": "summary", "cite_sources": true}}}}

User: "summarize everything the team discussed"
Output: {{"goal": "summarize", "retrieval_steps": [{{"tool": "semantic_search", "query": "team discussions", "filters": {{}}, "limit": 40}}], "answer_requirements": {{"format": "summary", "cite_sources": true}}}}

Today's date: {today}
"""


def _safe_fallback(user_query: str) -> QueryPlan:
    return QueryPlan(
        goal="answer",
        retrieval_steps=[RetrievalStep(tool="semantic_search", query=user_query, limit=40)],
        answer_requirements=AnswerRequirements(format="direct", cite_sources=True)
    )

def _apply_semantic_rules(plan: QueryPlan) -> QueryPlan:
    if plan.goal == "reject":
        plan.retrieval_steps = []
    elif plan.goal == "clarify":
        plan.retrieval_steps = []

    for step in plan.retrieval_steps:
        if step.tool == "recent_threads":
            step.query = None
        elif step.tool == "author_search" and not step.filters.author:
            # If they picked author_search but no author, fallback to semantic search
            step.tool = "semantic_search"

    return plan


def plan_query(user_query: str) -> QueryPlan:
    """Parse a user query into structured intent via Groq.
    Returns a QueryPlan instance.
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
        plan = QueryPlan.model_validate(parsed)
        plan = _apply_semantic_rules(plan)
        return plan

    except (json.JSONDecodeError, ValidationError) as e:
        logging.warning(f"[query_planner] Validation failed: {e}")
        return _safe_fallback(user_query)
    except Exception as e:
        logging.error(f"[query_planner] Groq call failed: {e}")
        return _safe_fallback(user_query)
