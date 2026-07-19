"""Groq API wrapper for answer generation."""

import os
import logging
from groq import Groq
from memory.settings import settings
from memory.shared import strip_reasoning

logger = logging.getLogger("groq_client")

_client = Groq(api_key=settings.GROQ_API_KEY)
_MODEL = settings.GROQ_MODEL

def generate_answer(query: str, context: str, answer_reqs: dict = None) -> str:
    answer_reqs = answer_reqs or {}
    format_str = answer_reqs.get("format", "direct")
    cite = answer_reqs.get("cite_sources", True)
    
    cite_rule = (
        "- When citing, reference the author and timestamp only: e.g. '@alice (2026-05-03 14:30)'.\n"
        "- Do NOT invent permalinks or write 'Permalink: Not available'.\n"
        "- Do NOT paste raw Slack URLs; the system appends source links separately.\n"
        "- Quote message body only — do not repeat the author name inside the quote."
    ) if cite else "- No need for explicit citations."
    format_rule = (
        f"- Format your response as a {format_str}. "
        "Keep it concise (2-4 sentences for direct answers)."
    )

    prompt = (
        "You are a helpful assistant for a software engineering team. "
        "You answer questions based on archived Slack conversations.\n\n"
        "Important rules:\n"
        "- Base your answer ONLY on the provided Slack threads.\n"
        "- If the threads are not relevant, say so clearly.\n"
        "- Do not fabricate information.\n"
        f"{cite_rule}\n"
        f"{format_rule}\n\n"
        f"--- Retrieved Slack threads ---\n{context}\n"
        f"--- End of threads ---\n\n"
        f"User question: {query}\n\nAnswer:"
    )
    
    try:
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_MODEL,
            temperature=0.3,
            max_tokens=2048,
        )
        return strip_reasoning(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"[groq_client] Generation failed: {e}")
        return ""
