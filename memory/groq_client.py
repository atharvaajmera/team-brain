"""Groq API wrapper for answer generation."""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Using the model we decided on
_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def generate_answer(query: str, context: str) -> str:
    prompt = (
        "You are a helpful assistant for a software engineering team. "
        "You answer questions based on archived Slack conversations.\n\n"
        "Important rules:\n"
        "- Base your answer ONLY on the provided Slack threads.\n"
        "- If the threads are not relevant, say so clearly.\n"
        "- Do not fabricate information.\n"
        "- If one thread clearly answers the question, be specific.\n"
        "- If multiple threads are relevant, summarize across them.\n"
        "- Reference specific authors and timestamps when useful.\n\n"
        f"--- Retrieved Slack threads ---\n{context}\n"
        f"--- End of threads ---\n\n"
        f"User question: {query}\n\nAnswer:"
    )
    
    try:
        response = _client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[groq_client] Generation failed: {e}")
        return ""
