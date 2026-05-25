from memory.decision import query_text_phase_2
from memory.llm import generate_response
from memory.storage import collection


def _tid_label(thread_id):
    return f"thread:{str(int(float(thread_id)))}"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _estimate_confidence(intent, stats):
    entropy = _safe_float(stats.get("entropy"), 0.0)
    coherence = _safe_float(stats.get("coherence"), 0.0)
    rel_gap = _safe_float(stats.get("rel_gap"), 0.0)
    abs_ratio = _safe_float(stats.get("abs_ratio"), 1.0)

    if intent == "NARROW":
        score = (0.45 * rel_gap) + (0.30 * coherence) + (0.25 * (1.0 - entropy))
    elif intent == "AMBIGUOUS":
        score = (0.35 * entropy) + (0.35 * coherence) + (0.30 * (1.0 - abs_ratio))
    elif intent == "BROAD":
        score = (0.40 * entropy) + (0.30 * (1.0 - rel_gap)) + (0.30 * (1.0 - abs_ratio))
    else:
        score = (0.40 * entropy) + (0.35 * (1.0 - coherence)) + (0.25 * abs_ratio)

    return _clamp(round(score, 2))


def _snippet(text, limit=120):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _thread_title(thread):
    messages = thread.get("messages", [])
    if not messages:
        return "(empty thread)"
    return _snippet(messages[0].get("document", ""), limit=80) or "(no text)"


def _format_top_threads(threads):
    lines = []
    for idx, thread in enumerate(threads, 1):
        title = _thread_title(thread)
        lines.append(f"{idx}. {_tid_label(thread.get('thread_id', '?'))} - {title}")
    return "\n".join(lines) if lines else "None"


def _format_evidence(threads, max_threads=3, max_msgs=2):
    lines = []
    for thread in threads[:max_threads]:
        label = _tid_label(thread.get("thread_id", "?"))
        for msg in thread.get("messages", [])[:max_msgs]:
            meta = msg.get("metadata", {})
            author = meta.get("author", meta.get("user", "unknown"))
            ts = meta.get("ts", "?")
            snippet = _snippet(msg.get("document", ""), limit=110)
            lines.append(f"- {label} @{author} [{ts}]: {snippet}")
    return "\n".join(lines) if lines else "- No supporting message snippets available."


def _fallback_summary(intent, threads):
    if intent == "REJECT" or not threads:
        return "I could not find relevant Slack discussions for that question."
    if intent == "NARROW":
        return f"The most relevant discussion is {_tid_label(threads[0].get('thread_id', '?'))}."
    if intent == "AMBIGUOUS":
        return "I found multiple plausible discussions. The top threads below show the strongest candidates."
    return "I found several relevant discussions and grouped the strongest threads below."


def _render_result(query, result):
    intent = result["type"].upper()
    stats = result.get("stats", {})
    threads = result.get("threads", [])
    confidence = _estimate_confidence(intent, stats)

    try:
        summary = generate_response(query, intent, threads).strip()
        if not summary:
            summary = _fallback_summary(intent, threads)
    except Exception:
        summary = _fallback_summary(intent, threads)

    sections = [
        f"Intent: {intent}",
        f"Confidence: {confidence:.2f}",
    ]

    if result.get("is_fallback"):
        sections.append(f"Fallback: {result.get('fallback_reason', 'retrieval fallback used')}")

    sections.extend(
        [
            "",
            "Top Threads:",
            _format_top_threads(threads),
            "",
            "Summary:",
            summary,
            "",
            "Evidence:",
            _format_evidence(threads),
        ]
    )

    return "\n".join(sections)


def _check_corpus_ready():
    try:
        return collection.count() > 0
    except Exception:
        return False


if __name__ == "__main__":
    if not _check_corpus_ready():
        print("No indexed Slack corpus was found. Build the local index first, then try again.")
        raise SystemExit(1)

    print("Ask about your Slack archive. Type 'exit' to quit.")

    while True:
        user_query = input(">> ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "close"]:
            print("Exiting...")
            break

        result = query_text_phase_2(user_query)
        if not result or not result.get("threads"):
            print(
                "\n".join(
                    [
                        "Intent: REJECT",
                        "Confidence: 0.00",
                        "",
                        "Summary:",
                        "I could not find relevant Slack discussions for that question.",
                        "",
                    ]
                )
            )
            continue

        print()
        print(_render_result(user_query, result))
        print()
            
