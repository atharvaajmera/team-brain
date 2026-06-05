import argparse

from memory.decision import process_query
from memory.llm import generate_response, build_context
from memory.groq_client import generate_answer as cloud_generate_answer
from memory.storage import collection


def _tid_label(thread_id):
    return f"thread:{str(int(float(thread_id)))}"


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


def _format_debug(result):
    plan = result.get("plan", {})
    scan = result.get("scan")
    
    lines = ["Debug:"]
    lines.append(f"  Plan: {plan}")
    if scan:
        lines.append(f"  Scan: PII count={scan.total_pii_count}, High sensitivity={scan.high_sensitivity_found}")
        if scan.findings:
            lines.append(f"  Findings: {list(scan.findings.keys())}")
            
    return "\n".join(lines)


def _format_profile(timings):
    """Format step timings as a visual breakdown."""
    total = timings.get("total", 0.001)
    labels = [
        ("plan",          "Query Planner (Groq)"),
        ("retrieve",      "Vector Retrieval + PRF"),
        ("rank",          "Ranking & MMR"),
        ("fetch_threads", "Thread Fetch (ChromaDB)"),
        ("privacy_scan",  "Privacy Scan"),
        ("redact",        "Redaction"),
        ("summary",       "Summary Generation"),
    ]
    lines = ["Profile:"]
    lines.append(f"  {'Step':<28} {'Time':>8}  {'%':>5}  Bar")
    lines.append(f"  {'─'*28} {'─'*8}  {'─'*5}  {'─'*20}")
    for key, label in labels:
        t = timings.get(key, 0)
        pct = (t / total) * 100 if total > 0 else 0
        bar_len = int(pct / 5)  # 20 chars = 100%
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {label:<28} {t*1000:>7.0f}ms  {pct:>4.1f}%  {bar}")
    lines.append(f"  {'─'*28} {'─'*8}  {'─'*5}  {'─'*20}")
    lines.append(f"  {'TOTAL':<28} {total*1000:>7.0f}ms")
    return "\n".join(lines)


def _generate_summary(query, result, threads):
    intent = result.get("action", "SEARCH")
    route = result.get("route", "local")
    
    from time import perf_counter
    t = perf_counter()
    try:
        if route == "cloud":
            context = build_context(threads)
            summary = cloud_generate_answer(query, context).strip()
            if summary:
                elapsed = perf_counter() - t
                return summary, False, elapsed
                
        # fallback to local
        summary = generate_response(query, intent, threads).strip()
        if summary:
            elapsed = perf_counter() - t
            return summary, False, elapsed
    except Exception as e:
        print(f"[summary] Generation error: {e}")
        pass
        
    elapsed = perf_counter() - t
    return _fallback_summary(intent, threads), True, elapsed


def _render_result(query, result, debug=False, profile=False):
    intent = result.get("action", "SEARCH")
    route = result.get("route", "local")
    scan = result.get("scan")
    threads = result.get("threads", [])
    
    summary, used_summary_fallback, summary_time = _generate_summary(query, result, threads)

    sections = [
        f"Intent: {intent}",
        f"Route: {route.upper()}",
    ]

    if scan and scan.high_sensitivity_found:
        sections.append("Privacy Warning: High sensitivity PII detected, forced local routing.")

    if used_summary_fallback:
        sections.append("Summary mode: fallback")

    if intent == "REJECT":
        sections.extend(
            [
                "",
                "Summary:",
                summary,
            ]
        )
    else:
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

    if debug:
        sections.extend(["", _format_debug(result)])

    if profile:
        timings = result.get("timings", {})
        timings["summary"] = summary_time
        timings["total"] = timings.get("total", 0) + summary_time
        sections.extend(["", _format_profile(timings)])

    return "\n".join(sections)


def _check_corpus_ready():
    try:
        return collection.count() > 0
    except Exception:
        return False


def _print_no_result():
    print(
        "\n".join(
            [
                "Intent: REJECT",
                "Route: LOCAL",
                "",
                "Summary:",
                "I could not find relevant Slack discussions for that question.",
                "",
            ]
        )
    )


def _run_query(user_query, debug=False, profile=False):
    result = process_query(user_query)
    if not result or result.get("action") == "REJECT":
        _print_no_result()
        return

    print()
    print(_render_result(user_query, result, debug=debug, profile=profile))
    print()


def _parse_args():
    parser = argparse.ArgumentParser(description="Query the indexed Slack archive.")
    parser.add_argument("query", nargs="*", help="Optional one-shot query.")
    parser.add_argument("--debug", action="store_true", help="Show compact runtime debug stats.")
    parser.add_argument("--profile", action="store_true", help="Show per-step timing breakdown.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not _check_corpus_ready():
        print("No indexed Slack corpus was found. Build the local index first, then try again.")
        raise SystemExit(1)

    one_shot_query = " ".join(args.query).strip()
    if one_shot_query:
        _run_query(one_shot_query, debug=args.debug, profile=args.profile)
        raise SystemExit(0)

    print("Ask about your Slack archive. Type 'exit' to quit.")

    while True:
        user_query = input(">> ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "close"]:
            print("Exiting...")
            break

        _run_query(user_query, debug=args.debug, profile=args.profile)
