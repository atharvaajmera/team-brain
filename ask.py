import argparse
import json
import sys

from memory.service import answer_query, is_corpus_ready


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
    from memory.citations import ts_to_readable, make_permalink
    lines = []
    for idx, thread in enumerate(threads, 1):
        title = _thread_title(thread)
        thread_id = thread.get('thread_id', '?')
        readable_ts = ts_to_readable(thread_id)
        # Get channel_id from first message metadata if available
        msgs = thread.get("messages", [])
        channel_id = msgs[0].get("metadata", {}).get("channel_id", "") if msgs else ""
        permalink = make_permalink(channel_id, thread_id)
        
        header = f"{idx}. [{readable_ts}]"
        if permalink:
            header += f" ({permalink})"
        header += f" - {title}"
        lines.append(header)
    return "\n".join(lines) if lines else "None"


def _format_evidence(citations):
    lines = []
    for cite in citations:
        line = f"- @{cite.author} [{cite.readable_ts}]"
        if cite.permalink:
            line += f" ({cite.permalink})"
        line += f": {cite.snippet}"
        lines.append(line)
    return "\n".join(lines) if lines else "- No supporting message snippets available."


def _format_debug(result):
    plan = result.plan
    debug = result.debug
    
    lines = ["Debug:"]
    lines.append(f"  Goal: {plan.get('goal', 'N/A')}")
    lines.append(f"  Answer Reqs: {plan.get('answer_requirements', {})}")
    lines.append("  Retrieval Steps:")
    for step in plan.get("retrieval_steps", []):
        lines.append(f"    - Tool: {step.get('tool')}")
        lines.append(f"      Query: {step.get('query')}")
        lines.append(f"      Filters: {step.get('filters')}")
        lines.append(f"      Limit: {step.get('limit')}")

    # Show decomposition info for summarize queries
    decomp = result.timings.get("_decomposition")
    if decomp and decomp.get("decomposed"):
        lines.append("  Decomposition:")
        lines.append(f"    Reasoning: {decomp.get('reasoning', 'N/A')}")
        lines.append(f"    Sub-queries:")
        for i, sq in enumerate(decomp.get('sub_queries', []), 1):
            lines.append(f"      {i}. {sq}")

    if debug:
        scan = debug.scan
        evidence = debug.evidence
        lines.append(f"  Scan: PII count={scan.pii_count}, High sensitivity={scan.high_sensitivity}")
        if scan.findings:
            lines.append(f"  Findings: {scan.findings}")
        lines.append(f"  Evidence: confidence={evidence.confidence:.2f}, reason={evidence.reason}")
            
    return "\n".join(lines)


def _format_profile(timings):
    """Format step timings as a visual breakdown."""
    total = timings.get("total", 0.001)
    labels = [
        ("plan",          "Query Planner (Groq)"),
        ("decompose",     "Query Decomposition"),
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


def _render_result(result, debug=False, profile=False):
    sections = [
        f"Goal: {result.goal.upper()}",
        f"Route: {result.route.upper()}",
    ]
    
    if result.status == "clarify":
        sections.extend([
            "",
            "Clarification Needed:",
            result.clarification_question,
            "",
            "Top Threads (for context):",
            _format_top_threads(result.threads)
        ])
    elif result.status == "reject":
        sections.extend([
            "",
            "Summary:",
            result.answer,
        ])
    else:
        sections.extend([
            "",
            "Top Threads:",
            _format_top_threads(result.threads),
            "",
            "Summary:",
            result.answer,
            "",
            "Evidence:",
            _format_evidence(result.citations),
        ])

    if debug:
        sections.extend(["", _format_debug(result)])

    if profile:
        sections.extend(["", _format_profile(result.timings)])

    return "\n".join(sections)


def _print_no_result():
    print(
        "\n".join(
            [
                "Goal: REJECT",
                "Route: LOCAL",
                "",
                "Summary:",
                "I could not find relevant Slack discussions for that question.",
                "",
            ]
        )
    )


def _run_query(user_query, debug=False, profile=False, no_cloud=False, json_output=False):
    result = answer_query(user_query, source="cli", no_cloud=no_cloud, debug=debug)
        
    if json_output:
        out_data = {
            "status": result.status,
            "goal": result.goal,
            "route": result.route,
            "answer": result.answer,
            "clarification_question": result.clarification_question,
            "threads": result.threads,
            "plan": result.plan,
            "timings": result.timings,
        }
        try:
            json_bytes = json.dumps(out_data, ensure_ascii=False).encode("utf-8")
            sys.stdout.buffer.write(json_bytes)
            sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()
        except OSError:
            pass
        return

    output = "\n" + _render_result(result, debug=debug, profile=profile) + "\n"
    try:
        sys.stdout.buffer.write(output.encode("utf-8"))
        sys.stdout.buffer.flush()
    except OSError:
        pass


def _parse_args():
    parser = argparse.ArgumentParser(description="Query the indexed Slack archive.")
    parser.add_argument("query", nargs="*", help="Optional one-shot query.")
    parser.add_argument("--debug", action="store_true", help="Show compact runtime debug stats.")
    parser.add_argument("--profile", action="store_true", help="Show per-step timing breakdown.")
    parser.add_argument("--no-cloud", action="store_true", help="Force all queries through local Ollama, never cloud.")
    parser.add_argument("--json", action="store_true", help="Output result as JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not is_corpus_ready():
        if args.json:
            print(json.dumps({"error": "No indexed Slack corpus was found. Build the local index first."}))
        else:
            print("No indexed Slack corpus was found. Build the local index first, then try again.")
        raise SystemExit(1)

    one_shot_query = " ".join(args.query).strip()
    if one_shot_query:
        _run_query(one_shot_query, debug=args.debug, profile=args.profile, no_cloud=args.no_cloud, json_output=args.json)
        raise SystemExit(0)

    if args.json:
        print(json.dumps({"error": "JSON mode requires a one-shot query."}))
        raise SystemExit(1)

    print("Ask about your Slack archive. Type 'exit' to quit.")

    while True:
        user_query = input(">> ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "close"]:
            print("Exiting...")
            break

        _run_query(user_query, debug=args.debug, profile=args.profile, no_cloud=args.no_cloud, json_output=args.json)

