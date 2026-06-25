import logging
import re
from datetime import datetime, timedelta
from time import perf_counter

from memory.decision import execute_plan
from memory.evidence import evaluate_evidence
from memory.groq_client import generate_answer as cloud_generate_answer
from memory.llm import build_context, generate_response
from memory.models import (
    AnswerRequirements,
    FilterSpec,
    QueryPlan,
    QueryResponse,
    RetrievalStep,
)
from memory.privacy import redact_threads, scan_text, scan_threads
from memory.query_planner import plan_query
from memory.storage import collection


def is_corpus_ready() -> bool:
    try:
        return collection.count() > 0
    except Exception:
        return False


def _build_citations(threads: list[dict]) -> list[dict]:
    # We will just return the raw threads for now, and let `ask.py` format them,
    # or we can build the citation models here. Let's do it in `ask.py` for now
    # to avoid changing `ask.py` format logic too much, or we can use the Citation model.
    # The models.py has a Citation model. Let's build it.
    from memory.citations import make_permalink, ts_to_readable
    from memory.models import Citation

    citations = []
    for thread in threads[:3]:
        thread_id = thread.get("thread_id", "?")
        for msg in thread.get("messages", [])[:2]:
            meta = msg.get("metadata", {})
            author = meta.get("author_display", meta.get("author", "unknown"))
            ts = meta.get("ts", "?")
            channel_id = meta.get("channel_id", "")
            readable_ts = ts_to_readable(ts)
            permalink = make_permalink(channel_id, ts)

            text = " ".join((msg.get("document", "")).split())
            snippet = text if len(text) <= 120 else text[:117] + "..."

            citations.append(
                Citation(
                    author=author,
                    ts=ts,
                    readable_ts=readable_ts,
                    channel_id=channel_id,
                    permalink=permalink,
                    snippet=snippet,
                    thread_id=thread_id,
                )
            )
    return citations


def _fallback_summary(goal: str, threads: list[dict]) -> str:
    if goal == "reject" or not threads:
        return "I could not find relevant Slack discussions for that question."

    # Extract first message snippet from top thread
    top = threads[0]
    top_msgs = top.get("messages", [])
    first_text = " ".join((top_msgs[0].get("document", "") if top_msgs else "").split())
    snippet = first_text[:200] + "..." if len(first_text) > 200 else first_text
    author = top_msgs[0].get("metadata", {}).get("author_display",
             top_msgs[0].get("metadata", {}).get("author", "someone")) if top_msgs else "someone"

    if len(threads) == 1:
        return (
            f"I couldn't generate a full summary, but here's the most relevant thread "
            f"(started by @{author}):\n\n> {snippet}"
        )

    # Multiple threads — list the top 2-3
    lines = [
        "I couldn't generate a full summary, but here are the most relevant threads:"
    ]
    for i, t in enumerate(threads[:3], 1):
        msgs = t.get("messages", [])
        text = " ".join((msgs[0].get("document", "") if msgs else "").split())
        snip = text[:120] + "..." if len(text) > 120 else text
        a = msgs[0].get("metadata", {}).get("author_display",
            msgs[0].get("metadata", {}).get("author", "unknown")) if msgs else "unknown"
        lines.append(f"{i}. @{a}: {snip}")
    return "\n".join(lines)


def _looks_like_catch_up(query: str) -> bool:
    lower = query.lower()
    return any(
        phrase in lower
        for phrase in (
            "catch me up",
            "what happened today",
            "what happened recently",
            "latest",
            "recent updates",
            "what's new",
            "whats new",
        )
    )


def _looks_like_summary(query: str) -> bool:
    lower = query.lower()
    return any(
        phrase in lower
        for phrase in (
            "summarize",
            "summary",
            "overview",
            "everything",
            "all the issues",
            "all issues",
            "all the things",
        )
    )


def _extract_author_query(query: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"\bwhat did\s+([\w .\-]+?)\s+say(?:\s+about\s+(.+))?$",
        query.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None

    author = (match.group(1) or "").strip(" ?")
    topic = (match.group(2) or "").strip(" ?") or None
    return author or None, topic


def _local_plan_query(user_query: str) -> QueryPlan:
    query = user_query.strip()
    lower = query.lower()
    today = datetime.now().strftime("%Y-%m-%d")
    week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    if not query:
        return QueryPlan(
            goal="reject",
            retrieval_steps=[],
            answer_requirements=AnswerRequirements(format="direct", cite_sources=True),
        )

    author, author_topic = _extract_author_query(query)
    if author:
        return QueryPlan(
            goal="answer",
            focus="person",
            retrieval_steps=[
                RetrievalStep(
                    tool="author_search",
                    query=author_topic,
                    filters=FilterSpec(author=author),
                    limit=40,
                )
            ],
            answer_requirements=AnswerRequirements(format="direct", cite_sources=True),
        )

    if _looks_like_catch_up(query):
        after = today if "today" in lower else week_start if "week" in lower else None
        return QueryPlan(
            goal="catch_up",
            focus="timeline",
            retrieval_steps=[
                RetrievalStep(
                    tool="recent_threads",
                    query=None,
                    filters=FilterSpec(after=after),
                    limit=40,
                )
            ],
            answer_requirements=AnswerRequirements(
                format="timeline", cite_sources=True
            ),
        )

    answer_format = (
        "decision"
        if any(
            term in lower
            for term in ("decision", "decide", "decided", "agreed", "final call")
        )
        else "direct"
    )
    if _looks_like_summary(query):
        return QueryPlan(
            goal="summarize",
            focus="topic",
            retrieval_steps=[
                RetrievalStep(
                    tool="semantic_search",
                    query=query,
                    filters=FilterSpec(after=week_start if "week" in lower else None),
                    limit=40,
                )
            ],
            answer_requirements=AnswerRequirements(
                format=answer_format, cite_sources=True
            ),
        )

    return QueryPlan(
        goal="answer",
        focus="decision" if answer_format == "decision" else "topic",
        retrieval_steps=[RetrievalStep(tool="semantic_search", query=query, limit=40)],
        answer_requirements=AnswerRequirements(format=answer_format, cite_sources=True),
    )


def _generate_answer(
    query: str, threads: list[dict], route: str, plan: QueryPlan, redacted_query: str
) -> tuple[str, float]:
    answer_reqs = plan.answer_requirements.model_dump(exclude_none=True)
    t = perf_counter()
    try:
        if route == "cloud":
            safe_query = redacted_query or query
            context = build_context(threads, include_permalinks=False)
            summary = cloud_generate_answer(safe_query, context, answer_reqs).strip()
            if summary:
                return summary, perf_counter() - t

        # fallback to local
        summary = generate_response(
            query, plan.goal.upper(), threads, answer_reqs=answer_reqs
        ).strip()
        if summary:
            return summary, perf_counter() - t
    except Exception as e:
        logging.error(f"[summary] Generation error: {e}")

    return _fallback_summary(plan.goal, threads), perf_counter() - t


def answer_query(
    query: str,
    source: str = "cli",
    user_id: str | None = None,
    channel_id: str | None = None,
    allowed_channel_ids: list[str] | None = None,
    no_cloud: bool = False,
    debug: bool = False,
) -> QueryResponse:
    timings = {}
    t0 = perf_counter()

    query_scan = scan_text(query)
    allow_cloud_planning = (not no_cloud) and query_scan.route == "cloud"

    planner_query = query_scan.redacted_text or query

    t = perf_counter()
    plan = (
        plan_query(planner_query) if allow_cloud_planning else _local_plan_query(query)
    )
    timings["plan"] = perf_counter() - t

    if plan.goal == "reject":
        timings["total"] = perf_counter() - t0
        return QueryResponse(
            status="reject",
            goal=plan.goal,
            route="local",
            answer="I could not find relevant Slack discussions for that question.",
            plan=plan.model_dump(exclude_none=True),
            timings=timings,
        )

    threads = execute_plan(
        plan,
        query,
        timings,
        allowed_channel_ids=allowed_channel_ids,
        allow_query_decomposition=allow_cloud_planning,
    )

    evidence = evaluate_evidence(plan, threads, query)
    if not evidence.strong_enough:
        timings["total"] = perf_counter() - t0
        return QueryResponse(
            status="clarify",
            goal=plan.goal,
            route="local",
            answer="",
            clarification_question=evidence.clarification_question,
            threads=threads,
            plan=plan.model_dump(exclude_none=True),
            timings=timings,
        )

    t = perf_counter()
    scan = scan_threads(query, threads)
    timings["privacy_scan"] = perf_counter() - t

    t = perf_counter()
    route = (
        "local"
        if (no_cloud or query_scan.route == "local" or scan.route == "local")
        else "cloud"
    )
    llm_threads = redact_threads(threads, scan.redactor) if route == "cloud" else threads
    timings["redact"] = perf_counter() - t

    answer, summary_time = _generate_answer(
        query, llm_threads, route, plan, scan.redacted_query
    )
    
    if route == "cloud" and scan.redactor:
        answer = scan.redactor.unredact(answer)

    if evidence.strong_enough and evidence.confidence < 0.5:
        answer += "\n\n_Note: I found limited evidence for this query. The answer may be incomplete._"

    broadened_from = timings.get("_broadened")
    if broadened_from:
        from datetime import datetime
        try:
            dt = datetime.strptime(broadened_from, "%Y-%m-%d")
            today = datetime.now().strftime("%Y-%m-%d")
            if broadened_from == today:
                scope_label = "today"
            else:
                scope_label = f"since {broadened_from}"
        except ValueError:
            scope_label = "that time range"
        answer = (
            f"_No recent activity was found for {scope_label}. "
            "Here are the latest updates:_\n\n" + answer
        )
    timings["summary"] = summary_time

    citations = _build_citations(threads)

    timings["total"] = perf_counter() - t0

    from memory.models import Diagnostics, EvidenceDiagnostic, PrivacyScanDiagnostic

    return QueryResponse(
        status="ok",
        goal=plan.goal,
        route=route,
        answer=answer,
        citations=citations,
        threads=threads,
        plan=plan.model_dump(exclude_none=True),
        timings=timings,
        debug=Diagnostics(
            scan=PrivacyScanDiagnostic(
                pii_count=scan.total_pii_count,
                high_sensitivity=scan.high_sensitivity_found,
                findings=list(scan.findings.keys()) if scan.findings else [],
            ),
            evidence=EvidenceDiagnostic(
                confidence=evidence.confidence, reason=evidence.reason
            ),
        ),
    )
