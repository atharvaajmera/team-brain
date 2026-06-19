import logging
from time import perf_counter

from memory.models import QueryResponse, QueryPlan
from memory.query_planner import plan_query
from memory.decision import execute_plan
from memory.evidence import evaluate_evidence
from memory.privacy import scan_threads, redact_threads
from memory.citations import format_citation
from memory.llm import generate_response, build_context
from memory.groq_client import generate_answer as cloud_generate_answer
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
    from memory.models import Citation
    from memory.citations import ts_to_readable, make_permalink
    
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
            
            citations.append(Citation(
                author=author,
                ts=ts,
                readable_ts=readable_ts,
                channel_id=channel_id,
                permalink=permalink,
                snippet=snippet,
                thread_id=thread_id
            ))
    return citations

def _fallback_summary(goal: str, threads: list[dict]) -> str:
    if goal == "reject" or not threads:
        return "I could not find relevant Slack discussions for that question."
    if len(threads) == 1:
        return f"The most relevant discussion is thread:{str(int(float(threads[0].get('thread_id', '?'))))}."
    return "I found several relevant discussions and grouped the strongest threads below."

def _generate_answer(query: str, threads: list[dict], route: str, plan: QueryPlan, redacted_query: str) -> tuple[str, float]:
    answer_reqs = plan.answer_requirements.model_dump(exclude_none=True)
    t = perf_counter()
    try:
        if route == "cloud":
            safe_query = redacted_query or query
            context = build_context(threads)
            summary = cloud_generate_answer(safe_query, context, answer_reqs).strip()
            if summary:
                return summary, perf_counter() - t
                
        # fallback to local
        summary = generate_response(query, plan.goal.upper(), threads, answer_reqs=answer_reqs).strip()
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

    t = perf_counter()
    plan = plan_query(query)
    timings["plan"] = perf_counter() - t

    if plan.goal == "reject":
        timings["total"] = perf_counter() - t0
        return QueryResponse(
            status="reject",
            goal=plan.goal,
            route="local",
            answer="I could not find relevant Slack discussions for that question.",
            plan=plan.model_dump(exclude_none=True),
            timings=timings
        )

    threads = execute_plan(plan, query, timings, allowed_channel_ids=allowed_channel_ids)
    
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
            timings=timings
        )

    t = perf_counter()
    scan = scan_threads(query, threads)
    timings["privacy_scan"] = perf_counter() - t

    t = perf_counter()
    route = "local" if (no_cloud or scan.route == "local") else "cloud"
    final_threads = redact_threads(threads) if route == "cloud" else threads
    timings["redact"] = perf_counter() - t

    answer, summary_time = _generate_answer(query, final_threads, route, plan, scan.redacted_query)
    timings["summary"] = summary_time

    citations = _build_citations(final_threads)
    
    timings["total"] = perf_counter() - t0

    from memory.models import Diagnostics, PrivacyScanDiagnostic, EvidenceDiagnostic
    
    return QueryResponse(
        status="ok",
        goal=plan.goal,
        route=route,
        answer=answer,
        citations=citations,
        threads=final_threads,
        plan=plan.model_dump(exclude_none=True),
        timings=timings,
        debug=Diagnostics(
            scan=PrivacyScanDiagnostic(
                pii_count=scan.total_pii_count,
                high_sensitivity=scan.high_sensitivity_found,
                findings=list(scan.findings.keys()) if scan.findings else []
            ),
            evidence=EvidenceDiagnostic(
                confidence=evidence.confidence,
                reason=evidence.reason
            )
        )
    )
