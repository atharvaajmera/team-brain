import time
import pytest
from pydantic import ValidationError

from memory.models import QueryPlan, RetrievalStep, FilterSpec, AnswerRequirements
from memory.evidence import evaluate_evidence, EvidenceResult


# ── Helper to build threads with retrieval scores ──

def _make_thread(thread_id, messages=None, min_distance=0.3, avg_distance=0.4,
                 thread_score=0.2, message_count=3):
    """Build a fake thread dict with _retrieval_score attached."""
    if messages is None:
        messages = [
            {"id": f"{thread_id}_m1", "document": "some relevant discussion about the topic",
             "metadata": {"ts": str(time.time() - 100), "author": "alice"}},
            {"id": f"{thread_id}_m2", "document": "follow up message with more details",
             "metadata": {"ts": str(time.time() - 50), "author": "bob"}},
        ]
    return {
        "thread_id": thread_id,
        "messages": messages,
        "_retrieval_score": {
            "min_distance": min_distance,
            "avg_distance": avg_distance,
            "thread_score": thread_score,
            "message_count": message_count,
        }
    }


# ── Model Validation Tests ──

def test_query_plan_validation_valid():
    """Test valid instantiation of QueryPlan."""
    plan = QueryPlan(
        goal="answer",
        retrieval_steps=[
            RetrievalStep(
                tool="semantic_search",
                query="test topic",
                limit=10,
                filters=FilterSpec(author="alice")
            )
        ]
    )
    assert plan.goal == "answer"
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "semantic_search"
    assert plan.retrieval_steps[0].filters.author == "alice"


def test_query_plan_validation_invalid_limit():
    """Test clamping of invalid limit values."""
    plan = QueryPlan(
        goal="catch_up",
        retrieval_steps=[
            RetrievalStep(
                tool="recent_threads",
                limit=999
            )
        ]
    )
    assert plan.retrieval_steps[0].limit == 100
    
    plan2 = QueryPlan(
        goal="catch_up",
        retrieval_steps=[
            RetrievalStep(
                tool="recent_threads",
                limit=-5
            )
        ]
    )
    assert plan2.retrieval_steps[0].limit == 1


def test_query_plan_validation_invalid_goal():
    """Test invalid goal."""
    with pytest.raises(ValidationError):
        QueryPlan(goal="invalid_goal")


# ── Evidence Scoring Tests ──

def test_evidence_no_threads():
    """No threads → clarify with confidence 0."""
    plan = QueryPlan(goal="answer")
    result = evaluate_evidence(plan, [], "my query")
    
    assert result.strong_enough is False
    assert result.reason == "no_threads"
    assert result.confidence == 0.0
    assert "couldn't find any discussions" in result.clarification_question


def test_evidence_strong_close_match():
    """Thread with low distance and high overlap → strong, high confidence."""
    plan = QueryPlan(goal="answer")
    threads = [
        _make_thread("t1", min_distance=0.25, messages=[
            {"id": "m1", "document": "redis outage caused by connection pool exhaustion",
             "metadata": {"ts": str(time.time() - 100), "author": "alice"}},
        ]),
    ]
    result = evaluate_evidence(plan, threads, "redis outage")
    
    assert result.strong_enough is True
    assert result.confidence > 0.5
    assert result.reason in ("high_relevance", "good_distance", "good_overlap")


def test_evidence_weak_distance():
    """Thread with high distance → clarify."""
    plan = QueryPlan(goal="answer")
    threads = [
        _make_thread("t1", min_distance=1.1, messages=[
            {"id": "m1", "document": "something unrelated about lunch plans",
             "metadata": {"ts": str(time.time() - 100), "author": "alice"}},
        ]),
    ]
    result = evaluate_evidence(plan, threads, "redis outage")
    
    assert result.strong_enough is False
    assert result.reason in ("weak_distance", "low_overlap")


def test_evidence_summarize_needs_multiple_threads():
    """Summarize with only 1 thread → clarify (needs >= 2)."""
    plan = QueryPlan(goal="summarize")
    threads = [_make_thread("t1", min_distance=0.3)]
    result = evaluate_evidence(plan, threads, "summarize all issues")
    
    assert result.strong_enough is False
    assert result.reason == "too_few_threads"


def test_evidence_summarize_with_enough_threads():
    """Summarize with 2+ threads → can proceed."""
    plan = QueryPlan(goal="summarize")
    threads = [
        _make_thread("t1", min_distance=0.3),
        _make_thread("t2", min_distance=0.4),
    ]
    result = evaluate_evidence(plan, threads, "summarize all issues")
    
    assert result.strong_enough is True
    assert result.confidence > 0.3


def test_evidence_decision_format_without_markers():
    """Decision format but no decision language in threads → clarify."""
    plan = QueryPlan(
        goal="answer",
        answer_requirements=AnswerRequirements(format="decision")
    )
    threads = [
        _make_thread("t1", min_distance=0.3, messages=[
            {"id": "m1", "document": "we talked about auth token rotation options",
             "metadata": {"ts": str(time.time() - 100), "author": "alice"}},
            {"id": "m2", "document": "there are several approaches we could take",
             "metadata": {"ts": str(time.time() - 50), "author": "bob"}},
        ]),
    ]
    result = evaluate_evidence(plan, threads, "what did we decide about auth rotation")
    
    assert result.strong_enough is False
    assert result.reason == "no_decision_markers"
    assert "no clear decision" in result.clarification_question


def test_evidence_decision_format_with_markers():
    """Decision format with decision language → pass."""
    plan = QueryPlan(
        goal="answer",
        answer_requirements=AnswerRequirements(format="decision")
    )
    threads = [
        _make_thread("t1", min_distance=0.3, messages=[
            {"id": "m1", "document": "we discussed auth token rotation approaches",
             "metadata": {"ts": str(time.time() - 100), "author": "alice"}},
            {"id": "m2", "document": "agreed we will go with rotating tokens every 24h",
             "metadata": {"ts": str(time.time() - 50), "author": "bob"}},
        ]),
    ]
    result = evaluate_evidence(plan, threads, "what did we decide about auth rotation")
    
    assert result.strong_enough is True


def test_evidence_catch_up_with_stale_threads():
    """Catch_up with threads older than 72h → clarify as stale."""
    plan = QueryPlan(goal="catch_up")
    old_ts = str(time.time() - (80 * 3600))  # 80 hours ago
    threads = [
        _make_thread("t1", min_distance=0.3, messages=[
            {"id": "m1", "document": "old discussion about deploys",
             "metadata": {"ts": old_ts, "author": "alice"}},
        ]),
    ]
    result = evaluate_evidence(plan, threads, "catch me up on deploys")
    
    assert result.strong_enough is False
    assert result.reason == "stale_threads"
    assert "hours old" in result.clarification_question


def test_evidence_catch_up_with_recent_threads():
    """Catch_up with recent threads → pass."""
    plan = QueryPlan(goal="catch_up")
    recent_ts = str(time.time() - 3600)  # 1 hour ago
    threads = [
        _make_thread("t1", min_distance=0.3, messages=[
            {"id": "m1", "document": "just deployed the new auth service",
             "metadata": {"ts": recent_ts, "author": "alice"}},
        ]),
    ]
    result = evaluate_evidence(plan, threads, "catch me up on deploys")
    
    assert result.strong_enough is True


def test_evidence_gap_boosts_confidence():
    """Large gap between top and second thread should boost confidence."""
    plan = QueryPlan(goal="answer")
    threads_dominant = [
        _make_thread("t1", min_distance=0.25),
        _make_thread("t2", min_distance=0.85),
    ]
    threads_flat = [
        _make_thread("t1", min_distance=0.25),
        _make_thread("t2", min_distance=0.27),
    ]
    
    result_dominant = evaluate_evidence(plan, threads_dominant, "redis outage issues")
    result_flat = evaluate_evidence(plan, threads_flat, "redis outage issues")
    
    # Dominant gap should produce higher confidence
    assert result_dominant.confidence > result_flat.confidence
