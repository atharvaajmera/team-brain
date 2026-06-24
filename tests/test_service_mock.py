from unittest.mock import MagicMock, patch

import pytest

from memory.models import QueryResponse
from memory.privacy import PrivacyScan
from memory.service import answer_query


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service._generate_answer")
@patch("memory.service.scan_threads")
def test_answer_query_reject(mock_scan, mock_gen, mock_exec, mock_plan):
    """Test that answer_query correctly handles a 'reject' plan without executing retrieval."""

    # Mock planner returning a reject plan
    from memory.models import AnswerRequirements, QueryPlan

    plan = QueryPlan(
        goal="reject",
        answer_requirements=AnswerRequirements(format="direct", cite_sources=False),
    )
    mock_plan.return_value = plan

    response = answer_query("Tell me a joke")

    assert isinstance(response, QueryResponse)
    assert response.status == "reject"
    assert response.goal == "reject"
    assert "I could not find relevant Slack discussions" in response.answer

    # Should short-circuit before execution
    mock_exec.assert_not_called()
    mock_gen.assert_not_called()
    mock_scan.assert_not_called()


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
@patch("memory.service._generate_answer")
@patch("memory.service.scan_threads")
def test_answer_query_clarify(mock_scan, mock_gen, mock_evid, mock_exec, mock_plan):
    """Test that answer_query returns a clarification response when evidence is weak."""

    from memory.models import QueryPlan

    plan = QueryPlan(goal="answer")
    mock_plan.return_value = plan

    # Mock retrieval returning some threads
    mock_exec.return_value = [{"thread_id": "123"}]

    # Mock evidence evaluator returning weak evidence
    from memory.evidence import EvidenceResult

    mock_evid.return_value = EvidenceResult(
        confidence=0.1,
        strong_enough=False,
        reason="weak_distance",
        candidate_topics=[],
        clarification_question="Did you mean X?",
    )

    response = answer_query("Ambiguous question")

    assert response.status == "clarify"
    assert response.clarification_question == "Did you mean X?"
    assert response.answer == ""

    # Should short-circuit before scan and generation
    mock_scan.assert_not_called()
    mock_gen.assert_not_called()


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
@patch("memory.service._generate_answer")
@patch("memory.service.scan_threads")
def test_answer_query_allowed_channels_passed(
    mock_scan, mock_gen, mock_evid, mock_exec, mock_plan
):
    from memory.models import QueryPlan

    plan = QueryPlan(goal="answer")
    mock_plan.return_value = plan
    mock_exec.return_value = [{"thread_id": "123"}]

    from memory.evidence import EvidenceResult

    mock_evid.return_value = EvidenceResult(
        confidence=0.9, strong_enough=True, reason="high_relevance"
    )

    from memory.privacy import PrivacyScan

    mock_scan.return_value = PrivacyScan(
        route="local",
        redacted_query="query",
        findings={},
        total_pii_count=0,
        high_sensitivity_found=False,
    )
    mock_gen.return_value = ("Answer", 0.5)

    answer_query("test", allowed_channel_ids=["C123"])

    mock_exec.assert_called_once()
    kwargs = mock_exec.call_args.kwargs
    assert kwargs.get("allowed_channel_ids") == ["C123"]


@patch("memory.decomposition.decompose_query")
@patch("memory.decision.execute_semantic_search")
@patch("memory.decision._fetch_full_threads")
def test_summarize_triggers_decomposition(mock_fetch, mock_search, mock_decomp):
    from memory.decision import execute_plan
    from memory.models import AnswerRequirements, QueryPlan, RetrievalStep

    plan = QueryPlan(
        goal="summarize",
        retrieval_steps=[RetrievalStep(tool="semantic_search")],
        answer_requirements=AnswerRequirements(format="summary"),
    )

    mock_decomp.return_value = {"sub_queries": ["sub1", "sub2"]}

    mock_search.return_value = [
        {
            "id": "doc1",
            "document": "test doc",
            "metadata": {"thread_id": "t1", "ts": "10", "channel_id": "C1"},
            "distance": 0.2,
            "embedding": None,
        },
    ]
    mock_fetch.return_value = [{"thread_id": "t1"}]

    threads = execute_plan(plan, "broad query", timings={}, allowed_channel_ids=["C1"])

    # Verify decompose was called
    mock_decomp.assert_called_once_with("broad query")

    # Verify search was called for each sub-query
    assert mock_search.call_count == 2
    mock_search.assert_any_call("sub1", {}, limit=40, allowed_channel_ids=["C1"])
    mock_search.assert_any_call("sub2", {}, limit=40, allowed_channel_ids=["C1"])


@patch("memory.service._local_plan_query")
@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
def test_answer_query_no_cloud_uses_local_planner(
    mock_evid, mock_exec, mock_plan, mock_local_plan
):
    from memory.evidence import EvidenceResult
    from memory.models import QueryPlan

    mock_local_plan.return_value = QueryPlan(goal="answer")
    mock_exec.return_value = []
    mock_evid.return_value = EvidenceResult(
        confidence=0.0,
        strong_enough=False,
        reason="no_threads",
        clarification_question="Need more detail",
    )

    response = answer_query("redis issue", no_cloud=True)

    assert response.status == "clarify"
    mock_local_plan.assert_called_once_with("redis issue")
    mock_plan.assert_not_called()


@patch("memory.service.scan_text")
@patch("memory.service._local_plan_query")
@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
def test_sensitive_query_uses_local_planner(
    mock_evid, mock_exec, mock_plan, mock_local_plan, mock_scan_text
):
    from memory.evidence import EvidenceResult
    from memory.models import QueryPlan

    mock_scan_text.return_value = PrivacyScan(
        route="local",
        redacted_query="[API_KEY]",
        findings={"api_key": ["secret"]},
        total_pii_count=1,
        high_sensitivity_found=True,
    )
    mock_local_plan.return_value = QueryPlan(goal="answer")
    mock_exec.return_value = []
    mock_evid.return_value = EvidenceResult(
        confidence=0.0,
        strong_enough=False,
        reason="no_threads",
        clarification_question="Need more detail",
    )

    response = answer_query("rotate sk-live_abc123xyz456", no_cloud=False)

    assert response.status == "clarify"
    mock_local_plan.assert_called_once_with("rotate sk-live_abc123xyz456")
    mock_plan.assert_not_called()


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
def test_low_sensitivity_query_is_redacted_before_cloud_planner(
    mock_evid, mock_exec, mock_plan
):
    from memory.evidence import EvidenceResult
    from memory.models import QueryPlan

    mock_plan.return_value = QueryPlan(goal="answer")
    mock_exec.return_value = []
    mock_evid.return_value = EvidenceResult(
        confidence=0.0,
        strong_enough=False,
        reason="no_threads",
        clarification_question="Need more detail",
    )

    answer_query("email alice@example.com about deploys")

    mock_plan.assert_called_once_with("email [EMAIL] about deploys")


@patch("memory.decomposition.decompose_query")
@patch("memory.decision.execute_semantic_search")
def test_execute_plan_can_disable_decomposition(mock_search, mock_decompose):
    from memory.decision import execute_plan
    from memory.models import AnswerRequirements, QueryPlan, RetrievalStep

    plan = QueryPlan(
        goal="summarize",
        retrieval_steps=[RetrievalStep(tool="semantic_search")],
        answer_requirements=AnswerRequirements(format="summary"),
    )
    mock_search.return_value = []

    execute_plan(plan, "broad query", timings={}, allow_query_decomposition=False)

    mock_decompose.assert_not_called()
    mock_search.assert_called_once_with(
        "broad query", {}, limit=40, allowed_channel_ids=None
    )


def test_access_control_falls_back_to_current_channel():
    from unittest.mock import MagicMock

    from memory.slack_access import _get_allowed_channels

    mock_client = MagicMock()
    mock_client.users_conversations.side_effect = Exception("missing scope")

    # User ID U123, Channel ID C456
    allowed = _get_allowed_channels(mock_client, "U123", "C456")

    assert allowed == ["C456"]


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
@patch("memory.service.scan_threads")
@patch("memory.service._generate_answer")
def test_cloud_route_returns_original_threads_and_citations(
    mock_gen, mock_scan, mock_evid, mock_exec, mock_plan
):
    from memory.evidence import EvidenceResult
    from memory.models import QueryPlan
    from memory.privacy import PrivacyScan

    mock_plan.return_value = QueryPlan(goal="answer")
    
    original_threads = [{
        "thread_id": "123",
        "messages": [{
            "document": "Contact alice@corp.com",
            "metadata": {"author": "alice", "ts": "123456.789", "email": "alice@corp.com"}
        }]
    }]
    mock_exec.return_value = original_threads

    mock_evid.return_value = EvidenceResult(
        confidence=0.9, strong_enough=True, reason="high_relevance"
    )

    mock_scan.return_value = PrivacyScan(
        route="cloud",
        redacted_query="Contact [EMAIL]",
        findings={"email": ["alice@corp.com"]},
        total_pii_count=1,
        high_sensitivity_found=False,
    )

    mock_gen.return_value = ("Generated answer", 0.5)

    response = answer_query("Contact alice@corp.com")

    # 1. Assert response.threads contains original PII
    assert response.threads[0]["messages"][0]["document"] == "Contact alice@corp.com"
    
    # 2. Assert citations contain original PII
    assert len(response.citations) > 0
    assert "alice@corp.com" in response.citations[0].snippet
    assert "[EMAIL]" not in response.citations[0].snippet

    # 3. Assert _generate_answer was called with REDACTED threads
    gen_call_threads = mock_gen.call_args[0][1]
    assert gen_call_threads[0]["messages"][0]["document"] == "Contact [EMAIL]"


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service.evaluate_evidence")
@patch("memory.service.scan_threads")
@patch("memory.service._generate_answer")
def test_broadened_catchup_uses_correct_scope_label(
    mock_gen, mock_scan, mock_evid, mock_exec, mock_plan
):
    from memory.evidence import EvidenceResult
    from memory.models import QueryPlan
    from memory.privacy import PrivacyScan

    mock_plan.return_value = QueryPlan(goal="catch_up")

    def fake_execute(plan, query, timings, *args, **kwargs):
        timings["_broadened"] = "2026-06-17"
        return [{"thread_id": "123", "messages": [{"document": "test", "metadata": {"ts": "123", "author": "a"}}]}]

    mock_exec.side_effect = fake_execute

    mock_evid.return_value = EvidenceResult(
        confidence=0.9, strong_enough=True, reason="high_relevance"
    )

    mock_scan.return_value = PrivacyScan(
        route="cloud", redacted_query="query", findings={}, total_pii_count=0, high_sensitivity_found=False
    )

    mock_gen.return_value = ("Here is the summary of recent things.", 0.5)

    response = answer_query("catch me up this week")

    assert "since 2026-06-17" in response.answer
    assert "for today" not in response.answer
