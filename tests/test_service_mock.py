import pytest
from unittest.mock import patch, MagicMock

from memory.models import QueryResponse
from memory.service import answer_query


@patch("memory.service.plan_query")
@patch("memory.service.execute_plan")
@patch("memory.service._generate_answer")
@patch("memory.service.scan_threads")
def test_answer_query_reject(mock_scan, mock_gen, mock_exec, mock_plan):
    """Test that answer_query correctly handles a 'reject' plan without executing retrieval."""
    
    # Mock planner returning a reject plan
    from memory.models import QueryPlan, AnswerRequirements
    plan = QueryPlan(
        goal="reject",
        answer_requirements=AnswerRequirements(format="direct", cite_sources=False)
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
        clarification_question="Did you mean X?"
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
def test_answer_query_allowed_channels_passed(mock_scan, mock_gen, mock_evid, mock_exec, mock_plan):
    from memory.models import QueryPlan
    plan = QueryPlan(goal="answer")
    mock_plan.return_value = plan
    mock_exec.return_value = [{"thread_id": "123"}]
    
    from memory.evidence import EvidenceResult
    mock_evid.return_value = EvidenceResult(confidence=0.9, strong_enough=True, reason="high_relevance")
    
    from memory.privacy import PrivacyScan
    mock_scan.return_value = PrivacyScan(route="local", redacted_query="query", findings={}, total_pii_count=0, high_sensitivity_found=False)
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
    from memory.models import QueryPlan, RetrievalStep, AnswerRequirements
    
    plan = QueryPlan(
        goal="summarize",
        retrieval_steps=[RetrievalStep(tool="semantic_search")],
        answer_requirements=AnswerRequirements(format="summary")
    )
    
    mock_decomp.return_value = {
        "sub_queries": ["sub1", "sub2"]
    }
    
    mock_search.return_value = [
        {"id": "doc1", "document": "test doc", "metadata": {"thread_id": "t1", "ts": "10", "channel_id": "C1"}, "distance": 0.2, "embedding": None},
    ]
    mock_fetch.return_value = [{"thread_id": "t1"}]
    
    threads = execute_plan(plan, "broad query", timings={}, allowed_channel_ids=["C1"])
    
    # Verify decompose was called
    mock_decomp.assert_called_once_with("broad query")
    
    # Verify search was called for each sub-query
    assert mock_search.call_count == 2
    mock_search.assert_any_call("sub1", {}, limit=40, allowed_channel_ids=["C1"])
    mock_search.assert_any_call("sub2", {}, limit=40, allowed_channel_ids=["C1"])

