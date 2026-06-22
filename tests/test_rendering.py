import pytest
import sys
from pathlib import Path

# Ensure root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.models import QueryResponse, Citation
from ask import _render_result
from slack_bot import _format_slack_response

def _mock_thread(tid="123", text="Hello", author="alice", email="alice@example.com"):
    return {
        "thread_id": tid,
        "messages": [
            {
                "document": text,
                "metadata": {
                    "author": author,
                    "author_display": author,
                    "ts": "1700000000.0001",
                    "text": text,
                    "email": email
                }
            }
        ]
    }

def test_cli_ok_response():
    resp = QueryResponse(
        status="ok",
        goal="answer",
        route="cloud",
        answer="This is the generated answer.",
        citations=[
            Citation(
                author="bob",
                ts="1700000000.0001",
                readable_ts="2023-11-14 12:00:00",
                snippet="Some context text",
                thread_id="123",
                permalink="https://slack.com/archives/C1/p1700000000000100"
            )
        ],
        threads=[_mock_thread(text="Some context text")]
    )
    output = _render_result(resp)
    
    assert "Goal: ANSWER" in output
    assert "Route: CLOUD" in output
    assert "Top Threads:" in output
    assert "Summary:" in output
    assert "This is the generated answer." in output
    assert "Evidence:" in output
    assert "@bob [2023-11-14 12:00:00] (https://slack.com/archives/C1/p1700000000000100): Some context text" in output

def test_cli_clarify_response():
    resp = QueryResponse(
        status="clarify",
        goal="summarize",
        route="local",
        answer="",
        clarification_question="What specific part do you want?",
        threads=[_mock_thread()]
    )
    output = _render_result(resp)
    
    assert "Clarification Needed:" in output
    assert "What specific part do you want?" in output
    assert "Top Threads (for context):" in output
    assert "Summary:" not in output
    assert "Evidence:" not in output

def test_cli_reject_response():
    resp = QueryResponse(
        status="reject",
        goal="reject",
        route="local",
        answer="I could not find relevant discussions."
    )
    output = _render_result(resp)
    
    assert "Goal: REJECT" in output
    assert "Summary:" in output
    assert "I could not find relevant discussions." in output
    assert "Top Threads:" not in output

def test_slack_ok_response():
    resp = QueryResponse(
        status="ok",
        goal="answer",
        route="cloud",
        answer="Slack generated answer.",
        citations=[
            Citation(
                author="bob",
                ts="1700000000.0001",
                readable_ts="2023-11-14",
                snippet="Test snippet",
                thread_id="123",
                permalink="https://slack.com/archives/123"
            )
        ]
    )
    output = _format_slack_response(resp)
    
    assert output.startswith("Slack generated answer.")
    assert "*Sources:*" in output
    assert "[1] bob at <https://slack.com/archives/123|2023-11-14>" in output

def test_slack_clarify_response():
    resp = QueryResponse(
        status="clarify",
        goal="answer",
        route="local",
        answer="",
        clarification_question="Did you mean X or Y?"
    )
    output = _format_slack_response(resp)
    
    assert output.startswith("❓ Did you mean X or Y?")

def test_slack_reject_response():
    resp = QueryResponse(
        status="reject",
        goal="reject",
        route="local",
        answer="No relevant threads."
    )
    output = _format_slack_response(resp)
    
    assert output.startswith("❌ No relevant threads.")

def test_citations_show_original_not_redacted():
    # If the issue 1 fix works, service.py passes original threads to _build_citations
    # So we construct original threads with PII, build citations, and ensure PII is intact
    from memory.service import _build_citations
    
    threads = [
        _mock_thread(text="Contact me at secret@corp.com please.")
    ]
    
    # We pretend this was routed cloud, but _build_citations should receive unredacted threads
    citations = _build_citations(threads)
    
    assert len(citations) == 1
    assert "secret@corp.com" in citations[0].snippet
    assert "[EMAIL]" not in citations[0].snippet
