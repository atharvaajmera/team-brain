"""Evidence quality scoring for retrieved threads.

Evaluates whether the retrieval results are strong enough to generate a
confident answer, or whether the system should ask a clarifying question.

Scoring signals:
  1. Best thread distance (from ChromaDB cosine distance)
  2. Gap between top and second thread (dominant vs. flat distribution)
  3. Query-term overlap with top thread content
  4. Thread count relative to goal requirements
  5. Format-specific markers (e.g. decision markers for format="decision")
  6. Recency for catch_up goals
"""

import time
import logging
from dataclasses import dataclass, field

from memory.models import QueryPlan, EvidenceReason
from memory.shared import tokenize

logger = logging.getLogger("evidence")

# ── Thresholds (tunable) ──

# ChromaDB cosine distance: 0 = identical, 2 = opposite
# Typical good match: 0.3–0.6, weak match: 0.8+, unrelated: 1.2+
DISTANCE_STRONG = 0.55       # best distance below this = strong signal
DISTANCE_WEAK = 0.90         # best distance above this = weak signal
GAP_DOMINANT = 0.15          # gap > this means top thread is clearly dominant
OVERLAP_STRONG = 0.40        # query-token overlap > this = strong signal
OVERLAP_WEAK = 0.15          # query-token overlap < this = weak signal
CATCH_UP_STALENESS_HOURS = 72  # threads older than this are "stale" for catch_up

# Per-goal minimum thread counts
MIN_THREADS = {
    "summarize": 2,
    "catch_up": 1,
    "answer": 1,
    "analysis": 2,
}

# Decision-format marker words
DECISION_MARKERS = frozenset({
    "decided", "agreed", "decision", "conclusion", "resolved",
    "verdict", "going with", "we'll go", "let's go with", "final call",
    "approved", "shipped", "merged", "consensus",
})


@dataclass
class EvidenceResult:
    confidence: float          # 0.0–1.0
    strong_enough: bool
    reason: EvidenceReason
    candidate_topics: list[str] = field(default_factory=list)
    clarification_question: str | None = None


def _extract_thread_text(thread: dict) -> str:
    """Concatenate all message documents in a thread into a single string."""
    return " ".join(
        m.get("document", "") for m in thread.get("messages", [])
    )


def _compute_overlap(query: str, thread_text: str) -> float:
    """Fraction of query tokens found in the thread text."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    thread_tokens = set(tokenize(thread_text))
    return len(query_tokens & thread_tokens) / len(query_tokens)


def _has_decision_markers(thread_text: str) -> bool:
    """Check if the thread contains any decision-indicating language."""
    text_lower = thread_text.lower()
    return any(marker in text_lower for marker in DECISION_MARKERS)


def _newest_ts(threads: list[dict]) -> float:
    """Get the most recent timestamp across all threads."""
    newest = 0.0
    for thread in threads:
        for msg in thread.get("messages", []):
            ts = float(msg.get("metadata", {}).get("ts", 0))
            if ts > newest:
                newest = ts
    return newest


def _build_clarification(reason: str, query: str, **kwargs) -> str:
    """Generate a user-facing clarification question based on the failure reason."""
    if reason == "no_threads":
        return (
            f"I couldn't find any discussions about '{query}'. "
            "Could you try different keywords or check if this topic was discussed in Slack?"
        )
    elif reason == "too_few_threads":
        return (
            f"I only found a single brief mention related to '{query}'. "
            "Could you be more specific about what aspect you're looking for?"
        )
    elif reason == "low_overlap":
        return (
            f"I found some threads, but they don't seem closely related to '{query}'. "
            "Could you rephrase or narrow your question?"
        )
    elif reason == "no_decision_markers":
        return (
            f"I found discussions related to '{query}', but no clear decision or conclusion. "
            "Are you asking about the discussion itself, or was there a specific decision you expected?"
        )
    elif reason == "stale_threads":
        hours = kwargs.get("staleness_hours", CATCH_UP_STALENESS_HOURS)
        return (
            f"The most recent threads I found about '{query}' are over {int(hours)} hours old. "
            "There may not be recent activity on this topic. Want a historical summary instead?"
        )
    elif reason == "weak_distance":
        return (
            f"I found some results, but none are a strong match for '{query}'. "
            "Could you try being more specific?"
        )
    else:
        return (
            f"I'm not confident I have enough information to answer '{query}' well. "
            "Could you provide more context?"
        )


def evaluate_evidence(plan: QueryPlan, threads: list, query: str) -> EvidenceResult:
    """Score evidence quality and decide if clarification is needed.
    
    Each thread should have a `_retrieval_score` dict attached by decision.py
    containing: thread_score, min_distance, avg_distance, message_count.
    """
    
    # ── Signal 0: No threads at all ──
    if not threads:
        return EvidenceResult(
            confidence=0.0,
            strong_enough=False,
            reason="no_threads",
            clarification_question=_build_clarification("no_threads", query)
        )
    
    goal = plan.goal
    answer_format = plan.answer_requirements.format
    num_threads = len(threads)
    
    # ── Extract retrieval scores ──
    scores = [t.get("_retrieval_score", {}) for t in threads]
    best_distance = scores[0].get("min_distance", 1.0) if scores else 1.0
    
    if len(scores) >= 2:
        second_distance = scores[1].get("min_distance", 1.0)
        gap = second_distance - best_distance
    else:
        gap = 0.0
    
    # ── Signal 1: Thread count vs. goal requirement ──
    min_required = MIN_THREADS.get(goal, 1)
    if num_threads < min_required:
        return EvidenceResult(
            confidence=0.25,
            strong_enough=False,
            reason="too_few_threads",
            clarification_question=_build_clarification("too_few_threads", query)
        )
    
    # ── Signal 2: Query-term overlap with top thread ──
    top_text = _extract_thread_text(threads[0])
    overlap = _compute_overlap(query, top_text)
    
    # ── Signal 3: Format-specific checks ──
    if answer_format == "decision":
        # For decision format, at least one thread must contain decision markers
        any_has_decision = any(
            _has_decision_markers(_extract_thread_text(t)) for t in threads
        )
        if not any_has_decision:
            return EvidenceResult(
                confidence=0.3,
                strong_enough=False,
                reason="no_decision_markers",
                clarification_question=_build_clarification("no_decision_markers", query)
            )
    
    # ── Signal 4: Recency for catch_up ──
    if goal == "catch_up":
        newest = _newest_ts(threads)
        if newest > 0:
            age_hours = (time.time() - newest) / 3600
            if age_hours > CATCH_UP_STALENESS_HOURS:
                return EvidenceResult(
                    confidence=0.3,
                    strong_enough=False,
                    reason="stale_threads",
                    clarification_question=_build_clarification(
                        "stale_threads", query, staleness_hours=age_hours
                    )
                )
    
    # ── Composite confidence score ──
    # Distance component (0-0.4): lower distance = higher confidence
    if best_distance <= DISTANCE_STRONG:
        distance_score = 0.4
    elif best_distance >= DISTANCE_WEAK:
        distance_score = 0.05
    else:
        # Linear interpolation between strong and weak
        ratio = (DISTANCE_WEAK - best_distance) / (DISTANCE_WEAK - DISTANCE_STRONG)
        distance_score = 0.05 + ratio * 0.35
    
    # Overlap component (0-0.3): higher overlap = higher confidence
    overlap_score = min(overlap, 1.0) * 0.3
    
    # Gap component (0-0.15): larger gap = more confidence in top result
    gap_score = min(gap / 0.3, 1.0) * 0.15 if gap > 0 else 0.0
    
    # Thread count component (0-0.15): more supporting threads = higher confidence
    count_score = min(num_threads / 4.0, 1.0) * 0.15
    
    confidence = distance_score + overlap_score + gap_score + count_score
    confidence = max(0.0, min(1.0, confidence))
    
    # ── Final gate ──
    # Require at least moderate confidence to answer
    if best_distance >= DISTANCE_WEAK and overlap < OVERLAP_WEAK:
        return EvidenceResult(
            confidence=confidence,
            strong_enough=False,
            reason="low_overlap",
            clarification_question=_build_clarification("low_overlap", query)
        )
    
    if best_distance >= DISTANCE_WEAK:
        return EvidenceResult(
            confidence=confidence,
            strong_enough=False,
            reason="weak_distance",
            clarification_question=_build_clarification("weak_distance", query)
        )
    
    # Determine reason string for logging
    if best_distance <= DISTANCE_STRONG and overlap >= OVERLAP_STRONG:
        reason = "high_relevance"
    elif best_distance <= DISTANCE_STRONG:
        reason = "good_distance"
    elif overlap >= OVERLAP_STRONG:
        reason = "good_overlap"
    else:
        reason = "moderate_match"
    
    logger.debug(
        f"Evidence: confidence={confidence:.2f}, reason={reason}, "
        f"best_dist={best_distance:.3f}, overlap={overlap:.2f}, "
        f"gap={gap:.3f}, threads={num_threads}"
    )
    
    return EvidenceResult(
        confidence=confidence,
        strong_enough=True,
        reason=reason,
        clarification_question=None
    )
