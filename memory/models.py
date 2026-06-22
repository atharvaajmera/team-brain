from pydantic import BaseModel, Field, field_validator
from typing import Literal

# Plan types
class FilterSpec(BaseModel):
    author: str | None = None
    after: str | None = None    # YYYY-MM-DD
    before: str | None = None   # YYYY-MM-DD

class RetrievalStep(BaseModel):
    tool: Literal["semantic_search", "recent_threads", "author_search"]
    query: str | None = None
    filters: FilterSpec = Field(default_factory=FilterSpec)
    limit: int = 40

    @field_validator("limit")
    @classmethod
    def clamp_limit(cls, v):
        return min(max(v, 1), 100)

class AnswerRequirements(BaseModel):
    format: Literal["direct", "summary", "timeline", "comparison", "decision"] = "direct"
    cite_sources: bool = True

class QueryPlan(BaseModel):
    goal: Literal["answer", "catch_up", "summarize", "analysis", "clarify", "reject"]
    focus: Literal["topic", "person", "decision", "timeline"] | None = None
    time_scope: str | None = None
    retrieval_steps: list[RetrievalStep] = Field(default_factory=list)
    answer_requirements: AnswerRequirements = Field(default_factory=AnswerRequirements)

# Response types
class Citation(BaseModel):
    author: str
    ts: float | str
    readable_ts: str
    channel_id: str | None = None
    permalink: str | None = None
    snippet: str
    thread_id: str | float

class PrivacyScanDiagnostic(BaseModel):
    pii_count: int
    high_sensitivity: bool
    findings: list[str]

EvidenceReason = Literal[
    "no_threads", 
    "too_few_threads", 
    "low_overlap", 
    "no_decision_markers", 
    "stale_threads", 
    "weak_distance", 
    "high_relevance", 
    "good_distance", 
    "good_overlap", 
    "moderate_match",
    "uncertain"
]

class EvidenceDiagnostic(BaseModel):
    confidence: float
    reason: EvidenceReason

class Diagnostics(BaseModel):
    scan: PrivacyScanDiagnostic
    evidence: EvidenceDiagnostic

class QueryResponse(BaseModel):
    status: Literal["ok", "reject", "clarify", "error"] = "ok"
    goal: Literal["answer", "catch_up", "summarize", "analysis", "clarify", "reject"]
    route: Literal["local", "cloud"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    threads: list[dict] = Field(default_factory=list)
    plan: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    clarification_question: str | None = None
    debug: Diagnostics | None = None
