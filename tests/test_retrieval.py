import pytest
from memory.retrieval import build_chroma_filter, _date_to_ts

def test_build_chroma_filter_no_filters():
    assert build_chroma_filter(None, None) is None
    assert build_chroma_filter({}, []) is None

def test_build_chroma_filter_author_normalized():
    # "Alice Smith" -> {"author": {"$eq": "alice_smith"}}
    filter_dict = build_chroma_filter({"author": "Alice Smith"})
    assert filter_dict == {"author": {"$eq": "alice_smith"}}

def test_build_chroma_filter_single_channel():
    filter_dict = build_chroma_filter(None, allowed_channel_ids=["C123"])
    assert filter_dict == {"channel_id": {"$eq": "C123"}}

def test_build_chroma_filter_multi_channel():
    filter_dict = build_chroma_filter(None, allowed_channel_ids=["C123", "C456"])
    assert filter_dict == {"channel_id": {"$in": ["C123", "C456"]}}

def test_build_chroma_filter_combined():
    filter_dict = build_chroma_filter(
        {"author": "bob", "after": "2026-05-01"}, 
        allowed_channel_ids=["C123"]
    )
    
    assert "$and" in filter_dict
    clauses = filter_dict["$and"]
    assert len(clauses) == 3
    assert {"author": {"$eq": "bob"}} in clauses
    assert {"channel_id": {"$eq": "C123"}} in clauses
    
    ts_clause = next((c for c in clauses if "ts" in c and "$gte" in c["ts"]), None)
    assert ts_clause is not None
