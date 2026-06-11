import pytest
from memory.privacy import scan_text, redact, scan_threads

def test_api_key_forces_local():
    scan = scan_text("use key sk-live_abc123xyz456")
    assert scan.route == "local"
    assert scan.high_sensitivity_found

def test_email_allows_cloud_redacted():
    scan = scan_text("email alice@example.com for help")
    assert scan.route == "cloud"
    assert "[EMAIL]" in scan.redacted_text

def test_query_pii_is_redacted():
    scan = scan_threads("rotate api key sk-live_abc123xyz456", [])
    assert scan.route == "local"
    assert "sk-live" not in scan.redacted_query
    assert "[API_KEY]" in scan.redacted_query

def test_connection_string_forces_local():
    scan = scan_text("redis://admin:pass@10.0.0.1:6379/0")
    assert scan.route == "local"
    assert scan.high_sensitivity_found

def test_high_pii_count_forces_local():
    emails = " ".join(f"user{i}@example.com" for i in range(15))
    scan = scan_text(emails)
    assert scan.route == "local"
