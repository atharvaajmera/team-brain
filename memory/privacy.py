"""PII detection, redaction, and cloud/local routing."""

import copy
import re
from dataclasses import dataclass, field

_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key": re.compile(r"(?:xoxb-|xoxp-|xapp-|sk-|sk_live_|pk_live_|Bearer |ghp_|gho_|AKIA|AIza)[A-Za-z0-9\-_]{10,}"),
    "aws_secret": re.compile(r"(?:aws_secret_access_key|AWS_SECRET)\s*[=:]\s*[A-Za-z0-9/+=]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
    "phone": re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "slack_user_id": re.compile(r"\bU[A-Z0-9]{8,12}\b"),
    "internal_url": re.compile(r"https?://(?:localhost|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)(?::\d+)?[/\w\-]*"),
    "connection_string": re.compile(r"(?:postgres|mysql|mongodb|redis)://[^\s]+"),
}

_HIGH_SENSITIVITY = {"api_key", "aws_secret", "private_key", "connection_string", "credit_card"}


@dataclass
class PrivacyScan:
    findings: dict = field(default_factory=dict)
    high_sensitivity_found: bool = False
    total_pii_count: int = 0
    route: str = "local"
    redacted_text: str = ""


def scan_text(text: str) -> PrivacyScan:
    result = PrivacyScan()
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            result.findings[pii_type] = matches
            result.total_pii_count += len(matches)
            if pii_type in _HIGH_SENSITIVITY:
                result.high_sensitivity_found = True
                
    if result.high_sensitivity_found:
        result.route = "local"
    elif result.total_pii_count > 10:
        result.route = "local"
    else:
        result.route = "cloud"
        result.redacted_text = redact(text)
        
    return result


def redact(text: str) -> str:
    redacted = text
    for pii_type, pattern in _PII_PATTERNS.items():
        redacted = pattern.sub(f"[{pii_type.upper()}]", redacted)
    return redacted


def scan_threads(query: str, threads: list) -> PrivacyScan:
    parts = [query]
    for thread in threads:
        for msg in thread.get("messages", []):
            text = msg.get("document", "")
            author = msg.get("metadata", {}).get("author", "")
            parts.append(f"{author}: {text}")
    return scan_text("\n".join(parts))


def redact_threads(threads: list) -> list:
    redacted = copy.deepcopy(threads)
    for thread in redacted:
        for msg in thread.get("messages", []):
            if "document" in msg:
                msg["document"] = redact(msg["document"])
            meta = msg.get("metadata", {})
            if "text" in meta:
                meta["text"] = redact(meta["text"])
            if "author" in meta:
                meta["author"] = redact(meta["author"])
    return redacted
