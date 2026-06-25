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
    redacted_query: str = ""
    redactor: "Redactor" = None


class Redactor:
    def __init__(self):
        self.mapping = {}
        self.counters = {}

    def redact(self, text: str) -> str:
        if not text:
            return text
        redacted = text
        for pii_type, pattern in _PII_PATTERNS.items():
            def repl(m):
                original = m.group(0)
                for token, val in self.mapping.items():
                    if val == original:
                        return token
                count = self.counters.get(pii_type, 0) + 1
                self.counters[pii_type] = count
                token = f"[{pii_type.upper()}_{count}]"
                self.mapping[token] = original
                return token
            redacted = pattern.sub(repl, redacted)
        return redacted

    def unredact(self, text: str) -> str:
        if not text:
            return text
        unredacted = text
        # Replace numbered tokens first: [EMAIL_1] -> security@acme.com
        for token, original in self.mapping.items():
            unredacted = unredacted.replace(token, original)
        
        # Fallback: LLMs sometimes drop the _N suffix and write [EMAIL] instead of [EMAIL_1].
        # Build a map from bare type -> first original value for that type.
        bare_fallbacks = {}
        for token, original in self.mapping.items():
            # Extract the type name: [EMAIL_1] -> EMAIL
            bare = re.sub(r"_\d+\]$", "]", token)
            if bare not in bare_fallbacks:
                bare_fallbacks[bare] = original
        for bare_token, original in bare_fallbacks.items():
            unredacted = unredacted.replace(bare_token, original)
        
        return unredacted


def scan_text(text: str, redactor: Redactor = None) -> PrivacyScan:
    result = PrivacyScan()
    redactor = redactor or Redactor()
    result.redactor = redactor
    
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
        result.redacted_text = redactor.redact(text)
        
    return result


def redact(text: str) -> str:
    # Backwards compatibility if needed, though we should use Redactor
    redactor = Redactor()
    return redactor.redact(text)


def scan_threads(query: str, threads: list) -> PrivacyScan:
    parts = [query]
    for thread in threads:
        for msg in thread.get("messages", []):
            text = msg.get("document", "")
            author = msg.get("metadata", {}).get("author", "")
            parts.append(f"{author}: {text}")
            
    redactor = Redactor()
    scan_result = scan_text("\n".join(parts), redactor=redactor)
    scan_result.redacted_query = redactor.redact(query)
    return scan_result


def redact_threads(threads: list, redactor: Redactor = None) -> list:
    redactor = redactor or Redactor()
    redacted = copy.deepcopy(threads)
    for thread in redacted:
        for msg in thread.get("messages", []):
            if "document" in msg:
                msg["document"] = redactor.redact(msg["document"])
            meta = msg.get("metadata", {})
            if "text" in meta:
                meta["text"] = redactor.redact(meta["text"])
            if "author" in meta:
                meta["author"] = redactor.redact(meta["author"])
    return redacted
