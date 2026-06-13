"""Utilities for formatting timestamps and generating Slack permalinks."""

from datetime import datetime, timezone
from memory.settings import settings


def ts_to_readable(ts) -> str:
    """Convert a Slack timestamp (float/str) to a human-readable datetime string.
    
    Example: 1777800606.830719 -> '2026-05-03 14:30'
    """
    try:
        ts_float = float(ts)
        dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return str(ts)


def make_permalink(channel_id: str, ts) -> str:
    """Build a Slack permalink URL from channel ID and timestamp.
    
    Slack permalink format:
        https://{workspace}.slack.com/archives/{channel_id}/p{timestamp_without_dot}
    
    Example:
        channel_id = 'C0B15BQ6N5R', ts = 1777800606.830719
        -> 'https://myteam.slack.com/archives/C0B15BQ6N5R/p1777800606830719'
    
    Returns empty string if workspace is not configured.
    """
    workspace = settings.SLACK_WORKSPACE
    if not workspace:
        return ""
    
    try:
        # Slack permalink uses ts without the dot
        ts_str = str(ts).replace(".", "")
        return f"https://{workspace}.slack.com/archives/{channel_id}/p{ts_str}"
    except Exception:
        return ""


def format_citation(author: str, ts, channel_id: str = "", thread_id=None) -> str:
    """Build a compact citation string for a message.
    
    Examples:
        With permalink:  '@alice, 2026-05-03 14:30 (https://team.slack.com/archives/C.../p...)'
        Without:         '@alice, 2026-05-03 14:30'
    """
    readable_ts = ts_to_readable(ts)
    permalink = make_permalink(channel_id, ts) if channel_id else ""
    
    citation = f"@{author}, {readable_ts}"
    if permalink:
        citation += f" ({permalink})"
    
    return citation
