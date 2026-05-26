from dateparser.search import search_dates
from datetime import datetime, timedelta
import re

aggregation=["issues","bugs","tasks","features","improvements","questions","discussions","announcements","updates"]
aggregation.sort(key=lambda x: -len(x))

_TEMPORAL_KEYWORDS = {
    "today",
    "yesterday",
    "tomorrow",
    "recent",
    "recently",
    "latest",
    "currently",
    "current",
    "now",
    "week",
    "month",
    "year",
    "quarter",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
}

_TEMPORAL_PHRASES = (
    "all time",
    "this week",
    "this month",
    "this year",
    "last week",
    "last month",
    "last year",
    "past week",
    "past month",
    "past year",
    "next week",
    "next month",
    "next year",
)


def _looks_like_temporal_match(query, matched_text):
    text = (matched_text or "").strip().lower()
    if not text:
        return False

    # Reject the noisy one-word matches that caused filters like "we", "me", "to", "on".
    if len(text) < 4 and not any(ch.isdigit() for ch in text):
        return False

    if text in _TEMPORAL_KEYWORDS:
        return True

    if any(phrase in query.lower() and phrase in text for phrase in _TEMPORAL_PHRASES):
        return True

    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    if not tokens:
        return False

    if any(token.isdigit() for token in tokens):
        return True

    if any(token in _TEMPORAL_KEYWORDS for token in tokens):
        return True

    # Accept ordinal / dated patterns like "may 5", "5 may", "2025-05-01".
    if re.search(r"\b\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?\b", text):
        return True

    return False

def extract_temporal_filter(query):
    """
    Extracts temporal information from query using dateparser.
    Returns Unix timestamp of the earliest date found, or None.
    """
    # Use dateparser to find date expressions in the query
    date_results = search_dates(
        query,
        settings={
            'PREFER_DATES_FROM': 'past',
            'RELATIVE_BASE': datetime.now()
        }
    )
    
    if date_results:
        # Get the first date match (dateparser returns list of tuples: (matched_text, datetime_obj))
        matched_text = None
        parsed_date = None
        for candidate_text, candidate_date in date_results:
            if _looks_like_temporal_match(query, candidate_text):
                matched_text = candidate_text
                parsed_date = candidate_date
                break

        if matched_text is None or parsed_date is None:
            return None, None
        
        # Handle special cases for relative dates to get start of period
        query_lower = query.lower()
        
        # Handle "this week" - get start of current week
        if "this week" in query_lower:
            parsed_date = parsed_date - timedelta(days=parsed_date.weekday())
            parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # Handle "this month" - get start of current month
        elif "this month" in query_lower:
            parsed_date = parsed_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Handle "this year" - get start of current year
        elif "this year" in query_lower:
            parsed_date = parsed_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        # Handle "today" - get start of day
        elif "today" in query_lower or "recent" in query_lower:
            parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return int(parsed_date.timestamp()), matched_text
    
    # Handle "all time" explicitly
    if "all time" in query.lower():
        return 0, "all time"
    
    return None, None

def analyze_query_intent(query):
    """
    Analyzes user query to extract temporal filters and aggregation intent.
    Uses dateparser for flexible date/time parsing.
    """
    intent = {"timeline": None, "aggregation": None, "filter_timeline": None}
    
    # Extract temporal filter using dateparser
    timestamp, matched_text = extract_temporal_filter(query)
    if timestamp is not None:
        intent["timeline"] = matched_text
        intent["filter_timeline"] = timestamp

    # Detect aggregation type
    user_query = query.lower()
    for agg in aggregation:
        if agg in user_query:
            intent["aggregation"] = agg
            break

    return intent
