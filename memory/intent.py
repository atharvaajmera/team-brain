from dateparser.search import search_dates
from datetime import datetime, timedelta

aggregation=["issues","bugs","tasks","features","improvements","questions","discussions","announcements","updates"]
aggregation.sort(key=lambda x: -len(x))

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
        matched_text, parsed_date = date_results[0]
        
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