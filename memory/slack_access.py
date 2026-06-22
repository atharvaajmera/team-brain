import time
import logging

logger = logging.getLogger("slack_access")

_user_channel_cache = {}

def _get_allowed_channels(client, user_id, current_channel_id):
    """Fetch and cache the list of channels a user has access to."""
    now = time.time()
    cached, ts = _user_channel_cache.get(user_id, (None, 0))
    if cached and now - ts < 300:
        return cached

    allowed = []
    try:
        cursor = None
        while True:
            response = client.users_conversations(
                user=user_id,
                exclude_archived=True,
                types="public_channel,private_channel",
                cursor=cursor
            )
            for channel in response.get("channels", []):
                allowed.append(channel["id"])
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        _user_channel_cache[user_id] = (allowed, now)
        return allowed
    except Exception as e:
        logger.warning(f"Could not fetch allowed channels for {user_id} (missing scope?): {e}")
        return [current_channel_id]  # Fail closed: fallback to current channel only
