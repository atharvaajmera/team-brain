import pytest
from unittest.mock import MagicMock, patch
from ingest import _get_author, get_user_map

def test_get_author_generator_metadata():
    msg = {
        "user": "U123",
        "metadata": {
            "event_payload": {
                "author": "System Bot"
            }
        }
    }
    author_tuple = _get_author(msg)
    # Expected (norm_name, display_name, user_id)
    assert author_tuple == ("system_bot", "System Bot", "U123")

def test_get_author_from_user_map():
    msg = {"user": "U123"}
    user_map = {
        "U123": {"norm": "alice", "display": "Alice"}
    }
    author_tuple = _get_author(msg, user_map)
    assert author_tuple == ("alice", "Alice", "U123")

def test_get_author_fallback():
    msg = {"user": "U999"}
    user_map = {
        "U123": {"norm": "alice", "display": "Alice"}
    }
    author_tuple = _get_author(msg, user_map)
    assert author_tuple == ("U999", "U999", "U999")

def test_get_author_missing_user():
    msg = {}
    author_tuple = _get_author(msg)
    assert author_tuple == ("unknown", "Unknown", "")

@patch("ingest.client")
def test_get_user_map_graceful_missing_scope(mock_client):
    from slack_sdk.errors import SlackApiError
    
    response = MagicMock()
    response.status_code = 200
    
    mock_client.users_list.side_effect = SlackApiError("missing_scope", response)
    
    user_map = get_user_map()
    assert user_map == {}

@patch("ingest.client")
def test_get_user_map_success(mock_client):
    mock_client.users_list.return_value = {
        "members": [
            {
                "id": "U123",
                "profile": {"real_name": "Alice Smith"}
            },
            {
                "id": "U456",
                "name": "bob_jones",
                "profile": {}
            }
        ],
        "response_metadata": {}
    }
    
    user_map = get_user_map()
    assert len(user_map) == 2
    assert user_map["U123"]["display"] == "Alice Smith"
    assert user_map["U123"]["norm"] == "alice_smith"
    assert user_map["U456"]["display"] == "bob_jones"
