import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import Request

from .config import MAX_HISTORY, RATE_LIMIT, RATE_WINDOW

ip_to_user_id: Dict[str, str] = {}
user_request_counts: Dict[str, List] = defaultdict(list)
conversation_history: Dict[str, list] = defaultdict(list)
last_cleanup = datetime.now()


def cleanup_expired_data() -> None:
    global last_cleanup
    current_time = datetime.now()

    if current_time - last_cleanup < timedelta(hours=24):
        return

    print("Performing 24-hour cleanup...")
    ip_to_user_id.clear()
    user_request_counts.clear()
    last_cleanup = current_time


def get_user_id(request: Request) -> str:
    ip = request.client.host
    if ip not in ip_to_user_id:
        ip_to_user_id[ip] = str(uuid.uuid4())
    return ip_to_user_id[ip]


def check_rate_limit(user_id: str) -> bool:
    cleanup_expired_data()

    current_time = datetime.now()
    user_request_counts[user_id] = [
        timestamp for timestamp in user_request_counts[user_id]
        if current_time - timestamp < RATE_WINDOW
    ]

    if len(user_request_counts[user_id]) >= RATE_LIMIT:
        return False

    user_request_counts[user_id].append(current_time)
    return True


def get_history(user_id: str) -> list:
    return conversation_history[user_id]


def save_exchange(user_id: str, query_text: str, response_text: str) -> None:
    history = conversation_history[user_id]
    history.append({
        "query": query_text,
        "response": response_text,
        "timestamp": datetime.now().isoformat(),
    })
    conversation_history[user_id] = history[-MAX_HISTORY:]


def clear_history(user_id: str) -> None:
    conversation_history[user_id] = []
