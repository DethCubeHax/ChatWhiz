import json
from typing import Iterator


def emit_status(text: str) -> str:
    return json.dumps({"type": "status", "text": text}) + "\n"


def emit_token(text: str) -> str:
    return json.dumps({"type": "token", "text": text}) + "\n"


def emit_done() -> str:
    return json.dumps({"type": "done"}) + "\n"


def stream_events(events: Iterator[str]) -> Iterator[str]:
    for event in events:
        yield event
