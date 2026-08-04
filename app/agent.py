import json
import re
from typing import Iterator, List

from .config import SOURCE_CATALOG, SOURCE_KEYS, SOURCE_STATUS_LABELS
from .data_manager import data_manager
from .gemini_client import gemini_client
from .prompts import build_answer_prompt, build_router_prompt, format_conversation_history
from .session import save_exchange
from .stream_events import emit_done, emit_status, emit_token


def fallback_sources(query_text: str) -> List[str]:
    query = query_text.lower()
    selected = set()

    project_terms = ("project", "built", "chatwhiz", "portfolio", "app", "github")
    work_terms = ("work", "job", "company", "kodifly", "intern", "experience", "role", "career", "employ")
    research_terms = ("research", "publication", "paper", "conference", "published", "isam")
    resume_terms = ("resume", "cv", "education", "degree", "skill", "graduate", "hku", "background", "about you", "who are you")

    if any(term in query for term in project_terms):
        selected.add("projects")
    if any(term in query for term in work_terms):
        selected.update({"work", "resume"})
    if any(term in query for term in research_terms):
        selected.update({"research", "resume"})
    if any(term in query for term in resume_terms):
        selected.add("resume")

    if not selected:
        selected.update({"resume", "work"})

    return [key for key in SOURCE_KEYS if key in selected]


def parse_router_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    sources = parsed.get("sources", [])
    valid_sources = [source for source in sources if source in SOURCE_CATALOG]
    if not valid_sources:
        raise ValueError("Router returned no valid sources")
    return {
        "sources": valid_sources,
        "reasoning": parsed.get("reasoning", ""),
    }


def select_sources(query_text: str, history) -> dict:
    history_context = format_conversation_history(history)
    prompt = (
        f"{build_router_prompt()}\n\n"
        f"{history_context}\n\n"
        f"User query: {query_text}\n\n"
        "JSON:"
    )

    try:
        raw = gemini_client.generate_text(prompt)
        return parse_router_response(raw)
    except Exception as error:
        print(f"Router fallback triggered: {error}")
        return {
            "sources": fallback_sources(query_text),
            "reasoning": "Using keyword routing fallback.",
        }


def gather_context(selected_sources: List[str], on_status=None) -> dict:
    context = {}

    for source in selected_sources:
        if on_status:
            on_status(SOURCE_STATUS_LABELS[source])
        context[source] = data_manager.fetch_source(source)

    return context


def answer_query(user_id: str, query_text: str) -> str:
    from .session import get_history

    history = get_history(user_id)
    plan = select_sources(query_text, history)
    context = gather_context(plan["sources"])
    prompt = build_answer_prompt(query_text, context, history)
    response_text = gemini_client.generate_text(prompt)
    save_exchange(user_id, query_text, response_text)
    return response_text


def stream_answer(user_id: str, query_text: str) -> Iterator[str]:
    from .session import get_history

    history = get_history(user_id)
    chunks: List[str] = []

    try:
        yield emit_status("Deciding which sources to check...")

        plan = select_sources(query_text, history)
        selected_sources = plan["sources"]

        if plan.get("reasoning"):
            yield emit_status(plan["reasoning"])

        context = {}
        for source in selected_sources:
            yield emit_status(SOURCE_STATUS_LABELS[source])
            context[source] = data_manager.fetch_source(source)

        yield emit_status("Drafting response...")

        prompt = build_answer_prompt(query_text, context, history)
        for chunk in gemini_client.stream_text(prompt):
            chunks.append(chunk)
            yield emit_token(chunk)

        yield emit_done()
    except Exception as error:
        print(f"Agent stream failed: {error}")
        message = "Sorry, something went wrong while generating a response. Please try again."
        chunks.append(message)
        yield emit_token(message)
        yield emit_done()
    finally:
        response_text = "".join(chunks).strip()
        if response_text:
            save_exchange(user_id, query_text, response_text)
