import json
import re
from typing import Dict, Iterator, List, Optional

from .config import SOURCE_CATALOG, SOURCE_KEYS, SOURCE_STATUS_LABELS, WEB_SEARCH_STATUS
from .data_manager import data_manager
from .gemini_client import gemini_client
from .prompts import build_answer_prompt, build_router_prompt, format_conversation_history
from .session import save_exchange
from .stream_events import emit_done, emit_status, emit_token
from .web_search import search_web


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


def normalize_web_search(value) -> Optional[str]:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


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
        "web_search": normalize_web_search(parsed.get("web_search")),
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
            "web_search": None,
        }


def gather_context(plan: dict) -> tuple[Dict, List[str]]:
    data_manager.ensure_loaded()
    context: Dict = {}
    statuses: List[str] = []

    for source in plan["sources"]:
        statuses.append(SOURCE_STATUS_LABELS[source])
        context[source] = data_manager.get_source(source)

    web_query = plan.get("web_search")
    if web_query:
        statuses.append(WEB_SEARCH_STATUS)
        results = search_web(web_query)
        context["web_search"] = {
            "query": web_query,
            "results": results,
            "result_count": len(results),
        }

    return context, statuses


def answer_query(user_id: str, query_text: str) -> str:
    from .session import get_history

    history = get_history(user_id)
    plan = select_sources(query_text, history)
    context, _ = gather_context(plan)
    prompt = build_answer_prompt(query_text, context, history)
    response_text = gemini_client.generate_text(prompt)
    save_exchange(user_id, query_text, response_text)
    return response_text


def stream_answer(user_id: str, query_text: str) -> Iterator[str]:
    from .session import get_history

    history = get_history(user_id)
    chunks: List[str] = []

    try:
        data_manager.ensure_loaded()
        yield emit_status("Deciding which sources to check...")

        plan = select_sources(query_text, history)

        if plan.get("reasoning"):
            yield emit_status(plan["reasoning"])

        context, statuses = gather_context(plan)
        for status in statuses:
            yield emit_status(status)

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
