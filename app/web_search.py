from typing import Any, Dict, List, Optional

from duckduckgo_search import DDGS

from .config import WEB_SEARCH_MAX_RESULTS


def search_web(query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
    limit = max_results or WEB_SEARCH_MAX_RESULTS
    cleaned_query = query.strip()

    if not cleaned_query:
        return []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(cleaned_query, max_results=limit))
    except Exception as error:
        print(f"Web search failed: {error}")
        return []

    results = []
    for item in raw_results:
        title = (item.get("title") or "").strip()
        body = (item.get("body") or item.get("snippet") or "").strip()
        href = (item.get("href") or item.get("link") or "").strip()
        if title or body:
            results.append({"title": title, "snippet": body, "url": href})

    return results
