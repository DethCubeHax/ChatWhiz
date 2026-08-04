import json
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List

import requests
from pypdf import PdfReader

from .config import DATA_CACHE_TTL, PORTFOLIO_DATA_BASE_URL, SOURCE_KEYS


def format_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


class DataManager:
    def __init__(self) -> None:
        self.base_url = PORTFOLIO_DATA_BASE_URL
        self.sources = {
            "projects": f"{self.base_url}/projects.json",
            "work": f"{self.base_url}/work.json",
            "research": f"{self.base_url}/research.json",
            "resume": f"{self.base_url}/Resume.pdf",
        }
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _is_stale(self, key: str) -> bool:
        entry = self.cache.get(key)
        if not entry:
            return True
        fetched_at = entry["fetched_at"]
        return datetime.now() - fetched_at > DATA_CACHE_TTL

    def fetch_json(self, url: str):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_resume_text(self) -> str:
        response = requests.get(self.sources["resume"], timeout=30)
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def fetch_source(self, key: str):
        if key not in self.sources:
            raise ValueError(f"Unknown source: {key}")

        if key in self.cache and not self._is_stale(key):
            return self.cache[key]["data"]

        if key == "resume":
            data = self.fetch_resume_text()
        else:
            data = self.fetch_json(self.sources[key])

        self.cache[key] = {"data": data, "fetched_at": datetime.now()}
        return data

    def fetch_sources(self, keys: Iterable[str]) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for key in keys:
            context[key] = self.fetch_source(key)
        return context

    def warm_cache(self) -> None:
        for key in SOURCE_KEYS:
            try:
                self.fetch_source(key)
            except Exception as error:
                print(f"Warm cache failed for {key}: {error}")

    def get_summary(self) -> Dict[str, Any]:
        if not self.cache:
            return {"loaded": False, "cached_sources": []}

        summary = {
            "loaded": True,
            "cached_sources": list(self.cache.keys()),
            "sources": self.sources,
            "counts": {},
        }

        if "projects" in self.cache:
            projects = self.cache["projects"]["data"]
            summary["counts"]["projects"] = len(projects) if isinstance(projects, list) else 0
        if "work" in self.cache:
            work = self.cache["work"]["data"]
            summary["counts"]["work"] = len(work) if isinstance(work, list) else 0
        if "research" in self.cache:
            research = self.cache["research"]["data"]
            research_items = research.get("projects", []) if isinstance(research, dict) else research
            summary["counts"]["research"] = len(research_items) if isinstance(research_items, list) else 0
        if "resume" in self.cache:
            resume = self.cache["resume"]["data"]
            summary["resume_chars"] = len(resume) if isinstance(resume, str) else 0

        return summary

    def probe_sources(self) -> Dict[str, Any]:
        results = {}

        for key, url in self.sources.items():
            started = time.perf_counter()
            try:
                if key == "resume":
                    text = self.fetch_resume_text()
                    results[key] = {
                        "ok": True,
                        "url": url,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        "resume_chars": len(text),
                    }
                else:
                    data = self.fetch_json(url)
                    item_count = len(data) if isinstance(data, list) else None
                    if isinstance(data, dict) and isinstance(data.get("projects"), list):
                        item_count = len(data["projects"])
                    results[key] = {
                        "ok": True,
                        "url": url,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        "items": item_count,
                    }
            except Exception as error:
                results[key] = {
                    "ok": False,
                    "url": url,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": format_error(error),
                }

        return results


data_manager = DataManager()
