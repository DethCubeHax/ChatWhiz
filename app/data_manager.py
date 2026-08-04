import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

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
        self.last_full_load: Optional[datetime] = None

    def _is_stale(self) -> bool:
        if not self.last_full_load:
            return True
        return datetime.now() - self.last_full_load > DATA_CACHE_TTL

    def _fetch_json(self, url: str):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _fetch_resume_text(self) -> str:
        response = requests.get(self.sources["resume"], timeout=30)
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def _fetch_from_network(self, key: str):
        if key == "resume":
            return self._fetch_resume_text()
        return self._fetch_json(self.sources[key])

    def is_fully_loaded(self) -> bool:
        return all(key in self.cache for key in SOURCE_KEYS) and not self._is_stale()

    def load_all_sources(self, force: bool = False) -> None:
        if not force and self.is_fully_loaded():
            return

        errors: List[str] = []
        new_cache: Dict[str, Dict[str, Any]] = dict(self.cache)

        for key in SOURCE_KEYS:
            try:
                data = self._fetch_from_network(key)
                new_cache[key] = {"data": data, "fetched_at": datetime.now()}
                print(f"Loaded source into memory: {key}")
            except Exception as error:
                errors.append(f"{key}: {error}")
                print(f"Failed loading source {key}: {error}")

        if not all(key in new_cache for key in SOURCE_KEYS):
            raise RuntimeError(
                "Unable to load all portfolio sources into memory. "
                + ("; ".join(errors) if errors else "No sources available.")
            )

        self.cache = new_cache
        self.last_full_load = datetime.now()
        print(f"All portfolio sources loaded into memory at {self.last_full_load.isoformat()}")

    def ensure_loaded(self) -> None:
        self.load_all_sources(force=self._is_stale())

    def get_source(self, key: str):
        if key not in SOURCE_KEYS:
            raise ValueError(f"Unknown source: {key}")
        if key not in self.cache:
            raise RuntimeError(f"Source '{key}' is not loaded in memory")
        return self.cache[key]["data"]

    def get_sources(self, keys: Iterable[str]) -> Dict[str, Any]:
        self.ensure_loaded()
        return {key: self.get_source(key) for key in keys}

    def get_all_sources(self) -> Dict[str, Any]:
        self.ensure_loaded()
        return {key: self.get_source(key) for key in SOURCE_KEYS}

    def warm_cache(self) -> None:
        self.load_all_sources(force=True)

    def get_summary(self) -> Dict[str, Any]:
        if not all(key in self.cache for key in SOURCE_KEYS):
            return {"loaded": False, "cached_sources": list(self.cache.keys())}

        summary = {
            "loaded": True,
            "cached_sources": list(self.cache.keys()),
            "last_full_load": self.last_full_load.isoformat() if self.last_full_load else None,
            "sources": self.sources,
            "counts": {},
        }

        projects = self.cache["projects"]["data"]
        work = self.cache["work"]["data"]
        research = self.cache["research"]["data"]
        resume = self.cache["resume"]["data"]

        summary["counts"]["projects"] = len(projects) if isinstance(projects, list) else 0
        summary["counts"]["work"] = len(work) if isinstance(work, list) else 0
        research_items = research.get("projects", []) if isinstance(research, dict) else research
        summary["counts"]["research"] = len(research_items) if isinstance(research_items, list) else 0
        summary["resume_chars"] = len(resume) if isinstance(resume, str) else 0

        return summary

    def probe_sources(self) -> Dict[str, Any]:
        results = {}

        for key, url in self.sources.items():
            started = time.perf_counter()
            try:
                if key == "resume":
                    text = self._fetch_resume_text()
                    results[key] = {
                        "ok": True,
                        "url": url,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        "resume_chars": len(text),
                    }
                else:
                    data = self._fetch_json(url)
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
