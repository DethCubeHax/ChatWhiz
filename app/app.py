import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import answer_query, stream_answer
from .config import GEMINI_API_KEY, GEMINI_MODEL, PORTFOLIO_DATA_BASE_URL, RATE_LIMIT, RATE_WINDOW
from .data_manager import data_manager, format_error
from .gemini_client import gemini_client
from .prompts import build_system_prompt, get_now_hkt
from .session import (
    check_rate_limit,
    cleanup_expired_data,
    clear_history,
    get_history,
    get_user_id,
    user_request_counts,
)


class Query(BaseModel):
    query_text: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.nafisui.com",
        "https://nafisui.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_context_stats():
    import json

    system_prompt = build_system_prompt()
    sample_context = data_manager.fetch_sources(["resume", "work"])
    context = json.dumps(sample_context, indent=2)
    sample_prompt = (
        f"{system_prompt}\n\nMy information:\n\n{context}\n\n"
        "Current User Query: What is your most recent work experience?"
    )

    return {
        "loaded": bool(data_manager.cache),
        "context_chars": len(context),
        "estimated_context_tokens": len(context) // 4,
        "sample_prompt_chars": len(sample_prompt),
        "estimated_sample_prompt_tokens": len(sample_prompt) // 4,
        "system_prompt_chars": len(system_prompt),
        "current_time_hkt": get_now_hkt().isoformat(),
        "agentic_routing": True,
    }


def test_gemini(prompt: str = "Reply with exactly: OK"):
    started = time.perf_counter()
    try:
        response = gemini_client.generate_text(prompt)
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": True,
            "model": GEMINI_MODEL,
            "duration_ms": duration_ms,
            "prompt": prompt,
            "response": response,
        }
    except Exception as error:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": False,
            "model": GEMINI_MODEL,
            "duration_ms": duration_ms,
            "prompt": prompt,
            "error": format_error(error),
        }


@app.get("/")
async def root():
    return {
        "service": "ChatWhiz",
        "status": "ok",
        "agentic": True,
        "endpoints": [
            "/health",
            "/query",
            "/query/stream",
            "/history",
            "/api/remaining-requests",
            "/diagnostics",
            "/diagnostics/config",
            "/diagnostics/data",
            "/diagnostics/data/sync",
            "/diagnostics/context",
            "/diagnostics/gemini",
        ],
    }


@app.get("/diagnostics")
async def diagnostics_index():
    return {
        "description": "Diagnostic endpoints for deployment testing",
        "endpoints": {
            "/diagnostics/config": "Environment and model configuration (no secrets)",
            "/diagnostics/data": "Cached portfolio data summary",
            "/diagnostics/data/sync": "Fetch each data source individually with timings",
            "/diagnostics/context": "Prompt/context size estimates",
            "/diagnostics/gemini": "Minimal Gemini API connectivity test",
        },
    }


@app.get("/diagnostics/config")
async def diagnostics_config():
    return {
        "timestamp": datetime.now().isoformat(),
        "portfolio_data_base_url": PORTFOLIO_DATA_BASE_URL,
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY else 0,
        "gemini_model": GEMINI_MODEL,
        "rate_limit": RATE_LIMIT,
        "rate_window_hours": RATE_WINDOW.total_seconds() / 3600,
        "agentic_routing": True,
    }


@app.get("/diagnostics/data")
async def diagnostics_data():
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": data_manager.get_summary(),
    }


@app.get("/diagnostics/data/sync")
async def diagnostics_data_sync():
    started = time.perf_counter()
    probe_results = data_manager.probe_sources()

    for key in probe_results:
        if probe_results[key].get("ok"):
            try:
                data_manager.fetch_source(key)
            except Exception as error:
                probe_results[key]["cache_error"] = format_error(error)

    return {
        "ok": all(result.get("ok") for result in probe_results.values()),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "sources": probe_results,
        "summary": data_manager.get_summary(),
    }


@app.get("/diagnostics/context")
async def diagnostics_context():
    return {
        "timestamp": datetime.now().isoformat(),
        "stats": get_context_stats(),
    }


@app.get("/diagnostics/gemini")
async def diagnostics_gemini():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    return {
        "timestamp": datetime.now().isoformat(),
        "result": test_gemini(),
    }


@app.on_event("startup")
async def startup_event():
    try:
        data_manager.warm_cache()
    except Exception as error:
        print(f"Startup cache warm failed (will fetch on demand): {error}")


@app.post("/query")
async def process_query(query: Query, request: Request):
    user_id = get_user_id(request)

    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW.total_seconds()/3600} hours.",
        )

    try:
        response_text = answer_query(user_id, query.query_text)
        return {"response": response_text}
    except Exception as error:
        print(f"Error processing query: {error}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")


@app.post("/query/stream")
async def process_query_stream(query: Query, request: Request):
    user_id = get_user_id(request)

    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW.total_seconds()/3600} hours.",
        )

    return StreamingResponse(
        stream_answer(user_id, query.query_text),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history")
async def get_conversation_history(request: Request):
    user_id = get_user_id(request)
    return {"history": get_history(user_id)}


@app.delete("/history")
async def clear_conversation_history(request: Request):
    user_id = get_user_id(request)
    clear_history(user_id)
    return {"message": "Conversation history cleared"}


@app.get("/health")
async def health_check():
    summary = data_manager.get_summary()
    return {
        "status": "healthy" if summary.get("loaded") else "degraded",
        "timestamp": datetime.now().isoformat(),
        "cached_sources": summary.get("cached_sources", []),
        "data_available": summary.get("loaded", False),
        "resume_available": "resume" in data_manager.cache,
        "data_source": PORTFOLIO_DATA_BASE_URL,
        "agentic_routing": True,
    }


@app.get("/api/remaining-requests")
async def get_remaining_requests(request: Request):
    user_id = get_user_id(request)
    cleanup_expired_data()

    current_time = datetime.now()
    user_request_counts[user_id] = [
        timestamp for timestamp in user_request_counts[user_id]
        if current_time - timestamp < RATE_WINDOW
    ]

    remaining = RATE_LIMIT - len(user_request_counts[user_id])
    return {
        "remaining_requests": remaining,
        "rate_limit": RATE_LIMIT,
        "window_hours": RATE_WINDOW.total_seconds() / 3600,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
