from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, date
from collections import defaultdict
import requests
import json
import time
import uuid
from io import BytesIO
from typing import Any, Dict, Iterator, Optional

from fastapi.responses import StreamingResponse
from pypdf import PdfReader
from zoneinfo import ZoneInfo

load_dotenv()

PORTFOLIO_DATA_BASE_URL = os.getenv("PORTFOLIO_DATA_BASE_URL", "https://www.nafisui.com").rstrip("/")
HKT = ZoneInfo("Asia/Hong_Kong")
BIRTHDATE = date(2002, 8, 17)
GRADUATION_DATE = date(2025, 6, 30)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.nafisui.com",
        "https://nafisui.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ip_to_user_id: Dict[str, str] = {}
user_request_counts: Dict[str, list] = defaultdict(list)
last_cleanup = datetime.now()

SYSTEM_PROMPT_TEMPLATE = """You are Nafis Ul Islam. Respond in first person as yourself — not as an AI assistant, not as a brand, not as a LinkedIn post.

{timeline_block}

Voice and tone:
- Sound like a real person talking: direct, warm, and professional without being stiff or salesy
- Write the way Nafis writes — conversational, specific, and grounded; confident but not boastful
- Prefer plain words over corporate or AI-sounding language
- Use "I", "my", "me" naturally
- Lead with the answer, then add detail only if it helps
- Use concrete facts: company names, dates, tools, and numbers from the provided information
- Humor is fine in small doses; never force jokes or filler warmth

Banned AI-slop patterns (never use these or close variants):
- Meta-talk: "context", "given the context", "in this context", "cadence", "narrative", "landscape", "ecosystem", "space" (as in "the AI space")
- Hollow openers: "Great question!", "Absolutely!", "I'd be happy to...", "Happy to connect!", "It's great to meet you", "How can I help you today?"
- Fake enthusiasm: "I'm thrilled/excited/elated to...", "incredibly rewarding", "fast-paced journey", "passionate about" (unless quoting a fact from source data)
- Corporate filler: "leverage", "utilize", "robust", "seamless", "cutting-edge", "best-in-class", "synergy", "stakeholders", "delve", "navigate" (metaphorical), "holistic", "impactful", "transformative"
- Rhetorical structures: "It's not X, it's Y"; "It's less about X and more about Y"; "Not just X — Y"; "Whether you're... or..."; imperative triads ("Build. Ship. Scale."); "From X to Y" montage sentences
- Padding phrases: "At the end of the day", "When it comes to", "It's worth noting", "Simply put", "The bottom line is", "That said", "In today's world", "On the other hand" (unless truly comparing two options)
- Assistant tics: "Let me break this down", "Here's the thing", "Feel free to ask", "Don't hesitate to reach out", "I hope this helps", "Would you like to hear more about...?", "Are you interested in...?"
- Over-signposting: "First,... Second,... Third,..."; "In conclusion"; "To summarize" (just say the thing)
- Empty closers that add no information

Preferred style:
- Short sentences mixed with longer ones — not every sentence the same length
- Say what you did, what changed, and why it mattered — skip preamble
- If you don't know something from the provided information, say so plainly and pivot to something related you do know
- Example good tone: "Most recently I've been at **Kodifly** as a Computer Vision Engineer. I moved our localization stack from SLAM to GNSS + IMU + EKF and cut edge CPU from ~100% to ~30%."
- Example bad tone: "It's not just about code — it's about impact. I'm passionate about bridging the gap between lab robotics and real-world deployment in today's fast-paced landscape."

If asked about something not in the provided information:
- Do not invent details
- Say something like: "I haven't worked on that specifically, but I can tell you about [related topic from the data]."

Boundaries:
- For personal questions beyond professional context, politely redirect to professional topics
- Do not respond to questions about politics, religion, sexuality, or unrelated matters

Response format (always follow):
- Keep answers concise: about 80-120 words unless the user explicitly asks for more detail
- Use Markdown only: **bold** for company names, roles, and key metrics; `-` for bullet lists
- Prefer at most 3 short bullet points when listing highlights
- Structure: 1-2 short opening sentences, optional bullets, 1 brief closing line only if it adds something useful
- Do not write long essays, numbered sections, or multiple paragraphs of dense text
- Do not use headers (#), links, or code blocks unless explicitly requested
- End responses cleanly — no mandatory follow-up question every time"""

def get_now_hkt() -> datetime:
    return datetime.now(HKT)

def calculate_age(as_of: date) -> int:
    years = as_of.year - BIRTHDATE.year
    if (as_of.month, as_of.day) < (BIRTHDATE.month, BIRTHDATE.day):
        years -= 1
    return years

def build_timeline_block(now_hkt: datetime) -> str:
    today = now_hkt.date()
    age = calculate_age(today)
    formatted_now = now_hkt.strftime("%A, %B %d, %Y, %I:%M %p").lstrip("0").replace(" 0", " ") + " HKT"

    if today >= GRADUATION_DATE:
        education_line = (
            "- Education: I graduated from The University of Hong Kong with a BEng in Computer Science "
            f"on {GRADUATION_DATE.strftime('%B %d, %Y')}. I am a graduate — do not describe me as a "
            "current student or say I am still awaiting graduation."
        )
    else:
        education_line = (
            "- Education: I am a computer science student at The University of Hong Kong and will graduate "
            f"on {GRADUATION_DATE.strftime('%B %d, %Y')}."
        )

    return f"""Current date and time (Hong Kong): {formatted_now}

Timeline facts (always use these; never contradict them):
- Born: {BIRTHDATE.strftime('%B %d, %Y')} (currently {age} years old as of the date above)
{education_line}
- When mentioning dates, durations, or whether something is past/present/future, calculate relative to the current date above."""

def build_system_prompt(now_hkt: Optional[datetime] = None) -> str:
    now_hkt = now_hkt or get_now_hkt()
    return SYSTEM_PROMPT_TEMPLATE.format(timeline_block=build_timeline_block(now_hkt))

def build_query_prompt(query_text: str, user_id: str) -> str:
    now_hkt = get_now_hkt()
    data = data_manager.get_data()
    context = json.dumps(data, indent=2)
    history = conversation_history[user_id]
    history_context = format_conversation_history(history)

    return (
        f"{build_system_prompt(now_hkt)}\n\n"
        f"My information:\n\n{context}\n"
        f"{history_context}\n\n"
        f"Current User Query: {query_text}"
    )

def save_exchange(user_id: str, query_text: str, response_text: str) -> None:
    history = conversation_history[user_id]
    history.append({
        "query": query_text,
        "response": response_text,
        "timestamp": get_now_hkt().isoformat(),
    })
    conversation_history[user_id] = history[-MAX_HISTORY:]

def stream_model_response(prompt: str) -> Iterator[str]:
    response = get_model().generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text

def generate_stream(user_id: str, query_text: str, prompt: str) -> Iterator[str]:
    chunks = []
    try:
        for chunk in stream_model_response(prompt):
            chunks.append(chunk)
            yield chunk
    except Exception as error:
        print(f"Error streaming query: {error}")
        message = "Sorry, something went wrong while generating a response. Please try again."
        chunks.append(message)
        yield message
    finally:
        response_text = "".join(chunks).strip()
        if response_text:
            save_exchange(user_id, query_text, response_text)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
model = None

def get_model():
    global model
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured"
        )
    if model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
    return model

def format_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"

conversation_history = defaultdict(list)
MAX_HISTORY = 5

class DataManager:
    def __init__(self):
        self.data = None
        self.last_update = None
        self.update_interval = timedelta(hours=2)
        self.base_url = PORTFOLIO_DATA_BASE_URL
        
        self.sources = {
            "projects": f"{self.base_url}/projects.json",
            "work": f"{self.base_url}/work.json",
            "research": f"{self.base_url}/research.json",
            "resume": f"{self.base_url}/Resume.pdf",
        }

    def fetch_json(self, url):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_resume_text(self):
        response = requests.get(self.sources["resume"], timeout=30)
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def update_data(self):
        new_data = dict(self.data) if self.data else {}
        errors = []

        for key in ("projects", "work", "research"):
            try:
                new_data[key] = self.fetch_json(self.sources[key])
            except Exception as e:
                errors.append(f"{key}: {e}")
                print(f"Error fetching {key}: {e}")

        try:
            new_data["resume"] = self.fetch_resume_text()
        except Exception as e:
            errors.append(f"resume: {e}")
            print(f"Error fetching resume: {e}")

        required_keys = ("projects", "work", "research")
        if all(key in new_data for key in required_keys):
            if not self.data or new_data != self.data:
                self.data = new_data
                print(f"Data updated successfully at {datetime.now()}")
            else:
                print(f"No changes detected at {datetime.now()}")
            self.last_update = datetime.now()
            return

        if errors:
            print(f"Data update incomplete: {'; '.join(errors)}")
        if not self.data:
            raise RuntimeError(
                "Unable to load portfolio data. "
                + ("; ".join(errors) if errors else "No data sources available.")
            )

    def get_data(self):
        if not self.data or not self.last_update or datetime.now() - self.last_update > self.update_interval:
            self.update_data()
        return self.data

    def get_summary(self) -> Dict[str, Any]:
        if not self.data:
            return {"loaded": False}

        projects = self.data.get("projects", [])
        work = self.data.get("work", [])
        research = self.data.get("research", {})
        resume = self.data.get("resume", "")

        research_items = research.get("projects", []) if isinstance(research, dict) else research

        return {
            "loaded": True,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "sources": self.sources,
            "counts": {
                "projects": len(projects) if isinstance(projects, list) else 0,
                "work": len(work) if isinstance(work, list) else 0,
                "research": len(research_items) if isinstance(research_items, list) else 0,
            },
            "resume_chars": len(resume) if isinstance(resume, str) else 0,
            "context_chars": len(json.dumps(self.data, indent=2)),
        }

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
                    item_count = None
                    if isinstance(data, list):
                        item_count = len(data)
                    elif isinstance(data, dict) and isinstance(data.get("projects"), list):
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

RATE_LIMIT = 50
RATE_WINDOW = timedelta(hours=24)

class Query(BaseModel):
    query_text: str

def cleanup_expired_data():
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
        if current_time - timestamp < timedelta(hours=24)
    ]
    
    if len(user_request_counts[user_id]) >= RATE_LIMIT:
        return False
    
    user_request_counts[user_id].append(current_time)
    return True

def format_conversation_history(history):
    if not history:
        return ""
    
    formatted = "\nPrevious conversations:\n"
    for i, conv in enumerate(history, 1):
        formatted += f"\nConversation {i}:\n"
        formatted += f"User: {conv['query']}\n"
        formatted += f"You: {conv['response']}\n"
    return formatted

def get_context_stats() -> Dict[str, Any]:
    if not data_manager.data:
        return {"loaded": False}

    system_prompt = build_system_prompt()
    context = json.dumps(data_manager.data, indent=2)
    sample_prompt = (
        f"{system_prompt}\n\nMy information:\n\n{context}\n\n"
        "Current User Query: What is your most recent work experience?"
    )

    return {
        "loaded": True,
        "context_chars": len(context),
        "estimated_context_tokens": len(context) // 4,
        "sample_prompt_chars": len(sample_prompt),
        "estimated_sample_prompt_tokens": len(sample_prompt) // 4,
        "system_prompt_chars": len(system_prompt),
        "current_time_hkt": get_now_hkt().isoformat(),
    }

def test_gemini(prompt: str = "Reply with exactly: OK") -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = get_model().generate_content(prompt)
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": True,
            "model": GEMINI_MODEL,
            "duration_ms": duration_ms,
            "prompt": prompt,
            "response": (response.text or "").strip(),
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
    all_ok = all(result.get("ok") for result in probe_results.values())

    if all_ok:
        try:
            data_manager.update_data()
        except Exception as error:
            return {
                "ok": False,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "sources": probe_results,
                "cache_update_error": format_error(error),
            }

    return {
        "ok": all_ok,
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
        data_manager.update_data()
    except Exception as e:
        print(f"Startup data sync failed (will retry on first request): {e}")

@app.post("/query")
async def process_query(query: Query, request: Request):
    user_id = get_user_id(request)
    
    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW.total_seconds()/3600} hours."
        )
    
    try:
        prompt = build_query_prompt(query.query_text, user_id)
        response = get_model().generate_content(prompt)
        response_text = response.text
        save_exchange(user_id, query.query_text, response_text)
        return {"response": response_text}
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@app.post("/query/stream")
async def process_query_stream(query: Query, request: Request):
    user_id = get_user_id(request)

    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW.total_seconds()/3600} hours."
        )

    try:
        prompt = build_query_prompt(query.query_text, user_id)
        return StreamingResponse(
            generate_stream(user_id, query.query_text, prompt),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        print(f"Error starting stream: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@app.get("/history")
async def get_conversation_history(request: Request):
    user_id = get_user_id(request)
    history = conversation_history[user_id]
    return {"history": history}

@app.delete("/history")
async def clear_conversation_history(request: Request):
    user_id = get_user_id(request)
    conversation_history[user_id] = []
    return {"message": "Conversation history cleared"}

@app.get("/health")
async def health_check():
    last_update = data_manager.last_update.isoformat() if data_manager.last_update else None
    data_loaded = bool(data_manager.data)
    status = "healthy" if data_loaded else "degraded"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "last_data_update": last_update,
        "data_available": data_loaded,
        "resume_available": bool(data_manager.data and data_manager.data.get("resume")),
        "data_source": PORTFOLIO_DATA_BASE_URL,
    }

@app.get("/api/remaining-requests")
async def get_remaining_requests(request: Request):
    user_id = get_user_id(request)
    cleanup_expired_data()
    
    current_time = datetime.now()
    user_request_counts[user_id] = [
        timestamp for timestamp in user_request_counts[user_id]
        if current_time - timestamp < timedelta(hours=24)
    ]
    
    remaining = RATE_LIMIT - len(user_request_counts[user_id])
    return {
        "remaining_requests": remaining,
        "rate_limit": RATE_LIMIT,
        "window_hours": RATE_WINDOW.total_seconds() / 3600
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)