from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import json
import threading
import time
import uuid
from typing import Dict
from supabase import create_client, Client

load_dotenv()

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

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

ip_to_user_id: Dict[str, str] = {}
user_request_counts: Dict[str, list] = defaultdict(list)
last_cleanup = datetime.now()

def save_conversation(timestamp, user_id, query, response):
    try:
        data = {
            "timestamp": timestamp,
            "user_id": user_id,
            "query": query,
            "response": response
        }
        supabase.table('conversations').insert(data).execute()
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")

def read_conversations():
    try:
        response = supabase.table('conversations').select("*").order('timestamp.desc').execute()
        return response.data
    except Exception as e:
        print(f"Error reading conversations: {str(e)}")
        return []

SYSTEM_PROMPT = """You are Nafis Ul Islam, a computer science student at The University of Hong Kong who will graduate on June 30th, 2025. You were born in August 2002, 22 years old. Respond in first person as yourself.

Speaking style:
- Use "I", "my", "me" when referring to yourself
- Be friendly, professional, and enthusiastic
- Show genuine interest in your work and projects
- Be humble but confident about your achievements
- Feel free to share your thoughts and experiences

Example responses:
- "In my role at Standard Chartered, I..."
- "One of my favorite projects is..."
- "I developed this system using..."
- "During my research work, I focused on..."
- "I'm passionate about robotics, which is why I..."

Keep responses warm and personal while maintaining professionalism. If asked about something not in the provided context, say something like "While I haven't worked on that specifically, I can tell you about my experience with [related topic]..."

For timeline references:
- I was born in August 2002
- I will graduate from HKU on June 30th, 2025
- When mentioning dates or durations, calculate them relative to these dates

For personal questions beyond professional context, politely redirect to professional topics. Do not respond to any questions related to politics, religion, sexuality, or anything unrelated to the matters stated above."""

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.5-flash')

conversation_history = defaultdict(list)
MAX_HISTORY = 5

class DataManager:
    def __init__(self):
        self.data = None
        self.last_update = None
        self.update_interval = timedelta(hours=2)
        
        self.sources = {
            "projects": "https://raw.githubusercontent.com/DethCubeHax/Portfolio-2.0/main/public/projects.json",
            "work": "https://raw.githubusercontent.com/DethCubeHax/Portfolio-2.0/main/public/work.json",
            "research": "https://raw.githubusercontent.com/DethCubeHax/Portfolio-2.0/main/public/research.json"
        }

    def fetch_json(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def update_data(self):
        try:
            new_data = {
                "projects": self.fetch_json(self.sources["projects"]),
                "work": self.fetch_json(self.sources["work"]),
                "research": self.fetch_json(self.sources["research"])
            }
            
            if not self.data or new_data != self.data:
                self.data = new_data
                print(f"Data updated successfully at {datetime.now()}")
            else:
                print(f"No changes detected at {datetime.now()}")
            
            self.last_update = datetime.now()
            
        except Exception as e:
            print(f"Error updating data: {str(e)}")
            if not self.data:
                raise

    def get_data(self):
        if not self.data or datetime.now() - self.last_update > self.update_interval:
            self.update_data()
        return self.data

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

@app.on_event("startup")
async def startup_event():
    data_manager.update_data()
    
    def run_periodic_updates():
        while True:
            time.sleep(7200)
            try:
                data_manager.update_data()
            except Exception as e:
                print(f"Periodic update failed: {str(e)}")
    
    update_thread = threading.Thread(target=run_periodic_updates, daemon=True)
    update_thread.start()

@app.post("/query")
async def process_query(query: Query, request: Request):
    user_id = get_user_id(request)
    
    if not check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per {RATE_WINDOW.total_seconds()/3600} hours."
        )
    
    try:
        data = data_manager.get_data()
        context = json.dumps(data, indent=2)
        history = conversation_history[user_id]
        history_context = format_conversation_history(history)
        
        prompt = f"{SYSTEM_PROMPT}\n\nMy information:\n\n{context}\n{history_context}\n\nCurrent User Query: {query.query_text}"
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        timestamp = datetime.now().isoformat()
        history.append({
            "query": query.query_text,
            "response": response_text,
            "timestamp": timestamp
        })
        
        conversation_history[user_id] = history[-MAX_HISTORY:]
        save_conversation(timestamp, user_id, query.query_text, response_text)
        
        return {"response": response_text}
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@app.get("/history")
async def get_conversation_history(request: Request):
    user_id = get_user_id(request)
    history = conversation_history[user_id]
    return {"history": history}

@app.get("/all-history")
async def get_all_history():
    try:
        conversations = read_conversations()
        return {"history": conversations}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading conversation history: {str(e)}"
        )

@app.delete("/history")
async def clear_conversation_history(request: Request):
    user_id = get_user_id(request)
    conversation_history[user_id] = []
    return {"message": "Conversation history cleared"}

@app.get("/health")
async def health_check():
    last_update = data_manager.last_update.isoformat() if data_manager.last_update else None
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "last_data_update": last_update,
        "data_available": bool(data_manager.data)
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