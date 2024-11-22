import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
import random
import string
import json

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are an AI assistant. Your task is to provide accurate and concise answers based on the user's previous conversation history and the provided context.

---

RAG Context:
{context}

Today's date is {today_date}.
Your graduation date is: 2025-06-30.
The date format is YYYY-MM-DD.

---

Previous Interaction History:
{history}
User Question: {question}

Instructions: 
1. Answer the user's question based on their previous conversation history to understand the context, along with the RAG context provided.
3. Always answer in first person.
4. Answer the question directly. Do not add anything irrelevant to the answer.
5. Nafis = He = You
6. Do not make up something you do not know.
6. Do not write "Based on the context..." at any point.
7. Always answer in first person.
"""

# In-memory storage for user conversations
conversations = {}

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive - allows all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query_text: str

def generate_random_string(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def load_conversations():
    try:
        with open('conversations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_conversations():
    with open('conversations.json', 'w') as f:
        json.dump(conversations, f)

@app.on_event("startup")
async def startup_event():
    global conversations
    conversations = load_conversations()

@app.post("/query")
async def query_rag(request: Request, query: Query, response: Response):
    query_text = query.query_text
    session_id = request.headers.get('session_id')

    if not session_id:
        session_id = generate_random_string()
        response.headers['session_id'] = session_id

    # Initialize conversation history if it doesn't exist
    if session_id not in conversations:
        conversations[session_id] = []

    # Get previous conversation history
    history = conversations[session_id]
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Construct the history with the most recent question-answer pairs
    recent_history = history[-4:]
    formatted_history = ""
    for i in range(0, len(recent_history), 2):
        if i+1 < len(recent_history):
            formatted_history += f"Question: {recent_history[i]}\nAnswer: {recent_history[i+1]}\n"

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(history=formatted_history, context=context_text, question=query_text, today_date=today_date)

    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.page_content for doc, _score in results]
    formatted_response = {
        "response": response_text,
        "sources": sources
    }

    # Log the question and answer to the conversation history
    conversations[session_id].append(f"Question: {query_text}\nAnswer: {response_text}")

    # Save conversation history to a single JSON file
    save_conversations()

    # Log the question, answer, and sources to a CSV file
    log_to_csv(session_id, query_text, response_text, sources)

    # Print the question and response to the terminal
    print(f"Session ID: {session_id}")
    print(f"Previous Statements: {history[:-4]}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response_text}")

    return formatted_response

def log_to_csv(session_id, question, answer, sources):
    with open('query_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [now, session_id, question, answer] + sources[:5]  # Ensure only up to 5 sources are logged
        while len(row) < 9:  # Fill empty columns if less than 5 sources
            row.append('')
        writer.writerow(row)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
