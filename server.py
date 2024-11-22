import csv
import argparse
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
Previous Interaction History:
{history}

---

User Question: {question}

---

Instructions: 
1. Answer the question based only on the above context.
2. Always answer in first person.
3. Answer the question directly. Do not add anything irrelevant to the answer.
4. Nafis = He = You
5. Do not write "Based on the context..." at any point.
6. Always answer in first person.
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
    message_id = request.cookies.get('message_id')

    if not message_id:
        message_id = generate_random_string()
        response.set_cookie(key='message_id', value=message_id)

    # Initialize conversation history if it doesn't exist
    if message_id not in conversations:
        conversations[message_id] = []

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Get previous conversation history
    history = conversations[message_id]
    context_text = "\n\n---\n\n".join(history + [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(history=context_text, question=query_text)

    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = {
        "response": response_text,
        "sources": sources
    }

    # Log the question and answer to the conversation history
    conversations[message_id].append(f"Question: {query_text}\nAnswer: {response_text}")

    # Save conversation history to a single JSON file
    save_conversations()

    # Log the question, answer, and sources to a CSV file
    log_to_csv(query_text, response_text, sources)

    # Print the question and response to the terminal
    print(f"Message ID: {message_id}")
    print(f"Previous Statements: {history}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response_text}")

    return formatted_response

def log_to_csv(question, answer, sources):
    with open('query_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, question, f'"{answer}"', f'"{sources}"'])  # Wrap the answer and sources in double quotes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
