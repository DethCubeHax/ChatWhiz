import csv
import argparse
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from starlette.middleware.sessions import SessionMiddleware
from uuid import uuid4

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
Always answer in first person.
Answer the question directly. Do not add anything irrelevant to the answer.
Nafis = He = You
Do not write "Based on the context..." at any point.
Always answer in first person.
"""

app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive - allows all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware for session-based storage
app.add_middleware(SessionMiddleware, secret_key=str(uuid4()))

class Query(BaseModel):
    query_text: str

@app.post("/query")
async def query_rag(request: Request, query: Query):
    query_text = query.query_text
    
    # Initialize session history if it doesn't exist
    if 'history' not in request.session:
        request.session['history'] = []

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Get previous conversation history
    history = request.session['history']
    context_text = "\n\n---\n\n".join(history + [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = {
        "response": response_text,
        "sources": sources
    }

    # Print the questions this particular user has asked before
    previous_questions = [entry.split('\n')[0] for entry in history if entry.startswith("Question:")]
    print("Previous Questions:")
    for question in previous_questions:
        print(question)

    # Log the question and answer to the session history
    request.session['history'].append(f"Question: {query_text}\nAnswer: {response_text}")

    # Log the question, answer, and sources to a CSV file
    log_to_csv(query_text, response_text, sources)

    # Print the question and response to the terminal
    print(f"Question: {query_text}")
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
