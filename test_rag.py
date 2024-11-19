import csv
from datetime import datetime
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

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

def query_rag(query_text):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        return {"response": "No relevant documents found", "sources": []}

    # Build the context from the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [{"id": doc.metadata.get("id", None), "content": doc.page_content} for doc, _score in results]
    formatted_response = {
        "response": response_text,
        "sources": sources
    }

    # Log the question, answer, and sources to a CSV file
    log_to_csv(query_text, response_text, sources)

    return formatted_response

def log_to_csv(question, answer, sources):
    with open('query_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now, question, f'"{answer}"', f'"{[source["id"] for source in sources]}"'])  # Log the source IDs

if __name__ == "__main__":
    question = input("Ask your question: ")
    response = query_rag(question)
    print(f"Answer: {response['response']}")
    print(f"Sources:")
    for source in response['sources']:
        print(f"ID: {source['id']}\nContent: {source['content']}\n")
