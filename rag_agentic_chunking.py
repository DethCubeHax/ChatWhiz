import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from langchain_community.llms.ollama import Ollama  # Ensure you have the correct library installed
from tqdm import tqdm  # Import tqdm for the progress bar

CHROMA_PATH = "chroma"
DATA_PATH = "data"
LLAMA_TOKENIZER_PATH = "/Users/nafis/.llama/checkpoints/Llama3.2-3B"  # Path to your tokenizer

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    mini_chunks = create_mini_chunks(documents)
    chunks = agentic_chunking(mini_chunks)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def create_mini_chunks(documents: list[Document]):
    mini_chunks = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), 300):
            mini_chunk = text[i:i+300]
            mini_chunks.append(Document(page_content=mini_chunk, metadata=doc.metadata))
    return mini_chunks


def agentic_chunking(mini_chunks: list[Document]):
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_PATH)  # Use LLaMa tokenizer
    model = Ollama(model="llama3.2")  # Initialize the Ollama model

    grouped_chunks = []
    current_group = []
    current_length = 0

    print("🚀 Starting agentic chunking...")
    for chunk in tqdm(mini_chunks, desc="Processing mini-chunks"):
        text = chunk.page_content
        inputs = tokenizer(text, return_tensors='pt')['input_ids']
        prompt = tokenizer.decode(inputs[0], skip_special_tokens=True)
        response_text = model.invoke(prompt)  # Invoke the model

        current_length += len(response_text)
        current_group.append(chunk)

        if current_length >= 800:  # Define your chunk size
            grouped_chunks.append(combine_chunks(current_group))
            current_group = []
            current_length = 0

    if current_group:
        grouped_chunks.append(combine_chunks(current_group))

    return grouped_chunks


def combine_chunks(chunks: list[Document]):
    combined_text = " ".join(chunk.page_content for chunk in chunks)
    combined_metadata = chunks[0].metadata  # Assuming all chunks have the same metadata
    return Document(page_content=combined_text, metadata=combined_metadata)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
