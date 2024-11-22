import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from transformers import AutoTokenizer
from langchain_community.llms.ollama import Ollama  # Ensure you have the correct library installed
from tqdm import tqdm  # Import tqdm for the progress bar
import torch
import numpy as np

CHROMA_PATH = "chroma"
DATA_PATH = "data"
LLAMA_TOKENIZER_PATH = "/Users/nafis/.llama/checkpoints/Llama3.2-3B"  # Path to your tokenizer

MAX_SECTION_LENGTH = 500
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

def main():
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
    save_chunks_to_file(chunks, "chunks.txt")
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def create_mini_chunks(documents: list[Document]):
    mini_chunks = []
    for doc in tqdm(documents, desc="Loading documents"):
        text = doc.page_content
        mini_chunks.extend(split_text(text, doc.metadata))
    return mini_chunks


def split_text(text, metadata):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORD_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", " "]
    length = len(text)
    start = 0
    end = length
    chunks = []

    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and text[end] not in SENTENCE_ENDINGS:
                if text[end] in WORD_BREAKS:
                    last_word = end
                end += 1
            if end < length and text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word

        if end < length:
            end += 1

        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and text[start] not in SENTENCE_ENDINGS:
            if text[start] in WORD_BREAKS:
                last_word = start
            start -= 1
        if text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = text[start:end]
        chunk = Document(page_content=section_text, metadata=metadata)
        chunks.append(chunk)

        start = end - SECTION_OVERLAP

    if start + SECTION_OVERLAP < end:
        section_text = text[start:end]
        chunk = Document(page_content=section_text, metadata=metadata)
        chunks.append(chunk)

    return chunks


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
            combined_chunk = combine_chunks(current_group)
            grouped_chunks.append(combined_chunk)
            current_group = []
            current_length = 0
            print(f"Chunk created: {combined_chunk.page_content}")
            print("="*50)

    if current_group:
        combined_chunk = combine_chunks(current_group)
        grouped_chunks.append(combined_chunk)
        print(f"Chunk created: {combined_chunk.page_content}")
        print("="*50)

    return grouped_chunks


def combine_chunks(chunks: list[Document]):
    combined_text = " ".join(chunk.page_content for chunk in chunks)
    combined_metadata = chunks[0].metadata  # Assuming all chunks have the same metadata
    return Document(page_content=combined_text, metadata=combined_metadata)


def save_chunks_to_file(chunks, filename):
    with open(filename, 'w') as f:
        for chunk in chunks:
            f.write(f"{chunk.page_content}\n\n")


def add_to_chroma(chunks: list[Document]):
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in tqdm(chunks_with_ids, desc="Adding to Chroma"):
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

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
