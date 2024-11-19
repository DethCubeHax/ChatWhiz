from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"

def print_all_embeddings_and_documents():
    # Prepare the DB.
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Fetch all documents, their embeddings, and metadata.
    all_docs = db.get(include=["documents", "embeddings", "metadatas"])
    documents = all_docs["documents"]
    embeddings = all_docs["embeddings"]
    metadatas = all_docs["metadatas"]

    # Print out the documents, their metadata, and corresponding embeddings.
    for idx, doc in enumerate(documents):
        print(f"Document ID: {metadatas[idx].get('id', 'N/A')}")
        print(f"Content: {doc}")
        print(f"Embedding: {embeddings[idx]}")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print_all_embeddings_and_documents()
