# ChatWhiz

This project implements scripts for Retrieval-Augmented Generation (RAG) using the LLaMa tokenizer and the Ollama model for chunking PDF documents, adding them to a Chroma database, and querying that database using FastAPI. I will update it as I learn more about RAG and LLM's

## Requirements

- Python 3.7+
- Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

## Setup

1. **Install Dependencies**: Install the necessary libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Files**: Ensure you have the following files in `/path/to/llama/folder/root` (you can change the path from the one given):
   - `checklist.chk`
   - `consolidated.00.pth`
   - `params.json`
   - `tokenizer.model`

3. **Data Directory**: Place your PDF files in the `data/` directory.

4. **Chroma Path**: Ensure the `chroma/` directory exists or update `CHROMA_PATH` in the scripts to your desired path.

## Running the Scripts

### Chunking and Adding to Chroma Database

To run the script for chunking PDF documents and adding them to the Chroma database, use:

```bash
python rag_agentic_chunking.py --reset
```

The `--reset` flag clears the existing database before processing the documents. If you want to update the database without clearing it, run the script without the `--reset` flag:

```bash
python rag_agentic_chunking.py
```

### Running the FastAPI Server

To run the FastAPI server, use:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://0.0.0.0:8000`.

### Querying with Test Script

To test querying the Chroma database with a standalone script, use:

```bash
python test_rag.py
```

Follow the on-screen prompt to input your query.

## API Endpoints

### POST /query

Query the Chroma database using RAG.

**Request Body**:
```json
{
  "query_text": "Your question here"
}
```

**Response**:
```json
{
  "response": "Generated response based on the context",
  "sources": [
    "source1",
    "source2",
    "..."
  ]
}
```

## Script Explanation

### rag_agentic_chunking.py

- **load_documents**: Loads all PDF documents from the `data/` directory.
- **create_mini_chunks**: Splits documents into mini chunks of 300 characters.
- **agentic_chunking**: Uses the LLaMa tokenizer and the Ollama model to process the mini chunks and group them into larger chunks.
- **add_to_chroma**: Adds or updates the chunks in the Chroma database.
- **calculate_chunk_ids**: Assigns unique IDs to each chunk based on its source and position.
- **clear_database**: Clears the Chroma database.

### server.py

- **query_rag**: The main endpoint that handles the RAG query process.
  - **Embedding Function**: Prepares the Chroma database using the embedding function.
  - **Similarity Search**: Searches the Chroma database for relevant documents.
  - **History Management**: Maintains session history and uses it to build context.
  - **Prompt Generation**: Creates a prompt for the Ollama model using the context.
  - **Model Invocation**: Gets the response from the Ollama model.
  - **Response Formatting**: Formats the response and logs it to a CSV file.

### test_rag.py

- **query_rag**: Queries the Chroma database and returns the response along with the sources.
- **log_to_csv**: Logs the question, answer, and sources to a CSV file.


## LICENSE

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Directory Structure

Here is the recommended structure for your project:

```plaintext
project/
│
├── data/
│   └── your_pdf_files.pdf
├── chroma/
│
├── rag_agentic_chunking.py
├── server.py
├── test_rag.py
├── requirements.txt
├── LICENSE
└── README.md
```
