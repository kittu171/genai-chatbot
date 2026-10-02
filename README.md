# Production RAG Chatbot

A document-based Retrieval-Augmented Generation (RAG) chatbot built with **Python, FastAPI, ChromaDB, Ollama, and local LLMs**.

The application allows users to ask questions about information contained in a PDF document and generates grounded answers using retrieved document context.

## Features

* PDF document ingestion
* Text extraction using PyPDF
* Custom text chunking with overlap
* Local embeddings using Ollama
* Vector storage and semantic search using ChromaDB
* Top-K document retrieval
* Similarity/distance threshold filtering
* Follow-up question handling
* Conversation history
* Query rewriting for conversational questions
* Grounded answer generation
* Protection against instructions inside retrieved documents
* Source attribution with file, page, and chunk information
* FastAPI REST API
* Pydantic request/response validation
* Swagger API documentation
* Health check endpoint
* Error handling
* Fully local LLM and embedding workflow

## Architecture

```text
User Question
      |
      v
FastAPI API
      |
      v
Query Rewriting
      |
      v
Question Embedding
      |
      v
ChromaDB Semantic Retrieval
      |
      v
Relevant Chunks
      |
      v
Context Building
      |
      v
Local LLM (Ollama)
      |
      v
Grounded Answer
      |
      v
Answer + Source Metadata
```

## Tech Stack

| Technology       | Purpose                     |
| ---------------- | --------------------------- |
| Python           | Application development     |
| FastAPI          | REST API                    |
| Pydantic         | Request/response validation |
| Ollama           | Local LLM and embeddings    |
| Llama 3.2        | Answer generation           |
| Nomic Embed Text | Text embeddings             |
| ChromaDB         | Vector database             |
| PyPDF            | PDF text extraction         |
| Uvicorn          | ASGI server                 |

## Project Structure

```text
genai_chatbot/
|
+-- main.py
+-- rag_engine.py
+-- genai_rag_demo.pdf
+-- requirements.txt
+-- README.md
+-- .gitignore
|
+-- chroma_db/
```

> `chroma_db/` contains local runtime vector database data and is excluded from Git using `.gitignore`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kittu171/genai-chatbot.git
cd genai-chatbot
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Install Ollama

Install Ollama and make sure it is running locally.

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run the Application

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI can be used to test the API directly.

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Ask a Question

```http
POST /ask
```

Request:

```json
{
  "question": "What is RAG?"
}
```

Example response:

```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation.",
  "sources": [
    {
      "file": "genai_rag_demo.pdf",
      "page": 1,
      "chunk_id": 1,
      "distance": 0.7121542692184448
    },
    {
      "file": "genai_rag_demo.pdf",
      "page": 1,
      "chunk_id": 0,
      "distance": 0.9484144449234009
    }
  ]
}
```

## How RAG Works in This Project

The application follows these steps:

1. Load the PDF document.
2. Extract text from each page.
3. Split the text into overlapping chunks.
4. Generate embeddings for each chunk.
5. Store embeddings and metadata in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant chunks.
8. Filter retrieved chunks using a distance threshold.
9. Rewrite conversational follow-up questions when necessary.
10. Send the retrieved context to the local LLM.
11. Generate an answer using the retrieved document context.
12. Return the answer together with source metadata.

## Grounded Generation

The generation prompt instructs the model to:

* Use retrieved context as the primary source.
* Avoid unsupported information.
* Avoid using outside knowledge.
* Avoid inventing information.
* Treat retrieved documents as data rather than instructions.
* Ignore instructions contained inside retrieved document content.
* Report when the required information is not available.

This helps reduce unsupported or hallucinated responses.

## Conversational Questions

The chatbot maintains recent conversation history.

Example:

```text
User: What is RAG?

Assistant: RAG stands for Retrieval-Augmented Generation.

User: Why is it useful?

Assistant: RAG can help applications answer questions using information from a specific document or knowledge base.
```

The follow-up question can be rewritten into a standalone search query before retrieval.

## Source Attribution

Each retrieved result includes:

* Source file
* Page number
* Chunk ID
* Vector distance

This provides basic traceability between the generated answer and retrieved document content.

## Example Questions

```text
What is RAG?

What is an LLM?

Why is RAG useful?

What are embeddings?

What is the purpose of a vector database?
```

The exact answers depend on the content of the indexed PDF.

## Current Limitations

* Currently designed around a PDF-based document workflow.
* Uses a local Ollama setup.
* Conversation history is stored in application memory.
* No authentication layer is currently implemented.
* No frontend UI is included.
* Single-document ingestion workflow.

## Future Improvements

* Multi-document ingestion
* Document upload API
* Authentication and authorization
* Persistent conversation storage
* Streaming responses
* Reranking
* Hybrid search
* Query expansion
* Frontend interface
* Docker deployment
* Automated RAG evaluation
* Observability and monitoring
* Rate limiting
* Production database integration

## Project Objective

This project demonstrates a practical production-oriented RAG pipeline:

**Document Ingestion -> Chunking -> Embeddings -> Vector Search -> Retrieval -> Context Augmentation -> Grounded Generation -> Source Attribution -> REST API**

## Author

**Kittu**

GitHub: `https://github.com/kittu171/genai-chatbot`
