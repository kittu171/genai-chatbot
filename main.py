from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import (
    load_pdf,
    create_chunks,
    index_chunks,
    ask_question
)


app = FastAPI(
    title="Production RAG Chatbot API",
    description="A document-based RAG question answering API",
    version="1.0.0"
)


conversation_history = []


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: int
    chunk_id: int
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.on_event("startup")
def initialize_rag():
    from rag_engine import collection

    if collection.count() == 0:
        pages = load_pdf()
        chunks = create_chunks(pages)
        index_chunks(chunks)

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = ask_question(
            question,
            conversation_history
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question."
        )