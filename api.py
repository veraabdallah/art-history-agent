"""
api.py — FastAPI web server wrapping the multi-agent art history assistant.
Run locally:  uvicorn api:app --reload
Run in Docker: uvicorn api:app --host 0.0.0.0 --port 8000
Access docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import chat, reset_chat

app = FastAPI(
    title="Art History Multi-Agent",
    description="A multi-agent RAG system for art history questions.",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    """Health check — confirms the server is running."""
    return {"status": "Art History Agent is running"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Send a message to the multi-agent system and get an answer.
    Example body: {"message": "What is Impressionism?"}
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    answer = chat(request.message, verbose=False)
    return ChatResponse(answer=answer)


@app.post("/reset")
def reset_endpoint():
    """Clear the conversation history."""
    reset_chat()
    return {"status": "Chat history cleared."}