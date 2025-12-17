from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    chat_history: Optional[List[dict]] = None # e.g. [{"role": "user", "parts": ["Hello"]}, {"role": "model", "parts": ["Hi there!"]}]

class ChatResponse(BaseModel):
    response: str
    source_documents: Optional[List[dict]] = None

class SelectContextRequest(BaseModel):
    selected_text: str

class HealthResponse(BaseModel):
    status: str

