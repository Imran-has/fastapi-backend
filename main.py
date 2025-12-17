from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse, SelectContextRequest, HealthResponse
from .rag import search_qdrant, generate_response
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for a RAG chatbot with Qdrant and Cohere.",
    version="1.0.0",
)

# --- CORS Configuration ---
# WARNING: This is a permissive CORS configuration for development.
# For production, you should restrict the origins to your actual frontend URL.
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint to ensure the service is running.
    """
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat_handler(chat_request: ChatRequest):
    """
    Main chat endpoint. Receives a query, finds relevant context,
    and generates a response.
    """
    try:
        query = chat_request.query
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # 1. Search for relevant documents in Qdrant
        print(f"Searching for: {query}")
        search_results = search_qdrant(query, top_k=5)
        print(f"Search results count: {len(search_results)}")

        if not search_results:
            # Fallback response if no context is found
            return ChatResponse(
                response="I'm sorry, I couldn't find any relevant information in the documentation to answer your question.",
                source_documents=[]
            )

        # 2. Generate a response using the context
        response_text = generate_response(
            query=query,
            context_docs=search_results,
            chat_history=chat_request.chat_history
        )
        
        # 3. Return the response and source documents
        return ChatResponse(
            response=response_text,
            source_documents=search_results
        )
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.post("/api/select-context", response_model=ChatResponse, tags=["Chat"])
def select_context_handler(request: SelectContextRequest):
    """
    Endpoint to handle queries based on user-selected text.
    The selected text itself becomes the primary context.
    """
    selected_text = request.selected_text
    if not selected_text:
        raise HTTPException(status_code=400, detail="Selected text cannot be empty.")
        
    # In this simplified version, we treat the selected text as the only context.
    # A more advanced implementation might still search Qdrant to enrich this context.
    
    # The user's implicit query is "What does this mean?" or "Explain this."
    # We can create a meta-query to generate a summary or explanation.
    meta_query = f"Please explain the following text in simple terms:\n\n---\n{selected_text}\n---"

    # We provide the selected text itself as the "context document".
    context_docs = [{"text": selected_text, "source": "User Selection", "score": 1.0}]

    response_text = generate_response(query=meta_query, context_docs=context_docs)

    return ChatResponse(
        response=response_text,
        source_documents=context_docs
    )

# To run the server, use the command:
# uvicorn fastapi_server.main:app --reload

