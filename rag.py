import os
import cohere
from .db import get_qdrant_client
from qdrant_client.http.models import ScoredPoint
from typing import List, Dict

# Configure Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = None

try:
    co = cohere.Client(COHERE_API_KEY)
    print("Cohere API configured successfully.")
except Exception as e:
    print(f"Failed to configure Cohere API: {e}")

# Constants
COLLECTION_NAME = "book_docs"

# --- Local Embedding Configuration ---
USE_LOCAL_EMBEDDING = os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true"
local_embedding_model = None

if USE_LOCAL_EMBEDDING:
    print("Initializing local embedding model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Local embedding model initialized successfully.")
    except Exception as e:
        print(f"Failed to load local embedding model: {e}")
        USE_LOCAL_EMBEDDING = False


def get_embedding(text: str) -> List[float]:
    """Generates embeddings for a given text, using local or Cohere."""
    if not text.strip():
        print("Attempted to embed empty text.")
        return []

    if USE_LOCAL_EMBEDDING and local_embedding_model:
        try:
            embedding = local_embedding_model.encode(text, convert_to_tensor=False).tolist()
            return embedding
        except Exception as e:
            print(f"Error generating local embedding: {e}")
            print("Falling back to Cohere embedding.")

    # Use Cohere for embeddings
    try:
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error generating Cohere embedding: {e}")
        return []


def search_qdrant(query: str, top_k: int = 5) -> List[Dict]:
    """Searches Qdrant for similar vectors."""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        print("Qdrant client not available.")
        return []

    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    try:
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

        results = [
            {
                "text": point.payload.get("text"),
                "source": point.payload.get("source"),
                "score": point.score
            }
            for point in search_results.points
        ]
        return results
    except Exception as e:
        print(f"Error searching Qdrant: {e}")
        return []


def generate_response(query: str, context_docs: List[Dict], chat_history: List[Dict] = None) -> str:
    """Generates a response using Cohere based on the query and context."""

    context_str = "\n---\n".join([doc["text"] for doc in context_docs])

    prompt = f"""You are a helpful assistant for a technical book. Your task is to answer the user's question based *only* on the provided context documents. If the answer is not found in the context, say "I'm sorry, I don't have enough information to answer that question." Do not use any external knowledge.

**Context Documents:**
{context_str}

**User's Question:**
{query}"""

    try:
        # Build chat history for Cohere (filter out empty messages)
        chat_history_cohere = []
        if chat_history:
            for msg in chat_history:
                role = "USER" if msg.get("role") == "user" else "CHATBOT"
                text = msg.get("parts", [""])[0] if isinstance(msg.get("parts"), list) else str(msg.get("parts", ""))
                # Only add if message is not empty
                if text and text.strip():
                    chat_history_cohere.append({"role": role, "message": text})

        response = co.chat(
            message=prompt,
            chat_history=chat_history_cohere if chat_history_cohere else None,
            model="command-r-plus-08-2024"
        )
        return response.text
    except Exception as e:
        print(f"Error generating Cohere response: {e}")
        return "Sorry, I encountered an error while generating a response."
