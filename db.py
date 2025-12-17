import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant client
# In a production environment, you might want to handle exceptions and retries
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    print("Qdrant client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Qdrant client: {e}")
    # In a real app, you might want to exit or use a fallback
    qdrant_client = None

# --- Optional: Neon Postgres for Chat History ---
# For now, we will use an in-memory store for simplicity.
# To use Postgres, you would uncomment the following lines and install psycopg2-binary
# import psycopg2

# DATABASE_URL = os.getenv("DATABASE_URL")
# db_conn = None
# try:
#     if DATABASE_URL:
#         db_conn = psycopg2.connect(DATABASE_URL)
#         print("PostgreSQL connection established.")
# except Exception as e:
#     print(f"Failed to connect to PostgreSQL: {e}")

# In-memory chat history store (as a simple replacement for Postgres)
chat_history_store = {} # e.g. { "session_id": [{"role": "user", ...}] }

def get_qdrant_client():
    """Returns the Qdrant client instance."""
    return qdrant_client

# Add functions to interact with Postgres here if you enable it.
# e.g., save_chat_message, get_chat_history
