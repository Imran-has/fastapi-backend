import os
import argparse
import time
from typing import List, Dict
import glob
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, UpdateStatus
from .rag import get_embedding, COLLECTION_NAME
from .db import get_qdrant_client

# --- TEXT PROCESSING ---

def load_markdown_docs(docs_path: str) -> List[Dict]:
    """Loads all markdown files from a given path recursively."""
    glob_pattern = os.path.join(docs_path, "**", "*.md")
    md_files = glob.glob(glob_pattern, recursive=True)

    loaded_docs = []
    for f in md_files:
        with open(f, "r", encoding="utf-8") as file:
            loaded_docs.append({
                "path": f,
                "content": file.read()
            })
    return loaded_docs


def split_text(documents: List[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Splits documents into smaller chunks using simple character-based splitting.
    """
    all_chunks = []

    for doc in documents:
        text = doc["content"]
        # Simple recursive splitting by paragraphs, then by size
        paragraphs = text.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    all_chunks.append({
                        "text": current_chunk.strip(),
                        "source": doc["path"]
                    })
                # Start new chunk with overlap
                if len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        # Don't forget the last chunk
        if current_chunk.strip():
            all_chunks.append({
                "text": current_chunk.strip(),
                "source": doc["path"]
            })

    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
    return all_chunks

# --- QDRANT OPERATIONS ---

def setup_qdrant_collection(recreate: bool = False):
    """Ensures the Qdrant collection exists."""
    qdrant_client = get_qdrant_client()

    if recreate:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024,  # Cohere embed-english-v3.0 uses 1024 dimensions
                distance=models.Distance.COSINE,
            ),
        )
        print("Collection recreated successfully.")
        return

    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating it...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024,  # Cohere embed-english-v3.0 uses 1024 dimensions
                distance=models.Distance.COSINE,
            ),
        )
        print("Collection created successfully.")

def ingest_data(chunks: List[Dict]):
    """Embeds and uploads data chunks to Qdrant."""
    qdrant_client = get_qdrant_client()
    points = []

    print(f"Preparing {len(chunks)} points for ingestion...")
    print("Note: Rate limiting enabled (90 calls/min for Cohere Trial key)")

    # Generate embeddings with rate limiting
    valid_embeddings_count = 0
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])

        if embedding:
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                    },
                )
            )
            valid_embeddings_count += 1

        # Rate limiting: 90 calls per minute = ~0.67 seconds between calls
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks...")

        time.sleep(0.7)  # Wait 0.7 seconds between API calls

    print(f"Generated {valid_embeddings_count} valid embeddings.")

    if not points:
        print("No valid points to ingest.")
        return

    # Batch upsert to Qdrant
    print("Upserting points to Qdrant...")
    try:
        operation_info = qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points,
        )
        if operation_info.status == UpdateStatus.COMPLETED:
            print("Data ingestion successful.")
        else:
            print(f"Data ingestion failed with status: {operation_info.status}")
    except Exception as e:
        print(f"An error occurred during Qdrant upsert: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ingest markdown documentation into Qdrant.")
    parser.add_argument(
        "docs_path",
        type=str,
        help="Path to the directory containing markdown files (e.g., '../docs').",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection (deletes existing data).",
    )
    args = parser.parse_args()

    print("--- Starting Ingestion Process ---")

    # 1. Ensure collection exists
    setup_qdrant_collection(recreate=args.recreate)
    
    # 2. Load markdown files
    raw_docs = load_markdown_docs(args.docs_path)
    if not raw_docs:
        print(f"No markdown files found in '{args.docs_path}'. Exiting.")
        return
        
    print(f"Found {len(raw_docs)} markdown documents.")

    # 3. Split documents into chunks
    chunks = split_text(raw_docs)
    
    # 4. Ingest chunks into Qdrant
    if chunks:
        ingest_data(chunks)
    else:
        print("No chunks to ingest.")
        
    print("--- Ingestion Process Finished ---")


if __name__ == "__main__":
    # This allows running the script directly, e.g.,
    # python -m fastapi_server.ingest ../../docs
    main()
