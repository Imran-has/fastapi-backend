---
title: RAG Chatbot API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# RAG Chatbot API

FastAPI backend for RAG (Retrieval Augmented Generation) chatbot using Cohere and Qdrant.

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Chat with the bot
- `POST /api/select-context` - Query with selected context

## Environment Variables

Set these in your Hugging Face Space settings:

- `COHERE_API_KEY` - Your Cohere API key
- `QDRANT_URL` - Qdrant cloud URL
- `QDRANT_API_KEY` - Qdrant API key
