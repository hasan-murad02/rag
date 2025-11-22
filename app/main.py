"""FastAPI application entry point."""
from fastapi import FastAPI
from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="PreMed RAG API",
    description="RAG API for querying premed questions using OpenAI embeddings and Qdrant",
    version="1.0.0"
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["rag"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PreMed RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

