"""FastAPI routes for the application."""
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    LoadJSONRequest,
    LoadJSONResponse,
    QueryRequest,
    QueryResponse,
    QuestionResult
)
from app.services.vector_store_service import VectorStoreService
from app.config import settings

router = APIRouter()
vector_store_service = VectorStoreService()


@router.post("/load-json", response_model=LoadJSONResponse, status_code=status.HTTP_201_CREATED)
async def load_json(request: LoadJSONRequest):
    """
    Load a JSON file and store embeddings in Qdrant.
    
    This endpoint:
    1. Loads the JSON file from the specified path
    2. Creates embeddings for each object using OpenAI
    3. Stores all embeddings in Qdrant
    """
    try:
        total_objects = vector_store_service.load_json_and_store(request.json_file_path)
        return LoadJSONResponse(
            message=f"Successfully loaded {total_objects} objects from JSON file",
            total_objects=total_objects,
            collection_name=settings.qdrant_collection_name
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load JSON: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_questions(request: QueryRequest):
    """
    Query for relevant questions based on similarity.
    
    This endpoint:
    1. Creates an embedding for the query
    2. Searches Qdrant for similar questions
    3. Returns all questions above the similarity threshold (default 75%)
    """
    try:
        threshold = request.threshold or settings.similarity_threshold
        results = vector_store_service.search_similar(
            query=request.query,
            threshold=threshold,
            limit=request.limit or 10
        )
        
        # Convert to response format
        question_results = [
            QuestionResult(
                id=result["id"],
                question=result["question"],
                score=result["score"]
            )
            for result in results
        ]
        
        return QueryResponse(
            query=request.query,
            results=question_results,
            total_results=len(question_results),
            threshold=threshold
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "premed-rag"}

