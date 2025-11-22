"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class LoadJSONRequest(BaseModel):
    """Request schema for loading JSON file."""
    json_file_path: str = Field(..., description="Path to the JSON file to load")
    batch_size: Optional[int] = Field(100, description="Number of items to process per batch", gt=0, le=1000)


class LoadJSONResponse(BaseModel):
    """Response schema for loading JSON file."""
    message: str
    total_objects: int
    collection_name: str


class QueryRequest(BaseModel):
    """Request schema for querying questions."""
    query: str = Field(..., description="Query string to search for relevant questions")
    threshold: Optional[float] = Field(0.40, description="Similarity threshold (0-1)", ge=0.0, le=1.0)
    limit: Optional[int] = Field(10, description="Maximum number of results to return", gt=0)


class QuestionResult(BaseModel):
    """Schema for a single question result."""
    question: Dict[str, Any] = Field(..., description="The original question object from JSON")
    score: float = Field(..., description="Similarity score (0-1)", ge=0.0, le=1.0)
    id: str = Field(..., description="Unique identifier for the question")


class QueryResponse(BaseModel):
    """Response schema for querying questions."""
    query: str
    results: List[QuestionResult]
    total_results: int
    threshold: float

