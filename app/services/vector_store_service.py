"""Service for managing Qdrant vector store operations."""
import json
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Query
from app.config import settings
from app.services.embedding_service import EmbeddingService


class VectorStoreService:
    """Service for managing vector store operations with Qdrant."""
    
    def __init__(self):
        """Initialize the vector store service."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.embedding_service = EmbeddingService()
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Get embedding dimension from OpenAI
                sample_embedding = self.embedding_service.create_embedding("sample")
                vector_size = len(sample_embedding)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed to ensure collection exists: {str(e)}")
    
    def load_json_and_store(self, json_file_path: str) -> int:
        """
        Load JSON file and store embeddings in Qdrant.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            Number of objects processed and stored
        """
        # Load JSON file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {str(e)}")
        
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
        
        # Prepare texts for embedding (only QuestionText)
        texts = []
        objects = []
        for obj in data:
            text = self.embedding_service.prepare_text_for_embedding(obj)
            if not text:
                # Skip objects without QuestionText
                continue
            texts.append(text)
            objects.append(obj)
        
        # Create embeddings
        embeddings = self.embedding_service.create_embeddings(texts)
        
        # Store in Qdrant with QuestionText as embedding and rest as metadata
        points = []
        for idx, (embedding, obj) in enumerate(zip(embeddings, objects)):
            # Convert MongoDB ObjectId or other string IDs to UUID format
            # Qdrant accepts UUIDs or integers, so we'll use integer index as ID
            # and store the original _id in metadata
            point_id = idx  # Use integer index as point ID
            
            # Create metadata payload with all fields except QuestionText
            metadata = {k: v for k, v in obj.items() if k != "QuestionText"}
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "QuestionText": obj.get("QuestionText", ""),
                        "metadata": metadata,
                        "index": idx
                    }
                )
            )
        
        # Batch upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return len(objects)
    
    def search_similar(
        self, 
        query: str, 
        threshold: float = 0.75, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar questions based on query.
        
        Args:
            query: Query string
            threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of results with question objects and scores
        """
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query)
        
        # Search in Qdrant using query_points
        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=threshold
        )
        
        # Format results
        results = []
        for point in search_response.points:
            payload = point.payload
            # Reconstruct the full question object
            question_obj = {
                "QuestionText": payload.get("QuestionText", ""),
                **payload.get("metadata", {})
            }
            
            # Get score from the ScoredPoint
            score = point.score if hasattr(point, 'score') else 1.0
            
            results.append({
                "id": str(point.id),
                "question": question_obj,
                "score": float(score)
            })
        
        return results
    
    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
        except Exception as e:
            raise RuntimeError(f"Failed to clear collection: {str(e)}")

