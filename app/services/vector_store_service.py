"""Service for managing Qdrant vector store operations."""
import json
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Query, Filter, FieldCondition, MatchValue
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
            # Check if collection exists
            try:
                self.client.get_collection(self.collection_name)
                # Collection exists, return
                return
            except Exception:
                # Collection doesn't exist, create it
                pass
            
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
    
    def _get_existing_ids(self) -> set:
        """
        Get all existing _id values from the collection.
        
        Returns:
            Set of existing _id values
        """
        existing_ids = set()
        try:
            # Scroll through all points to get existing _id values
            offset = None
            while True:
                result, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not result:
                    break
                
                for point in result:
                    payload = point.payload
                    metadata = payload.get("metadata", {})
                    question_id = metadata.get("_id")
                    if question_id:
                        existing_ids.add(str(question_id))
                
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            # If collection is empty or doesn't exist, return empty set
            pass
        
        return existing_ids
    
    def load_json_and_store(self, json_file_path: str, batch_size: int = 100) -> int:
        """
        Load JSON file and store embeddings in Qdrant in batches.
        Skips questions that already exist (based on _id).
        
        Args:
            json_file_path: Path to the JSON file
            batch_size: Number of items to process in each batch (default: 100)
            
        Returns:
            Number of objects processed and stored
        """
        # Ensure collection exists before processing
        self._ensure_collection_exists()
        
        # Get existing _id values to skip duplicates
        existing_ids = self._get_existing_ids()
        
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
        
        # Filter objects with QuestionText and exclude existing ones
        valid_objects = []
        skipped_count = 0
        for obj in data:
            # Check if _id already exists
            obj_id = obj.get("_id")
            if obj_id and str(obj_id) in existing_ids:
                skipped_count += 1
                continue
            
            text = self.embedding_service.prepare_text_for_embedding(obj)
            if text:
                valid_objects.append(obj)
        
        total_objects = len(valid_objects)
        if total_objects == 0:
            return 0
        
        # Process in batches to avoid timeout
        total_stored = 0
        point_id_counter = 0  # Global counter for point IDs
        
        for batch_start in range(0, total_objects, batch_size):
            batch_end = min(batch_start + batch_size, total_objects)
            batch_objects = valid_objects[batch_start:batch_end]
            
            # Prepare texts for embedding (only QuestionText)
            texts = []
            texts_with_visual = []
            objects_with_visual = []
            visual_indices = []  # Track original indices for objects with visual_context
            
            for local_idx, obj in enumerate(batch_objects):
                # Always create embedding for QuestionText
                text = self.embedding_service.prepare_text_for_embedding(obj)
                texts.append(text)
                
                # Check if visual_context exists and create separate embedding
                text_with_visual = self.embedding_service.prepare_text_with_visual_context(obj)
                if text_with_visual:
                    texts_with_visual.append(text_with_visual)
                    objects_with_visual.append(obj)
                    visual_indices.append(batch_start + local_idx)
            
            # Create embeddings for QuestionText only
            embeddings = self.embedding_service.create_embeddings(texts)
            
            # Create embeddings for QuestionText + visual_context (if any)
            embeddings_with_visual = []
            if texts_with_visual:
                embeddings_with_visual = self.embedding_service.create_embeddings(texts_with_visual)
            
            # Store in Qdrant with QuestionText as embedding and rest as metadata
            points = []
            
            # Store embeddings for QuestionText only
            for idx, (embedding, obj) in enumerate(zip(embeddings, batch_objects)):
                point_id = point_id_counter
                point_id_counter += 1
                
                # Create metadata payload with all fields except QuestionText
                metadata = {k: v for k, v in obj.items() if k != "QuestionText"}
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "QuestionText": obj.get("QuestionText", ""),
                            "metadata": metadata,
                            "index": batch_start + idx,
                            "embedding_type": "question_text_only"
                        }
                    )
                )
            
            # Store embeddings for QuestionText + visual_context (if any)
            for idx, (embedding, obj) in enumerate(zip(embeddings_with_visual, objects_with_visual)):
                point_id = point_id_counter
                point_id_counter += 1
                
                # Create metadata payload with all fields except QuestionText
                metadata = {k: v for k, v in obj.items() if k != "QuestionText"}
                original_index = visual_indices[idx] if idx < len(visual_indices) else batch_start
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "QuestionText": obj.get("QuestionText", ""),
                            "visual_context": obj.get("visual_context", ""),
                            "metadata": metadata,
                            "index": original_index,
                            "embedding_type": "question_text_with_visual_context"
                        }
                    )
                )
            
            # Batch upsert points
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            total_stored += len(batch_objects)
        
        return total_stored
    
    def search_similar(
        self, 
        query: str, 
        threshold: float = 0.75, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar questions based on query.
        Returns unique questions based on _id from metadata.
        
        Args:
            query: Query string
            threshold: Minimum similarity score (0-1)
            limit: Maximum number of unique results
            
        Returns:
            List of unique results with question objects and scores
        """
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query)
        
        # Search with higher limit to account for potential duplicates
        # We'll deduplicate and then limit to the requested number
        search_limit = limit * 3  # Get more results to account for duplicates
        
        # Search in Qdrant using query_points
        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=search_limit,
            score_threshold=threshold
        )
        
        # Format results and deduplicate by _id
        results_by_id = {}  # Dictionary to store best result per _id
        
        for point in search_response.points:
            payload = point.payload
            metadata = payload.get("metadata", {})
            
            # Get _id from metadata (original question ID)
            question_id = metadata.get("_id", None)
            if not question_id:
                # Fallback: use point ID if _id not found
                question_id = str(point.id)
            
            question_id = str(question_id)
            
            # Get score from the ScoredPoint
            score = point.score if hasattr(point, 'score') else 1.0
            score = float(score)
            
            # Reconstruct the full question object
            question_obj = {
                "QuestionText": payload.get("QuestionText", ""),
                **metadata
            }
            
            # Keep the result with the highest score for each _id
            if question_id not in results_by_id:
                results_by_id[question_id] = {
                    "id": question_id,
                    "question": question_obj,
                    "score": score,
                    "point_id": str(point.id)
                }
            else:
                # If we already have this question, keep the one with higher score
                if score > results_by_id[question_id]["score"]:
                    results_by_id[question_id] = {
                        "id": question_id,
                        "question": question_obj,
                        "score": score,
                        "point_id": str(point.id)
                    }
        
        # Convert to list and sort by score (descending)
        results = list(results_by_id.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return only the requested number of results
        return results[:limit]
    
    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
        except Exception as e:
            raise RuntimeError(f"Failed to clear collection: {str(e)}")

