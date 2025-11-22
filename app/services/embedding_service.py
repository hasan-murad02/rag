"""Service for creating embeddings using OpenAI."""
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from app.config import settings


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        """Initialize the embedding service with OpenAI."""
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def prepare_text_for_embedding(self, obj: Dict[str, Any]) -> str:
        """
        Extract only QuestionText for embedding.
        All other fields will be stored as metadata.
        
        Args:
            obj: Dictionary object from JSON
            
        Returns:
            QuestionText string for embedding
        """
        if isinstance(obj, dict):
            # Extract only QuestionText for embedding
            question_text = obj.get("QuestionText", "")
            if not question_text:
                # Fallback: try other common question field names
                question_text = obj.get("question", obj.get("Question", ""))
            return str(question_text) if question_text else ""
        return str(obj)
    
    def prepare_text_with_visual_context(self, obj: Dict[str, Any]) -> str:
        """
        Prepare text combining QuestionText and visual_context for embedding.
        
        Args:
            obj: Dictionary object from JSON
            
        Returns:
            Combined QuestionText + visual_context string, or empty if no visual_context
        """
        if isinstance(obj, dict):
            question_text = obj.get("QuestionText", "")
            if not question_text:
                question_text = obj.get("question", obj.get("Question", ""))
            
            visual_context = obj.get("visual_context", "")
            
            # Only return combined text if visual_context exists and is not empty
            if visual_context and str(visual_context).strip():
                return f"{question_text} {visual_context}".strip()
        
        return ""

