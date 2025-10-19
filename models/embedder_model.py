"""
Embedding Model initialization for document embeddings
"""
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbedderModel:
    """Handles text embedding generation using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence transformer model
                       (default: all-MiniLM-L6-v2 - fast and efficient)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model with error handling"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.model = None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats, or None if embedding fails
        """
        if not self.model:
            logger.error("Embedding model not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors, or None if embedding fails
        """
        if not self.model:
            logger.error("Embedding model not initialized")
            return None
        
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return None
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return None
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of the embedding vectors"""
        if not self.model:
            return None
        return self.model.get_sentence_embedding_dimension()
    
    def is_available(self) -> bool:
        """Check if embedding model is properly initialized"""
        return self.model is not None