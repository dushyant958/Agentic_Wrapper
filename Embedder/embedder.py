"""
Vector Store and RAG functionality using FAISS
"""
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document embeddings and retrieval"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Dimension of embedding vectors (default: 384 for MiniLM)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        logger.info(f"FAISS index initialized with dimension {dimension}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadata: Optional[List[Dict]] = None):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
        """
        try:
            if len(texts) != len(embeddings):
                logger.error("Number of texts and embeddings must match")
                return
            
            # Convert embeddings to numpy array
            embeddings_np = np.array(embeddings, dtype='float32')
            
            # Add to FAISS index
            self.index.add(embeddings_np)
            
            # Store documents
            self.documents.extend(texts)
            
            # Store metadata
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(texts))
            
            logger.info(f"Added {len(texts)} documents to vector store. Total: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (document_text, distance, metadata)
        """
        try:
            if len(self.documents) == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Convert query to numpy array
            query_np = np.array([query_embedding], dtype='float32')
            
            # Search in FAISS
            k = min(k, len(self.documents))
            distances, indices = self.index.search(query_np, k)
            
            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        float(dist),
                        self.metadata[idx] if idx < len(self.metadata) else {}
                    ))
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def clear(self):
        """Clear all documents from the vector store"""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            self.metadata = []
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
    
    def save(self, index_path: str, docs_path: str):
        """
        Save vector store to disk
        
        Args:
            index_path: Path to save FAISS index
            docs_path: Path to save documents and metadata
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save documents and metadata
            with open(docs_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Vector store saved to {index_path} and {docs_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def load(self, index_path: str, docs_path: str):
        """
        Load vector store from disk
        
        Args:
            index_path: Path to FAISS index file
            docs_path: Path to documents file
        """
        try:
            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                logger.warning("Vector store files not found")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents and metadata
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
            
            logger.info(f"Vector store loaded with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'index_size': self.index.ntotal
        }


class RAGPipeline:
    """Complete RAG pipeline combining embeddings and vector store"""
    
    def __init__(self, embedder_model, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize RAG pipeline
        
        Args:
            embedder_model: Instance of EmbedderModel
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.embedder = embedder_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        dimension = embedder_model.get_embedding_dimension()
        if dimension:
            self.vector_store = VectorStore(dimension=dimension)
        else:
            logger.error("Could not get embedding dimension")
            self.vector_store = None
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a document to the RAG system
        
        Args:
            text: Document text
            metadata: Optional metadata about the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty document provided")
                return False
            
            # Chunk the text
            chunks = self._chunk_text(text)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.embedder.embed_batch(chunks)
            if not embeddings:
                return False
            
            # Prepare metadata for each chunk
            chunk_metadata = [metadata or {} for _ in chunks]
            
            # Add to vector store
            self.vector_store.add_documents(chunks, embeddings, chunk_metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to RAG: {str(e)}")
            return False
    
    def query(self, query_text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Query the RAG system
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (context_text, relevance_score)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query_text)
            if not query_embedding:
                return []
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k)
            
            # Return context and scores
            return [(doc, score) for doc, score, _ in results]
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return []
    
    def clear(self):
        """Clear the RAG system"""
        if self.vector_store:
            self.vector_store.clear()