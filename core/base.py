"""
Abstract base classes for the AI Embedding Framework.
Defines interfaces for embedding providers, chunkers, vector stores, and document parsers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


@dataclass
class EmbeddingResult:
    """Represents an embedding result with vector and metadata."""
    embedding: List[float]
    chunk: Chunk
    model_name: str
    embedding_id: Optional[str] = None


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""
    chunk: Chunk
    score: float
    embedding_id: Optional[str] = None


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding provider with configuration."""
        self.config = config
        self.model_name = config.get('model_name', 'default')
        self.max_tokens = config.get('max_tokens', 8192)
        self.batch_size = config.get('batch_size', 100)
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass
    
    def validate_text_length(self, text: str) -> bool:
        """
        Validate if text length is within model limits.
        
        Args:
            text: Input text to validate
            
        Returns:
            True if text is valid, False otherwise
        """
        # Basic token estimation (rough approximation)
        estimated_tokens = len(text.split()) * 1.3
        return estimated_tokens <= self.max_tokens


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the chunker with configuration."""
        self.config = config
        self.chunk_size = config.get('chunk_size', 1000)
        self.overlap = config.get('overlap', 200)
        self.preserve_structure = config.get('preserve_structure', True)
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any], 
                     start_index: int = None, end_index: int = None) -> Chunk:
        """
        Helper method to create a Chunk object.
        
        Args:
            content: Chunk content
            metadata: Chunk metadata
            start_index: Start position in original text
            end_index: End position in original text
            
        Returns:
            Chunk object
        """
        return Chunk(
            content=content,
            metadata=metadata,
            start_index=start_index,
            end_index=end_index
        )


class BaseVectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store with configuration."""
        self.config = config
        self.collection_name = config.get('collection_name', 'default')
        self.dimension = config.get('dimension', 1536)
    
    @abstractmethod
    async def add_embeddings(self, embeddings: List[EmbeddingResult]) -> List[str]:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of EmbeddingResult objects
            
        Returns:
            List of embedding IDs
        """
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    async def delete_embeddings(self, embedding_ids: List[str]) -> bool:
        """
        Delete embeddings by IDs.
        
        Args:
            embedding_ids: List of embedding IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def update_embedding(self, embedding_id: str, 
                              embedding_result: EmbeddingResult) -> bool:
        """
        Update an existing embedding.
        
        Args:
            embedding_id: ID of embedding to update
            embedding_result: New embedding data
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        pass


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document parser with configuration."""
        self.config = config
        self.extract_metadata = config.get('extract_metadata', True)
        self.preserve_formatting = config.get('preserve_formatting', False)
    
    @abstractmethod
    def parse_file(self, file_path: str) -> Document:
        """
        Parse a file and extract text content.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Document object with extracted content and metadata
        """
        pass
    
    @abstractmethod
    def parse_bytes(self, file_bytes: bytes, filename: str = None) -> Document:
        """
        Parse file bytes and extract text content.
        
        Args:
            file_bytes: File content as bytes
            filename: Optional filename for metadata
            
        Returns:
            Document object with extracted content and metadata
        """
        pass
    
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.txt'])
        """
        pass
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if parser can handle the file, False otherwise
        """
        extension = file_path.lower().split('.')[-1]
        return f'.{extension}' in self.supported_extensions()


class EmbeddingFrameworkError(Exception):
    """Base exception for embedding framework errors."""
    pass


class ProviderError(EmbeddingFrameworkError):
    """Exception raised by embedding providers."""
    pass


class ChunkerError(EmbeddingFrameworkError):
    """Exception raised by chunkers."""
    pass


class VectorStoreError(EmbeddingFrameworkError):
    """Exception raised by vector stores."""
    pass


class ParserError(EmbeddingFrameworkError):
    """Exception raised by document parsers."""
    pass