"""
Core module for the AI Embedding Framework.
Contains base classes and core functionality.
"""

from .base import BaseEmbeddingProvider, BaseChunker, BaseVectorStore, BaseDocumentParser
from .processor import DocumentProcessor
from .config import EmbeddingConfig

__all__ = [
    'BaseEmbeddingProvider',
    'BaseChunker', 
    'BaseVectorStore',
    'BaseDocumentParser',
    'DocumentProcessor',
    'EmbeddingConfig'
]