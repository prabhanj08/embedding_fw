"""
AI Embedding Framework - A comprehensive text embedding system
supporting multiple providers, chunking strategies, and vector stores.
"""

from .embeddings import EmbeddingSystem
from .core.config import EmbeddingConfig

__version__ = "1.0.0"
__author__ = "AI Embedding Framework"

__all__ = [
    'EmbeddingSystem',
    'EmbeddingConfig'
]