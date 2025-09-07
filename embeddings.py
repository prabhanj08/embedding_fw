"""
Main EmbeddingSystem class for the AI Embedding Framework.
Orchestrates all components including providers, chunkers, parsers, and vector stores.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .core.config import EmbeddingConfig, create_default_config
from .core.processor import DocumentProcessor, BatchProcessor
from .core.base import (
    BaseEmbeddingProvider, BaseChunker, BaseVectorStore, BaseDocumentParser,
    EmbeddingFrameworkError, SearchResult
)

# Import providers
from .providers.openai_provider import OpenAIProvider, AzureOpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.huggingface_provider import HuggingFaceProvider, LocalEmbeddingProvider
from .providers.cohere_provider import CohereProvider, CohereSearchProvider, CohereMultilingualProvider

# Import parsers
from .parsers.text_parser import TextParser, CSVParser, JSONParser
from .parsers.pdf_parser import PDFParser
from .parsers.ppt_parser import PowerPointParser

logger = logging.getLogger(__name__)


class EmbeddingSystem:
    """
    Main embedding system that orchestrates all components.
    Provides a high-level interface for document processing and embedding operations.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], EmbeddingConfig]] = None):
        """
        Initialize the embedding system.
        
        Args:
            config: Configuration dictionary or EmbeddingConfig object
        """
        # Initialize configuration
        if config is None:
            self.config = EmbeddingConfig(create_default_config())
        elif isinstance(config, dict):
            self.config = EmbeddingConfig(config)
        else:
            self.config = config
        
        # Initialize components
        self.embedding_provider = None
        self.chunker = None
        self.vector_store = None
        self.parsers = {}
        self.processor = None
        self.batch_processor = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize all components
        self._initialize_components()
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'EmbeddingSystem':
        """
        Create EmbeddingSystem from configuration file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            EmbeddingSystem instance
        """
        config = EmbeddingConfig.from_file(config_path)
        return cls(config)
    
    @classmethod
    def from_env(cls) -> 'EmbeddingSystem':
        """
        Create EmbeddingSystem from environment variables.
        
        Returns:
            EmbeddingSystem instance
        """
        config = EmbeddingConfig.from_env()
        return cls(config)
    
    @classmethod
    def quick_setup(cls, 
                   provider: str = 'openai',
                   model: str = None,
                   api_key: str = None,
                   vector_store: str = 'chroma',
                   **kwargs) -> 'EmbeddingSystem':
        """
        Quick setup with minimal configuration.
        
        Args:
            provider: Embedding provider ('openai', 'huggingface', 'cohere', 'anthropic')
            model: Model name (uses provider default if not specified)
            api_key: API key for the provider
            vector_store: Vector store type ('chroma', 'pinecone', 'opensearch')
            **kwargs: Additional configuration options
            
        Returns:
            EmbeddingSystem instance
        """
        config = create_default_config()
        
        # Update provider settings
        config['provider']['provider'] = provider
        if model:
            config['provider']['model_name'] = model
        if api_key:
            config['provider']['api_key'] = api_key
        
        # Update vector store settings
        config['vector_store']['store_type'] = vector_store
        
        # Apply additional kwargs
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'provider.batch_size'
                parts = key.split('.')
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value
        
        return cls(config)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        processing_config = self.config.get_processing_config()
        log_level = getattr(logging, processing_config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize embedding provider
            self.embedding_provider = self._create_embedding_provider()
            
            # Initialize chunker (placeholder - will be implemented next)
            self.chunker = self._create_chunker()
            
            # Initialize vector store (placeholder - will be implemented next)
            self.vector_store = self._create_vector_store()
            
            # Initialize parsers
            self.parsers = self._create_parsers()
            
            # Initialize processor
            self.processor = DocumentProcessor(
                embedding_provider=self.embedding_provider,
                chunker=self.chunker,
                vector_store=self.vector_store,
                parsers=self.parsers,
                config=self.config
            )
            
            # Initialize batch processor
            self.batch_processor = BatchProcessor(self.processor)
            
            logger.info("EmbeddingSystem initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingSystem: {e}")
            raise EmbeddingFrameworkError(f"System initialization failed: {e}")
    
    def _create_embedding_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider based on configuration."""
        provider_config = self.config.get_provider_config()
        
        provider_map = {
            'openai': OpenAIProvider,
            'azure_openai': AzureOpenAIProvider,
            'anthropic': AnthropicProvider,
            'huggingface': HuggingFaceProvider,
            'cohere': CohereProvider,
            'cohere_search': CohereSearchProvider,
            'cohere_multilingual': CohereMultilingualProvider,
            'local': LocalEmbeddingProvider
        }
        
        provider_class = provider_map.get(provider_config.provider)
        if not provider_class:
            raise EmbeddingFrameworkError(f"Unknown embedding provider: {provider_config.provider}")
        
        # Convert config to dictionary
        config_dict = {
            'model_name': provider_config.model_name,
            'api_key': provider_config.api_key,
            'api_base': provider_config.api_base,
            'max_tokens': provider_config.max_tokens,
            'batch_size': provider_config.batch_size,
            'timeout': provider_config.timeout,
            'retry_attempts': provider_config.retry_attempts,
            **provider_config.extra_params
        }
        
        return provider_class(config_dict)
    
    def _create_chunker(self) -> BaseChunker:
        """Create chunker based on configuration."""
        # Placeholder - will implement chunkers next
        from .chunkers.recursive_chunker import RecursiveChunker
        
        chunker_config = self.config.get_chunker_config()
        config_dict = {
            'chunk_size': chunker_config.chunk_size,
            'overlap': chunker_config.overlap,
            'preserve_structure': chunker_config.preserve_structure,
            'min_chunk_size': chunker_config.min_chunk_size,
            'max_chunk_size': chunker_config.max_chunk_size,
            'separators': chunker_config.separators,
            **chunker_config.extra_params
        }
        
        return RecursiveChunker(config_dict)
    
    def _create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on configuration."""
        # Placeholder - will implement vector stores next
        from .vector_stores.chroma_store import ChromaStore
        
        store_config = self.config.get_vector_store_config()
        config_dict = {
            'collection_name': store_config.collection_name,
            'dimension': store_config.dimension,
            'metric': store_config.metric,
            'host': store_config.host,
            'port': store_config.port,
            'api_key': store_config.api_key,
            'index_name': store_config.index_name,
            **store_config.extra_params
        }
        
        return ChromaStore(config_dict)
    
    def _create_parsers(self) -> Dict[str, BaseDocumentParser]:
        """Create document parsers."""
        parser_config = self.config.get_parser_config()
        config_dict = {
            'extract_metadata': parser_config.extract_metadata,
            'preserve_formatting': parser_config.preserve_formatting,
            'ocr_enabled': parser_config.ocr_enabled,
            'extract_images': parser_config.extract_images,
            'extract_tables': parser_config.extract_tables,
            'language': parser_config.language,
            **parser_config.extra_params
        }
        
        parsers = {}
        
        # Text parser
        text_parser = TextParser(config_dict)
        for ext in text_parser.supported_extensions():
            parsers[ext] = text_parser
        
        # CSV parser
        csv_parser = CSVParser(config_dict)
        for ext in csv_parser.supported_extensions():
            parsers[ext] = csv_parser
        
        # JSON parser
        json_parser = JSONParser(config_dict)
        for ext in json_parser.supported_extensions():
            parsers[ext] = json_parser
        
        # PDF parser
        try:
            pdf_parser = PDFParser(config_dict)
            for ext in pdf_parser.supported_extensions():
                parsers[ext] = pdf_parser
        except Exception as e:
            logger.warning(f"PDF parser not available: {e}")
        
        # PowerPoint parser
        try:
            ppt_parser = PowerPointParser(config_dict)
            for ext in ppt_parser.supported_extensions():
                parsers[ext] = ppt_parser
        except Exception as e:
            logger.warning(f"PowerPoint parser not available: {e}")
        
        return parsers
    
    # High-level API methods
    
    async def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process raw text and return embedding IDs.
        
        Args:
            text: Input text to process
            metadata: Optional metadata to attach
            
        Returns:
            List of embedding IDs
        """
        return await self.processor.process_text(text, metadata)
    
    async def process_file(self, file_path: Union[str, Path], 
                          metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process a file and return embedding IDs.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata to attach
            
        Returns:
            List of embedding IDs
        """
        return await self.processor.process_file(file_path, metadata)
    
    async def process_files(self, file_paths: List[Union[str, Path]], 
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            metadata: Optional metadata to attach to all files
            
        Returns:
            Dictionary mapping file paths to embedding IDs
        """
        return await self.processor.process_files_batch(file_paths, metadata)
    
    async def process_directory(self, directory_path: Union[str, Path], 
                               recursive: bool = True,
                               file_patterns: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            file_patterns: Optional list of file patterns to match
            metadata: Optional metadata to attach to all files
            
        Returns:
            Dictionary mapping file paths to embedding IDs
        """
        return await self.batch_processor.process_directory(
            directory_path, recursive, file_patterns, metadata
        )
    
    async def search(self, query: str, top_k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for similar content.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        return await self.processor.search(query, top_k, filters)
    
    async def similarity_search(self, text: str, top_k: int = 10) -> List[SearchResult]:
        """
        Find similar content to the given text.
        
        Args:
            text: Input text to find similar content for
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        return await self.search(text, top_k)
    
    # Utility methods
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.processor.get_stats()
        stats['system_info'] = {
            'provider': self.config.get_provider_config().provider,
            'model': self.config.get_provider_config().model_name,
            'chunker': self.config.get_chunker_config().strategy,
            'vector_store': self.config.get_vector_store_config().store_type
        }
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processor.reset_stats()
    
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'config': {
                'provider': self.config.get_provider_config().__dict__,
                'chunker': self.config.get_chunker_config().__dict__,
                'vector_store': self.config.get_vector_store_config().__dict__,
                'parser': self.config.get_parser_config().__dict__,
                'processing': self.config.get_processing_config().__dict__
            },
            'components': {
                'embedding_provider': await self.embedding_provider.get_model_info(),
                'vector_store': await self.vector_store.get_collection_stats(),
                'parsers': list(self.parsers.keys())
            },
            'stats': self.get_stats()
        }
        return info
    
    def save_config(self, config_path: Union[str, Path], format: str = 'yaml'):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        self.config.save_to_file(config_path, format)
    
    async def close(self):
        """Close the system and cleanup resources."""
        if self.processor:
            await self.processor.cleanup()
        
        logger.info("EmbeddingSystem closed")
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience functions for quick usage

async def embed_text(text: str, provider: str = 'openai', **kwargs) -> List[float]:
    """
    Quick function to embed a single text.
    
    Args:
        text: Text to embed
        provider: Embedding provider to use
        **kwargs: Additional configuration
        
    Returns:
        Embedding vector
    """
    system = EmbeddingSystem.quick_setup(provider=provider, **kwargs)
    try:
        embedding_ids = await system.process_text(text)
        # For quick usage, we'd need to return the actual embedding
        # This would require accessing the vector store directly
        return embedding_ids  # Placeholder
    finally:
        await system.close()


async def embed_file(file_path: Union[str, Path], provider: str = 'openai', **kwargs) -> List[str]:
    """
    Quick function to embed a file.
    
    Args:
        file_path: Path to file to embed
        provider: Embedding provider to use
        **kwargs: Additional configuration
        
    Returns:
        List of embedding IDs
    """
    system = EmbeddingSystem.quick_setup(provider=provider, **kwargs)
    try:
        return await system.process_file(file_path)
    finally:
        await system.close()


async def search_similar(query: str, top_k: int = 10, provider: str = 'openai', **kwargs) -> List[SearchResult]:
    """
    Quick function to search for similar content.
    
    Args:
        query: Search query
        top_k: Number of results to return
        provider: Embedding provider to use
        **kwargs: Additional configuration
        
    Returns:
        List of search results
    """
    system = EmbeddingSystem.quick_setup(provider=provider, **kwargs)
    try:
        return await system.search(query, top_k)
    finally:
        await system.close()


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Example usage
        system = EmbeddingSystem.quick_setup(
            provider='openai',
            api_key='your-api-key-here'
        )
        
        try:
            # Process some text
            embedding_ids = await system.process_text("This is a test document.")
            print(f"Generated {len(embedding_ids)} embeddings")
            
            # Search for similar content
            results = await system.search("test document", top_k=5)
            print(f"Found {len(results)} similar documents")
            
            # Get system stats
            stats = system.get_stats()
            print(f"System stats: {stats}")
            
        finally:
            await system.close()
    
    # Run example
    # asyncio.run(main())
    print("EmbeddingSystem module loaded successfully")