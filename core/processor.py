"""
Document processor for the AI Embedding Framework.
Orchestrates the entire pipeline from document parsing to vector storage.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .base import (
    Document, Chunk, EmbeddingResult, SearchResult,
    BaseEmbeddingProvider, BaseChunker, BaseVectorStore, BaseDocumentParser,
    EmbeddingFrameworkError
)
from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main processor that orchestrates the embedding pipeline."""
    
    def __init__(self, 
                 embedding_provider: BaseEmbeddingProvider,
                 chunker: BaseChunker,
                 vector_store: BaseVectorStore,
                 parsers: Dict[str, BaseDocumentParser],
                 config: EmbeddingConfig):
        """
        Initialize the document processor.
        
        Args:
            embedding_provider: Provider for generating embeddings
            chunker: Strategy for chunking text
            vector_store: Backend for storing vectors
            parsers: Dictionary mapping file extensions to parsers
            config: Configuration object
        """
        self.embedding_provider = embedding_provider
        self.chunker = chunker
        self.vector_store = vector_store
        self.parsers = parsers
        self.config = config
        self.processing_config = config.get_processing_config()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.processing_config.max_workers)
        
        # Statistics tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    async def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process raw text through the embedding pipeline.
        
        Args:
            text: Input text to process
            metadata: Optional metadata to attach
            
        Returns:
            List of embedding IDs
        """
        start_time = time.time()
        
        try:
            # Create document
            document = Document(
                content=text,
                metadata=metadata or {},
                doc_id=str(uuid.uuid4()),
                source="text_input"
            )
            
            # Chunk the document
            chunks = self.chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Generate embeddings and store
            embedding_ids = await self._process_chunks(chunks)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(embedding_ids)
            self.stats['processing_time'] += time.time() - start_time
            
            return embedding_ids
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error processing text: {e}")
            raise EmbeddingFrameworkError(f"Failed to process text: {e}")
    
    async def process_file(self, file_path: Union[str, Path], 
                          metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process a file through the embedding pipeline.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata to attach
            
        Returns:
            List of embedding IDs
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        try:
            # Find appropriate parser
            parser = self._get_parser_for_file(file_path)
            if not parser:
                raise EmbeddingFrameworkError(f"No parser available for file: {file_path}")
            
            # Parse the document
            document = parser.parse_file(str(file_path))
            
            # Add file metadata
            if metadata:
                document.metadata.update(metadata)
            document.metadata.update({
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix
            })
            
            # Chunk the document
            chunks = self.chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # Generate embeddings and store
            embedding_ids = await self._process_chunks(chunks)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(embedding_ids)
            self.stats['processing_time'] += time.time() - start_time
            
            return embedding_ids
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error processing file {file_path}: {e}")
            raise EmbeddingFrameworkError(f"Failed to process file {file_path}: {e}")
    
    async def process_files_batch(self, file_paths: List[Union[str, Path]], 
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            metadata: Optional metadata to attach to all files
            
        Returns:
            Dictionary mapping file paths to embedding IDs
        """
        results = {}
        batch_size = self.processing_config.batch_size
        
        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batch_tasks = []
            
            for file_path in batch:
                task = self.process_file(file_path, metadata)
                batch_tasks.append((str(file_path), task))
            
            # Wait for batch completion
            for file_path, task in batch_tasks:
                try:
                    embedding_ids = await task
                    results[file_path] = embedding_ids
                    logger.info(f"Successfully processed {file_path}")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results[file_path] = []
        
        return results
    
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
        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.embed_text(query)
            
            # Search in vector store
            results = await self.vector_store.search(query_embedding, top_k, filters)
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise EmbeddingFrameworkError(f"Search failed: {e}")
    
    async def _process_chunks(self, chunks: List[Chunk]) -> List[str]:
        """
        Process chunks through embedding and storage pipeline.
        
        Args:
            chunks: List of chunks to process
            
        Returns:
            List of embedding IDs
        """
        if not chunks:
            return []
        
        # Generate embeddings in batches
        embedding_results = []
        batch_size = self.embedding_provider.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            try:
                # Generate embeddings for batch
                embeddings = await self.embedding_provider.embed_batch(batch_texts)
                
                # Create embedding results
                for chunk, embedding in zip(batch_chunks, embeddings):
                    result = EmbeddingResult(
                        embedding=embedding,
                        chunk=chunk,
                        model_name=self.embedding_provider.model_name,
                        embedding_id=str(uuid.uuid4())
                    )
                    embedding_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Try individual processing for failed batch
                for chunk in batch_chunks:
                    try:
                        embedding = await self.embedding_provider.embed_text(chunk.content)
                        result = EmbeddingResult(
                            embedding=embedding,
                            chunk=chunk,
                            model_name=self.embedding_provider.model_name,
                            embedding_id=str(uuid.uuid4())
                        )
                        embedding_results.append(result)
                    except Exception as individual_error:
                        logger.error(f"Failed to embed individual chunk: {individual_error}")
        
        # Store embeddings
        if embedding_results:
            try:
                embedding_ids = await self.vector_store.add_embeddings(embedding_results)
                logger.info(f"Stored {len(embedding_ids)} embeddings")
                return embedding_ids
            except Exception as e:
                logger.error(f"Error storing embeddings: {e}")
                raise EmbeddingFrameworkError(f"Failed to store embeddings: {e}")
        
        return []
    
    def _get_parser_for_file(self, file_path: Path) -> Optional[BaseDocumentParser]:
        """
        Get the appropriate parser for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser instance or None if no parser available
        """
        extension = file_path.suffix.lower()
        
        # Try exact extension match first
        if extension in self.parsers:
            return self.parsers[extension]
        
        # Try to find a parser that can handle this file
        for parser in self.parsers.values():
            if parser.can_parse(str(file_path)):
                return parser
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.stats.copy()
        if stats['documents_processed'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['documents_processed']
            stats['avg_chunks_per_document'] = stats['chunks_created'] / stats['documents_processed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['avg_chunks_per_document'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'processing_time': 0.0,
            'errors': 0
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.vector_store, 'close'):
            await self.vector_store.close()
        
        if hasattr(self.embedding_provider, 'close'):
            await self.embedding_provider.close()
        
        self.executor.shutdown(wait=True)
        logger.info("Document processor cleanup completed")


class BatchProcessor:
    """Utility class for batch processing operations."""
    
    def __init__(self, processor: DocumentProcessor):
        """
        Initialize batch processor.
        
        Args:
            processor: Document processor instance
        """
        self.processor = processor
    
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
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all files
        files = []
        if recursive:
            for pattern in (file_patterns or ['*']):
                files.extend(directory_path.rglob(pattern))
        else:
            for pattern in (file_patterns or ['*']):
                files.extend(directory_path.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        logger.info(f"Found {len(files)} files to process in {directory_path}")
        
        # Process files
        return await self.processor.process_files_batch(files, metadata)
    
    async def reprocess_failed(self, failed_files: List[str], 
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Reprocess files that failed in previous batch.
        
        Args:
            failed_files: List of file paths that failed
            metadata: Optional metadata to attach
            
        Returns:
            Dictionary mapping file paths to embedding IDs
        """
        logger.info(f"Reprocessing {len(failed_files)} failed files")
        return await self.processor.process_files_batch(failed_files, metadata)