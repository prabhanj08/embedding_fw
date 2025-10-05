"""
Configuration management for the AI Embedding Framework.
Handles loading and validation of configuration settings.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for embedding providers."""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 8192
    batch_size: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkerConfig:
    """Configuration for text chunkers."""
    strategy: str
    chunk_size: int = 1000
    overlap: int = 200
    preserve_structure: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    separators: Optional[list] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""
    store_type: str
    collection_name: str = "default"
    dimension: int = 1536
    metric: str = "cosine"
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    index_name: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParserConfig:
    """Configuration for document parsers."""
    extract_metadata: bool = True
    preserve_formatting: bool = False
    ocr_enabled: bool = False
    extract_images: bool = False
    extract_tables: bool = True
    language: str = "en"
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    batch_size: int = 10
    max_workers: int = 4
    timeout: int = 300
    enable_caching: bool = True
    cache_dir: str = ".embedding_cache"
    log_level: str = "INFO"


class EmbeddingConfig:
    """Main configuration class for the embedding framework."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from dictionary or environment variables.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        self._config = config_dict or {}
        self._load_environment_variables()
        self._validate_config()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'EmbeddingConfig':
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            EmbeddingConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            return cls(config_dict)
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """
        Create configuration from environment variables only.
        
        Returns:
            EmbeddingConfig instance
        """
        return cls({})
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Provider settings
            'EMBEDDING_PROVIDER': ('provider', 'provider'),
            'EMBEDDING_MODEL': ('provider', 'model_name'),
            'OPENAI_API_KEY': ('provider', 'api_key'),
            'ANTHROPIC_API_KEY': ('provider', 'api_key'),
            'COHERE_API_KEY': ('provider', 'api_key'),
            'HF_API_KEY': ('provider', 'api_key'),
            
            # Vector store settings
            'VECTOR_STORE_TYPE': ('vector_store', 'store_type'),
            'VECTOR_STORE_HOST': ('vector_store', 'host'),
            'VECTOR_STORE_PORT': ('vector_store', 'port'),
            'PINECONE_API_KEY': ('vector_store', 'api_key'),
            'OPENSEARCH_HOST': ('vector_store', 'host'),
            
            # Chunker settings
            'CHUNKER_STRATEGY': ('chunker', 'strategy'),
            'CHUNK_SIZE': ('chunker', 'chunk_size'),
            'CHUNK_OVERLAP': ('chunker', 'overlap'),
            
            # Processing settings
            'BATCH_SIZE': ('processing', 'batch_size'),
            'MAX_WORKERS': ('processing', 'max_workers'),
            'LOG_LEVEL': ('processing', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                
                # Type conversion
                if key in ['chunk_size', 'overlap', 'batch_size', 'max_workers', 'port']:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                elif key in ['preserve_structure', 'extract_metadata', 'enable_caching']:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                
                self._config[section][key] = value
    
    def _validate_config(self):
        """Validate the configuration."""
        # Set defaults if not provided
        if 'provider' not in self._config:
            self._config['provider'] = {}
        if 'chunker' not in self._config:
            self._config['chunker'] = {}
        if 'vector_store' not in self._config:
            self._config['vector_store'] = {}
        if 'parser' not in self._config:
            self._config['parser'] = {}
        if 'processing' not in self._config:
            self._config['processing'] = {}
        
        # Validate required fields
        provider_config = self._config['provider']
        if 'provider' not in provider_config:
            logger.warning("No embedding provider specified, defaulting to 'openai'")
            provider_config['provider'] = 'openai'
        
        vector_store_config = self._config['vector_store']
        if 'store_type' not in vector_store_config:
            logger.warning("No vector store type specified, defaulting to 'chroma'")
            vector_store_config['store_type'] = 'chroma'
        
        chunker_config = self._config['chunker']
        if 'strategy' not in chunker_config:
            logger.warning("No chunker strategy specified, defaulting to 'recursive'")
            chunker_config['strategy'] = 'recursive'
    
    def get_provider_config(self) -> ProviderConfig:
        """Get provider configuration."""
        config = self._config.get('provider', {})
        return ProviderConfig(
            provider=config.get('provider', 'openai'),
            model_name=config.get('model_name', self._get_default_model(config.get('provider', 'openai'))),
            api_key=config.get('api_key'),
            api_base=config.get('api_base'),
            max_tokens=config.get('max_tokens', 8192),
            batch_size=config.get('batch_size', 100),
            timeout=config.get('timeout', 30),
            retry_attempts=config.get('retry_attempts', 3),
            extra_params=config.get('extra_params', {})
        )
    
    def get_chunker_config(self) -> ChunkerConfig:
        """Get chunker configuration."""
        config = self._config.get('chunker', {})
        return ChunkerConfig(
            strategy=config.get('strategy', 'recursive'),
            chunk_size=config.get('chunk_size', 1000),
            overlap=config.get('overlap', 200),
            preserve_structure=config.get('preserve_structure', True),
            min_chunk_size=config.get('min_chunk_size', 100),
            max_chunk_size=config.get('max_chunk_size', 2000),
            separators=config.get('separators'),
            extra_params=config.get('extra_params', {})
        )
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        config = self._config.get('vector_store', {})
        return VectorStoreConfig(
            store_type=config.get('store_type', 'chroma'),
            collection_name=config.get('collection_name', 'default'),
            dimension=config.get('dimension', 1536),
            metric=config.get('metric', 'cosine'),
            host=config.get('host'),
            port=config.get('port'),
            api_key=config.get('api_key'),
            index_name=config.get('index_name'),
            extra_params=config.get('extra_params', {})
        )
    
    def get_parser_config(self) -> ParserConfig:
        """Get parser configuration."""
        config = self._config.get('parser', {})
        return ParserConfig(
            extract_metadata=config.get('extract_metadata', True),
            preserve_formatting=config.get('preserve_formatting', False),
            ocr_enabled=config.get('ocr_enabled', False),
            extract_images=config.get('extract_images', False),
            extract_tables=config.get('extract_tables', True),
            language=config.get('language', 'en'),
            extra_params=config.get('extra_params', {})
        )
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        config = self._config.get('processing', {})
        return ProcessingConfig(
            batch_size=config.get('batch_size', 10),
            max_workers=config.get('max_workers', 4),
            timeout=config.get('timeout', 300),
            enable_caching=config.get('enable_caching', True),
            cache_dir=config.get('cache_dir', '.embedding_cache'),
            log_level=config.get('log_level', 'INFO')
        )
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            'openai': 'text-embedding-3-small',
            'anthropic': 'claude-3-haiku-20240307',
            'huggingface': 'sentence-transformers/all-MiniLM-L6-v2',
            'cohere': 'embed-english-v3.0'
        }
        return defaults.get(provider, 'text-embedding-3-small')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def save_to_file(self, config_path: Union[str, Path], format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        'provider': {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'batch_size': 100,
            'max_tokens': 8192
        },
        'chunker': {
            'strategy': 'recursive',
            'chunk_size': 1000,
            'overlap': 200,
            'preserve_structure': True
        },
        'vector_store': {
            'store_type': 'chroma',
            'collection_name': 'default',
            'dimension': 1536,
            'metric': 'cosine'
        },
        'parser': {
            'extract_metadata': True,
            'preserve_formatting': False,
            'extract_tables': True
        },
        'processing': {
            'batch_size': 10,
            'max_workers': 4,
            'enable_caching': True,
            'log_level': 'INFO'
        }
    }