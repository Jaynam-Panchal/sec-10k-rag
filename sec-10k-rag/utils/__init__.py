# ============================================
"""Utility functions and classes."""
from .logger import setup_logger, LoggerMixin
from .exceptions import (
    RAGSystemError,
    SECAPIError,
    DownloadError,
    DocumentProcessingError,
    ParsingError,
    ChunkingError,
    EmbeddingError,
    IndexError,
    SearchError,
    ValidationError,
    ConfigurationError
)
from .retry import retry_with_backoff, RetryContext
from .validation import (
    Filing,
    ChunkMetadata,
    Chunk,
    EmbeddingData,
    SearchQuery,
    SearchResult,
    SearchResponse
)

__all__ = [
    # Logger
    'setup_logger',
    'LoggerMixin',
    
    # Exceptions
    'RAGSystemError',
    'SECAPIError',
    'DownloadError',
    'DocumentProcessingError',
    'ParsingError',
    'ChunkingError',
    'EmbeddingError',
    'IndexError',
    'SearchError',
    'ValidationError',
    'ConfigurationError',
    
    # Retry
    'retry_with_backoff',
    'RetryContext',
    
    # Validation
    'Filing',
    'ChunkMetadata',
    'Chunk',
    'EmbeddingData',
    'SearchQuery',
    'SearchResult',
    'SearchResponse',
]
