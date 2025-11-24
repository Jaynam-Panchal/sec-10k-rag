"""
Custom exceptions for the 10K RAG system.
"""

class RAGSystemError(Exception):
    """Base exception for all RAG system errors."""
    pass


class ConfigurationError(RAGSystemError):
    """Raised when configuration is invalid."""
    pass


class SECAPIError(RAGSystemError):
    """Raised when SEC API requests fail."""
    pass


class DownloadError(RAGSystemError):
    """Raised when document download fails."""
    pass


class DocumentProcessingError(RAGSystemError):
    """Raised when document processing fails."""
    pass


class ParsingError(DocumentProcessingError):
    """Raised when document parsing fails."""
    pass


class ChunkingError(DocumentProcessingError):
    """Raised when text chunking fails."""
    pass


class EmbeddingError(RAGSystemError):
    """Raised when embedding generation fails."""
    pass


class IndexError(RAGSystemError):
    """Raised when FAISS index operations fail."""
    pass


class SearchError(RAGSystemError):
    """Raised when search operations fail."""
    pass


class ValidationError(RAGSystemError):
    """Raised when data validation fails."""
    pass