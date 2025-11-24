"""
Data validation utilities using Pydantic.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from config.config import config


class Filing(BaseModel):
    """10-K filing metadata."""
    ticker: str = Field(..., min_length=1, max_length=10)
    cik: str = Field(..., min_length=10, max_length=10)
    accession: str = Field(..., pattern=r'^\d{10}-\d{2}-\d{6}$')
    filing_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    primary_doc: Optional[str] = None

    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator('cik')
    @classmethod
    def validate_cik(cls, v: str) -> str:
        return v.zfill(10)

    class Config:
        frozen = True


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    chunk_id: str
    ticker: str = Field(..., min_length=1, max_length=10)
    file_name: str
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., gt=0)
    section: Optional[str] = None
    filing_date: Optional[str] = None

    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator('chunk_index', 'total_chunks')
    @classmethod
    def validate_chunk_numbers(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Chunk numbers must be non-negative")
        return v

    class Config:
        extra = 'forbid'


class Chunk(BaseModel):
    """Text chunk with metadata."""
    text: str = Field(..., min_length=10)
    metadata: ChunkMetadata

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class EmbeddingData(BaseModel):
    """Embedding vector with metadata."""
    id: str
    embedding: List[float] = Field(..., min_length=config.EMBEDDING_DIM)
    metadata: Dict[str, Any]

    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimension(cls, v: List[float]) -> List[float]:
        if len(v) != config.EMBEDDING_DIM:
            raise ValueError(
                f"Expected {config.EMBEDDING_DIM} dimensions, got {len(v)}"
            )
        return v

    class Config:
        arbitrary_types_allowed = True


class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    ticker_filter: Optional[str] = None
    section_filter: Optional[str] = None

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator('ticker_filter')
    @classmethod
    def ticker_uppercase(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else None


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: ChunkMetadata

    class Config:
        frozen = True


class SearchResponse(BaseModel):
    """Search API response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

    class Config:
        frozen = True
