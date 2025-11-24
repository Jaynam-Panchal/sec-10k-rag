"""
FastAPI application for 10K RAG system.
"""
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

from config.config import config
from utils.logger import setup_logger
from data_pipeline.build_faiss_index import FAISSIndexLoader
from data_pipeline.chunk_and_embed import EmbeddingGenerator

logger = setup_logger(__name__, config.LOGS_DIR / "api.log")

# Global variables for loaded models
index_loader = None
embedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global index_loader, embedder
    
    logger.info("Starting application...")
    
    try:
        # Load FAISS index
        logger.info("Loading FAISS index...")
        index_loader = FAISSIndexLoader()
        index_loader.load()
        logger.info("✓ Index loaded")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embedder = EmbeddingGenerator()
        logger.info("✓ Embedding model loaded")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="10K Financial Document RAG API",
    description="Semantic search over SEC 10-K filings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    ticker_filter: Optional[str] = Field(default=None, description="Filter by ticker")
    section_filter: Optional[str] = Field(default=None, description="Filter by section")


class SearchResultItem(BaseModel):
    """Single search result."""
    chunk_id: str
    text: str
    score: float
    ticker: str
    section: Optional[str]
    file_name: str
    chunk_index: int


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_loaded: bool
    model_loaded: bool
    n_vectors: Optional[int] = None


class StatsResponse(BaseModel):
    """Statistics response."""
    total_vectors: int
    dimension: int
    index_type: str
    companies: List[str]


# API Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "10K Financial Document RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        index_loaded=index_loader is not None and index_loader.index is not None,
        model_loaded=embedder is not None,
        n_vectors=index_loader.index.ntotal if index_loader and index_loader.index else None
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not index_loader or not index_loader.index:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Extract unique companies from metadata
    companies = list(set(
        meta.get('ticker', 'UNKNOWN') 
        for meta in index_loader.metadatas
    ))
    companies.sort()
    
    return StatsResponse(
        total_vectors=index_loader.index.ntotal,
        dimension=index_loader.index.d,
        index_type=config.FAISS_INDEX_TYPE,
        companies=companies
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic search over 10-K documents.
    
    Args:
        request: Search request with query and filters
        
    Returns:
        Search results with relevant chunks
    """
    if not index_loader or not index_loader.index:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    if not embedder:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    start_time = time.time()
    
    try:
        logger.info(f"Search query: '{request.query}' (top_k={request.top_k})")
        
        # Generate query embedding
        query_embedding = embedder.embed_texts([request.query])[0]
        
        # Search index (get more results for filtering)
        search_k = request.top_k * 3 if request.ticker_filter or request.section_filter else request.top_k
        distances, indices, metadatas = index_loader.search(
            query_embedding, 
            top_k=search_k
        )
        
        # Load full chunk data and apply filters
        results = []
        for score, idx, meta in zip(distances, indices, metadatas):
            # Apply filters
            if request.ticker_filter and meta.get('ticker') != request.ticker_filter.upper():
                continue
            
            if request.section_filter and request.section_filter.lower() not in meta.get('section', '').lower():
                continue
            
            # Load chunk text
            chunk_id = index_loader.ids[idx]
            chunk_file = config.CHUNKS_DIR / f"{chunk_id}.json"
            
            try:
                import json
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                
                result = SearchResultItem(
                    chunk_id=chunk_id,
                    text=chunk_data['text'],
                    score=float(score),
                    ticker=meta.get('ticker', 'UNKNOWN'),
                    section=meta.get('section'),
                    file_name=meta.get('file_name', ''),
                    chunk_index=meta.get('chunk_index', 0)
                )
                results.append(result)
                
                if len(results) >= request.top_k:
                    break
                    
            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_id}: {e}")
                continue
        
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"Found {len(results)} results in {search_time:.2f}ms")
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/companies", response_model=List[str])
async def get_companies():
    """Get list of available companies."""
    if not index_loader or not index_loader.metadatas:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    companies = list(set(
        meta.get('ticker', 'UNKNOWN') 
        for meta in index_loader.metadatas
    ))
    companies.sort()
    
    return companies


@app.get("/sections/{ticker}", response_model=List[str])
async def get_sections(ticker: str):
    """Get available sections for a ticker."""
    if not index_loader or not index_loader.metadatas:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    ticker = ticker.upper()
    
    sections = list(set(
        meta.get('section', 'UNKNOWN')
        for meta in index_loader.metadatas
        if meta.get('ticker') == ticker and meta.get('section')
    ))
    sections.sort()
    
    if not sections:
        raise HTTPException(status_code=404, detail=f"No sections found for {ticker}")
    
    return sections


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        log_level="info",
        reload=False
    )