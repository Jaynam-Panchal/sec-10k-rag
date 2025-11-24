"""
Centralized configuration for the 10K RAG system.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass
class Config:
    """Application configuration."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_10K_DIR: Path = DATA_DIR / "raw_10k"
    CLEAN_TXT_DIR: Path = DATA_DIR / "clean_txt"
    CHUNKS_DIR: Path = DATA_DIR / "chunks"
    INDEX_DIR: Path = DATA_DIR / "index"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # SEC API Configuration
    SEC_BASE_URL: str = "https://data.sec.gov"
    SEC_ARCHIVE_URL: str = "https://www.sec.gov/Archives/edgar/data"
    USER_AGENT: str = os.getenv(
        "SEC_USER_AGENT", 
        "MyCompany/1.0 (contact@example.com)"
    )
    RATE_LIMIT_DELAY: float = 0.3
    REQUEST_TIMEOUT: int = 15
    MAX_RETRIES: int = 3
    
    # Processing Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_WORKERS: int = 4
    MAX_FILE_SIZE: int = 500_000  # 500KB
    NUM_YEARS: int = 3
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    EMBEDDING_DIM: int = 768
    BATCH_SIZE: int = 32
    
    # FAISS Configuration
    FAISS_INDEX_TYPE: str = "IndexFlatIP"  # Inner product
    TOP_K_RESULTS: int = 5
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # Company Tickers
    TICKER_TO_CIK: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize directories and load tickers."""
        # Create all required directories
        for dir_path in [
            self.RAW_10K_DIR, 
            self.CLEAN_TXT_DIR,
            self.CHUNKS_DIR, 
            self.INDEX_DIR,
            self.LOGS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Default tickers if not loaded from file
        if self.TICKER_TO_CIK is None:
            self.TICKER_TO_CIK = {
                'AAPL': '320193',
                'MSFT': '789019',
                'GOOGL': '1652044',
                'AMZN': '1018724',
                'NVDA': '1045810',
                'META': '1326801',
                'TSLA': '1318605',
                'JNJ': '200406',
                'V': '1403161',
                'WMT': '104169',
                'JPM': '19617',
                'XOM': '34088',
                'PG': '80424',
                'MA': '1141391',
            }

# Global config instance
config = Config()